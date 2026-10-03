# 基于标注框训练轨迹的无监督错误排序：设计与实施方案

## 0. 本文档的范围与目标

本方案只对**训练标注中实际存在的 anno** 排序。排序目标是区分 `clean` 与
`fault`；`fault` 包括类别错误、位置错误和冗余框，但排序器**不需要判断错误的具体类型**。
缺失标注在有噪声的训练标注中没有 anno，本方案不处理它，也不把图片或虚拟框混进
anno 排名。

希望检验的研究假设是：一个 anno 在训练过程中持续得到的**类别与位置联合支持**，
能否用于判断该 anno 是否有错。因此，方法的主体应当是

```text
每个 anno × 每个 epoch 的过程度量
              ↓
同一个 anno 的 50 轮训练轨迹
              ↓
过程特征矩阵
              ↓
不使用 clean/fault 真值的排序分数
```

`fault_type`、注错记录和原始正确标注只能用于**实验结束后的评价**，不能进入
候选框关联、特征构造、归一化分组或排序过程。所有默认阈值与权重应在查看目标实验的
错误标签之前确定。

下面首先给出一套能直接实现的 YOLOv7 版本；Faster R-CNN、RT-DETR 需要各自实现
统一的候选输出接口，再复用关联、特征和排序部分。

## 1. 当前方法的问题与数据边界

现有 [匹配与度量脚本](match_metrics/match_and_collect_metrics.py) 将预测框与 anno
按“类别相同、IoU ≥ 0.5”做正式匹配。只有发生过正式匹配的 anno 才写入
`metrics.json`；某一轮未匹配时，`conf_list` 和 `iou_list` 的对应位置为 0。
随后 [排序脚本](rank/rank.py) 对 50 轮均未匹配的 anno 补上完全相同的八个值：
六个 0、两个 1。于是这些 anno 的八维过程特征和 TOPSIS 分数相同。

以当前 VOC / YOLOv7 / 注错率 0.1 / repeat 10 的文件为例，13,609 个现存 anno
中有 4,973 个在 50 轮都没有正式匹配，其中 1,853 个是 clean。这是**该次实验的
观察结果**，不是对其他数据集或模型的普遍断言。检查当前已保存的最终预测框后，
上述 1,853 个 clean 中有 1,820 个在 50 轮里连一次与任意类别预测框的重叠都没有。
所以仅放宽当前匹配的 IoU 阈值，不能解决大多数零匹配 clean 的信息缺失。

YOLOv7 的 [收集脚本](../yolov7/collect_train_info.py) 实际执行顺序为：

```python
out, _ = model(imgs, augment=False)                 # 已解码、NMS 前的候选
out = non_max_suppression(out, 0.25, 0.65, ...)  # 筛选、NMS
# 只把此时剩余的框写入 epoch_*_predicted_bboxs.json
```

本方案使用第一行的 `out` 计算 anno 轨迹，而不是从已过滤的 JSON 中寻找候选。
可以复用现有 50 个 checkpoint 和数据加载器，只需**重新推理，无需重新训练**。

## 2. 核心概念：关联候选与正式匹配分开

### 2.1 两个术语

- **关联候选**：为一个 anno 在某个 epoch 从模型输出中选出的、最能代表该位置
  模型响应的候选。所有 anno 都应用同一选择规则；候选可以很弱，也可以预测成
  其他类别。它提供本文主方法的过程度量。
- **正式匹配**：现有方法定义的同类别、IoU ≥ 0.5 的一对一匹配。它可以作为
  诊断字段或消融特征，但**不得**决定这一轮改用另一套候选选择规则。

关联候选不声称是一次成功检测。如果模型在该位置没有目标，关联算法仍可能选到一个
得分很低的原始候选；这个低得分正是观测到的模型响应。因而文中应称其为
“anno 关联候选”，`conf` 应明确为“对 anno 标注类别的模型得分”，避免与 NMS 后
最终检测框的最高类别置信度混用。

### 2.2 为什么主过程不使用一对一分配

一对一分配会使没有足够正式检测框的 anno 再次成为空记录。主过程是逐 anno 测量
模型响应，允许同一个原始候选关联到两个相邻 anno。共享候选可能暗示重叠或冗余，
但仅凭共享不能判错：密集真实目标也可能共享原始候选。需要的话，可另存
`shared_candidate_count` 做后续消融，不在首版八个主特征中计分。

## 3. 每个 epoch 的输入与坐标约定

对 epoch `t`，加载该轮 checkpoint，使用和现有收集脚本相同的训练图片、推理
预处理、图像顺序和 `model.eval()`。每张图取得：

1. 该图的全部现存 anno：`id`、`image_id`、`category_id`、COCO 格式
   `bbox=[x,y,w,h]`。不得读取 `fault_type`。
2. 模型的过滤前候选。YOLOv7 的 `out[image_index]` 每行包含模型输入尺度上的
   解码框 `[cx,cy,w,h]`、objectness `o_j` 和各类别分数 `p_j(c)`。
3. 原图尺寸以及数据加载器返回的缩放和 padding 信息。

对每个原始候选 `j`，先把 `[cx,cy,w,h]` 变成 `xyxy`，再按现有收集脚本使用的
`scale_coords(...)` 还原到原图像素坐标；anno 的 COCO `[x,y,w,h]` 也变成原图
`xyxy`。裁剪越界坐标，并丢弃还原后宽或高不为正的候选。

**类别编号必须先核对。**当前所检查的 VOC 注错文件和 YOLOv7 预测都使用从 0
开始的类别编号，不能再减 1。若换数据集或模型，应显式提供类别映射，并在第一个
batch 用已知样本检查。不要沿用 Faster R-CNN 路径中的偏移逻辑。

对 YOLOv7，以下是同一原始候选的两种得分：

\[
q_j(a)=o_j\,p_j(y_a),\qquad
r_j=o_j\max_c p_j(c).
\]

这里 `y_a` 是 anno 的**标注类别**。`q_j(a)` 衡量该候选对该标注类别的支持；
`r_j` 衡量该候选是否强烈预测某个类别，不要求该类别与 anno 相同。两者均在
`[0,1]` 内。要使用 NMS 前的原始 `o_j` 和 `p_j(c)`，不能对已被 NMS 修改过的
类别分数再次乘 objectness。

## 4. 逐 anno、逐 epoch 的统一候选选择规则

### 4.1 空间接近度

记 anno 框为 `A`，候选框为 `B_j`；宽高分别为 `w_A,h_A` 与 `w_j,h_j`，中心为
`c_A,c_j`。先计算真实的 `u_j=IoU(A,B_j)`。为了在 IoU 为 0 时仍能区分“紧邻
anno”与“图像另一端”，定义归一化中心距离与尺寸差：

\[
d_j^2=\left(\frac{c_{jx}-c_{Ax}}{\max(w_A,1)}\right)^2+
      \left(\frac{c_{jy}-c_{Ay}}{\max(h_A,1)}\right)^2,
\]

\[
e_j=\left|\log\frac{\max(w_j,1)}{\max(w_A,1)}\right|+
    \left|\log\frac{\max(h_j,1)}{\max(h_A,1)}\right|.
\]

一个**待验证的初始定义**是：

\[
g_j=u_j+(1-u_j)\exp(-4d_j^2-e_j).
\]

`g_j∈[0,1]`；框完全一致时为 1，IoU 为 0 但位置和尺度接近时仍大于 0，远离
anno 时趋近 0。系数 4 是预设的空间衰减参数，不是从错误标签学习得到；实验中要
报告例如 `2/4/8` 的敏感性，不能仅展示效果最好的设置。**真实 IoU 仍单独保存为
`u_j`；`g_j` 不能被称为 IoU。**

### 4.2 搜索范围与选择

以 anno 中心为中心，把它的宽高各扩为原来的 2 倍，首先只考虑**预测框中心**落在
该区域内的原始候选。这个限制用于避免远处高置信度目标抢占候选，同时减少计算量。
对区域内的候选计算

\[
v_j=r_j g_j,
\qquad
j^*=\arg\max_j v_j.
\]

`v_j` 的选择**与 anno 类别无关**。例如一个真实的猫被标成狗，高置信度的“猫”
候选仍可能被选中；随后记录的“狗”类别得分 `q_j(a)` 会很低。

若扩展区域内没有候选，从全图原始候选中取中心距离最近的固定数量候选，再按同一
`v_j` 规则选择，并标记 `fallback_used=True`。YOLOv7 通常产生大量过滤前候选，
但不能在文档中假定所有模型、所有图片一定非空。**若模型输出确实为空**，写入
`candidate_available=False` 和缺失值；不得虚构 conf、IoU 或候选框。这个情况应
单独计数和报告。

并列候选按 `g_j` 较大、`r_j` 较大、原始候选索引较小的顺序确定，保证复现性。
每个 anno × epoch 执行完全相同的规则。前后 epoch 的候选索引不需要相同，
因为轨迹追踪的是**同一个 anno 所在区域的模型响应**，而非模型内部固定的 anchor。

### 4.3 保存的逐轮过程度量

选择 `j*` 后记录：

\[
\begin{aligned}
\operatorname{conf}_{a,t}&=q_{j^*}(a),\\
\operatorname{iou}_{a,t}&=u_{j^*},\\
\operatorname{geom}_{a,t}&=g_{j^*},\\
\operatorname{support}_{a,t}&=q_{j^*}(a)g_{j^*},\\
\operatorname{topconf}_{a,t}&=r_{j^*}.
\end{aligned}
\]

`conf` 与 `iou` 来自**同一个**候选框，`support` 是由它们及连续空间接近度导出
的联合支持。候选预测类别、候选框坐标、候选索引和是否用了 fallback 可作为排查
字段。正式匹配的布尔值 `hard_matched` 若同时计算，应放在独立字段；主排序初版
不使用它选框或打分。

这是一条**待实验验证的关联规则**，不能预先断言它必然区分所有 fault。尤其是
过滤前低得分候选数量庞大，必须通过候选可视化、阈值敏感性和消融实验检查它是否
选到了与 anno 有意义的局部响应。

### 4.4 一个数值例子

| anno 情况 | 关联候选的最高类别得分 `r` | anno 类别得分 `conf` | 真实 IoU | 预期的联合支持 |
| --- | ---: | ---: | ---: | ---: |
| clean，类别与位置吻合 | 高 | 高 | 高 | 高 |
| 类别错误，模型预测另一类 | 高 | 低 | 高 | 低 |
| 位置错误，模型预测附近目标 | 高 | 较高 | 低至中等 | 较低 |
| 冗余框附近没有可信目标 | 低 | 低 | 不固定 | 低 |

表格只是用于解释三个量的作用，不作为故障类别的判定规则。困难的 clean 目标也
可能长期只有低响应，这是最终排序的主要误报风险。

## 5. 从过程度量到八个过程特征

设训练轮数为 `T`，取前 `K=ceil(0.2T)` 轮为早期，后 `K` 轮为后期。YOLOv7 的
`T=50` 时分别是 epoch `0–9` 和 `40–49`。对每个 anno 得到等长的
`conf[0:T]`、`iou[0:T]`、`support[0:T]`；缺失值与数值 0 必须区分。

首版采用如下八个过程特征：

| 序号 | 特征 | 计算方式 | 越大表示 |
| ---: | --- | --- | --- |
| 1 | `early_conf` | 前 `K` 轮 `conf` 均值 | 更受标注类别支持 |
| 2 | `late_conf` | 后 `K` 轮 `conf` 均值 | 更受标注类别支持 |
| 3 | `early_iou` | 前 `K` 轮真实 `iou` 均值 | 早期位置更吻合 |
| 4 | `late_iou` | 后 `K` 轮真实 `iou` 均值 | 后期位置更吻合 |
| 5 | `early_support` | 前 `K` 轮 `support` 均值 | 早期联合支持更强 |
| 6 | `late_support` | 后 `K` 轮 `support` 均值 | 后期联合支持更强 |
| 7 | `support_persistence` | 在同轮可比 anno 中持续高支持的比例 | 支持更持续 |
| 8 | `longest_low_run` | 连续低支持的最长轮数除以 `T` | 缺乏支持更持续 |

第 7、8 项不需要已知 clean anno。每个 epoch 按标注类别和**相对框面积**
`w_A h_A / (image_width × image_height)` 建可比组，取该组全部 anno 的
`support` 中位数 `m_{group,t}`。对 `support_persistence`，该轮高于中位数记 1，
等于中位数记 0.5，低于记 0；再取 50 轮平均。`longest_low_run` 只把严格低于
中位数的轮次视为低支持。全部值相等时大家都为 0.5，**不会凭空制造顺序**。

默认把相对面积按所有 anno 的三分位分成小、中、大三组；类别 × 面积组少于 30
个 anno 时，先合并该类别的面积组，再不足则使用面积组，最后回退到全部 anno。
分组只用类别、面积和未标注对错的过程值。分组阈值、最小组大小和窗口比例都应
固定并做敏感性实验。第 7、8 项如果因为数据分布缺少区分度，可在开发实验中
剔除；不能为了维持“八个特征”而填常数。

**缺失处理：**如果某个模型真的没有输出任何原始候选，特征均值只基于有效轮次，
同时记录有效轮次比例。全程没有有效观测的 anno 不能凭空获得八个特征；在排名
里只能被标成“证据不足”并与同类情况并列。YOLOv7 实现首先应统计这种情况是否
出现，再决定是否需要独立的缺失处理。这里的 0 只表示实际观测到的 0。

## 6. 不用 clean 参照的 TOPSIS 排序

这一步输入全部现存 anno 的八维特征矩阵，**不需要知道其中谁是 clean**。
为了让不同类别和尺寸的 anno 公平比较，对每个特征先在上节定义的可比组内做
带并列平均名次的百分位变换。变换后所有维度都统一为“值越大，越可疑”：

- `early_conf`、`late_conf`、`early_iou`、`late_iou`、`early_support`、
  `late_support`、`support_persistence`：原值越小，异常百分位越大；
- `longest_low_run`：原值越大，异常百分位越大。

可用如下中间名次定义；对低值可疑的特征 `x`：

\[
z_a(x)=\frac{\#\{i:x_i>x_a\}+0.5\#\{i:x_i=x_a\}}{n}.
\]

高值可疑时把 `>` 换成 `<`。所有同值的 anno 得到相同的 `0.5`；这只是在
**未标明对错的候选总体中做相对比较**，并非拿 clean 参考集来拟合正常分布。

以八个百分位组成 `Z_a`，首版预先固定每个特征权重为 `1/8`，计算 TOPSIS
到“最可疑理想点”和“最不疑似理想点”的距离：

\[
V_{ak}=\tfrac18 Z_{ak},\quad
D_a^+=\|V_a-\max_i V_i\|_2,\quad
D_a^-=\|V_a-\min_i V_i\|_2,
\]

\[
\operatorname{score}(a)=\frac{D_a^-}{D_a^++D_a^-}.
\]

按 `score` 降序排序。数值大的 anno 优先人工检查，但这个数**不是错误概率**。
某维若在全部 anno 中恒定，应删除该维并对剩余预设权重重新归一化；如果全部
维度恒定，则所有 anno 的分数统一为 0.5 并列。分数相同按 anno ID 升序输出，
仅为了结果可复现，不能把 ID 顺序解释为质量差异。

`conf`、`iou`、`support` 互有相关性，等权 TOPSIS 可能重复计算某些证据。
因此必须报告至少三组消融：仅 `conf/iou`、仅联合支持的过程特征、全部八项。
如果将来调整权重，只能在独立开发数据或预注册规则上完成，不能利用最终测试集的
错误标签挑选最好看的权重。

## 7. 在当前仓库中的实施顺序

### 步骤 A：新增轨迹收集器

建议新增 `yolov7/collect_anno_trajectory.py`，复用
[现有模型加载和 dataloader](../yolov7/collect_train_info.py)，但在
`non_max_suppression` **之前**处理 `out`。不要直接改写现有收集路径，以便新旧
方法在同样 checkpoint、图片和标注上复现实验。

预处理时，从 `annotations_no_miss.json` 建立 `image_name → annos` 索引，只拷贝
`id/image_id/category_id/bbox` 四类必要字段。每个 epoch 遍历所有训练图片；
每张图先一次性还原原始候选坐标并计算 `o_j`、类别分数，再对该图的每个 anno
向量化计算 `u_j,g_j,v_j`，取 `j*` 并保存过程记录。不要为每个 anno 重复运行模型。

原始候选数量较大，**边推理边归约到 anno 记录**即可；不必写出每轮全部原始框。
推荐使用 GPU 张量批量计算区域筛选、IoU、中心距离和得分；最终只把每个 anno
每轮的一条结果转到 CPU。实现时先在少量图上检查内存，再决定按图或按 anno 分块。

可以按以下伪代码实现主循环。`decode_and_restore`、`geometry` 和 `associate`
必须作为独立函数，方便用人工构造的框检查：

```python
annos_by_name = load_visible_annos_without_fault_type(annotations_no_miss)
for epoch in range(T):
    model = load_checkpoint(epoch).eval()
    for imgs, paths, shapes in dataloader:
        raw_batch = model(imgs)[0]  # YOLOv7: decoded, before NMS
        for image_index, path in enumerate(paths):
            image_name = basename(path)
            proposals = decode_and_restore(
                raw_batch[image_index], imgs[image_index].shape,
                shapes[image_index], class_mapping,
            )
            for anno in annos_by_name[image_name]:
                record = associate(anno, proposals)  # same rule for every anno
                write(epoch, anno["id"], record)
```

其中 `associate` 返回候选索引及 `conf/iou/geom/support/topconf`，不能访问
`fault_type`；`decode_and_restore` 不能调用默认 `conf_thres=0.25` 的 NMS。可以
额外在同一批推理中运行旧 NMS，仅为了输出 `hard_matched` 诊断字段，且不能用其
结果覆盖主轨迹。若模型输出形状与上述 YOLOv7 假设不一致，应立即报错，而不是
静默地选择错误列。

将来支持其他检测器时，先做模型专用 adapter，使它统一返回：

```text
boxes_xyxy_original: [N, 4]
class_scores:       [N, C]  # 每个候选对每个数据集类别的得分
top_score:          [N]     # 每个候选的最高类别得分
candidate_index:    [N]     # 可复核的模型输出索引
```

YOLOv7 可直接从 `out` 的 objectness 和类别列构造该接口。RT-DETR 应在其查询
输出过滤前导出框与类别分数；Faster R-CNN 则需在 RoI 预测进入阈值筛选和 NMS
之前导出提案、类别分数及对应的回归框。这两者的具体张量及类别映射必须分别核实；
不能把 YOLOv7 的列下标和坐标变换照搬过去。各模型**分别排序和评价**，不能
未经校准把不同模型的得分放在同一个 TOPSIS 矩阵中。

### 步骤 B：保存可复核的轨迹

建议按 `dataset/model/inject_ratio/repeat_id` 隔离输出，保存版本化的 `manifest`
和紧凑的轨迹表。每个 `anno_id,epoch` 至少包含：

```text
anno_id, image_name, epoch, category_id,
conf, iou, geom, support, topconf,
candidate_available, fallback_used, candidate_index
```

调试版本还应存候选框的原图 `xyxy` 和预测最高类别，以便抽样可视化。
`manifest` 记录数据集、checkpoint 路径或哈希、50 个 epoch 编号、输入图像尺度、
类别映射、关联规则版本、空间衰减系数以及软件版本。可用 Parquet 或按 epoch 的
紧凑数组保存；核心要求是无需重新推理即可复算过程特征。

### 步骤 C：构建过程特征与排名

建议新增独立的 `ours/rank/rank_anno_trajectory.py`：

1. 检查 `anno_id` 集合与输入的所有现存 anno 完全一致；
2. 检查每个 anno 恰有 `T` 个 epoch 位置，数值范围和缺失标记合法；
3. 计算八个过程特征与可比组中位数；
4. 做组内百分位、TOPSIS、稳定排序；
5. 输出 `annoid_features.csv`、含过程度量摘要的排名 CSV 和 `rank.joblib`。

这些新结果与现有 [rank.py](rank/rank.py) 的旧方法分目录保存。首版只评价
anno 排名，不调用旧代码把图片级排名混入其中。

### 步骤 D：只在评价阶段打开错误真值

评价脚本单独读取注错标签，将 `fault_type == 0` 视为 clean，
`fault_type ∈ {1,2,3}` 视为 fault。不得把该字段加入收集器或排序器参数。
在同一批现存 anno 上，与当前八特征 TOPSIS 排名比较：

- PR-AUC / AP、ROC-AUC，以及固定检查预算下的 `Precision@K`；
- `Recall@K`，避免只降低误报却把 fault 一起压到后面；
- 原旧方法中“50 轮零正式匹配”的子集，单独报告其中 clean 的前 `K` 占比、
  `Precision@K` 与并列分数数量；
- 按类别、框面积和数据集分层的结果，检查是否只对大而容易的目标有效；
- 对候选选择、`g_j` 衰减、前后窗口、特征组和 TOPSIS 权重做消融或敏感性分析。

如果使用注错标签调整候选规则、阈值或权重，必须将数据按**图像与实验重复**划分
开发集和最终测试集，并把调参后的方案与“预设无监督方案”分开报告。否则会把
目标实验的答案泄漏进排序方法。

## 8. 实施前后的核对清单

1. **坐标一致性：**随机抽图把 anno、NMS 前候选、NMS 后框画在原图上；
   检查 letterbox 的缩放与 padding 恢复是否正确。
2. **类别一致性：**核对 COCO `category_id`、YOLOv7 输出列与当前数据的类别名；
   防止静默的 ±1 错位。
3. **得分一致性：**确认 `conf=o_j p_j(y_a)` 只乘一次 objectness；记录
   `topconf` 以便区分类别不一致与无目标响应。
4. **覆盖完整性：**不管原有 `match.json` 有没有该 anno，都应有 50 轮轨迹
   记录或明确的缺失状态；不能用 0/1 常数补齐缺失的整个 anno。
5. **选择质量：**人工查看 clean、类别错误、位置错误、冗余框各若干个例子，
   核查 `j*` 是否确为局部模型响应；这些例子只用于检查实现，不用于调整最终
   测试集上的方法。
6. **排名稳定性：**恒定特征、并列特征和空候选都不应产生 NaN；固定输入应
   得到固定顺序。

## 9. 方法能够与不能够证明什么

成功的实验可以支持：“由统一采集的 anno 类别得分、几何一致性及其 50 轮过程
特征，能把一部分 fault anno 排在 clean anno 之前，并改善零正式匹配组的误报。”

它不能保证把每个困难 clean 与每个冗余 anno 分开。如果二者对该模型的所有
过滤前输出、空间关系和 50 轮变化都相同，任何只使用这些观测的打分方法也无法
产生有根据的不同结论。并列应被保留和报告；不能再通过人为指定极端特征或按
anno ID 排序制造区分度。

本方案与 ObjectLab/CLOD 一样，属于通过模型输出来审查目标检测标注的研究方向，
但这里的**按 checkpoint 收集过程轨迹、统一关联、
八项过程特征与无监督 TOPSIS 排序**是本项目拟实现和验证的具体方案，不应写成
已经得到实验支持的结论。

## 参考研究

- [ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data](https://arxiv.org/abs/2309.00832)
- [Combating noisy labels in object detection datasets (CLOD)](https://arxiv.org/abs/2211.13993)

两篇研究说明“通过检测模型的预测证据审查标注”是已有研究方向；本方案的具体
关联规则、过程特征和排序方式需由本项目自己的实验验证。
