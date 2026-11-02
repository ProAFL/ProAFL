# ObjectLab 复现与任务适配

依据仓库根目录 `objectlab.pdf`（arXiv:2309.00832v1），第 3–4 页算法 1–4。
入口为 `python -m baselines_new.objectLab`，实验使用文件顶部 Python 常量。
默认读取 epoch 49；`build_ranking` 直接返回原始 anno ID 和图片名
混合组成的列表。主程序将该列表通过 joblib 保存到
`DATA_ROOT/rank/objectlab/{DATASET_NAME}/{MODEL_NAME}/{FAULT_RATIO}/repeat_{REPEAT_ID}/rank.joblib`。
使用 `joblib.load(path)` 即可得到 `[42, "example.jpg", 7]` 形式的排名列表。
列表位置代表最终名次，不输出分数及条目字典。

## 论文基础公式

- 预测集合：只保留 `conf > 0.5`；高置信度证据要求 `conf > 0.95`。
- 相似度：`sim = 0.1 * exp(-||b1-b2|| / 0.1) + 0.9 * IoU`。
  `b` 是按图片宽高归一化的四维 xyxy 坐标；距离使用欧氏范数，不平方。
  零 IoU 框仍有位置相似度，因此不会全部得到相同的零分。
- 定位分：同类且有重叠的预测框中最大 sim；不存在时为 1（算法 2）。
- 类别分：`1 - max(sim)`，候选为高置信度异类预测框；不存在时为 1。
  **0.95 是置信度阈值，不是 IoU 阈值**（算法 3）。
- 漏标候选：高置信度且与任何类别标注均无重叠的预测框。
  若存在同类标注，候选质量为最大同类 sim；否则为 `sim* * (1-conf)`。
  `sim*` 是本次数据集所有同图标注/有效预测框配对的最小 sim；无配对时取 1。
  极端坐标导致核下溢时，以 float64 最小正正规数保护该值。
  候选用 `softmin(q) = sum(q * softmax(1-q))` 聚合；无候选为 1。

PDF 算法 4 只列出同类零重叠标注集合，没有显式排除已经匹配的预测。
为避免正确标注的目标也成为漏标证据，本实现采用作者公开 Cleanlab 实现中的
重叠保护：与任一标注 IoU > 0 的预测不作为漏标候选；通过保护后，同类标注
自然都满足算法 4 的零重叠条件。这比上一版自行设定 IoU < 0.5 更贴近作者实现。
参考：[Cleanlab 源码](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/object_detection/rank.py)，
`_compute_overlooked_box_scores_for_image`。其余相似度参数以 PDF 为准，
不混用当前 Cleanlab 版本的距离定义或参数。

## anno 与 image 排名适配

论文算法 1 原本只定义整张图片的总质量：

`(softmin(定位分) * softmin(类别分) * softmin(漏标候选分)) ** (1/3)`。

本任务需要分别检查显式标注和漏标图片，因此采用：

- `anno_score = sqrt(badloc_score * badcls_score)`：对每个 anno 的两类质量做
  等权几何平均。这是沿用论文聚合思想的任务适配，论文未直接定义 anno 总分。
- `img_score = miss_score`：仅代表漏标质量，避免图片条目重复表达已有 anno 的问题。
- 所有条目按各自 score 稳定升序排列；越小越可疑，同分保留插入顺序。
  这类混合排名也是本任务适配，不是论文原本的纯图片排名。

代码只计算最终排序需要的 anno 分数和图片漏标分数，不再计算论文的完整图片总分。
打分使用全部有效预测框；不再记录单框匹配信息，也不需要 `probs`。
预测框只需提供 `bbox`、`predicted_cls` 和 `conf`。

## 复现范围

论文使用 5 折交叉验证产生 out-of-sample 预测，避免模型记住训练标注。
本脚本使用用户已收集的 epoch 49 预测，不能由这些 JSON 判断它们是否为
out-of-sample，也不会重新训练模型。若预测来自同一训练集上的模型，
就不具备论文的预测独立性条件，不能声称完整复现了论文实验。

输入若在收集时被更高置信度阈值或其他规则剪枝，脚本无法恢复已丢弃框。
图片必须提供有效 `width`、`height`，用于坐标归一化。保留 COCO `images` 中
的零标注图片，也可对它们计算漏标分。无重叠预测的 anno 依算法 2 默认质量 1，
本方法不额外加入冗余标注检测分支。
