# 单 feature 的 AUC 为什么这么高？—— 标签泄漏（label leakage）分析

## 现象

在 `ours/discussion/features/box_features.py::plot_roc_auc` 中，把每个 box 级 feature 单独拿来区分 `error_gid` / `correct_gid`，AUC 远高于"特征本身应该具有的判别力"。这并不是 feature 设计得多好，而是**特征在构造过程中混入了与标签强相关的非特征信息**。下面按数据流向逐步拆解。

---

## 数据流回顾

1. `ours/match_and_collect_metrics.py::match` —— 把每张图的 `gt_boxes` 和每个 epoch 的 `predicted_boxes` 做 PG 匹配（类别一致 + IoU ≥ 0.5）。匹配上的对存进 `match.json`。
2. `ours/match_and_collect_metrics.py::collect_metrics_for_gboxs` —— 把每个 `g_box_id` 在所有 epoch 上的 `(conf, iou)` 序列汇总成 `conf_list` / `iou_list`。
3. `ours/discussion/features/box_features.py::build_gid_feature` —— 用 `(conf_list, iou_list)` 计算 `early_conf_mean / lastly_conf_mean / conf_mean / D_conf / ... ` 等 8 个 feature。
4. `plot_roc_auc` —— 用每个 feature 当作 score 跑 ROC。

错误标签的定义（`data_organization_tools.get_all_errored_g_box_id_set`）：`g_box["fault_type"] != 0` 即为 error gid。`fault_type` 包含类别错误、bbox 偏移、缺失等。

---

## 为什么 AUC 异常高 —— 三个泄漏点

### 泄漏 1（最严重）：`build_gid_feature` 对"从未匹配上的 gid"赋了"完美可疑"的哨兵值

`box_features.py:86-98`：

```python
for g_id in all_gids:
    if g_id not in g_id_to_features:
        # 没有匹配上的gid都是最可疑的
        g_id_to_features[g_id] = {
            "early_conf_mean": 0,
            "early_iou_mean":  0,
            "lastly_conf_mean":0,
            "lastly_iou_mean": 0,
            "conf_mean":       0,
            "iou_mean":        0,
            "D_conf":          1,
            "D_iou":           1,
        }
```

- "在所有 epoch 上都没匹配到任何预测框"的 gid 会拿到这一组哨兵值。
- 而**"完全没匹配到"的 gid 几乎都是 error gid**：因为 `search_match_PG` 要求"类别一致 + IoU ≥ 0.5"，类别打错的 gid 永远不会匹配；bbox 错得离谱的 gid 也永远不会匹配。
- 结果：哨兵值 `(0,0,0,0,0,0,1,1)` 本身就是一个近乎完美的 error 指示器。**任何**单个 feature 把"等于哨兵值"的样本都排在最前面 / 最后面，ROC 曲线一开局就直接拉到很高的 TPR 而几乎不引入 FPR，AUC 自然就被推高。

这一步把"label 的一个充分条件（match 失败）"直接写进了 feature 值里，等于把答案泄漏给了分类器。

### 泄漏 2：`collect_metrics_for_gboxs` 用 0 填充未匹配 epoch

`match_and_collect_metrics.py:339-349`：

```python
for epoch in range(epochs):
    matched_info = temp_dict.get(epoch)
    if matched_info is None:
        conf = 0
        iou  = 0
    else:
        conf = matched_info["p_box"]["conf"]
        iou  = matched_info["iou_val"]
    instance["conf_list"].append(conf)
    instance["iou_list"].append(iou)
```

- 这是"部分 epoch 没匹配"的情况，不是泄漏 1 的"完全没匹配"。
- 但 error gid 整体上"匹配次数更少"，所以零的占比更高，`*_mean` 系列被拉低，`D_conf / D_iou`（起量延迟）被拉高，正好对齐 sign 方向。
- 这部分本来就是特征想刻画的信号（"涨得晚 / 涨得低就更可疑"），但它同时也是匹配机制的副产物，所以仍带有"标签信息"的成分，不算干净的"模型行为"信号。

### 泄漏 3：匹配机制本身就携带标签信息

`search_match_PG` 同时使用了 `predicted_cls == gt_cls` 和 `IoU ≥ 0.5`：

- `fault_type` 中的"类别错"会直接导致永远匹配不上（→ 泄漏 1 触发）。
- `fault_type` 中的"框位置错"会导致 IoU 长期低于阈值（→ 泄漏 1 或泄漏 2 触发）。

也就是说，error 的定义本身就和"是否能被匹配上"高度耦合，而特征又是从"匹配后的 (conf, iou) 序列"里抽出来的，从信息论上看，feature 的输入和 label 之间存在共同的因果上游（错误标注）。这是结构性问题，不是哪一行代码的 bug。

---

## 怎么验证以上判断

可以在 `box_features.py::main` 里加几行诊断：

1. 统计"完全没匹配"的 gid 中 error / correct 的比例：
   ```python
   never_matched = set(all_gids) - set(g_id_to_metric.keys())
   err_in_nm  = len(never_matched & error_gid_set)
   cor_in_nm  = len(never_matched & correct_gid_set)
   print(f"never matched: total={len(never_matched)} err={err_in_nm} cor={cor_in_nm}")
   ```
   如果 `err_in_nm / len(error_gid_set)` 很高，泄漏 1 就坐实了。

2. **只对 matched gid 跑 ROC**（剔除哨兵值样本），AUC 应该会显著下降，下降幅度就是泄漏 1 贡献的部分：
   ```python
   matched_gids = set(g_id_to_metric.keys())
   correct_eval = correct_gid_set & matched_gids
   error_eval   = error_gid_set   & matched_gids
   plot_roc_auc(g_id_to_features, feature_name_to_sign, correct_eval, error_eval, ...)
   ```

3. **同时跑两条 TOPSIS 曲线**：一条带哨兵 gid，一条只在 matched gid 上跑。两条 AUC 的差，就是哨兵值带来的"白送的判别力"。

---

## 实测结果（`ours/discussion/features/leak_check.py`，YOLOv7）

### unmatched gid 在 error / correct 中的分布

| dataset  | #all gid | #matched | unmatched ratio | error 中 unmatched | correct 中 unmatched | "unmatched=1" 单二值特征 AUC |
|----------|---------:|---------:|----------------:|-------------------:|---------------------:|----------------------------:|
| VOC2012  |   13,609 |   10,345 |           0.240 |  3168/4080 = 0.776 |    96/9529 = 0.010   |                       0.883 |
| KITTI_8  |   20,423 |   14,925 |           0.269 | 5142/6126 = 0.839  |   356/14297 = 0.025  |                       0.907 |
| VisDrone |  343,203 |  189,722 |           0.447 | 94476/102960 = 0.918 | 59005/240243 = 0.246 |                     0.836 |

**结论**：error / correct 在"是否匹配"上极度不平衡，只看"未匹配 = 1"这一个二值特征 AUC 就 ≈ 0.84–0.91 —— 这就是泄漏 1 白送的判别力。

按 fault_type 拆分（VisDrone 为例）：

| fault_type        | total   | unmatched | ratio |
|-------------------|--------:|----------:|------:|
| no_fault          | 240,243 |    59,005 | 0.246 |
| cls_fault         |  34,320 |    32,522 | 0.948 |
| loc_fault         |  34,320 |    27,648 | 0.806 |
| redundancy_fault  |  34,320 |    34,306 | 1.000 |

`cls_fault` 类别错 → 几乎全部过不了 `predicted_cls==gt_cls`；`redundancy_fault` 冗余框 → 几乎 100% 未匹配。哨兵值的判别力主要来自这两类。

### AUC 比较：全集 vs 仅 matched gid（剔除哨兵）

#### VOC2012

| feature           | AUC(all) | AUC(matched only) |     Δ      |
|-------------------|---------:|------------------:|-----------:|
| early_conf_mean   |   0.9511 |            0.9266 |    -0.0245 |
| early_iou_mean    |   0.9542 |            0.9404 |    -0.0138 |
| lastly_conf_mean  |   0.9658 |            0.8840 |    -0.0818 |
| lastly_iou_mean   |   0.9577 |            0.8471 |    -0.1106 |
| conf_mean         |   0.9848 |            0.9593 |    -0.0255 |
| iou_mean          |   0.9872 |            0.9698 |    -0.0174 |
| D_conf            |   0.9692 |            0.8886 |    -0.0806 |
| D_iou             |   0.9761 |            0.9201 |    -0.0560 |
| **TOPSIS_score**  | **0.9866** |          **0.9672** | **-0.0194** |

#### KITTI_8

| feature           | AUC(all) | AUC(matched only) |     Δ      |
|-------------------|---------:|------------------:|-----------:|
| early_conf_mean   |   0.9451 |            0.9024 |    -0.0427 |
| early_iou_mean    |   0.9478 |            0.9200 |    -0.0278 |
| lastly_conf_mean  |   0.9654 |            0.9004 |    -0.0650 |
| lastly_iou_mean   |   0.9658 |            0.9028 |    -0.0630 |
| conf_mean         |   0.9787 |            0.9565 |    -0.0222 |
| iou_mean          |   0.9813 |            0.9729 |    -0.0084 |
| D_conf            |   0.9483 |            0.7622 |    -0.1861 |
| D_iou             |   0.9535 |            0.7954 |    -0.1581 |
| **TOPSIS_score**  | **0.9790** |          **0.9582** | **-0.0208** |

#### VisDrone

| feature           | AUC(all) | AUC(matched only) |     Δ      |
|-------------------|---------:|------------------:|-----------:|
| early_conf_mean   |   0.7959 |            0.7467 |    -0.0492 |
| early_iou_mean    |   0.7977 |            0.7762 |    -0.0215 |
| lastly_conf_mean  |   0.8420 |            0.8728 |    +0.0308 |
| lastly_iou_mean   |   0.8437 |            0.9001 |    +0.0564 |
| conf_mean         |   0.8587 |            0.8648 |    +0.0061 |
| iou_mean          |   0.8607 |            0.8970 |    +0.0363 |
| D_conf            |   0.8412 |            0.5839 |    -0.2573 |
| D_iou             |   0.8419 |            0.5951 |    -0.2468 |
| **TOPSIS_score**  | **0.8571** |          **0.8391** | **-0.0180** |

### 解读

1. **`D_conf / D_iou` 跌得最狠**（VisDrone 上 0.84 → 0.58/0.60，几乎随机）。原因直接：`build_gid_feature` 给哨兵 gid 赋 `D_*=1`，正好等于"最可疑"的上限，剔除后这两个 feature 信息就所剩无几。这两个 feature 是泄漏 1 的主要受益者。
2. **`*_mean` 系列跌幅中等**（0.01 – 0.11 不等）。它们的判别力一部分来自"零填充比例"（泄漏 2），一部分来自"匹配上之后" conf/iou 自身的差异（真信号）。
3. **`lastly_*` / `iou_mean` 在 VisDrone 上反而上升**（+0.03 ~ +0.06）。原因是 VisDrone 难例多、correct gid 里也有 24.6% 未匹配（被赋为 0），这部分 correct 样本在全集下被错排到"最可疑端"，反而拖低了 AUC；剔除后变干净了。说明"哨兵值"在 correct gid 上也是噪声源，不只对 error 有效。
4. **TOPSIS_score 跌幅最小**（≈ -0.02）。综合 score 把多种信号平均，对"哨兵值集中暴击单一 feature"有稀释作用，鲁棒性更好，但仍然吃了 ≈ 2 个百分点的泄漏红利。
5. **`iou_mean` 在所有数据集 matched-only 下都是单 feature 最强**（0.97 / 0.97 / 0.90），且跌幅最小 —— 说明它是这套 feature 里最"实在"的信号。

---

## 为什么综合考虑全部 feature 的 TOPSIS score 反而打不过单 feature（如 iou_mean）？

直觉上"多 feature 综合 ≥ 单 feature"应当成立，但 matched-only 下 TOPSIS_score 几乎处处被 `iou_mean` 压住：

| dataset  | TOPSIS_score | iou_mean | 差距     |
|----------|-------------:|---------:|---------:|
| VOC2012  |       0.9672 |   0.9698 |   -0.003 |
| KITTI_8  |       0.9582 |   0.9729 |   -0.015 |
| VisDrone |       0.8391 |   0.8970 |   -0.058 |

简短答案：**等权 TOPSIS 不是分类器，它只是"标准化 feature 的加权平均"，并不优化 AUC**。加入弱 / 噪声 feature 会稀释强 feature 的判别力。具体到这套数据有三个叠加原因：

### 1. TOPSIS 不学权重 —— 等权 = 把噪声平均进去

`box_rank.py:74` 用的是 `weights = np.ones(n_features) / n_features`，所有 8 个 feature 各占 1/8 权重。但实测各 feature 在 matched-only 下的 AUC（即"真信号强度"）差异很大（以 VisDrone 为例，AUC 从高到低）：

| feature           | VisDrone AUC(matched) |
|-------------------|---------------------:|
| lastly_iou_mean   |                0.900 |
| iou_mean          |                0.897 |
| lastly_conf_mean  |                0.873 |
| conf_mean         |                0.865 |
| early_iou_mean    |                0.776 |
| early_conf_mean   |                0.747 |
| **D_iou**         |            **0.595** |
| **D_conf**        |            **0.584** |

`D_conf / D_iou` 在 matched-only 下基本就是随机噪声（0.58 ≈ 0.5），但 TOPSIS 仍给它们各占 1/8 话语权 —— 这 2/8 ≈ 25% 的权重都在往随机方向拖。

### 2. Feature 之间高度相关，方向被"双重计票"

8 个 feature 其实只是 3 个方向：

- conf 方向：`early_conf_mean / lastly_conf_mean / conf_mean`（3 个）
- iou 方向：`early_iou_mean / lastly_iou_mean / iou_mean`（3 个）
- 延迟方向：`D_conf / D_iou`（2 个）

等权后实际方向权重 = **conf 3/8 + iou 3/8 + 延迟 2/8**。这并不是"看 8 个独立证据"，而是同一信号被多算了几遍。同一方向加再多相关 feature 也不会带来新信息，但会改变综合分的"重心"。

### 3. TOPSIS 不是为 AUC 设计的

TOPSIS 用欧氏距离到理想点 / 反理想点，输出的是 `d⁻/(d⁺+d⁻)`，本质上对"标准化后的加权和"做单调变换。它解决的是"多准则决策"问题（在多个不可比的指标下排序方案），不是监督学习。要"综合多 feature 拿到更好的 AUC"，需要**学**权重 —— 逻辑回归 / GBDT / 甚至按 AUC 加权都行。

### 直觉：什么时候多 feature 才能赢

> 当且仅当：被组合的 feature **都比随机好** **且** 带来的是 **互补**（不同方向）信号 **且** 组合方式给它们的权重 **大致正比于各自判别力**。

这套数据三条都不满足：D_* 接近随机、6 个 feature 集中在 2 个方向、weights 是死的 1/8。

### 可以做的对照实验

- (a) 把 `D_conf / D_iou` 从 TOPSIS 里丢掉，看综合 AUC 是否反超 `iou_mean` —— 验证"是 D_* 在拖后腿"。
- (b) 用各 feature 的 matched-only AUC 作权重再跑 TOPSIS —— 验证"等权本身是次优"。
- (c) 把 8 个 feature 喂进逻辑回归 / GBDT，得到一个"学过的"综合 score —— 给出"理论上限"，再跟 TOPSIS_score 比较，差距就是"等权 + 不学"的代价。

---

## 修复建议（不影响 box_rank 的现有排序，只是分析时区分干净）

- **评估时**显式排除 `never_matched` gid，得到的 AUC 才反映 feature 真正的判别力。
- **特征定义上**，与其用哨兵值 `(0,0,...,1,1)`，不如显式新增一个二值 feature `matched_any_epoch ∈ {0,1}`，然后 `*_mean` / `D_*` 只在 `matched=1` 的样本上有定义。这样 TOPSIS 仍然可以用，但泄漏来源透明、可控。
- 把"完全没匹配 = 最可疑"这个 prior 显式当成 ranking 的兜底规则（rule-based fallback），而不是塞进数值特征里——保持"特征"和"先验规则"的边界清晰。

---

## 顺带：`match_and_collect_metrics.py` 中加剧泄漏的实现细节

主体已经按"特征侧"列出了三个泄漏点，这里只补一下**收集侧**有哪些具体写法把上面那三个泄漏点变得更尖锐。

### 1. 零填充 vs 哨兵值的语义重叠

`collect_metrics_for_gboxs` 用 `conf=0 / iou=0` 表示"该 epoch 没匹配上"（泄漏 2 的源头），而 `build_gid_feature` 又用 `*_mean=0` 表示"所有 epoch 都没匹配上"（泄漏 1 的哨兵值）。

- 两种语义被压成同一个数值 `0`，下游 feature 无法区分"偶尔没匹配"和"从来没匹配"。
- 更干净的做法：`collect_metrics_for_gboxs` 把未匹配 epoch 标成 `NaN`（而不是 `0`），下游聚合时显式选择忽略 / 计数 / 单独建 feature。把"是否匹配"和"匹配上之后的 conf / iou 数值"分成两路特征，泄漏来源就透明可控了。

### 2. 匹配条件与 fault_type 定义耦合（结构性，无法只靠收集侧绕开）

`search_match_PG` 的匹配条件 `predicted_cls == gt_cls && IoU ≥ 0.5` 同时被两类 fault_type 直接破坏：

- 类别错 → 永远匹配不上 → 进哨兵值通道（泄漏 1）。
- 位置错 → IoU 长期低于阈值 → 进零填充通道（泄漏 2）。

也就是说 error 的定义本身就和"是否能被匹配"高度耦合，feature 又是从匹配结果里抽的 —— feature 的输入和 label 之间存在共同的因果上游（错误标注）。这是结构问题，不是哪一行代码的 bug，但要意识到 AUC 里有多少是 feature 真本事、多少是这条因果通路白送的。

---

## 一句话总结

单 feature AUC 异常高，是**数据泄漏**：`build_gid_feature` 把"从未匹配的 gid"赋了哨兵值 `(0,0,0,0,0,0,1,1)`，而"从未匹配"几乎等价于"error gid"（类别错就永远过不了 `cls==cls && IoU≥0.5` 的匹配条件）—— 等于把 label 直接抄进了特征里。次要泄漏来源是 `collect_metrics_for_gboxs` 对未匹配 epoch 的零填充，以及匹配条件本身就与 fault_type 的定义耦合。想看 feature 真正的判别力，请在评估时剔除哨兵样本（或单独评估 matched 子集）。

