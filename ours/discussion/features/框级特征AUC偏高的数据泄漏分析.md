# 框级特征 AUC 偏高的数据泄漏分析

## TL;DR

把 `box_features.py::plot_roc_auc` 里每个 feature 单拎出来当 score，区分 `error_gid` / `correct_gid`，AUC 普遍在 **0.95 – 0.99**（VisDrone 上 0.80 – 0.87）。这并不是 feature 真有这么强，而是**特征构造过程把 label 信息抄进了数值里**。

剔除"从未匹配上预测框的 gid"（哨兵样本）后，AUC 普遍下降 0.02 – 0.26 不等；其中 `D_conf / D_iou` 在 VisDrone 上从 0.84 跌到 0.58 / 0.60（≈ 随机），说明它们的判别力几乎全部来自泄漏。

**泄漏的核心位置**：`box_features.py:86-98` 对从未匹配上预测框的 gid 直接赋哨兵值 `(0,0,0,0,0,0,1,1)`；而"从未匹配"几乎等同于"error gid"（错类别 / 错位置 / 冗余框都过不了 `cls==cls && IoU≥0.5` 的匹配条件）。

---

## 1. 现象

`box_features.py::plot_roc_auc` 中各 feature 单独作 score 算 AUC（YOLOv7）：

| dataset  | iou_mean | TOPSIS_score | "未匹配 = 1" 二值特征 |
|----------|---------:|-------------:|---------------------:|
| VOC2012  |   0.9872 |       0.9866 |                0.883 |
| KITTI_8  |   0.9813 |       0.9790 |                0.907 |
| VisDrone |   0.8607 |       0.8571 |                0.836 |

**关键反常**：只看"该 gid 是否在任何 epoch 都没匹配上"这一个 0/1 特征，AUC 就达到 0.84 – 0.91。这说明大部分判别力来自"是否匹配"本身，而非 conf / iou 序列。

---

## 2. 数据流水线

```
gt_json + 各 epoch 预测框
  ↓ match_and_collect_metrics.py::match            (PG 匹配, cls 一致 + IoU≥0.5)
match.json
  ↓ collect_metrics_for_gboxs                       (按 epoch 汇总 conf_list/iou_list, 未匹配 epoch 填 0)
collection_metrics.json
  ↓ box_features.py::build_gid_feature             (算 *_mean / D_* 共 8 个 feature; 全程未匹配的 gid 赋哨兵值)
g_id_to_features
  ↓ topsis(等权)  +  单 feature ROC
排序 score / ROC 曲线
```

`error_gid` 定义（`data_organization_tools.get_all_errored_g_box_id_set`）：`g_box["fault_type"] != 0`，包含 `cls_fault` / `loc_fault` / `redundancy_fault` / `missing_fault`。

---

## 3. 三层数据泄漏

### 3.1 泄漏 1（主因）：`build_gid_feature` 用哨兵值表示"从未匹配"

`box_features.py:86-98`：

```python
for g_id in all_gids:
    if g_id not in g_id_to_features:
        g_id_to_features[g_id] = {
            "early_conf_mean": 0, "early_iou_mean":  0,
            "lastly_conf_mean":0, "lastly_iou_mean": 0,
            "conf_mean":       0, "iou_mean":        0,
            "D_conf":          1, "D_iou":           1,
        }
```

- 这组哨兵值刚好等于"最可疑的端点"：6 个 mean 取 0（sign = -1，越小越可疑），2 个 D_* 取 1（sign = +1，越大越可疑）。
- 而**"从未匹配"几乎等价于"error gid"**：`cls_fault` 永远过不了 `predicted_cls==gt_cls`；`redundancy_fault` 是数据集里凭空多出的框，对不上任何真实预测；`loc_fault` 位置错得离谱时 IoU 长期低于 0.5。
- 结果：哨兵值本身就是一个近乎完美的 error 指示器，且每个 feature 都"独立"地携带了这个指示 —— 任何单个 feature 把"等于哨兵值"的样本排到端点，ROC 一开局就拉到很高 TPR 而几乎不引入 FPR，AUC 自然被推高。

> 等于把 label 的一个充分条件（match 失败）直接写进了 feature 数值。

### 3.2 泄漏 2：`collect_metrics_for_gboxs` 用 0 填充未匹配 epoch

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
```

- 这是"部分 epoch 没匹配上"的情况，不是泄漏 1 的"全程未匹配"。
- 但 error gid 整体匹配次数更少 → 零的占比更高 → `*_mean` 被拉低、`D_*` 被拉高，恰好对齐 sign 方向。
- 与泄漏 1 语义重叠：`conf=0` 既可能是"该 epoch 没匹配"，也可能是"全程没匹配" —— 同一数值承担两个含义，下游 feature 无法区分。

### 3.3 泄漏 3（结构性）：匹配条件与 fault_type 的定义耦合

`search_match_PG` 同时使用 `predicted_cls == gt_cls` 和 `IoU ≥ 0.5`。而 fault_type 恰好就是破坏这两个条件的东西：

- `cls_fault` → 永远过不了类别检查 → 触发泄漏 1。
- `loc_fault` → IoU 长期低于阈值 → 触发泄漏 1 或 2。
- `redundancy_fault` → 没有对应的真实预测 → 触发泄漏 1。

也就是说 **error 的定义本身就和"是否能被匹配"高度耦合**。feature 是从匹配结果里抽的，feature 和 label 之间存在共同的因果上游（错误标注）。这是结构问题，不是某一行代码的 bug，但要意识到 AUC 里有多少是 feature 真本事、多少是这条因果通路白送的。

---

## 4. 实测证据（脚本：`ours/discussion/features/leak_check.py`）

### 4.1 未匹配 gid 在 error / correct 中的分布

| dataset  | #all gid | #matched | unmatched 比例 | error 中未匹配 | correct 中未匹配 |
|----------|---------:|---------:|--------------:|---------------:|-----------------:|
| VOC2012  |   13,609 |   10,345 |         0.240 |          0.776 |            0.010 |
| KITTI_8  |   20,423 |   14,925 |         0.269 |          0.839 |            0.025 |
| VisDrone |  343,203 |  189,722 |         0.447 |          0.918 |            0.246 |

> error gid 中未匹配比例 0.78 – 0.92，correct gid 中只有 0.01 – 0.25。**两者悬殊就是泄漏 1 成立的直接证据**。

按 fault_type 拆分（VisDrone）：

| fault_type        | total   | unmatched | 未匹配率 |
|-------------------|--------:|----------:|--------:|
| no_fault          | 240,243 |    59,005 |   0.246 |
| cls_fault         |  34,320 |    32,522 |   0.948 |
| loc_fault         |  34,320 |    27,648 |   0.806 |
| redundancy_fault  |  34,320 |    34,306 |   1.000 |

`cls_fault` 几乎全不匹配（0.95），`redundancy_fault` 100% 不匹配 —— 这两类 fault 是哨兵值"白送 AUC"的主要来源。

### 4.2 AUC 对比：全集 vs 仅 matched gid（剔除哨兵）

**VOC2012**

| feature           | AUC(全集) | AUC(matched only) |       Δ |
|-------------------|---------:|------------------:|--------:|
| early_conf_mean   |   0.9511 |            0.9266 | -0.0245 |
| early_iou_mean    |   0.9542 |            0.9404 | -0.0138 |
| lastly_conf_mean  |   0.9658 |            0.8840 | -0.0818 |
| lastly_iou_mean   |   0.9577 |            0.8471 | -0.1106 |
| conf_mean         |   0.9848 |            0.9593 | -0.0255 |
| iou_mean          |   0.9872 |            0.9698 | -0.0174 |
| D_conf            |   0.9692 |            0.8886 | -0.0806 |
| D_iou             |   0.9761 |            0.9201 | -0.0560 |
| **TOPSIS_score**  | **0.9866** |        **0.9672** | **-0.0194** |

**KITTI_8**

| feature           | AUC(全集) | AUC(matched only) |       Δ |
|-------------------|---------:|------------------:|--------:|
| early_conf_mean   |   0.9451 |            0.9024 | -0.0427 |
| early_iou_mean    |   0.9478 |            0.9200 | -0.0278 |
| lastly_conf_mean  |   0.9654 |            0.9004 | -0.0650 |
| lastly_iou_mean   |   0.9658 |            0.9028 | -0.0630 |
| conf_mean         |   0.9787 |            0.9565 | -0.0222 |
| iou_mean          |   0.9813 |            0.9729 | -0.0084 |
| D_conf            |   0.9483 |            0.7622 | -0.1861 |
| D_iou             |   0.9535 |            0.7954 | -0.1581 |
| **TOPSIS_score**  | **0.9790** |        **0.9582** | **-0.0208** |

**VisDrone**

| feature           | AUC(全集) | AUC(matched only) |       Δ |
|-------------------|---------:|------------------:|--------:|
| early_conf_mean   |   0.7959 |            0.7467 | -0.0492 |
| early_iou_mean    |   0.7977 |            0.7762 | -0.0215 |
| lastly_conf_mean  |   0.8420 |            0.8728 | +0.0308 |
| lastly_iou_mean   |   0.8437 |            0.9001 | +0.0564 |
| conf_mean         |   0.8587 |            0.8648 | +0.0061 |
| iou_mean          |   0.8607 |            0.8970 | +0.0363 |
| D_conf            |   0.8412 |            0.5839 | -0.2573 |
| D_iou             |   0.8419 |            0.5951 | -0.2468 |
| **TOPSIS_score**  | **0.8571** |        **0.8391** | **-0.0180** |

### 4.3 结果解读

1. **`D_conf / D_iou` 跌得最狠**：VisDrone 上 0.84 → 0.58 / 0.60，几乎掉到随机。原因直接 —— 哨兵值给它们的 `D_*=1` 正是"最可疑"的上限，剔除哨兵后这两个 feature 信息所剩无几。它们是泄漏 1 的最大受益者。
2. **`*_mean` 系列跌幅中等**（-0.01 – -0.11）。判别力一部分来自零填充泄漏（泄漏 2），一部分来自"匹配上之后" conf / iou 自身的差异（真信号）。
3. **VisDrone 上 `lastly_*` / `iou_mean` 剔除哨兵后反而上升**（+0.03 – +0.06）。原因是 VisDrone 中 correct gid 也有 24.6% 未匹配（被赋哨兵 0），这部分 correct 在全集下被错排到"最可疑端"反而拖低 AUC；剔除后变干净。说明**哨兵值对 correct gid 也是噪声源**，不仅对 error 有"帮助"。
4. **`TOPSIS_score` 跌幅最小**（≈ -0.02）。综合 score 把多源信号平均，对"哨兵值集中暴击单一 feature"有稀释作用，但仍然吃了 ≈ 2 个百分点的泄漏红利。
5. **`iou_mean` 是 matched-only 下最稳的单 feature**（0.97 / 0.97 / 0.90），跌幅最小 —— 它是这套 feature 里最"实在"的信号。

---

## 5. 为什么综合 TOPSIS score 在 matched-only 下打不过 iou_mean？

直觉上"多 feature 综合 ≥ 单 feature"应当成立，但 matched-only 下 TOPSIS_score 处处被 `iou_mean` 压住：

| dataset  | TOPSIS_score | iou_mean |   差距 |
|----------|-------------:|---------:|-------:|
| VOC2012  |       0.9672 |   0.9698 | -0.003 |
| KITTI_8  |       0.9582 |   0.9729 | -0.015 |
| VisDrone |       0.8391 |   0.8970 | -0.058 |

**简短答案**：等权 TOPSIS 不是分类器，它只是"标准化 feature 的加权平均"，并不优化 AUC。三个叠加原因：

### 5.1 等权放大噪声

`box_rank.py:74` 用 `weights = np.ones(n_features) / n_features`。但各 feature 的真实判别力差异很大（VisDrone matched-only AUC）：

| feature           | AUC(matched) |
|-------------------|-------------:|
| lastly_iou_mean   |        0.900 |
| iou_mean          |        0.897 |
| lastly_conf_mean  |        0.873 |
| conf_mean         |        0.865 |
| early_iou_mean    |        0.776 |
| early_conf_mean   |        0.747 |
| **D_iou**         |    **0.595** |
| **D_conf**        |    **0.584** |

`D_conf / D_iou` 基本就是随机噪声，但 TOPSIS 仍给它们各 1/8 权重 —— 2/8 ≈ 25% 的权重在往随机方向拖。

### 5.2 Feature 高度相关，方向被"双重计票"

8 个 feature 其实只是 3 个方向：

- conf 方向：`early_conf_mean / lastly_conf_mean / conf_mean`（3 个）
- iou 方向：`early_iou_mean / lastly_iou_mean / iou_mean`（3 个）
- 延迟方向：`D_conf / D_iou`（2 个）

等权后方向权重 = conf 3/8 + iou 3/8 + 延迟 2/8。同一信号被多算几遍，但不带来新信息，反而改变综合分的"重心"。

### 5.3 TOPSIS 不为 AUC 设计

TOPSIS 用欧氏距离到理想点 / 反理想点，输出 `d⁻ / (d⁺ + d⁻)`，本质上对"标准化加权和"做单调变换。它解决的是"多准则决策"问题，不是监督学习。

> **"多 feature > 单 feature" 成立的三个条件**：被组合 feature **都比随机好** **且** 信号 **互补**（不同方向）**且** 权重 **正比于判别力**。这套数据三条都不满足。

### 5.4 可做的对照实验

- **(a)** 丢掉 `D_conf / D_iou` 再跑 TOPSIS，看综合 AUC 是否反超 `iou_mean` —— 验证"是 D_* 在拖后腿"。
- **(b)** 用各 feature 的 matched-only AUC 作权重再跑 TOPSIS —— 验证"等权本身次优"。
- **(c)** 把 8 个 feature 喂逻辑回归 / GBDT 得到"学过的"综合 score，作为理论上限。

---

## 6. 修复建议

不影响 `box_rank.py` 现有排序，只是让评估和未来重构更干净。

### 评估侧

- 报 AUC / 做 ablation 时**显式剔除 `never_matched` gid**，得到的数字才反映 feature 真正的判别力。
- 同时报"全集 AUC"和"matched-only AUC"，差值就是哨兵值贡献的部分。

### 特征定义侧

- 不要用哨兵值 `(0,0,...,1,1)`。改为显式新增二值 feature `matched_any_epoch ∈ {0,1}`，`*_mean` / `D_*` 只在 `matched=1` 时定义。
- `collect_metrics_for_gboxs` 把未匹配 epoch 标成 `NaN`（而非 `0`），下游聚合时显式选择忽略 / 计数 / 单独建 feature。把"是否匹配"和"匹配上后的 conf / iou 数值"分成两路 feature，泄漏来源透明可控。

### Ranking 侧

- 把"完全没匹配 = 最可疑"作为显式的兜底规则（rule-based fallback），而不是塞进数值 feature 里 —— 保持"特征"和"先验规则"的边界清晰。
- 如果想让综合 score 真的超过单 feature，把 TOPSIS 等权改成 AUC 加权，或换成监督学习模型。

---

## 7. 一句话总结

单 feature AUC 异常高，是**数据泄漏**：`build_gid_feature` 把"从未匹配的 gid"赋了哨兵值 `(0,0,0,0,0,0,1,1)`，而"从未匹配"几乎等价于"error gid"（错类别 / 错位置 / 冗余框都过不了 `cls==cls && IoU≥0.5` 的匹配条件）—— 等于把 label 直接抄进了特征里。次要泄漏来源是 `collect_metrics_for_gboxs` 对未匹配 epoch 的零填充，以及匹配条件本身就与 fault_type 定义耦合。想看 feature 真正的判别力，请在评估时剔除哨兵样本（脚本：`leak_check.py`）。
