# YOLOv7 框级训练动态特征重要性分析报告

## 1. 目标

本文档量化 `ours/discussion/features/box_features.py` 中 8 个 GT box 级训练动态特征在 3 个数据集、1 个模型设置下区分 `error / correct` box 的能力。

- 数据集：`VOC2012`、`KITTI_8`、`VisDrone`
- 模型：`YOLOv7`
- 任务：给定 GT box 的训练动态特征，判断该 box 是否为注错 box
- 正类：`error box`，即 `fault_type != 0`
- 负类：`correct box`，即 `fault_type == 0`
- 主要指标：AUC。AUC 越高，说明该特征越能把 error box 排在 correct box 前面。

本文重点区分两类评估：

1. **All GT boxes**：包含所有 GT box。未匹配上的 GT box 会被 `build_gid_feature` 赋极端可疑值。
2. **Matched-only GT boxes**：只评估至少在某个 epoch 匹配到预测框的 GT box，用于观察排除 never-matched 哨兵样本后的真实训练动态信号。

## 2. 特征定义

`box_features.py::build_gid_feature` 为每个 GT box 构造 8 个特征：

| 特征 | 含义 | 方向 |
|---|---|---|
| `early_conf_mean` | 早期窗口内匹配预测框置信度均值 | 越小越可疑 |
| `early_iou_mean` | 早期窗口内匹配预测框 IoU 均值 | 越小越可疑 |
| `lastly_conf_mean` | 后期窗口内匹配预测框置信度均值 | 越小越可疑 |
| `lastly_iou_mean` | 后期窗口内匹配预测框 IoU 均值 | 越小越可疑 |
| `conf_mean` | 全训练过程置信度均值 | 越小越可疑 |
| `iou_mean` | 全训练过程 IoU 均值 | 越小越可疑 |
| `D_conf` | 置信度首次超过 `0.5 * lastly_conf_mean` 的相对 epoch | 越大越可疑 |
| `D_iou` | IoU 首次超过 `0.5 * lastly_iou_mean` 的相对 epoch | 越大越可疑 |

TOPSIS 使用上述 8 个特征等权综合，得到一个无监督综合可疑度分数 `TOPSIS_score`。

## 3. 数据划分与匹配耦合

`search_match_PG` 的匹配逻辑与常规目标检测评估中的核心思想一致：按置信度从高到低遍历预测框，要求类别一致、IoU 达到阈值，并保证一个 GT box 只能被一个预测框匹配。

这套匹配规则本身合理，但它与注入错误类型存在强耦合：

- `cls_fault` 会破坏类别一致条件；
- `loc_fault` 会降低 IoU；
- `redundancy_fault` 通常没有真实预测框对应；
- 因此 never-matched 与 error label 强相关。

### 3.1 Matched / unmatched 按错误类型拆分

**VOC2012**

| fault_type | total | matched | unmatched | matched% | unmatched% |
|---|---:|---:|---:|---:|---:|
| no_fault | 9529 | 9433 | 96 | 0.990 | 0.010 |
| cls_fault | 1360 | 580 | 780 | 0.426 | 0.574 |
| loc_fault | 1360 | 315 | 1045 | 0.232 | 0.768 |
| redundancy_fault | 1360 | 17 | 1343 | 0.013 | 0.988 |
| ALL_ERROR | 4080 | 912 | 3168 | 0.224 | 0.776 |

**KITTI_8**

| fault_type | total | matched | unmatched | matched% | unmatched% |
|---|---:|---:|---:|---:|---:|
| no_fault | 14297 | 13941 | 356 | 0.975 | 0.025 |
| cls_fault | 2042 | 362 | 1680 | 0.177 | 0.823 |
| loc_fault | 2042 | 527 | 1515 | 0.258 | 0.742 |
| redundancy_fault | 2042 | 95 | 1947 | 0.047 | 0.953 |
| ALL_ERROR | 6126 | 984 | 5142 | 0.161 | 0.839 |

**VisDrone**

| fault_type | total | matched | unmatched | matched% | unmatched% |
|---|---:|---:|---:|---:|---:|
| no_fault | 240243 | 181238 | 59005 | 0.754 | 0.246 |
| cls_fault | 34320 | 1798 | 32522 | 0.052 | 0.948 |
| loc_fault | 34320 | 6672 | 27648 | 0.194 | 0.806 |
| redundancy_fault | 34320 | 14 | 34306 | 0.000 | 1.000 |
| ALL_ERROR | 102960 | 8484 | 94476 | 0.082 | 0.918 |

结论：matched-only 中仍然保留一部分错误 GT box，主要来自 `cls_fault` 和 `loc_fault`。但大多数错误 box，尤其 `redundancy_fault`，集中在 unmatched 子集中。

## 4. Unmatched 二值规则的判别力

只使用一个二值规则：

```text
score = 1, if GT box never matched any predicted box
score = 0, otherwise
```

其 AUC 已经很高：

| Dataset | error 中 unmatched | correct 中 unmatched | AUC |
|---|---:|---:|---:|
| VOC2012 | 3168/4080 = 0.776 | 96/9529 = 0.010 | 0.8832 |
| KITTI_8 | 5142/6126 = 0.839 | 356/14297 = 0.025 | 0.9072 |
| VisDrone | 94476/102960 = 0.918 | 59005/240243 = 0.246 | 0.8360 |

这说明“是否从未匹配”本身已经是强判别信号。若把 never-matched GT box 的各特征直接设置为最可疑端值 `(0,0,0,0,0,0,1,1)`，就会把这个二值信号复制进每个 feature，导致 all-box AUC 偏高。

## 5. All vs Matched-only AUC

### 5.1 VOC2012

| feature | AUC(all) | AUC(matched-only) | delta |
|---|---:|---:|---:|
| early_conf_mean | 0.9511 | 0.9266 | -0.0245 |
| early_iou_mean | 0.9542 | 0.9404 | -0.0138 |
| lastly_conf_mean | 0.9658 | 0.8840 | -0.0818 |
| lastly_iou_mean | 0.9577 | 0.8471 | -0.1106 |
| conf_mean | 0.9848 | 0.9593 | -0.0255 |
| iou_mean | 0.9872 | 0.9698 | -0.0174 |
| D_conf | 0.9692 | 0.8886 | -0.0806 |
| D_iou | 0.9761 | 0.9201 | -0.0560 |
| TOPSIS_score | 0.9866 | 0.9672 | -0.0194 |

### 5.2 KITTI_8

| feature | AUC(all) | AUC(matched-only) | delta |
|---|---:|---:|---:|
| early_conf_mean | 0.9451 | 0.9024 | -0.0427 |
| early_iou_mean | 0.9478 | 0.9200 | -0.0278 |
| lastly_conf_mean | 0.9654 | 0.9004 | -0.0650 |
| lastly_iou_mean | 0.9658 | 0.9028 | -0.0630 |
| conf_mean | 0.9787 | 0.9565 | -0.0222 |
| iou_mean | 0.9813 | 0.9729 | -0.0084 |
| D_conf | 0.9483 | 0.7622 | -0.1861 |
| D_iou | 0.9535 | 0.7954 | -0.1581 |
| TOPSIS_score | 0.9790 | 0.9582 | -0.0208 |

### 5.3 VisDrone

| feature | AUC(all) | AUC(matched-only) | delta |
|---|---:|---:|---:|
| early_conf_mean | 0.7959 | 0.7467 | -0.0492 |
| early_iou_mean | 0.7977 | 0.7762 | -0.0215 |
| lastly_conf_mean | 0.8420 | 0.8728 | +0.0308 |
| lastly_iou_mean | 0.8437 | 0.9001 | +0.0564 |
| conf_mean | 0.8587 | 0.8648 | +0.0061 |
| iou_mean | 0.8607 | 0.8970 | +0.0363 |
| D_conf | 0.8412 | 0.5839 | -0.2573 |
| D_iou | 0.8419 | 0.5951 | -0.2468 |
| TOPSIS_score | 0.8571 | 0.8391 | -0.0180 |

## 6. Matched-only 下的特征重要性

matched-only 更能反映“匹配上之后的训练动态”对 error/correct 的区分能力。按 3 个数据集 matched-only AUC 的平均值排序：

| rank | feature | VOC2012 | KITTI_8 | VisDrone | mean AUC |
|---:|---|---:|---:|---:|---:|
| 1 | iou_mean | 0.9698 | 0.9729 | 0.8970 | 0.9466 |
| 2 | conf_mean | 0.9593 | 0.9565 | 0.8648 | 0.9269 |
| 3 | TOPSIS_score | 0.9672 | 0.9582 | 0.8391 | 0.9215 |
| 4 | lastly_conf_mean | 0.8840 | 0.9004 | 0.8728 | 0.8857 |
| 5 | lastly_iou_mean | 0.8471 | 0.9028 | 0.9001 | 0.8833 |
| 6 | early_iou_mean | 0.9404 | 0.9200 | 0.7762 | 0.8789 |
| 7 | early_conf_mean | 0.9266 | 0.9024 | 0.7467 | 0.8586 |
| 8 | D_iou | 0.9201 | 0.7954 | 0.5951 | 0.7702 |
| 9 | D_conf | 0.8886 | 0.7622 | 0.5839 | 0.7449 |

按 mean AUC 严格重排后，8 个原始 feature 的重要性为：

| rank | feature | mean matched-only AUC | 解读 |
|---:|---|---:|---|
| 1 | `iou_mean` | 0.9466 | 最稳定、最强的单特征，反映整个训练过程预测框与 GT box 的平均定位质量 |
| 2 | `conf_mean` | 0.9269 | 第二强，反映整个训练过程模型对该 GT box 的平均置信度 |
| 3 | `lastly_conf_mean` | 0.8857 | 后期置信度，说明训练收敛后的预测信心仍有明显区分力 |
| 4 | `lastly_iou_mean` | 0.8833 | 后期定位质量，VisDrone 上很强，但 VOC2012 上弱于全程均值 |
| 5 | `early_iou_mean` | 0.8789 | 早期定位质量仍有一定判别力 |
| 6 | `early_conf_mean` | 0.8586 | 早期置信度，整体弱于早期 IoU |
| 7 | `D_iou` | 0.7702 | 起量延迟类特征，对数据集敏感，VisDrone 上接近随机 |
| 8 | `D_conf` | 0.7449 | 最弱，matched-only 下不稳定，受 never-matched 哨兵值影响较大 |

注意：上表中 `TOPSIS_score` 不是原始 8 个 feature 之一，因此最终原始特征排序不包含它。`TOPSIS_score` 的 matched-only 平均 AUC 为 0.9215，低于 `iou_mean` 和 `conf_mean`。

## 7. 结果解释

### 7.1 `iou_mean` 是最重要的单特征

`iou_mean` 在三个数据集 matched-only 下均为最强或接近最强：

- VOC2012：0.9698
- KITTI_8：0.9729
- VisDrone：0.8970

这说明即使排除 never-matched 样本，error box 在“匹配上之后”的平均定位质量仍显著低于 correct box。对于 `loc_fault`，这是直接信号；对于部分 `cls_fault` 和 `redundancy_fault`，能被匹配上的样本也往往对应不稳定或低质量匹配。

### 7.2 全程均值优于窗口均值

`iou_mean` 和 `conf_mean` 的稳定性明显好于 early/lastly 特征。原因是全程均值整合了整个训练过程中的匹配质量，抗单个阶段波动能力更强。

### 7.3 后期特征仍有价值

`lastly_conf_mean` 和 `lastly_iou_mean` 在 matched-only 下平均 AUC 接近 0.88。尤其在 VisDrone 上，`lastly_iou_mean` 达到 0.9001，高于 `iou_mean` 的 0.8970。这说明在困难密集场景中，训练后期的定位质量可能比全程平均更能反映最终错误风险。

### 7.4 起量延迟特征不稳定

`D_conf` 和 `D_iou` 在 all-box 设置下看起来较强，但 matched-only 后明显下降，尤其 VisDrone：

- `D_conf`：0.8412 -> 0.5839
- `D_iou`：0.8419 -> 0.5951

这说明 `D_*` 的很多判别力来自 never-matched 样本的极端赋值，而非 matched 样本内部的稳定训练动态差异。

### 7.5 TOPSIS 的意义和限制

TOPSIS 的作用是提供一个无监督、多特征综合可疑度分数。它不使用 error/correct 标签学习权重，因此适合作为没有验证标签时的启发式排序方法。

但当前等权 TOPSIS 不应被解释为“必然优于最强单特征”。matched-only 下：

| Dataset | TOPSIS_score | iou_mean |
|---|---:|---:|
| VOC2012 | 0.9672 | 0.9698 |
| KITTI_8 | 0.9582 | 0.9729 |
| VisDrone | 0.8391 | 0.8970 |

TOPSIS 低于 `iou_mean`，说明等权综合把弱特征和强特征同等纳入，稀释了最强信号。特别是 `D_conf/D_iou` 在 matched-only 下较弱，却仍各占 1/8 权重。

因此，TOPSIS 的定位应是“无监督综合排序 baseline”，而不是“最优 error/correct 分类器”。

## 8. 结论

1. **是否 never-matched 本身就是强判别规则**：仅用 `unmatched=1` 的二值规则，AUC 已达 0.8360-0.9072。
2. **all-box AUC 会受到哨兵值放大**：never-matched GT box 被赋极端可疑值，会把匹配失败信号复制进所有 feature。
3. **matched-only 下仍存在真实训练动态信号**：`iou_mean`、`conf_mean` 等特征在排除 never-matched 后仍有高 AUC。
4. **最重要的原始单特征是 `iou_mean`**：三个数据集 matched-only 平均 AUC 为 0.9466。
5. **`conf_mean` 是第二重要特征**：matched-only 平均 AUC 为 0.9269。
6. **后期特征有补充价值**：`lastly_conf_mean` 和 `lastly_iou_mean` 平均 AUC 接近 0.88，在 VisDrone 上尤其有效。
7. **`D_conf/D_iou` 不稳定**：其 all-box 表现很大程度受 unmatched 哨兵值影响，matched-only 后尤其在 VisDrone 上接近随机。
8. **等权 TOPSIS 不是最优组合器**：它提供无监督综合分数，但当前配置下不如 `iou_mean`；后续应考虑去除弱特征、加权 TOPSIS 或监督学习权重作为对照。

## 9. 建议

后续报告和论文表述中建议同时给出：

- `unmatched=1` 二值规则 AUC；
- all-box AUC；
- matched-only AUC；
- matched / unmatched 按 `fault_type` 的数量拆分；
- TOPSIS 与最强单特征的对比。

若继续优化排序方法，建议优先做三组消融：

1. 只使用 `iou_mean`；
2. 使用 `iou_mean + conf_mean + lastly_iou_mean + lastly_conf_mean`；
3. 去掉 `D_conf/D_iou` 后重新跑 TOPSIS。

这样可以判断多特征综合是否真正优于最强单特征，以及哪些特征在综合排序中带来边际贡献。
