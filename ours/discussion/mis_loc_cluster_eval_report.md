# Miss Fault 精确定位与 Cluster 排名评估报告

## 1. 评估目标

本文档评估 `ours/discussion/mis_loc.py` 对 missing fault 的精确定位能力，并对定位阈值 `IoU > 0.5 / 0.6 / 0.7 / 0.8 / 0.9` 全部做 cluster-level 统计。

评估内容：

1. 候选预测框池能否覆盖真实 missing box；
2. TOPSIS 排名靠前的 cluster 是否真的定位到了 missing fault；
3. cluster TOPSIS 分数作为二分类 score 时的 ROC/AUC；
4. 基于 cluster score 阈值扫描得到的最佳 Precision / Recall / F1；
5. top-k cluster 的命中效果。

评估设置：

- 模型：`YOLOv7`
- 数据集：`VOC2012`、`KITTI_8`、`VisDrone`
- 定位阈值：`IoU > 0.5 / 0.6 / 0.7 / 0.8 / 0.9`
- 候选 cluster：最后 5 个 epoch 中未匹配到 GT box 且置信度大于 0.6 的预测框，经 IoU > 0.6 聚类得到
- cluster score：4 个 cluster 特征 `conf / stab / cls_consis / e_freq` 的等权 TOPSIS 分数

说明：默认 `python -m ours.discussion.mis_loc` 在读取 VisDrone 的 `match_v2.json` 时遇到 JSONDecodeError，因此本次统计直接使用已生成的 `img_to_nomatched_pboxs.json` 中间结果构建 cluster，并调用 `mis_loc.py` 中新增的 cluster-level 评估函数。

## 2. 指标定义

### 2.1 Missed Box 定位成功率

对每个真实 missing box，若同图像中存在任意候选预测框满足：

```text
IoU(predicted_box, missing_box) > threshold
```

则认为该 missing box 被成功定位。

```text
loc_success_rate = 成功定位的 missing box 数 / missing box 总数
```

该指标是 box-level 定位召回率，不统计误报，因此不是 precision。

### 2.2 Cluster Label

每个候选 cluster 是一条评估样本：

- `label = 1`：cluster 中至少一个预测框与任意真实 missing box 的 IoU 大于当前阈值；
- `label = 0`：cluster 没有定位到任何 missing box。

用 cluster TOPSIS score 作为排序分数，计算 ROC/AUC、最佳 F1 和 top-k 命中指标。

注意：`positive clusters` 与 `located boxes` 不要求相等。一个 cluster 可能覆盖多个 missing boxes，一个 missing box 也可能被多个 cluster 命中。

## 3. 阈值扫描总表

### 3.1 VOC2012

| IoU阈值 | positive clusters | located boxes | loc_success_rate | cluster AUC | best precision | best recall | best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 672 | 675 | 0.4963 | 0.5996 | 0.3356 | 0.8095 | 0.4745 |
| 0.6 | 664 | 666 | 0.4897 | 0.6031 | 0.3331 | 0.8133 | 0.4726 |
| 0.7 | 644 | 645 | 0.4743 | 0.6106 | 0.3263 | 0.8214 | 0.4671 |
| 0.8 | 588 | 588 | 0.4324 | 0.6173 | 0.3023 | 0.8333 | 0.4436 |
| 0.9 | 453 | 453 | 0.3331 | 0.6299 | 0.2856 | 0.5541 | 0.3769 |

### 3.2 KITTI_8

| IoU阈值 | positive clusters | located boxes | loc_success_rate | cluster AUC | best precision | best recall | best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1572 | 1570 | 0.7689 | 0.5620 | 0.3362 | 0.8887 | 0.4879 |
| 0.6 | 1551 | 1553 | 0.7605 | 0.5672 | 0.3338 | 0.8943 | 0.4862 |
| 0.7 | 1522 | 1524 | 0.7463 | 0.5733 | 0.3302 | 0.9014 | 0.4834 |
| 0.8 | 1421 | 1423 | 0.6969 | 0.5924 | 0.3133 | 0.9198 | 0.4674 |
| 0.9 | 941 | 941 | 0.4608 | 0.6251 | 0.2264 | 0.8672 | 0.3591 |

### 3.3 VisDrone

| IoU阈值 | positive clusters | located boxes | loc_success_rate | cluster AUC | best precision | best recall | best F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 9466 | 9468 | 0.2759 | 0.5263 | 0.3040 | 0.9988 | 0.4661 |
| 0.6 | 9378 | 9382 | 0.2734 | 0.5302 | 0.3013 | 0.9977 | 0.4628 |
| 0.7 | 9041 | 9045 | 0.2635 | 0.5428 | 0.3005 | 0.8960 | 0.4500 |
| 0.8 | 7942 | 7943 | 0.2314 | 0.5763 | 0.2778 | 0.8463 | 0.4182 |
| 0.9 | 4329 | 4330 | 0.1262 | 0.6542 | 0.2024 | 0.6556 | 0.3093 |

## 4. Top-K Cluster 命中摘要

### 4.1 Top10

| Dataset | IoU阈值 | hits@10 | precision@10 | recall@10 | F1@10 |
|---|---:|---:|---:|---:|---:|
| VOC2012 | 0.5 | 5 | 0.5000 | 0.0074 | 0.0147 |
| VOC2012 | 0.6 | 5 | 0.5000 | 0.0075 | 0.0148 |
| VOC2012 | 0.7 | 5 | 0.5000 | 0.0078 | 0.0153 |
| VOC2012 | 0.8 | 4 | 0.4000 | 0.0068 | 0.0134 |
| VOC2012 | 0.9 | 4 | 0.4000 | 0.0088 | 0.0173 |
| KITTI_8 | 0.5 | 5 | 0.5000 | 0.0032 | 0.0063 |
| KITTI_8 | 0.6 | 5 | 0.5000 | 0.0032 | 0.0064 |
| KITTI_8 | 0.7 | 5 | 0.5000 | 0.0033 | 0.0065 |
| KITTI_8 | 0.8 | 5 | 0.5000 | 0.0035 | 0.0070 |
| KITTI_8 | 0.9 | 5 | 0.5000 | 0.0053 | 0.0105 |
| VisDrone | 0.5 | 2 | 0.2000 | 0.0002 | 0.0004 |
| VisDrone | 0.6 | 2 | 0.2000 | 0.0002 | 0.0004 |
| VisDrone | 0.7 | 2 | 0.2000 | 0.0002 | 0.0004 |
| VisDrone | 0.8 | 2 | 0.2000 | 0.0003 | 0.0005 |
| VisDrone | 0.9 | 2 | 0.2000 | 0.0005 | 0.0009 |

### 4.2 Top20%

| Dataset | IoU阈值 | hits@20% | precision@20% | recall@20% | F1@20% |
|---|---:|---:|---:|---:|---:|
| VOC2012 | 0.5 | 178 | 0.3964 | 0.2649 | 0.3176 |
| VOC2012 | 0.6 | 178 | 0.3964 | 0.2681 | 0.3199 |
| VOC2012 | 0.7 | 176 | 0.3920 | 0.2733 | 0.3220 |
| VOC2012 | 0.8 | 163 | 0.3630 | 0.2772 | 0.3143 |
| VOC2012 | 0.9 | 139 | 0.3096 | 0.3068 | 0.3082 |
| KITTI_8 | 0.5 | 369 | 0.3712 | 0.2347 | 0.2876 |
| KITTI_8 | 0.6 | 368 | 0.3702 | 0.2373 | 0.2892 |
| KITTI_8 | 0.7 | 366 | 0.3682 | 0.2405 | 0.2910 |
| KITTI_8 | 0.8 | 359 | 0.3612 | 0.2526 | 0.2973 |
| KITTI_8 | 0.9 | 274 | 0.2757 | 0.2912 | 0.2832 |
| VisDrone | 0.5 | 2031 | 0.3259 | 0.2146 | 0.2588 |
| VisDrone | 0.6 | 2029 | 0.3256 | 0.2164 | 0.2600 |
| VisDrone | 0.7 | 2021 | 0.3243 | 0.2235 | 0.2646 |
| VisDrone | 0.8 | 1968 | 0.3158 | 0.2478 | 0.2776 |
| VisDrone | 0.9 | 1490 | 0.2391 | 0.3442 | 0.2822 |

## 5. ROC 曲线文件

已为每个数据集和每个 IoU 阈值生成 ROC 图：

```text
ours/discussion/features/results/mis_loc/<dataset>/YOLOv7/cluster_loc_roc_iou_<threshold>.png
```

具体包括：

| Dataset | ROC files |
|---|---|
| VOC2012 | `cluster_loc_roc_iou_0.5.png`, `cluster_loc_roc_iou_0.6.png`, `cluster_loc_roc_iou_0.7.png`, `cluster_loc_roc_iou_0.8.png`, `cluster_loc_roc_iou_0.9.png` |
| KITTI_8 | `cluster_loc_roc_iou_0.5.png`, `cluster_loc_roc_iou_0.6.png`, `cluster_loc_roc_iou_0.7.png`, `cluster_loc_roc_iou_0.8.png`, `cluster_loc_roc_iou_0.9.png` |
| VisDrone | `cluster_loc_roc_iou_0.5.png`, `cluster_loc_roc_iou_0.6.png`, `cluster_loc_roc_iou_0.7.png`, `cluster_loc_roc_iou_0.8.png`, `cluster_loc_roc_iou_0.9.png` |

结构化结果也保存为：

```text
ours/discussion/features/results/mis_loc/cluster_eval_threshold_summary.json
```

## 6. 结果解读

### 6.1 IoU 阈值越高，missed-box 定位召回越低

这是预期结果。以 `loc_success_rate` 为例：

- VOC2012：0.4963 -> 0.3331
- KITTI_8：0.7689 -> 0.4608
- VisDrone：0.2759 -> 0.1262

阈值从 0.5 提高到 0.9 后，需要预测框与真实 missing box 高度重合，因此成功定位的 missing box 数量明显下降。

### 6.2 IoU 阈值越高，cluster AUC 反而上升

三个数据集都呈现这个趋势：

- VOC2012：0.5996 -> 0.6299
- KITTI_8：0.5620 -> 0.6251
- VisDrone：0.5263 -> 0.6542

原因是高阈值下的 positive cluster 更“纯”：只有非常准确覆盖 missing box 的 cluster 才被标为正样本。它们更容易与普通误报 cluster 拉开 TOPSIS 分数差距，因此 AUC 上升。

但这不代表定位能力变强，因为正样本数量和 located boxes 数量同时大幅减少。

### 6.3 最佳 F1 在较低 IoU 阈值下更高

三个数据集的 best F1 都在 IoU=0.5 附近最高：

- VOC2012：0.4745
- KITTI_8：0.4879
- VisDrone：0.4661

这说明如果目标是“发现 missing fault 区域”，较宽松的 IoU 阈值更符合候选定位任务；如果目标是“框级精确重合”，则 IoU=0.9 更严格但召回损失明显。

### 6.4 Precision 仍是主要瓶颈

即使在最佳 F1 阈值下，precision 大多只有 0.20-0.34。说明高分 cluster 中仍有大量没有真正覆盖 missing box 的误报区域。

这意味着当前 TOPSIS cluster score 可以提供一定排序信号，但还不足以作为精确定位 miss fault 的高精度判别器。

### 6.5 Top-k 结果显示高分 cluster 有命中倾向，但召回增长慢

Top10 的 precision 在 VOC2012/KITTI_8 上可达到 0.4-0.5，但 recall 很低，因为 positive cluster 总量较多。Top20% 能获得更高 recall，但 precision 下降到约 0.24-0.40。

因此，如果用于人工检查，top-k cluster 可以作为优先查看区域；但如果用于自动定位，还需要更强的过滤或重排序机制。

## 7. 结论

1. **候选框池确实能覆盖一部分 missing boxes**，但覆盖率对 IoU 阈值非常敏感。
2. **TOPSIS cluster score 有排序能力，但整体偏弱**，AUC 在 0.53-0.65 之间。
3. **严格定位阈值 IoU=0.9 下召回下降明显**，尤其 VisDrone 只有 0.1262。
4. **较宽松阈值 IoU=0.5 下 best F1 更高**，更适合作为“发现可能漏标区域”的评估。
5. **precision 是当前方法的主要短板**，需要进一步减少高分 cluster 中的误报。

## 8. 后续建议

1. 报告中同时给出 IoU=0.5 和 IoU=0.9：前者衡量区域发现能力，后者衡量精确框定位能力。
2. 增加 cluster-level 特征消融，分别看 `conf / stab / cls_consis / e_freq` 对定位命中的 AUC。
3. 加入候选 cluster 数量、cluster size、跨 epoch 覆盖率等特征，训练或加权一个更适合定位任务的 score。
4. 做 image-level top-k 后的 box-level localization recall，例如“只检查 top 10% 图像时能定位多少 missing boxes”。
