# YOLOv7 图像级 best-cluster 特征重要性分析报告

## 1. 目标

本文档分析 `img_rank_2` 口径下 4 个图像级特征对区分图像是否包含 `missing_fault` 的重要程度。

- 数据集：`VOC2012`、`KITTI_8`、`VisDrone`
- 模型：`YOLOv7`
- 正类：包含 miss fault 的图像
- 负类：不包含 miss fault 的图像
- 评估范围：全部图像一起评估，不再按是否存在未匹配预测簇拆分
- 主要指标：AUC。AUC 越高，说明该特征越能把 miss fault 图像排在 clean 图像前面。

## 2. 特征构造流程

图像级 miss fault 检测基于未匹配高置信预测框：

1. 对每张图像收集最后若干 epoch 中未被任何 GT box 匹配到、且置信度超过阈值的预测框。
2. 对这些未匹配预测框按 IoU 聚类。
3. 对每个 cluster 计算 4 个 cluster-level 特征。
4. 对 cluster 使用 TOPSIS 得到综合分数。
5. 每张图像取 TOPSIS 分数最高的 cluster，即 best cluster。
6. best cluster 的 4 个特征作为该图像的 image-level 特征。

若某张图没有未匹配预测簇，则 4 个特征均为 0。

## 3. 4 个特征定义

| Feature | 含义 | 方向 |
|---|---|---|
| `conf` | best cluster 内预测框平均置信度 | 越大越可疑 |
| `stab` | best cluster 内预测框两两 IoU 平均值，表示空间稳定性 | 越大越可疑 |
| `cls_consis` | best cluster 内预测类别一致性 | 越大越可疑 |
| `e_freq` | best cluster 跨 epoch 出现频率 | 越大越可疑 |

## 4. 全图像 AUC 结果

| Dataset | `conf` | `e_freq` | `cls_consis` | `stab` |
|---|---:|---:|---:|---:|
| VOC2012 | **0.7188** | 0.7180 | 0.7081 | 0.7029 |
| KITTI_8 | **0.7557** | 0.7403 | 0.7199 | 0.7225 |
| VisDrone | **0.6753** | 0.6729 | 0.6203 | 0.6074 |
| Mean | **0.7166** | 0.7104 | 0.6828 | 0.6776 |

## 5. 特征重要性排序

| Rank | Feature | Mean AUC | 结论 |
|---:|---|---:|---|
| 1 | `conf` | **0.7166** | 最重要、最稳定的单特征 |
| 2 | `e_freq` | 0.7104 | 与 `conf` 接近，第二重要 |
| 3 | `cls_consis` | 0.6828 | 有一定区分力，但明显弱于前两个 |
| 4 | `stab` | 0.6776 | 最弱，跨数据集稳定性一般 |

## 6. 结果解释

### 6.1 `conf` 最重要

`conf` 在三个数据集上都是 AUC 最高的特征：

- VOC2012：0.7188
- KITTI_8：0.7557
- VisDrone：0.6753

这说明 miss fault 图像中的 best cluster 往往具有更高平均置信度。直观上，如果图像中存在漏标目标，模型会在该区域持续预测出一个高置信目标，但由于标注中缺少对应 GT box，该预测框无法被匹配，因此形成高置信未匹配簇。

### 6.2 `e_freq` 是第二重要信号

`e_freq` 表示 best cluster 在最后若干 epoch 中出现的频率。它在三个数据集上的 AUC 与 `conf` 很接近：

- VOC2012：0.7180
- KITTI_8：0.7403
- VisDrone：0.6729

持续跨 epoch 出现的未匹配预测簇更可能对应真实漏标对象，而不是偶然误检。因此 `e_freq` 是稳定的辅助特征。

### 6.3 `cls_consis` 和 `stab` 较弱

`cls_consis` 和 `stab` 也有一定区分能力，但整体弱于 `conf` 和 `e_freq`。

原因是：一旦形成 best cluster，不管图像是否真的包含 miss fault，该 cluster 内的预测框通常已经具有较高的类别一致性和空间稳定性。因此这两个特征在 miss/clean 之间的额外差异有限。

在 VisDrone 上这一点尤其明显：

- `cls_consis` AUC：0.6203
- `stab` AUC：0.6074

说明在密集、小目标场景中，类别一致性和空间稳定性容易同时出现在真实漏标和普通误检中，判别力下降。

## 7. 结论

在全部图像一起评估的设置下，4 个 image-level best-cluster 特征的重要性排序为：

```text
conf > e_freq > cls_consis > stab
```

最关键的两个信号是：

1. **高置信度**：漏标目标更可能产生高置信未匹配预测簇。
2. **跨 epoch 持续出现**：真实漏标目标对应的预测簇更稳定地出现在多个 epoch 中。

因此，后续如果需要简化 image-level miss fault 检测特征，优先保留 `conf` 和 `e_freq`；`cls_consis` 和 `stab` 可作为辅助特征，但不应作为主要判别依据。
