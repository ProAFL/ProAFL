# 六特征双分支流程

本目录独立实现六特征双分支方案，不依赖实验对比脚本，不读取正确标注或故障类型来打分。
原有 `ours/process_*.py`、`ours/merge_rank.py` 及其结果不变。

## 配置和流程

统一修改 `config.py` 中的数据集、模型、实验编号、轮数及阈值。
默认对应 YOLOv7 / VisDrone / repeat_1 / 1791968956。
已列出前面八组实验的编号；RT-DETR / VOC 使用 100 轮，其余使用 50 轮。

一条命令顺序执行全部五步（包括重新收集过程度量）：

```bash
python -m ours.dual_branch.run_pipline
```

`run_pipline.py` 只有一个 `main()`，直接复用五个模块中的采集、特征提取、
评分、保存和合并函数，不调用这五个模块的 `main()`，也不启动子进程。
任何一步失败都会保留原始异常并停止后续步骤；各步骤都使用 `config.py` 的当前配置。
过程度量仍保存为中间 JSON，避免大型标注轨迹和图片轨迹同时占用内存；
标注和图片排名通过 Python 对象直接传给合并函数，同时保留原有 JSON 便于检查。
旧名称 `run_pipeline.py` 仅作为兼容入口，指向同一个统一流程。
也可以在仓库根目录的 Python 环境中调用：

```python
from ours.dual_branch.run_pipline import main

main()
```

如需单独执行，五个脚本的顺序如下，均从仓库根目录启动：

```bash
python -m ours.dual_branch.process_metrics
python -m ours.dual_branch.process_metrics_img
python -m ours.dual_branch.process_feature
python -m ours.dual_branch.process_feature_img
python -m ours.dual_branch.merge_rank
```

以上仅为使用说明，编写代码时未执行这些命令。
依赖为 NumPy、SciPy、PyYAML；公共 TOPSIS 直接使用 NumPy 实现。

## 标注打分

六项特征完整保留：

- 类别分支：`late_conf_deficit`、`all_conf_deficit`、`first_conf_reach`。
- 几何分支：`late_iou_deficit`、`all_iou_deficit`、`first_iou_reach`。

两个分支分别在全部现存标注上进行等权 TOPSIS。
仅在至少有一轮预测重叠的标注中，将两分支得分分别转换为升序平均名次百分位，
再取两个百分位的最大值作为最终可疑分数，越大越可疑。
从未重叠的标注不参与百分位校准，融合分数设为 0，合并时仍按原规则单独置于最前。

`process_features.json` 保存原六特征，`branch_scores.json` 保存分支得分及融合过程。
`topsis_ranking.json` 的 `topsis_score` 字段保存融合分数，以兼容原有排名格式。
分支诊断字段不会混入最终合并排名。

## 图片与合并

图片分支沿用原规则：过滤已有标注附近的预测框，跨轮次关联位置轨迹，
选择全程置信度证据最大的代表轨迹，提取六项图片特征并进行 TOPSIS 排序。
候选筛选阈值独立配置，不能通过修改标注的 `CONF_THRESHOLD` 来改变图片筛选。

合并前，将标注融合分数、图片 TOPSIS 分数分别转换为降序平均名次百分位。
标注使用原百分位；图片使用百分位乘 `IMAGE_PERCENTILE_SCALE`，再按值升序混排。
默认仅 YOLOv7 / VisDrone 的图片系数为 0.1，其余已列实验为 1.0。
最终顺序仍为：无重叠标注 → 有证据条目的混合排名 → 无候选图片。

## 输出目录和格式

新结果统一保存到：

```text
<DATA_ROOT>/dual_branch/process_metrics/<数据集>/<模型>/0.1_repeat/repeat_1_<EXPID>/
<DATA_ROOT>/dual_branch/process_features/<数据集>/<模型>/0.1_repeat/repeat_1_<EXPID>/
```

如要复用已有过程度量，可将 `config.py` 的 `METRICS_DIR` 指向原实验的
`process_metrics` 目录，并从第三步开始；需确认轮数、实验编号及阈值与数据一致。
这不会修改已有过程度量，特征和排名仍写入新的 `FEATURES_DIR`。

最终 `merged_rank.json` 保持原有顶层字段：
`merge_rule`、`counts`、`merged_ranking`、`never_overlapped_annos`、
`no_candidate_images`、`all_ranking`。
条目的 `item_type`、`anno_id` / `image_name`、`source_rank`、
`source_topsis_score`、`total_rank`，以及中间混排组的
`group_rank`、`group_percentile`、`merge_percentile`、`merged_rank` 均保持原格式。
其中标注的 `source_topsis_score` 对应新的融合分数，图片仍对应其 TOPSIS 分数。
后续评估和修复脚本可沿用原来的读取逻辑，只需将排名路径指向新目录。
