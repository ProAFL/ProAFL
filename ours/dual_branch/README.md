# 双分支对称八维流程

正式入口 `process_feature.py` 默认采用上一版对称八维：类别 4 维、几何 4 维，
两分支等优先级融合。当前 VisDrone/YOLOv7 图片系数为 `0.5`。

| 过程属性 | 类别分支 | 几何分支 |
| --- | --- | --- |
| 后期分歧 | `late_conf_deficit` | `late_iou_deficit` |
| 全程分歧 | `all_conf_deficit` | `all_iou_deficit` |
| 首次达到阈值 | `first_conf_reach` | `first_iou_reach` |
| 持续分歧 | `persistent_class_margin` | `persistent_iou_deficit` |

六维模式使用表中前三行；八维模式再加入最后一行。各分支分别做三项或四项等权 TOPSIS，
只在至少有一次预测重叠的标注内计算异常百分位，
最终取 `max(class_percentile, geometry_percentile)`。从未匹配的标注沿用原规则，
合并时单独放在最前。特征和排序不使用正确标签、故障类型或干净 test 标签。

## 运行

流水线使用 Python 常量配置，不解析命令行参数。直接修改 `run_pipeline.py` 顶部：

```python
GEOMETRY_WEIGHT = 1.0         # 正式 4+4 对称八维，两分支等优先级
FEATURE_DIMENSIONS = 8        # 改为 6 使用原版 3+3 六维；8 使用扩展 4+4 八维
PREDICTIONS_DIR_OVERRIDE = None  # 使用 config.py 的预测目录
REUSE_METRICS = False         # 默认重新收集；同一预测来源的度量可设 True 复用
```

从仓库根目录运行：

```bash
python -m ours.dual_branch.run_pipeline
```

`run_pipeline.py` 的 `main` 读取上述常量，默认选择正式对称八维。
原六项特征始终保留。`FEATURE_DIMENSIONS = 6` 会独立运行原六维排序，
不计算新增两项，也不回查完整类别预测得分；两种模式都集中在 `process_feature.py`。
流水线依次收集标注轨迹、收集漏标候选轨迹、排序标注、排序图片、合并排名。
`REUSE_METRICS = True` 仅跳过前两步，仍重建特征与排名；八维模式会回查完整类别得分。
复用模式会先检查两份过程度量文件存在，缺失则在写入前报错。

Python 调用：

```python
from ours.dual_branch.run_pipeline import main

main(reuse_metrics=True)
```

如果已有图片排名，仅更新标注排序和最终合并排名：

```bash
python -m ours.dual_branch.process_feature
python -m ours.dual_branch.merge_rank
```

## 配置与输入

`config.py` 统一配置数据集、模型、故障比例、repeat、轮数及输入/输出目录。
本次默认是 VisDrone / YOLOv7 / 0.1 / repeat_1，50 轮，最后 10 轮作为后期窗口。
类别首次达到阈值为 `0.1`，IoU 首次达到阈值为 `0.45`，均使用严格大于判定。

预测目录统一使用不带实验编号的路径：

```text
<DATA_ROOT>/collection_process_info/<DATASETNAME>/<MODEL_NAME>/collected_predict_boxes/inject_<FAULT_RATIO>_repeat/repeat_<REPEAT_ID>
```

例如 VOC/YOLOv7 使用 `inject_0.1_repeat/repeat_1`。
切换预测来源后必须用 `REUSE_METRICS = False` 重新采集标注和图片过程度量，
避免旧缓存中的预测框 ID 与新预测不一致。默认配置会重新采集；
完成后，如预测和标注均未变化，可设为 `True` 复用。
也可以用流水线的 `PREDICTIONS_DIR_OVERRIDE` 常量显式覆盖预测目录。
`EPOCHS` 必须与实际采集轮数一致。

八维模式下，`process_feature.py` 内部按每轮已有的匹配预测框 ID 回查完整 `probs`，
逐轮释放预测数据，不重新训练或重新做框两两匹配；同时检查 ID 和标注类别得分，
防止静默混用其他实验。以下两项持续分歧都采用“平均冲突强度 × 最长连续冲突占比”：

- 类别：可靠匹配（IoU >= 0.5、最高类别得分 >= 0.25）下的平均相对类别 margin，
  乘以同一替代类别以 margin >= 0.5 连续获得支持的最长轮数 / 总轮数。
- 几何：`observed * (1 - IoU)` 的全程平均，乘以连续存在匹配且 IoU <= 0.45
  的最长轮数 / 总轮数。没有观测的轮次记零证据，并中断连续段。

六维模式输出原六项特征。八维模式输出九项统计，其中八项参与排序，
`alternative_class_persistence` 作为联合类别冲突强度的辅助量。
元数据分别记录实际打分维度和辅助统计。

## 输出与下游

当前过程度量和排序目录分别为：

```text
<DATA_ROOT>/dual_branch/process_metrics/visdrone/yolov7/0.1_repeat/repeat_1/
<DATA_ROOT>/dual_branch/process_features/visdrone/yolov7/0.1_repeat/repeat_1/
```

标注输出仍为 `process_features.json`、`topsis_ranking.json`、`branch_scores.json`、
`never_overlapped.json`；图片输出及最终 `merged_rank.json` 结构保持兼容。
`repair.py` 继续读取原来的 `merged_rank.json` 路径。
运行上述命令会更新配置目录中的对应结果，复用模式不改动两份过程度量文件。

仅生成独立预览：

```bash
python -m ours.dual_branch.process_feature --output-dir /tmp/symmetric_preview
```

默认 `merge_rank` 不会自动读取这个预览目录。

本次配置的同预算 137280 离线修复模拟为：分类 29856、定位 18220、
冗余 33930、漏标 22812，合计 104818，修复率 76.35%。
这是当前 repeat 的注入故障评估结果，尚未验证重训练 mAP。

正式实现已集中在 `process_feature.py`，流水线可用 `FEATURE_DIMENSIONS` 选择 6 或 8。
`GEOMETRY_WEIGHT` 可调整分支优先级；程序调用时，`build_results` 与
`save_results` 应传入相同的权重和 `feature_dimensions`。
