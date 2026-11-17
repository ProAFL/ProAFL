# 其他 baseline 的可选图片混排

运行 `python -m baselines_new.others`，修改文件顶部的 Python 常量选择实验。
每次运行自动依次执行 `BASELINES = ("entropy", "deepgini", "margin", "loss")`，
分别保存四份 `rank.joblib`，无需设置单个 `BASELINE_NAME`。

- `ENABLE_IMAGE_MIXING = False`：原 baseline 行为，未匹配 anno 置前，
  其余 anno 按各方法分数降序排列，所有图片按名称追加到末尾。
- `ENABLE_IMAGE_MIXING = True`：保留 anno 打分与匹配方式，嫁接 ObjectLab
  图片漏标分，用组内百分位混排。
- `IMAGE_PERCENTILE_SCALE = 1.0`：直接比较两组百分位；设置更小的正数会
  让图片相对提前。例如 0.1 表示图片百分位乘 0.1 后参加混排。

图片打分直接复用 `objectLab.py` 的 `box_matrices` 和 `image_miss_score`，
包括该文件当前的 `LOW_CONF_THRESHOLD`、`HIGH_CONF_THRESHOLD`、
`SIM_ALPHA`、`SIM_SIGMA`；按本次输入数据计算全局最小相似度。
只复用打分参数与函数，输入始终来自 `others.py` 指定的实验路径。
anno 仍使用全部采集预测进行 `choose_prediction` 匹配，不额外应用图片分支阈值。

开启后的总顺序：未匹配 anno → 有分数的 anno/有漏标证据图片混排 → 无漏标证据图片。
anno 分数越大越可疑，图片分数越小越可疑；先分别按可疑程度排序，再用
`(首个同分位置 + 最后同分位置 - 1) / (2 * 组内条目数)` 转为百分位，位置从 1 开始。
无漏标证据图片（质量分为 1）不参与图片组的百分位计算。
百分位越小越靠前，跨组同百分位时 anno 在前，组内同分按 ID 或图片名排列。
百分位表达组内相对名次，不代表故障概率；混排是嫁接后的方法，不是原 baseline。

结果仍为 joblib 保存的纯 anno ID / 图片名列表。关闭时路径保持：

`OUTPUT_ROOT/{baseline_name}/{DATASET_NAME}/{MODEL_NAME}/{FAULT_RATIO}/repeat_{REPEAT_ID}/rank.joblib`

`OUTPUT_ROOT` 默认是 `DATA_ROOT/rank`。无论混排开关是否开启，均使用上述路径，
再次运行会覆盖对应方法原有的 `rank.joblib`。
