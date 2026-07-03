### 数据错误注入：

fault_inject.py

## Ours method.

### Training process info collection

#### 对于 yolov7 model:

(1) 训练多轮次模型: yolov7/train.py
(2) 收集训练信息: yolov7/collect_train_info.py

#### 对于 frcnn model:

(1) 训练多轮次模型: frcnn/train.py
(2) 收集训练信息: frcnn/collect_train_info.py

#### 对于 rtdetr model:

(1) 训练多轮次模型: rtdetr/train.py
(2) 收集训练信息: rtdetr/collect_train_info.py

### build process metrics:

ours/match_metrics/match_and_collect_metrics.py

### Rank

ours/rank/rank.py

### Repair

ours/repair/repair.py

### Retrain:

(1)构建retrain annotaion: ours/build_retrain_label_for_yolov7.py
(2)retrain: yolov7/train.py

## Baselines.
### Datactive
我们复现的核心代码在: baselines/datactive/exp_reproduct目录下
(1) ./train_classmodel.py: 训练分类模型
(2) ./inference_classmodel.py: 推理
(3) ./datactive.py: 排序

### ObjectLab
我们复现的核心代码在: baselines/other_baselines/objectlab.py

### other baselines
(1) 收集预测信息
对于 yolov7: yolov7/for_baseline_collect.py
对于 frcnn: frcnn/for_baseline_collect.py
对于 rtdetr: rtdetr/for_baseline_collect.py

(2) match: 复用yolov7/for_baseline_collect_and_match.py中的match

(3) rank: baselines/other_baselines/rank.py

# 实验的数据

