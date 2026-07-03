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


