### text:

fault_inject.py

## Ours method.

### Training process info collection

#### text yolov7 model:

(1) text: yolov7/train.py
(2) text: yolov7/collect_train_info.py

#### text frcnn model:

(1) text: frcnn/train.py
(2) text: frcnn/collect_train_info.py

#### text rtdetr model:

(1) text: rtdetr/train.py
(2) text: rtdetr/collect_train_info.py

### build process metrics:

ours/match_metrics/match_and_collect_metrics.py

### Rank

ours/rank/rank.py

### Repair

ours/repair/repair.py

### Retrain:

(1)textretrain annotaion: ours/build_retrain_label_for_yolov7.py
(2)retrain: yolov7/train.py

## Baselines.
### Datactive
text: baselines/datactive/exp_reproducttext
(1) ./train_classmodel.py: text
(2) ./inference_classmodel.py: text
(3) ./datactive.py: ranking

### ObjectLab
text: baselines/other_baselines/objectlab.py

### other baselines
(1) text
text yolov7: yolov7/for_baseline_collect.py
text frcnn: frcnn/for_baseline_collect.py
text rtdetr: rtdetr/for_baseline_collect.py

(2) match: textyolov7/for_baseline_collect_and_match.pytextmatch

(3) rank: baselines/other_baselines/rank.py

# text

