### Annotation Fault Injection:

fault_inject.py

## Ours method.

### Training process info collection

#### For yolov7 model:

(1) Training multiple epochs: yolov7/train.py
(2) Collect training information: yolov7/collect_train_info.py

#### text frcnn model:

(1) Training multiple epochs: frcnn/train.py
(2) Collect training information: frcnn/collect_train_info.py

#### text rtdetr model:

(1) Training multiple epochs: rtdetr/train.py
(2) Collect training information: rtdetr/collect_train_info.py

### build process metrics:

ours/match_metrics/match_and_collect_metrics.py

### Rank

ours/rank/rank.py

### Repair

ours/repair/repair.py

### Retrain:

(1) Build retraining annotation: ours/build_retrain_label_for_yolov7.py
(2) retrain: yolov7/train.py

## Baselines.

### Datactive

The core reproduction code is in:: baselines/datactive/exp_reproduct
(1) ./train_classmodel.py: Training classifier
(2) ./inference_classmodel.py: infer
(3) ./datactive.py: ranking

### ObjectLab

The core reproduction code is in: baselines/other_baselines/objectlab.py

### other baselines

(1) model prediction collection
For yolov7: yolov7/for_baseline_collect.py
For frcnn: frcnn/for_baseline_collect.py
For rtdetr: rtdetr/for_baseline_collect.py

(2) match: yolov7/for_baseline_collect_and_match.py

(3) rank: baselines/other_baselines/rank.py
