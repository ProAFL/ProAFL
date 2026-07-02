# Case Study

This directory builds a compact case-study figure for ours.

The script selects:

- three explicit GT-box annotation errors and visualizes their 8 box-level
  process features;
- one implicit missing-annotation error and visualizes its 4 image-level
  process features.

All values in the figure are converted to suspicious percentiles, where larger
means the sample is more suspicious under the corresponding feature direction.
This makes features with different scales directly comparable.

Run:

```bash
PYTHONPATH=. python ours/casestudy/case_study_features.py \
  --dataset-name VOC2012 \
  --model-name YOLOv7
```

Default output:

```text
ours/casestudy/results/<dataset>/<model>/case_study_process_feature_contrast.png
ours/casestudy/results/<dataset>/<model>/selected_cases.csv
```
