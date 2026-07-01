# Motivation Experiment

This directory contains a small experiment that visualizes the core idea of
ours: annotation errors leave different training-process traces from correct
annotations.

The experiment uses the collected GT-box-level training metrics:

- `conf_list`: matched prediction confidence across epochs
- `iou_list`: matched prediction IoU across epochs

For each matched GT box, the script compares correct boxes (`fault_type == 0`)
against erroneous boxes (`fault_type != 0`) and visualizes:

1. mean confidence trajectory over epochs;
2. mean IoU trajectory over epochs;
3. distributions of simple process features such as mean/late confidence and
   mean/late IoU;
4. a confidence-vs-IoU scatter plot.

The plots are intended as a motivation figure: if the training process is
informative, correct boxes should become confident and well-localized more
consistently than erroneous boxes.

Run example:

```bash
PYTHONPATH=. python ours/motivation/motivation_process_features.py \
  --dataset-name VOC2012 \
  --model-name YOLOv7
```

Default output:

```text
ours/motivation/results/<dataset>/<model>/
```

Use `--output-dir` to write figures to another directory.
