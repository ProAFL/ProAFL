# Motivation Experiment Design

## Goal

Show the core intuition of ours:

> Training-process metrics carry signals about whether a training annotation is
> correct or erroneous.

If an annotation is correct, the model should learn it more consistently during
training. Its matched prediction should become more confident and better
localized. If an annotation is wrong, the matched prediction is expected to show
weaker or less stable confidence/IoU dynamics.

## Data

For each matched GT box, use collected process metrics:

- `conf_list`: matched prediction confidence across epochs;
- `iou_list`: matched prediction IoU across epochs;
- `fault_type`: annotation status from collected GT boxes.

Labels:

- `correct`: `fault_type == 0`;
- `error`: `fault_type != 0`.

The default experiment uses matched GT boxes only. This avoids turning
never-matched sentinel values into the motivation signal.

## Visual Evidence

The script produces four figures:

1. `process_conf_trajectory.png`
   - mean confidence curve for correct vs error boxes;
   - shaded area is a bootstrap 95% confidence interval.

2. `process_iou_trajectory.png`
   - mean IoU curve for correct vs error boxes;
   - shows whether localization dynamics differ.

3. `process_feature_distributions.png`
     - boxplots for derived process features:
     `early_conf_mean`, `early_iou_mean`, `conf_mean`, `iou_mean`,
     `late_conf_mean`, `late_iou_mean`, `conf_delay`, `iou_delay`.

4. `conf_iou_feature_space.png`
   - scatter plot of `conf_mean` vs `iou_mean`;
   - shows whether the two groups occupy different regions.

## Interpretation

The motivation is supported if:

- correct boxes have consistently higher confidence/IoU trajectories;
- error boxes show lower mean/late confidence or IoU;
- delay features are larger for error boxes;
- the scatter plot shows separable regions between correct and error boxes.

The experiment is not intended to be the final evaluation metric. It is a
diagnostic visualization explaining why process features are meaningful before
ranking is evaluated with APFD/FPR/FNR/Top1/Exam.
