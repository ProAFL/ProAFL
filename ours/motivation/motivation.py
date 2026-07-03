"""
Motivation experiment for ours.

Core question:
Do correct and faulty annotations show different training-process dynamics?

This script compares matched GT boxes with fault_type == 0 against matched GT
boxes with fault_type != 0 using collected per-epoch confidence and IoU lists.
It writes motivation figures and a metric summary to disk.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

from helper.base_data_manager import exp_data_root_dir, get_collected_gt_box_json_path


FAULT_TYPE_NAME = {
    0: "correct",
    1: "class_fault",
    2: "loc_fault",
    3: "redundancy_fault",
    4: "missing_fault",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_metric_path(dataset_name, model_name, metric_path=None):
    if metric_path is not None:
        return metric_path

    metric_dir = Path(exp_data_root_dir) / "collection_bbox_level" / dataset_name / model_name / "collection_metric"
    candidates = [
        "collection_metrics_v21.json",
        "collection_metrics_v3.json",
        "collection_metrics_v2.json",
        "collection_metrics.json",
        "metrics.json",
    ]
    for filename in candidates:
        path = metric_dir / filename
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"No metric json found in {metric_dir}")


def flatten_gt_boxes(gt_json):
    gid_to_info = {}
    for img_name, boxes in gt_json.items():
        for box in boxes:
            gid_to_info[box["box_id"]] = {
                "img_name": img_name,
                "fault_type": int(box["fault_type"]),
            }
    return gid_to_info


def load_metric_by_gid(metric_path):
    metric_items = read_json(metric_path)
    gid_to_metric = {}
    for item in metric_items:
        gid = item["g_box_id"]
        gid_to_metric[gid] = {
            "conf_list": np.asarray(item["conf_list"], dtype=float),
            "iou_list": np.asarray(item["iou_list"], dtype=float),
        }
    return gid_to_metric


def first_crossing_epoch(values, threshold):
    for idx, value in enumerate(values):
        if value > threshold:
            return idx
    return len(values)


def build_records(gid_to_info, gid_to_metric, k=0.2):
    records = []
    for gid, metric in gid_to_metric.items():
        if gid not in gid_to_info:
            continue
        conf = metric["conf_list"]
        iou = metric["iou_list"]
        if len(conf) == 0 or len(iou) == 0:
            continue
        epochs = min(len(conf), len(iou))
        conf = conf[:epochs]
        iou = iou[:epochs]
        win = max(1, int(k * epochs))
        early_conf = conf[:win]
        early_iou = iou[:win]
        late_conf = conf[-win:]
        late_iou = iou[-win:]
        conf_threshold = 0.5 * float(np.mean(late_conf))
        iou_threshold = 0.5 * float(np.mean(late_iou))
        fault_type = gid_to_info[gid]["fault_type"]
        records.append({
            "gid": gid,
            "img_name": gid_to_info[gid]["img_name"],
            "fault_type": fault_type,
            "fault_name": FAULT_TYPE_NAME.get(fault_type, str(fault_type)),
            "label": "correct" if fault_type == 0 else "fault",
            "conf_list": conf,
            "iou_list": iou,
            "early_conf_mean": float(np.mean(early_conf)),
            "early_iou_mean": float(np.mean(early_iou)),
            "conf_mean": float(np.mean(conf)),
            "iou_mean": float(np.mean(iou)),
            "late_conf_mean": float(np.mean(late_conf)),
            "late_iou_mean": float(np.mean(late_iou)),
            "conf_delay": first_crossing_epoch(conf, conf_threshold) / epochs,
            "iou_delay": first_crossing_epoch(iou, iou_threshold) / epochs,
        })
    return records


def bootstrap_mean_ci(matrix, n_boot=300, seed=0):
    if len(matrix) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    mean = matrix.mean(axis=0)
    if len(matrix) == 1:
        return mean, mean, mean
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(matrix), len(matrix))
        boot.append(matrix[idx].mean(axis=0))
    boot = np.asarray(boot)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return mean, lo, hi


def plot_trajectory(records, key, ylabel, output_path):
    groups = {
        "correct": [r[key] for r in records if r["label"] == "correct"],
        "fault": [r[key] for r in records if r["label"] == "fault"],
    }
    if not groups["correct"] or not groups["fault"]:
        return

    min_epochs = min(min(len(v) for v in values) for values in groups.values())
    plt.figure(figsize=(7, 4))
    palette = {"correct": "#2f7d32", "fault": "#c62828"}
    for label, values in groups.items():
        matrix = np.vstack([v[:min_epochs] for v in values])
        mean, lo, hi = bootstrap_mean_ci(matrix)
        x = np.arange(1, min_epochs + 1)
        plt.plot(x, mean, label=f"{label} (n={len(values)})", color=palette[label], lw=2)
        plt.fill_between(x, lo, hi, color=palette[label], alpha=0.18, linewidth=0)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Training-process {ylabel}: correct vs fault")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def cliffs_delta(values_pos, values_neg):
    pos = np.asarray(values_pos, dtype=float)
    neg = np.asarray(values_neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = 0
    less = 0
    for value in pos:
        greater += int(np.sum(value > neg))
        less += int(np.sum(value < neg))
    return (greater - less) / (len(pos) * len(neg))


def plot_feature_distributions(records, output_path):
    feature_names = ["conf_mean", "iou_mean"]
    feature_labels = {
        "conf_mean": "Confidence Mean",
        "iou_mean": "IoU Mean",
    }
    rows = []
    for record in records:
        for feature_name in feature_names:
            rows.append({
                "label": record["label"],
                "feature": feature_labels[feature_name],
                "value": record[feature_name],
            })

    plot_df = pd.DataFrame(rows)
    with plt.style.context(["science", "ieee", "no-latex"]):
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        sns.boxplot(
            data=plot_df,
            x="feature",
            y="value",
            hue="label",
            order=[feature_labels[name] for name in feature_names],
            hue_order=["correct", "fault"],
            palette={"correct": "#2f7d32", "fault": "#c62828"},
            showfliers=False,
            width=0.62,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_xlabel("Process feature")
        ax.set_ylabel("Value")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            ["correct", "fault"],
            loc="upper right",
            ncol=1,
            frameon=False,
            title=None,
            fontsize=8,
            handlelength=1.3,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.35,
        )
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_conf_iou_scatter(records, output_path, max_points=12000, seed=0):
    rng = np.random.default_rng(seed)
    if len(records) > max_points:
        idx = rng.choice(len(records), max_points, replace=False)
        sampled = [records[i] for i in idx]
    else:
        sampled = records

    x = [r["conf_mean"] for r in sampled]
    y = [r["iou_mean"] for r in sampled]
    labels = [r["label"] for r in sampled]
    plt.figure(figsize=(5.5, 5))
    sns.scatterplot(x=x, y=y, hue=labels, alpha=0.35, s=12, linewidth=0)
    plt.xlabel("Mean confidence across epochs")
    plt.ylabel("Mean IoU across epochs")
    plt.title("Training-process feature space")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(records, output_path):
    feature_names = [
        "early_conf_mean",
        "early_iou_mean",
        "conf_mean",
        "iou_mean",
        "late_conf_mean",
        "late_iou_mean",
        "conf_delay",
        "iou_delay",
    ]
    correct = [r for r in records if r["label"] == "correct"]
    fault = [r for r in records if r["label"] == "fault"]

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "feature",
            "correct_mean",
            "fault_mean",
            "correct_median",
            "fault_median",
            "cliffs_delta_fault_minus_correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feature_name in feature_names:
            correct_values = [r[feature_name] for r in correct]
            fault_values = [r[feature_name] for r in fault]
            writer.writerow({
                "feature": feature_name,
                "correct_mean": np.mean(correct_values) if correct_values else "",
                "fault_mean": np.mean(fault_values) if fault_values else "",
                "correct_median": np.median(correct_values) if correct_values else "",
                "fault_median": np.median(fault_values) if fault_values else "",
                "cliffs_delta_fault_minus_correct": cliffs_delta(fault_values, correct_values),
            })


def write_records_csv(records, output_path):
    scalar_fields = [
        "gid",
        "img_name",
        "fault_type",
        "fault_name",
        "label",
        "early_conf_mean",
        "early_iou_mean",
        "conf_mean",
        "iou_mean",
        "late_conf_mean",
        "late_iou_mean",
        "conf_delay",
        "iou_delay",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in scalar_fields})


def run(args):
    gt_path = args.gt_json_path or get_collected_gt_box_json_path(args.dataset_name)
    metric_path = resolve_metric_path(args.dataset_name, args.model_name, args.metric_path)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "results" / args.dataset_name / args.model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_json = read_json(gt_path)
    gid_to_info = flatten_gt_boxes(gt_json)
    gid_to_metric = load_metric_by_gid(metric_path)
    records = build_records(gid_to_info, gid_to_metric, k=args.window_ratio)

    correct_count = sum(1 for r in records if r["label"] == "correct")
    fault_count = sum(1 for r in records if r["label"] == "fault")
    print(f"dataset={args.dataset_name}, model={args.model_name}")
    print(f"gt boxes={len(gid_to_info)}, matched metric boxes={len(gid_to_metric)}")
    print(f"used matched boxes={len(records)}, correct={correct_count}, fault={fault_count}")
    print(f"output_dir={output_dir}")

    plot_trajectory(records, "conf_list", "confidence", output_dir / "process_conf_trajectory.png")
    plot_trajectory(records, "iou_list", "IoU", output_dir / "process_iou_trajectory.png")
    plot_feature_distributions(records, output_dir / "process_feature_distributions.pdf")
    plot_conf_iou_scatter(records, output_dir / "conf_iou_feature_space.png")
    write_summary(records, output_dir / "motivation_summary.csv")
    write_records_csv(records, output_dir / "matched_box_process_features.csv")

    print("saved:")
    for filename in [
        "process_conf_trajectory.png",
        "process_iou_trajectory.png",
        "process_feature_distributions.pdf",
        "conf_iou_feature_space.png",
        "motivation_summary.csv",
        "matched_box_process_features.csv",
    ]:
        print(f"  {output_dir / filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training-process motivation for annotation-fault ranking.")
    parser.add_argument("--dataset-name", default="VOC2012", choices=["VOC2012", "KITTI_8", "VisDrone"])
    parser.add_argument("--model-name", default="YOLOv7")
    parser.add_argument("--metric-path", default=None)
    parser.add_argument("--gt-json-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window-ratio", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
