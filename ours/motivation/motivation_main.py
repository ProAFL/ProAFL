from ours.small_utils import read_json
import numpy as np
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots


plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def build_gid_to_is_fault(gt_json):
    gid_to_is_fault = {}
    for _, g_box_list in gt_json.items():
        for g_box in g_box_list:
            gid_to_is_fault[int(g_box["box_id"])] = int(g_box["fault_type"]) != 0
    return gid_to_is_fault


def init_epoch_row():
    return {
        "correct": {"confi_list": [], "iou_list": [], "mean_confi": np.nan, "mean_iou": np.nan},
        "error": {"confi_list": [], "iou_list": [], "mean_confi": np.nan, "mean_iou": np.nan},
    }


def first_crossing_epoch(values, threshold):
    for idx, value in enumerate(values):
        if value > threshold:
            return idx
    return len(values)


def add_process_record(records, gid, is_fault, conf_list, iou_list):
    conf = np.asarray(conf_list, dtype=float)
    iou = np.asarray(iou_list, dtype=float)
    epoch_num = min(len(conf), len(iou))
    if epoch_num == 0:
        return
    conf = conf[:epoch_num]
    iou = iou[:epoch_num]
    late_window = max(1, int(0.2 * epoch_num))
    conf_threshold = 0.5 * float(np.mean(conf[-late_window:]))
    iou_threshold = 0.5 * float(np.mean(iou[-late_window:]))
    records.append({
        "gid": gid,
        "label": "error" if is_fault else "correct",
        "conf_list": conf,
        "iou_list": iou,
        "conf_mean": float(np.mean(conf)),
        "iou_mean": float(np.mean(iou)),
        "conf_delay": first_crossing_epoch(conf, conf_threshold) / epoch_num,
        "iou_delay": first_crossing_epoch(iou, iou_threshold) / epoch_num,
    })


def _sample_records_for_heatmap(records, max_per_group=260):
    sampled = []
    for label in ["correct", "error"]:
        group_records = [record for record in records if record["label"] == label]
        if not group_records:
            continue
        group_records = sorted(
            group_records,
            key=lambda record: (record["conf_mean"] + record["iou_mean"]) / 2,
            reverse=(label == "correct"),
        )
        if len(group_records) > max_per_group:
            idx = np.linspace(0, len(group_records) - 1, max_per_group).astype(int)
            group_records = [group_records[i] for i in idx]
        sampled.extend(group_records)
    return sampled


def plot_process_heatmap(ax, records, key, title):
    sampled = _sample_records_for_heatmap(records)
    if not sampled:
        return
    epoch_num = min(len(record[key]) for record in sampled)
    matrix = np.vstack([record[key][:epoch_num] for record in sampled])
    split_idx = sum(1 for record in sampled if record["label"] == "correct")

    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.axhline(split_idx - 0.5, color="white", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GT boxes")
    ax.set_yticks([
        max(0, split_idx / 2),
        split_idx + max(0, (len(sampled) - split_idx) / 2),
    ])
    ax.set_yticklabels(["correct", "error"])
    return im


def plot_epoch_metric(ax, epoch_map, metric_key, ylabel, colors, labels):
    epochs = sorted(epoch_map.keys())
    mean_key = f"mean_{metric_key}"
    list_key = f"{metric_key}_list"
    for group in ["correct", "error"]:
        mean_values = []
        q10_values = []
        q90_values = []
        for epoch in epochs:
            values = np.asarray(epoch_map[epoch][group][list_key], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                mean_values.append(np.nan)
                q10_values.append(np.nan)
                q90_values.append(np.nan)
                continue

            mean_values.append(float(epoch_map[epoch][group][mean_key]))
            q10, q90 = np.percentile(values, [10, 90])
            q10_values.append(float(q10))
            q90_values.append(float(q90))

        ax.fill_between(
            epochs,
            q10_values,
            q90_values,
            color=colors[group],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            epochs,
            mean_values,
            color=colors[group],
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            label=labels[group],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    # ax.set_title(f"{ylabel} over training")


def plot_combined_epoch_metrics(ax, epoch_map, colors):
    epochs = sorted(epoch_map.keys())
    metric_specs = [
        ("confi", "confidence", "-"),
        ("iou", "IoU", "--"),
    ]
    for group in ["correct", "error"]:
        for metric_key, metric_label, linestyle in metric_specs:
            mean_key = f"mean_{metric_key}"
            mean_values = [
                float(epoch_map[epoch][group][mean_key])
                for epoch in epochs
            ]
            ax.plot(
                epochs,
                mean_values,
                color=colors[group],
                linestyle=linestyle,
                linewidth=2.4,
                marker="o" if metric_key == "iou" else None,
                markersize=3.0,
                label=f"{group} {metric_label}",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean process value")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    # ax.set_title("Training dynamics behind process features")
    ax.legend(loc="lower right", frameon=False, ncol=2)


def plot_process_feature_space(ax, records, colors):
    rng = np.random.default_rng(0)
    for group, max_points, alpha, size in [
        ("correct", 2500, 0.18, 12),
        ("error", 1200, 0.38, 16),
    ]:
        group_records = [record for record in records if record["label"] == group]
        if len(group_records) > max_points:
            sampled_idx = rng.choice(len(group_records), size=max_points, replace=False)
            group_records = [group_records[idx] for idx in sampled_idx]

        x = np.asarray([record["conf_mean"] for record in group_records], dtype=float)
        y = np.asarray([record["iou_mean"] for record in group_records], dtype=float)
        ax.scatter(
            x,
            y,
            s=size,
            color=colors[group],
            alpha=alpha,
            edgecolors="none",
            label="Correct annotations" if group == "correct" else "Fault annotations",
        )

    ax.set_xlabel("Mean confidence over training")
    ax.set_ylabel("Mean IoU over training")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    # ax.set_title("Process-feature space from training dynamics")
    legend = ax.legend(loc="lower right", frameon=False, ncol=1)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([26])


def draw(epoch_map, records=None, save_path=None):
    # 基于epoch_map绘制以epoch为x轴,confi和iou值为y轴的两条曲线。
    # 均值用实线表示，每个epoch的取值分布用分位数阴影带表示。
    if save_path is None:
        save_dir = os.path.join(os.path.dirname(__file__), "results")
    else:
        save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)

    colors = {
        "correct": "#2f7d32",
        "error": "#c62828",
    }
    labels = {
        "correct": "Correct annotations",
        "error": "Fault annotations",
    }
    if not records:
        raise ValueError("records is required to draw process trajectory heatmaps.")

    output_paths = {
        "confidence": os.path.join(save_dir, "motivation_confidence_dynamics.pdf"),
        "iou": os.path.join(save_dir, "motivation_iou_dynamics.pdf"),
        "feature_space": os.path.join(save_dir, "motivation_process_feature_space.pdf"),
    }

    fig, ax = plt.subplots(figsize=(3.3, 2.45), constrained_layout=True)
    plot_epoch_metric(ax, epoch_map, "confi", "Confidence", colors, labels)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(output_paths["confidence"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.45), constrained_layout=True)
    plot_epoch_metric(ax, epoch_map, "iou", "IoU", colors, labels)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(output_paths["iou"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    plot_process_feature_space(ax, records, colors)
    fig.savefig(output_paths["feature_space"], dpi=220, bbox_inches="tight")
    plt.close(fig)

    for output_path in output_paths.values():
        print(f"figure saved: {output_path}")

    
def main():
    metric_json = read_json(metric_json_path)
    gt_json = read_json(gt_json_path)
    gid_to_is_fault = build_gid_to_is_fault(gt_json)

    max_epoch = 0
    epoch_map = {}
    records = []
    matched_correct_count = 0
    matched_error_count = 0

    for item in metric_json:
        gid = int(item["g_box_id"])
        if gid not in gid_to_is_fault:
            continue

        conf_list = item["conf_list"]
        iou_list = item["iou_list"]
        epoch_num = min(len(conf_list), len(iou_list))
        if epoch_num == 0:
            continue

        group = "error" if gid_to_is_fault[gid] else "correct"
        if group == "error":
            matched_error_count += 1
        else:
            matched_correct_count += 1

        add_process_record(records, gid, gid_to_is_fault[gid], conf_list, iou_list)

        max_epoch = max(max_epoch, epoch_num - 1)
        for epoch in range(epoch_num):
            epoch_map.setdefault(epoch, init_epoch_row())
            epoch_map[epoch][group]["confi_list"].append(float(conf_list[epoch]))
            epoch_map[epoch][group]["iou_list"].append(float(iou_list[epoch]))

    for epoch in range(max_epoch + 1):
        epoch_map.setdefault(epoch, init_epoch_row())
        for group in ["correct", "error"]:
            confi_list = epoch_map[epoch][group]["confi_list"]
            iou_list = epoch_map[epoch][group]["iou_list"]
            epoch_map[epoch][group]["mean_confi"] = np.mean(confi_list) if confi_list else np.nan
            epoch_map[epoch][group]["mean_iou"] = np.mean(iou_list) if iou_list else np.nan

    print(f"matched correct gbox count: {matched_correct_count}")
    print(f"matched error gbox count: {matched_error_count}")
    print(f"epoch range: 0-{max_epoch}")

    save_path = os.path.join(
        os.path.dirname(__file__),
        "results",
        "motivation_process_metric_separation.pdf",
    )
    draw(epoch_map, records, save_path)




if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    metric_json_path = os.path.join(
        exp_data_root_dir,
        "collection_bbox_level",
        "VOC2012",
        "YOLOv7",
        "collection_metric",
        "collection_metrics_v2.json",
    )
    gt_json_path = "/data/mml/data_debugging_data/collection_bbox_level/VOC2012/gt_bboxs.json"
    main()
