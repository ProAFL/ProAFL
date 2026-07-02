"""
Case-study visualization for ours.

The figure contrasts:
1. three explicit annotation errors at GT-box level using 8 process features;
2. one implicit missing-annotation error at image level using 4 process features.

All feature values are converted to suspicious percentiles, where larger means
more suspicious under the feature sign stored in the CSV.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ours.base_data_manager import exp_data_root_dir, get_collected_gt_box_json_path


BOX_FEATURES = [
    "early_conf_mean",
    "early_iou_mean",
    "lastly_conf_mean",
    "lastly_iou_mean",
    "conf_mean",
    "iou_mean",
    "D_conf",
    "D_iou",
]

IMG_FEATURES = ["conf", "stab", "cls", "epoch_cross"]

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


def load_gid_meta(dataset_name):
    gt_json = read_json(get_collected_gt_box_json_path(dataset_name))
    rows = []
    for img_name, boxes in gt_json.items():
        for box in boxes:
            fault_type = int(box["fault_type"])
            rows.append({
                "gid": box["box_id"],
                "img_name": img_name,
                "fault_type": fault_type,
                "fault_name": FAULT_TYPE_NAME.get(fault_type, str(fault_type)),
            })
    return pd.DataFrame(rows)


def add_suspicious_percentiles(df, feature_names):
    df = df.copy()
    for feature in feature_names:
        sign_col = f"{feature}_sign"
        sign = int(df[sign_col].dropna().iloc[0]) if sign_col in df.columns else 1
        suspicious = sign * df[feature].astype(float)
        df[f"{feature}_suspicious"] = suspicious
        df[f"{feature}_pct"] = suspicious.rank(method="average", pct=True) * 100.0
    pct_cols = [f"{feature}_pct" for feature in feature_names]
    df["mean_suspicious_pct"] = df[pct_cols].mean(axis=1)
    return df


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def _rounded_feature_pcts(row, feature_names):
    return {feature: round(float(row[f"{feature}_pct"]), 1) for feature in feature_names}


def _pct_collision_count(row, selected_rows, feature_names):
    if not selected_rows:
        return 0
    row_pcts = _rounded_feature_pcts(row, feature_names)
    used = {feature: set() for feature in feature_names}
    for selected in selected_rows:
        selected_pcts = _rounded_feature_pcts(selected, feature_names)
        for feature, value in selected_pcts.items():
            used[feature].add(value)
    return sum(1 for feature, value in row_pcts.items() if value in used[feature])


def _choose_diverse_row(candidates, selected_rows, feature_names):
    if candidates.empty:
        return None
    candidates = candidates.sort_values("mean_suspicious_pct", ascending=False)
    for _, row in candidates.iterrows():
        if _pct_collision_count(row, selected_rows, feature_names) == 0:
            return row

    scored = []
    for _, row in candidates.iterrows():
        scored.append((
            _pct_collision_count(row, selected_rows, feature_names),
            -float(row["mean_suspicious_pct"]),
            row,
        ))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def choose_explicit_cases(box_df, count=3, existing_cases=None):
    cases = []
    selected_rows = list(existing_cases) if existing_cases is not None else []
    matched_mask = bool_series(box_df["is_matched"]) if "is_matched" in box_df.columns else pd.Series(True, index=box_df.index)
    for fault_type in [1, 2, 3]:
        subset = box_df[(box_df["fault_type"] == fault_type) & matched_mask]
        if subset.empty:
            subset = box_df[box_df["fault_type"] == fault_type]
        if subset.empty:
            continue
        selected = _choose_diverse_row(subset, selected_rows, BOX_FEATURES)
        cases.append(selected)
        selected_rows.append(selected)

    if len(cases) < count:
        used = {int(row["gid"]) for row in cases}
        fallback = box_df[(box_df["is_error_bool"]) & matched_mask & (~box_df["gid"].isin(used))]
        if fallback.empty:
            fallback = box_df[(box_df["is_error_bool"]) & (~box_df["gid"].isin(used))]
        while len(cases) < count and not fallback.empty:
            row = _choose_diverse_row(fallback, selected_rows, BOX_FEATURES)
            if row is None:
                break
            cases.append(row)
            selected_rows.append(row)
            fallback = fallback[fallback["gid"] != row["gid"]]
    return pd.DataFrame(cases).head(count)


def choose_correct_box_case(box_df):
    matched_mask = bool_series(box_df["is_matched"]) if "is_matched" in box_df.columns else pd.Series(True, index=box_df.index)
    correct_df = box_df[(~box_df["is_error_bool"]) & matched_mask]
    if correct_df.empty:
        correct_df = box_df[~box_df["is_error_bool"]]
    if correct_df.empty:
        raise ValueError("No correct GT box found in box feature CSV.")
    target = correct_df["mean_suspicious_pct"].median()
    return correct_df.iloc[(correct_df["mean_suspicious_pct"] - target).abs().argsort().iloc[0]]


def choose_implicit_case(img_df, existing_cases=None):
    miss_df = img_df[img_df["is_missfault_bool"]]
    if miss_df.empty:
        raise ValueError("No implicit missing-fault image found in image feature CSV.")
    return miss_df.sort_values(
        ["mean_suspicious_pct", "img_name"],
        ascending=[False, True],
    ).iloc[0]


def choose_correct_image_case(img_df, reference_case=None):
    correct_df = img_df[~img_df["is_missfault_bool"]]
    if correct_df.empty:
        raise ValueError("No no-miss image found in image feature CSV.")

    diverse_candidates = []
    reference_vector = None
    if reference_case is not None:
        reference_vector = np.array(
            [float(reference_case[f"{feature}_pct"]) for feature in IMG_FEATURES],
            dtype=float,
        )

    for _, row in correct_df.iterrows():
        pct_vector = np.array(
            [float(row[f"{feature}_pct"]) for feature in IMG_FEATURES],
            dtype=float,
        )
        rounded_pcts = [round(value, 1) for value in pct_vector]
        if len(set(rounded_pcts)) <= 1:
            continue
        distance = (
            float(np.linalg.norm(pct_vector - reference_vector))
            if reference_vector is not None
            else -float(row["mean_suspicious_pct"])
        )
        diverse_candidates.append((distance, float(row["mean_suspicious_pct"]), str(row["img_name"]), row))

    if diverse_candidates:
        diverse_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return diverse_candidates[0][3]

    return correct_df.sort_values(
        ["mean_suspicious_pct", "img_name"],
        ascending=[True, True],
    ).iloc[0]


def plot_case_study(box_cases, image_cases, output_path, box_dataset_name, image_dataset_name):
    explicit_matrix = box_cases[[f"{f}_pct" for f in BOX_FEATURES]].to_numpy()
    explicit_labels = [
        f"{row.fault_name}\ngid={int(row.gid)}"
        for row in box_cases.itertuples(index=False)
    ]

    image_matrix = image_cases[[f"{f}_pct" for f in IMG_FEATURES]].to_numpy(dtype=float)
    image_labels = [
        f"{row.case_label}\n{row.img_name}"
        for row in image_cases.itertuples(index=False)
    ]

    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        explicit_matrix,
        ax=ax1,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        cbar=True,
        cbar_kws={"label": "Suspicious percentile"},
        xticklabels=BOX_FEATURES,
        yticklabels=explicit_labels,
    )
    ax1.set_title(f"{box_dataset_name}: correct vs explicit annotation errors")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", rotation=35)

    sns.heatmap(
        image_matrix,
        ax=ax2,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        cbar=False,
        xticklabels=IMG_FEATURES,
        yticklabels=image_labels,
    )
    ax2.set_title(f"{image_dataset_name}: correct vs implicit missing error")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=35)

    fig.suptitle("Case study: process features expose explicit and implicit annotation errors", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_selected_cases(box_cases, image_cases, output_path, box_dataset_name, image_dataset_name):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in box_cases.to_dict("records"):
        case_row = {
            "case_type": "correct_box" if row["fault_type"] == 0 else "explicit_box_error",
            "dataset_name": box_dataset_name,
            "id": row["gid"],
            "img_name": row["img_name"],
            "fault_name": row["fault_name"],
            "mean_suspicious_pct": row["mean_suspicious_pct"],
        }
        case_row.update({f"{feature}_pct": row[f"{feature}_pct"] for feature in BOX_FEATURES})
        rows.append(case_row)
    for row in image_cases.to_dict("records"):
        case_row = {
            "case_type": row["case_type"],
            "dataset_name": image_dataset_name,
            "id": row["img_name"],
            "img_name": row["img_name"],
            "fault_name": row["case_label"],
            "mean_suspicious_pct": row["mean_suspicious_pct"],
        }
        case_row.update({f"{feature}_pct": row[f"{feature}_pct"] for feature in IMG_FEATURES})
        rows.append(case_row)
    fieldnames = ["case_type", "dataset_name", "id", "img_name", "fault_name", "mean_suspicious_pct"]
    fieldnames.extend([f"{feature}_pct" for feature in BOX_FEATURES])
    fieldnames.extend([f"{feature}_pct" for feature in IMG_FEATURES])
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    input_dir = Path(args.input_dir)
    image_dataset_name = args.image_dataset_name or args.dataset_name
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "results" / f"{args.dataset_name}_img-{image_dataset_name}" / args.model_name
    )
    box_csv = input_dir / f"box_feature_table_{args.dataset_name}_{args.model_name}.csv"
    img_csv = input_dir / f"img_feature_table_{image_dataset_name}_{args.model_name}.csv"
    if not box_csv.exists():
        raise FileNotFoundError(box_csv)
    if not img_csv.exists():
        raise FileNotFoundError(img_csv)

    box_df = pd.read_csv(box_csv)
    img_df = pd.read_csv(img_csv)
    gid_meta = load_gid_meta(args.dataset_name)
    box_df = box_df.merge(gid_meta, on="gid", how="left", suffixes=("", "_gt"))
    if "img_name_gt" in box_df.columns:
        box_df["img_name"] = box_df["img_name"].fillna(box_df["img_name_gt"])
    box_df["is_error_bool"] = bool_series(box_df["is_error"])
    img_df["is_missfault_bool"] = bool_series(img_df["is_missfault"])

    box_df = add_suspicious_percentiles(box_df, BOX_FEATURES)
    img_df = add_suspicious_percentiles(img_df, IMG_FEATURES)

    correct_case = choose_correct_box_case(box_df)
    explicit_cases = choose_explicit_cases(box_df, count=3, existing_cases=[correct_case])
    box_cases = pd.concat([pd.DataFrame([correct_case]), explicit_cases], ignore_index=True)
    implicit_case = choose_implicit_case(img_df)
    correct_image_case = choose_correct_image_case(img_df, reference_case=implicit_case)
    image_cases = pd.DataFrame([
        dict(correct_image_case, case_label="no_miss", case_type="correct_image"),
        dict(implicit_case, case_label="missing_fault", case_type="implicit_missing_error"),
    ])

    figure_path = output_dir / "case_study_process_feature_contrast.png"
    csv_path = output_dir / "selected_cases.csv"
    plot_case_study(box_cases, image_cases, figure_path, args.dataset_name, image_dataset_name)
    write_selected_cases(box_cases, image_cases, csv_path, args.dataset_name, image_dataset_name)

    print(f"box_dataset={args.dataset_name}, image_dataset={image_dataset_name}, model={args.model_name}")
    print("selected box cases:")
    print(box_cases[["gid", "img_name", "fault_name", "mean_suspicious_pct"]].to_string(index=False))
    print("selected implicit case:")
    print(image_cases[["img_name", "case_label", "mean_suspicious_pct"]].to_string(index=False))
    print(f"saved figure: {figure_path}")
    print(f"saved cases: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create case-study visualization for process features.")
    parser.add_argument("--dataset-name", default="VOC2012", choices=["VOC2012", "KITTI_8", "VisDrone"])
    parser.add_argument("--image-dataset-name", default=None, choices=["VOC2012", "KITTI_8", "VisDrone"])
    parser.add_argument("--model-name", default="YOLOv7")
    parser.add_argument("--input-dir", default=f"{exp_data_root_dir}/discussion")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
