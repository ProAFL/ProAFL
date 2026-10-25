"""Compare six-feature dual_head_max on six saved repeat-1 experiments.

Run with the existing evaluation environment, from the repository root:
    /home/mml/anaconda3/envs/data_debugging/bin/python -B -m ours.compare_dual_rank_datasets

Original feature/rank/annotation files are read-only. Image merge scales and
no-evidence groups remain unchanged in each experiment; no per-dataset tuning.
Both branches retain all original features: three confidence and three IoU,
including first_conf_reach. Earlier five-feature reports remain unchanged.
Fault labels enter only the evaluation stage. Metrics use the existing common
functions, including legacy APFD/FNR behavior for missing Datactive items.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from ours.experiment_class_ranking import (
    FEATURES, CLASS_FEATURES, GEOMETRY_FEATURES, annotation_branch_scores,
    dual_scores, merge_ranking, read_json, topsis,
)
from helper.data_organization_tools import (
    get_all_error_imgset, get_all_explicit_fault_annoids,
    get_all_miss_error_img_name_set, get_all_annoids_detail,
    get_annoid_to_imgname,
)
from ours.rank.analyse.common import (
    compute_apfd, calc_fpr_fnr_f1, calc_top1, calc_exam,
    get_imgs_group_by_fault, get_imgname_to_ranked_components, exam_by_one_fault,
)


EXPERIMENTS = (
    ("voc", "yolov7", "1791094449"),
    ("kitti", "yolov7", "1792056154"),
    ("visdrone", "yolov7", "1791968956"),
    ("voc", "rtdetr", "1792398239"),
    ("kitti", "rtdetr", "1792556872"),
    ("visdrone", "rtdetr", "1792504350"),
)


def build_rankings(feature_dir):
    feature_table = read_json(feature_dir / "process_features.json")
    source = read_json(feature_dir / "topsis_ranking.json")
    rows = sorted(source["ranking"], key=lambda row: row["anno_id"])
    ids = np.array([row["anno_id"] for row in rows], dtype=np.int64)
    if len(set(ids)) != len(ids) or set(feature_table) != {str(i) for i in ids}:
        raise ValueError(f"Duplicate or mismatched annotation IDs: {feature_dir}")
    matrix = np.array([[feature_table[str(i)][name] for name in FEATURES] for i in ids])
    del feature_table
    original_score = np.array([row["topsis_score"] for row in rows])
    no_overlap = np.array([row["never_overlapped"] for row in rows])
    if not np.allclose(topsis(matrix), original_score, rtol=0, atol=1e-12):
        raise AssertionError(f"TOPSIS baseline reconstruction failed: {feature_dir}")
    reference = read_json(feature_dir / "merged_rank.json")
    scale = reference["merge_rule"].get("image_percentile_scale", 1.0)
    original = [row["anno_id"] if row["item_type"] == "anno" else row["image_name"]
                for row in reference["all_ranking"]]
    del reference
    images = read_json(feature_dir / "miss_image_ranking.json")["ranking"]
    reconstructed = merge_ranking(ids, original_score, no_overlap, images, scale)
    if original != reconstructed:
        raise AssertionError(f"Baseline merge reconstruction failed: {feature_dir}")
    del reconstructed
    # Keep all original features; only change scoring / fusion, not inputs.
    cls, geo = annotation_branch_scores(matrix)
    dual, _ = dual_scores(cls, geo, ~no_overlap)
    proposed = merge_ranking(ids, dual, no_overlap, images, scale)
    if set(proposed) != set(original) or len(set(proposed)) != len(proposed):
        raise AssertionError("Proposed ranking changes the original item universe")
    return original, proposed, scale


def score_ranking(rank, annotations, cutoff):
    faults = (set(get_all_explicit_fault_annoids(annotations))
              | get_all_miss_error_img_name_set(annotations))
    fault_images = get_all_error_imgset(annotations)
    fpr, fnr, f1 = calc_fpr_fnr_f1(rank, faults, cut_off=cutoff)
    result = dict(APFD=compute_apfd(faults, rank), FPR=fpr, FNR=fnr, F1=f1,
                  TOP1=calc_top1(annotations, rank, faults, fault_images),
                  EXAM=calc_exam(annotations, rank), ranking_length=len(rank),
                  duplicate_items=len(rank) - len(set(rank)),
                  missing_fault_items=len(faults - set(rank)))
    fault_names = {1: "class_fault", 2: "loc_fault", 3: "redun_fault", 4: "miss_fault"}
    by_fault = {name: set(ids) for name, ids in get_all_annoids_detail(annotations).items()}
    by_fault["miss_fault"] = get_all_miss_error_img_name_set(annotations)
    grouped = get_imgname_to_ranked_components(rank, get_annoid_to_imgname(annotations))
    breakdown = {fault_names[t]: exam_by_one_fault(images, grouped, by_fault[fault_names[t]])
                 for t, images in get_imgs_group_by_fault(annotations).items()}
    if round(sum(breakdown.values()) / len(breakdown), 3) != result["EXAM"]:
        raise AssertionError("EXAM breakdown disagrees with the original evaluator")
    return result, breakdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/mml/data_debugging_data/ProAFL_data"))
    parser.add_argument("--cutoff", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/dual_rank_comparison") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not 0 < args.cutoff < 1:
        parser.error("cutoff must lie in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results, breakdowns = [], {}
    for dataset, model, expid in EXPERIMENTS:
        print(f"Building {model}/{dataset}, experiment {expid}...", flush=True)
        feature_dir = args.data_root / f"process_features/{dataset}/{model}/0.1_repeat/repeat_1_{expid}"
        original, proposed, scale = build_rankings(feature_dir)
        # Ranking construction is finished before loading fault ground truth.
        annotations = read_json(args.data_root / f"fault_inject/0.1/{dataset}/coco_format/annotations_with_miss.json")
        datactive = joblib.load(args.data_root / f"rank/datactive/{dataset}/0.1/repeat_1/converted_rank.joblib")
        for method, rank in (("original_ours", original), ("dual_head_max_6features", proposed),
                             ("datactive_repeat_1", datactive)):
            metrics, detail = score_ranking(rank, annotations, args.cutoff)
            result = dict(dataset=dataset, model=model, expid=expid, method=method,
                          image_percentile_scale=scale, **metrics)
            results.append(result)
            breakdowns[f"{model}/{dataset}/{method}"] = detail
            print(json.dumps(result), flush=True)
    directions = {"APFD": 1, "FPR": -1, "FNR": -1, "TOP1": 1, "EXAM": -1}
    summaries = []
    for dataset, model, _ in EXPERIMENTS:
        group = {r["method"]: r for r in results if r["dataset"] == dataset and r["model"] == model}
        for reference in ("original_ours", "datactive_repeat_1"):
            changes = {metric: round((group["dual_head_max_6features"][metric] - group[reference][metric]) * direction, 3)
                       for metric, direction in directions.items()}
            summaries.append(dict(dataset=dataset, model=model, reference=reference,
                                  changes_positive_is_better=changes))
    report = dict(cutoff=args.cutoff, rule="Fixed dual_head_max with all six features (3 confidence + 3 IoU); no tuning.",
                  class_features=CLASS_FEATURES, geometry_features=GEOMETRY_FEATURES,
                  note="Original evaluation functions and rounding. Each original merge is reproduced entry by entry. Original image scales preserved. Missing Datactive fault items reported; legacy APFD/FNR handling preserved for comparability.",
                  results=results, exam_by_fault=breakdowns, comparisons=summaries)
    with open(args.output_dir / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with open(args.output_dir / "results.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved reports to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
