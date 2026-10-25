"""Standalone ranking ablation; never modifies existing rankings or annotations.

Run from the repository root:
    python -m ours.experiment_class_ranking
Add --with-conflict to reconstruct late-epoch class margins from saved boxes.
Fault labels and clean annotations are used ONLY by the final evaluator.
Default budgets match repair.py: fraction of the clean annotation count.
These are exploratory, in-sample ablations, not held-out validation.
The dual heads retain all six features: three confidence and three IoU.
Earlier saved experiments used only two confidence features; their reports
remain unchanged and are not results for this six-feature version.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata


FEATURES = (
    "late_conf_deficit", "all_conf_deficit", "first_conf_reach",
    "late_iou_deficit", "all_iou_deficit", "first_iou_reach",
)
CLASS_FEATURES = FEATURES[:3]
GEOMETRY_FEATURES = FEATURES[3:]


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def topsis(matrix, weights=None):
    """Vector-normalized, positive-direction TOPSIS, matching topsispy."""
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Nonfinite feature values")
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-15)
    if weights is not None:
        normalized *= np.asarray(weights, dtype=float)
    positive = np.linalg.norm(normalized - normalized.max(axis=0), axis=1)
    negative = np.linalg.norm(normalized - normalized.min(axis=0), axis=1)
    return negative / np.maximum(positive + negative, 1e-15)


def calibrate(scores, evidence):
    """Empirical midrank CDF among evidence-backed annotations; no labels."""
    result = np.zeros(len(scores))
    result[evidence] = (rankdata(scores[evidence], method="average") - 0.5) / evidence.sum()
    return result


def dual_scores(class_score, geometry_score, evidence):
    """Put heads on the same percentile scale before max / smooth-OR fusion."""
    cls = calibrate(class_score, evidence)
    geo = calibrate(geometry_score, evidence)
    return np.maximum(cls, geo), 1.0 - (1.0 - cls) * (1.0 - geo)


def annotation_branch_scores(matrix):
    """Retain all six inputs, including first_conf_reach in the class head."""
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
        raise ValueError("Dual heads require all six annotation features")
    return topsis(matrix[:, :3]), topsis(matrix[:, 3:])


def merge_ranking(ids, scores, no_overlap, image_rows, image_scale):
    """Preserve merge_rank.py's groups, midrank ties, and deterministic order."""
    evidence = np.flatnonzero(~no_overlap)
    percentiles = (rankdata(-scores[evidence], method="average") - 0.5) / len(evidence)
    # Sorting the evidence by score first also fixes equal-score ID ordering.
    order = np.lexsort((ids[evidence], -scores[evidence]))
    entries = [(float(percentiles[i]), 0, int(ids[evidence[i]])) for i in order]
    images = sorted((row for row in image_rows if row["has_candidate"]),
                    key=lambda row: (-row["topsis_score"], row["image_name"]))
    if images:
        image_percentiles = (rankdata([-row["topsis_score"] for row in images],
                                     method="average") - 0.5) / len(images)
        entries.extend((float(p * image_scale), 1, row["image_name"])
                       for p, row in zip(image_percentiles, images))
    entries.sort()
    return ([int(i) for i in ids[no_overlap]] + [entry[2] for entry in entries]
            + sorted(row["image_name"] for row in image_rows if not row["has_candidate"]))


def conflict_features(ids, categories, metrics_path, predictions_dir, late_epochs, min_iou):
    """Read selected prediction IDs; do not rematch or consult fault labels."""
    print("Loading annotation trajectories for class-conflict features...", flush=True)
    metrics = read_json(metrics_path)
    if set(metrics) != {str(i) for i in ids}:
        raise ValueError("Process metric IDs disagree with annotation IDs")
    epochs = len(metrics[str(ids[0])]["conf_list"])
    if not 1 <= late_epochs <= epochs:
        raise ValueError("late_epochs must lie within the saved trajectory")
    selected = np.full((len(ids), late_epochs), -1, dtype=np.int64)
    reliable = np.zeros((len(ids), late_epochs), dtype=bool)
    for j, annotation_id in enumerate(ids):
        row = metrics[str(annotation_id)]
        if any(len(row[key]) != epochs for key in
               ("conf_list", "iou_list", "selected_predicted_box_id_list")):
            raise ValueError(f"Inconsistent trajectory length for {annotation_id}")
        selected[j] = [-1 if pid is None else pid
                       for pid in row["selected_predicted_box_id_list"][-late_epochs:]]
        reliable[j] = np.asarray(row["iou_list"][-late_epochs:]) >= min_iou
    del metrics
    margins = np.zeros_like(selected, dtype=float)
    alternatives = np.full_like(selected, -1)
    for window_index, epoch in enumerate(range(epochs - late_epochs, epochs)):
        print(f"Reading class probabilities: epoch {epoch}", flush=True)
        records = read_json(predictions_dir / f"epoch_{epoch}_predicted_bboxs.json")
        needed = set(selected[:, window_index]) - {-1}
        lookup = {}
        for image in records.values():
            for prediction in image["predicted_bboxs"]:
                pid = prediction["predicted_box_id"]
                if pid in needed:
                    if pid in lookup:
                        raise ValueError(f"Duplicate prediction ID {pid}, epoch {epoch}")
                    lookup[pid] = prediction["probs"]
        if set(lookup) != needed:
            raise ValueError(f"Saved selected predictions missing in epoch {epoch}")
        del records
        for j in np.flatnonzero(reliable[:, window_index] & (selected[:, window_index] >= 0)):
            probs = np.asarray(lookup[selected[j, window_index]], dtype=float)
            if (not np.all(np.isfinite(probs)) or np.any(probs < 0)
                    or np.any(probs > 1) or not 0 <= categories[j] < len(probs)):
                raise ValueError(f"Invalid class scores for annotation {ids[j]}")
            target = probs[categories[j]]
            other = probs.copy()
            other[categories[j]] = -1
            alternative = int(np.argmax(other))
            margin = max(0.0, (other[alternative] - target) /
                         max(other[alternative] + target, 1e-8))
            margins[j, window_index] = margin
            if margin > 0:
                alternatives[j, window_index] = alternative
        del lookup
    coverage = reliable.mean(axis=1)
    # Divide by ALL late epochs: one reliable conflicting observation cannot
    # receive the same score as persistent reliable conflicting observations.
    strength = margins.mean(axis=1)
    persistence = (margins > 0).mean(axis=1)
    stable_alternative = np.array([
        max(Counter(row[row >= 0]).values(), default=0) / late_epochs
        for row in alternatives
    ])
    return np.column_stack((strength, persistence, stable_alternative)), coverage


def evaluation_context(fault_path, with_miss_path, correct_path):
    """Evaluation-only oracle; never passed to scoring or feature functions."""
    fault = read_json(fault_path)
    with_miss = read_json(with_miss_path)
    correct = read_json(correct_path)
    annotation_types = {row["id"]: row["fault_type"] for row in fault["annotations"]}
    names = {row["id"]: row["file_name"] for row in fault["images"]}
    missing = Counter(names[row["image_id"]] for row in correct["annotations"]
                      if row["id"] not in annotation_types)
    injected_missing = Counter(names[row["image_id"]] for row in with_miss["annotations"]
                               if row["fault_type"] == 4)
    if missing != injected_missing:
        raise ValueError("Oracle missing annotations disagree with injected faults")
    fault_items = {i for i, t in annotation_types.items() if t in (1, 2, 3)} | set(missing)
    total_faults = sum(row["fault_type"] != 0 for row in with_miss["annotations"])
    return annotation_types, missing, fault_items, total_faults, len(correct["annotations"]), set(names.values())


def evaluate(method, ranking, cutoffs, budget_base, context):
    types, missing, faults, total_faults, correct_count, image_names = context
    ranked_set = set(ranking)
    if len(ranked_set) != len(ranking):
        raise ValueError(f"{method}: duplicate rank entries")
    if ranked_set - (set(types) | image_names):
        raise ValueError(f"{method}: unknown rank entries")
    omitted = faults - ranked_set
    # Missing faults get a position AFTER the list, instead of disappearing
    # from the APFD sum or the FNR numerator as in the legacy evaluator.
    positions = sum(i for i, item in enumerate(ranking, 1) if item in faults)
    apfd = 1 - (positions + len(omitted) * (len(ranking) + 1)) / (len(ranking) * len(faults)) + 1 / (2 * len(ranking))
    costs = np.array([max(1, missing.get(item, 0)) if isinstance(item, str) else 1
                      for item in ranking], dtype=np.int64)
    cumulative = np.cumsum(costs)
    rows = []
    for cutoff in cutoffs:
        budget = int(cutoff * (correct_count if budget_base == "correct_annotations" else len(ranking)))
        completed = int(np.searchsorted(cumulative, budget, side="right"))
        counts = Counter()
        for item in ranking[:completed]:
            if isinstance(item, str):
                counts["miss"] += missing.get(item, 0)
            elif types[item]:
                counts[{1: "cls", 2: "loc", 3: "redun"}[types[item]]] += 1
        spent = int(cumulative[completed - 1]) if completed else 0
        touched = completed
        if completed < len(ranking) and spent < budget:
            item = ranking[completed]
            if not isinstance(item, str) or not missing.get(item, 0):
                raise AssertionError("Only missing-box repair can stop inside an item")
            counts["miss"] += budget - spent
            spent = budget
            touched += 1
        total = sum(counts.values())
        prefix = set(ranking[:int(cutoff * len(ranking))])
        fp = len(prefix - faults)
        rows.append(dict(method=method, cutoff=cutoff, budget=budget, spent=spent,
                         repaired=total, repair_rate=total / total_faults,
                         cls=counts["cls"], loc=counts["loc"], redun=counts["redun"],
                         miss=counts["miss"], nonrepair_cost=spent - total,
                         processed_items=touched, ranking_length=len(ranking),
                         omitted_fault_items=len(omitted), apfd=apfd,
                         item_fpr=fp / len(ranked_set - faults),
                         item_fnr=len(faults - prefix) / len(faults)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/mml/data_debugging_data/ProAFL_data"))
    parser.add_argument("--dataset", default="visdrone")
    parser.add_argument("--model", default="yolov7")
    parser.add_argument("--expid", default="1791968956")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    parser.add_argument("--budget-base", choices=["correct_annotations", "rank_items"], default="correct_annotations")
    parser.add_argument("--with-conflict", action="store_true")
    parser.add_argument("--late-epochs", type=int, default=10)
    parser.add_argument("--reliable-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/class_ranking") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--export-method", action="append", default=[], help="Export a candidate as a separate merged_rank.json; repeatable")
    args = parser.parse_args()
    if any(not 0 < c <= 1 for c in args.cutoffs) or not 0 <= args.reliable_iou <= 1:
        parser.error("Cutoffs must be in (0,1], reliable IoU in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.data_root
    suffix = f"{args.dataset}/{args.model}/0.1_repeat/repeat_{args.repeat}_{args.expid}"
    feature_dir = root / "process_features" / suffix
    fault_dir = root / f"fault_inject/0.1/{args.dataset}/coco_format"
    fault_path = fault_dir / "annotations_no_miss.json"
    # Strip oracle fault fields before constructing any features or rankings.
    observed = read_json(fault_path)
    annotations = sorted(((row["id"], row["category_id"]) for row in observed["annotations"]))
    del observed
    ids = np.array([row[0] for row in annotations], dtype=np.int64)
    categories = np.array([row[1] for row in annotations], dtype=np.int64)
    features = read_json(feature_dir / "process_features.json")
    matrix = np.array([[features[str(i)][name] for name in FEATURES] for i in ids])
    del features
    source = read_json(feature_dir / "topsis_ranking.json")
    source_rows = {row["anno_id"]: row for row in source["ranking"]}
    if set(source_rows) != set(ids):
        raise ValueError("Annotation source rank has missing or extra IDs")
    no_overlap = np.array([source_rows[i]["never_overlapped"] for i in ids])
    baseline = np.array([source_rows[i]["topsis_score"] for i in ids])
    if not np.allclose(topsis(matrix), baseline, rtol=0, atol=1e-12):
        raise AssertionError("TOPSIS reconstruction does not match the saved baseline")
    image_rows = read_json(feature_dir / "miss_image_ranking.json")["ranking"]
    reference = read_json(feature_dir / "merged_rank.json")
    image_scale = reference["merge_rule"].get("image_percentile_scale", 1.0)
    reference_items = [row["anno_id"] if row["item_type"] == "anno" else row["image_name"]
                       for row in reference["all_ranking"]]
    del reference, source, source_rows
    evidence = ~no_overlap
    cls, geo = annotation_branch_scores(matrix)
    dual_max, dual_or = dual_scores(cls, geo, evidence)
    scores = dict(baseline=baseline,
                  confidence_weight_2=topsis(matrix, [2, 2, 2, 1, 1, 1]),
                  dual_head_max_6features=dual_max,
                  dual_head_smooth_or_6features=dual_or)
    conflict_summary = None
    if args.with_conflict:
        prediction_dir = root / f"collection_process_info/{args.dataset}/{args.model}/collected_predict_boxes/inject_0.1_repeat/repeat_{args.repeat}_{args.expid}"
        conflict, coverage = conflict_features(ids, categories, root / "process_metrics" / suffix / "process_metrics.json",
                                               prediction_dir, args.late_epochs, args.reliable_iou)
        conflict_cls = topsis(np.column_stack((matrix[:, :3], conflict)))
        conflict_max, conflict_or = dual_scores(conflict_cls, geo, evidence)
        scores.update(dual_conflict_max_6features=conflict_max,
                      dual_conflict_smooth_or_6features=conflict_or,
                      joint_conflict_topsis=topsis(np.column_stack((matrix, conflict))))
        conflict_summary = dict(feature_names=["late_positive_margin_evidence", "late_conflict_presence", "late_stable_alternative_evidence"],
                                mean_reliable_coverage=float(coverage.mean()),
                                note="All conflict features divide by all late epochs, preserving coverage reliability.")
    unknown = set(args.export_method) - set(scores)
    if unknown:
        parser.error(f"Unknown export methods: {sorted(unknown)}; available: {sorted(scores)}")
    # Build EVERY candidate before loading evaluation-only oracle information.
    rankings = {name: merge_ranking(ids, score, no_overlap, image_rows, image_scale)
                for name, score in scores.items()}
    if rankings["baseline"] != reference_items:
        raise AssertionError("Baseline merge is not identical to the existing all_ranking")
    del reference_items
    datactive_path = root / f"rank/datactive/{args.dataset}/0.1/repeat_{args.repeat}/converted_rank.joblib"
    rankings[f"datactive_repeat_{args.repeat}"] = joblib.load(datactive_path)
    context = evaluation_context(fault_path, fault_dir / "annotations_with_miss.json",
                                 root / f"datasets/{args.dataset}-coco/train/_annotations.coco_correct.json")
    rows = [row for name, ranking in rankings.items()
            for row in evaluate(name, ranking, args.cutoffs, args.budget_base, context)]
    print(f"\nBudget base: {args.budget_base}; correct annotations: {context[4]}; fault boxes: {context[3]}")
    print(f"{'method':30} {'cutoff':>6} {'budget':>8} {'rate':>9} {'cls':>7} {'loc':>7} {'redun':>7} {'miss':>7}")
    for row in rows:
        print(f"{row['method']:30} {row['cutoff']:6.0%} {row['budget']:8} {row['repair_rate']:9.2%} {row['cls']:7} {row['loc']:7} {row['redun']:7} {row['miss']:7}")
    report = dict(settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  class_features=CLASS_FEATURES, geometry_features=GEOMETRY_FEATURES,
                  image_percentile_scale=image_scale, baseline_exactly_reproduced=True,
                  conflict=conflict_summary,
                  evaluation_note="Exploratory in-sample results. Fault labels never enter ranking. APFD/FNR penalize omitted faults. item_fpr/item_fnr use item-count prefixes, not strict repair budgets.",
                  results=rows)
    with open(args.output_dir / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with open(args.output_dir / "results.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in args.export_method:
        path = args.output_dir / name
        path.mkdir()
        exported = dict(experiment=name, source_experiment=args.expid,
                        note="Experimental ranking; original inputs unchanged.",
                        all_ranking=[dict(item_type="image", image_name=item) if isinstance(item, str)
                                     else dict(item_type="anno", anno_id=int(item))
                                     for item in rankings[name]])
        with open(path / "merged_rank.json", "w", encoding="utf-8") as stream:
            json.dump(exported, stream, separators=(",", ":"))
    print(f"\nReports saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
