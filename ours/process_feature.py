"""Build six annotation process features and rank them with equal-weight TOPSIS.

Edit the settings below, then run ``python -m ours.process_feature`` from the
repository root. Annotation fault labels are never used for feature extraction
or ranking. Annotations with no overlapping prediction in any epoch are also
written to a separate report; their identical scores remain tied.
"""

import json
from pathlib import Path

import numpy as np
import topsispy as tp


# Settings for the current VOC / YOLOv7 experiment.
DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
ANNOTATIONS_PATH = DATA_ROOT / "fault_inject/0.1/voc/coco_format/annotations_no_miss.json"
METRICS_PATH = DATA_ROOT / (
    "process_metrics/voc/yolov7/0.1_repeat/"
    "repeat_1_1791094449/process_metrics.json"
)
OUTPUT_DIR = DATA_ROOT / "process_features/voc/yolov7/0.1_repeat/repeat_1_1791094449"
EPOCHS = 50
LATE_EPOCHS = 10
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

FEATURE_NAMES = (
    "late_conf_deficit",
    "all_conf_deficit",
    "first_conf_reach",
    "late_iou_deficit",
    "all_iou_deficit",
    "first_iou_reach",
)


def first_reach_fraction(values, threshold):
    """First epoch strictly above threshold / epoch count; 1 if never reached."""
    return next((epoch for epoch, value in enumerate(values) if value > threshold), len(values)) / len(values)


def extract_features(metric):
    conf = np.asarray(metric["conf_list"], dtype=np.float64)
    iou = np.asarray(metric["iou_list"], dtype=np.float64)
    has_overlap = metric["has_overlap_list"]
    if conf.shape != (EPOCHS,) or iou.shape != (EPOCHS,) or len(has_overlap) != EPOCHS:
        raise ValueError(f"Expected {EPOCHS} values in each trajectory")
    if not np.all(np.isfinite(conf)) or not np.all(np.isfinite(iou)):
        raise ValueError("Confidence and IoU trajectories must be finite")
    if np.any((conf < 0) | (conf > 1)) or np.any((iou < 0) | (iou > 1)):
        raise ValueError("Confidence and IoU trajectories must lie in [0, 1]")
    if any(type(value) is not bool for value in has_overlap):
        raise ValueError("has_overlap_list must contain Boolean values")

    features = {
        "late_conf_deficit": 1.0 - float(np.mean(conf[-LATE_EPOCHS:])),
        "all_conf_deficit": 1.0 - float(np.mean(conf)),
        "first_conf_reach": first_reach_fraction(conf, CONF_THRESHOLD),
        "late_iou_deficit": 1.0 - float(np.mean(iou[-LATE_EPOCHS:])),
        "all_iou_deficit": 1.0 - float(np.mean(iou)),
        "first_iou_reach": first_reach_fraction(iou, IOU_THRESHOLD),
    }
    return features, not any(has_overlap)


def topsis_scores(feature_matrix):
    """Use topsispy with equal weights; larger scores mean more suspicious."""
    n_features = len(FEATURE_NAMES)
    weights = np.ones(n_features, dtype=np.float64) / n_features
    signs = np.ones(n_features, dtype=int)
    _, score_array = tp.topsis(feature_matrix, weights, signs)
    scores = np.asarray(score_array, dtype=np.float64)
    if scores.shape != (len(feature_matrix),) or not np.all(np.isfinite(scores)):
        raise ValueError("topsispy returned invalid annotation scores")
    return scores


def build_results(annotations_path, metrics_path):
    """Return feature table, complete ranking, and the no-overlap report."""
    with open(annotations_path, encoding="utf-8") as file:
        annotation_data = json.load(file)
    annotation_ids = [int(annotation["id"]) for annotation in annotation_data["annotations"]]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Annotation IDs must be unique")
    annotation_ids.sort()

    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)
    expected_ids = {str(annotation_id) for annotation_id in annotation_ids}
    if set(metrics) != expected_ids:
        missing = expected_ids - set(metrics)
        extra = set(metrics) - expected_ids
        raise ValueError(f"Metric/annotation ID mismatch: {len(missing)} missing, {len(extra)} extra")

    features_by_id = {}
    no_overlap_ids = []
    for annotation_id in annotation_ids:
        try:
            features, no_overlap = extract_features(metrics[str(annotation_id)])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid process metrics for anno {annotation_id}: {error}") from error
        features_by_id[str(annotation_id)] = features
        if no_overlap:
            no_overlap_ids.append(annotation_id)

    feature_matrix = np.asarray(
        [[features_by_id[str(annotation_id)][name] for name in FEATURE_NAMES]
         for annotation_id in annotation_ids],
        dtype=np.float64,
    )
    scores = topsis_scores(feature_matrix)
    order = sorted(range(len(annotation_ids)), key=lambda index: (-scores[index], annotation_ids[index]))

    ranking = []
    previous_score = None
    tied_rank = 0
    no_overlap_set = set(no_overlap_ids)
    for position, index in enumerate(order, start=1):
        score = float(scores[index])
        if previous_score is None or score != previous_score:
            tied_rank = position
            previous_score = score
        annotation_id = annotation_ids[index]
        ranking.append({
            "rank": tied_rank,
            "anno_id": annotation_id,
            "topsis_score": score,
            "never_overlapped": annotation_id in no_overlap_set,
        })

    no_overlap_report = {
        "definition": "has_overlap_list is false in every epoch",
        "count": len(no_overlap_ids),
        "annotation_ids": no_overlap_ids,
        "topsis_scores": {
            str(annotation_id): float(scores[index])
            for index, annotation_id in enumerate(annotation_ids)
            if annotation_id in no_overlap_set
        },
        "note": "Equal scores remain tied; annotation ID only fixes display order.",
    }
    # Fault labels, when present, are read only after ranking for this report.
    if all("fault_type" in annotation for annotation in annotation_data["annotations"]):
        fault_type_by_id = {
            int(annotation["id"]): annotation["fault_type"]
            for annotation in annotation_data["annotations"]
        }
        no_overlap_report["evaluation_only"] = {
            "clean_count": sum(fault_type_by_id[annotation_id] == 0 for annotation_id in no_overlap_ids),
            "fault_count": sum(fault_type_by_id[annotation_id] != 0 for annotation_id in no_overlap_ids),
            "note": "Fault labels are used only for this post-ranking summary.",
        }
    return features_by_id, ranking, no_overlap_report


def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def main():
    features, ranking, no_overlap_report = build_results(ANNOTATIONS_PATH, METRICS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "process_features.json", features)
    save_json(OUTPUT_DIR / "topsis_ranking.json", {
        "feature_names": FEATURE_NAMES,
        "weights": [1.0 / len(FEATURE_NAMES)] * len(FEATURE_NAMES),
        "method": "topsispy.topsis",
        "score_direction": "higher_is_more_suspicious",
        "epochs": EPOCHS,
        "late_epochs": LATE_EPOCHS,
        "conf_threshold_strictly_greater_than": CONF_THRESHOLD,
        "iou_threshold_strictly_greater_than": IOU_THRESHOLD,
        "ranking": ranking,
    })
    save_json(OUTPUT_DIR / "never_overlapped.json", no_overlap_report)
    print(f"Ranked {len(ranking)} annotations; results saved to {OUTPUT_DIR}")
    print(f"Annotations never overlapping a prediction: {no_overlap_report['count']}")


if __name__ == "__main__":
    main()
