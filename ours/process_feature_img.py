"""Build image-level missing-annotation features and rank images with TOPSIS.

Each image contributes the six features of one representative position track.
No missing-annotation ground truth or other fault labels enter scoring.

Edit the constants below and run from the repository root with
``python -m ours.process_feature_img``.
"""

import json
import math
from pathlib import Path

import numpy as np
import topsispy as tp

from ours.process_metrics_img import box_iou, validate_box


DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
METRICS_PATH = DATA_ROOT / (
    "process_metrics/voc/yolov7/0.1_repeat/"
    "repeat_1_1791094449/miss_process_metrics.json"
)
OUTPUT_DIR = DATA_ROOT / (
    "process_features/voc/yolov7/0.1_repeat/"
    "repeat_1_1791094449"
)
LATE_EPOCHS = 10

FEATURE_NAMES = (
    "all_presence_rate",
    "late_presence_rate",
    "all_confidence_evidence",
    "late_confidence_evidence",
    "longest_consecutive_presence_rate",
    "position_stability",
)


def track_features(track, epochs, late_epochs):
    """Six positive-direction features from one complete position track."""
    present = track["present_list"]
    confidence = track["conf_list"]
    boxes = track["bbox_list"]
    classes = track["predicted_cls_list"]
    prediction_ids = track["predicted_box_id_list"]
    anno_ious = track["max_anno_iou_list"]
    anno_coverages = track["max_anno_coverage_list"]
    sequences = (present, confidence, boxes, classes, prediction_ids,
                 anno_ious, anno_coverages)
    if any(not isinstance(values, list) or len(values) != epochs for values in sequences):
        raise ValueError("All trajectory lists must contain one entry per epoch")

    observed_boxes = []
    confidence_evidence = []
    longest_run = 0
    current_run = 0
    for epoch, appears in enumerate(present):
        if type(appears) is not bool:
            raise ValueError(f"Epoch {epoch}: present_list must contain booleans")
        if appears:
            score = confidence[epoch]
            if (not isinstance(score, (int, float)) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError(f"Epoch {epoch}: invalid prediction confidence")
            if classes[epoch] is None or prediction_ids[epoch] is None:
                raise ValueError(f"Epoch {epoch}: missing prediction identity")
            if anno_ious[epoch] is None or anno_coverages[epoch] is None:
                raise ValueError(f"Epoch {epoch}: missing annotation overlap metrics")
            observed_boxes.append(validate_box(boxes[epoch], f"track epoch {epoch}"))
            confidence_evidence.append(float(score))
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            if any(values[epoch] is not None for values in sequences[1:]):
                raise ValueError(f"Epoch {epoch}: absent prediction must use null values")
            confidence_evidence.append(0.0)
            current_run = 0

    if not observed_boxes:
        raise ValueError("An empty track should not be stored")
    representative = validate_box(track["representative_bbox"], "representative track box")
    # One observation cannot establish temporal position stability.
    position_stability = (
        sum(box_iou(box, representative) for box in observed_boxes) / len(observed_boxes)
        if len(observed_boxes) >= 2 else 0.0
    )
    return {
        "all_presence_rate": sum(present) / epochs,
        "late_presence_rate": sum(present[-late_epochs:]) / late_epochs,
        "all_confidence_evidence": sum(confidence_evidence) / epochs,
        "late_confidence_evidence": sum(confidence_evidence[-late_epochs:]) / late_epochs,
        "longest_consecutive_presence_rate": longest_run / epochs,
        "position_stability": position_stability,
    }


def build_image_features(metrics, late_epochs):
    epochs = metrics["epochs"]
    if type(epochs) is not int or epochs <= 0 or not 1 <= late_epochs <= epochs:
        raise ValueError("Invalid epoch count or late-epoch window")
    images = metrics["images"]
    if not isinstance(images, dict) or not images:
        raise ValueError("Metrics must contain images")

    features_by_image = {}
    for image_name, image in sorted(images.items()):
        tracks = image["tracks"]
        counts = image["candidate_count_per_epoch"]
        if (not isinstance(tracks, list) or not isinstance(counts, list)
                or len(counts) != epochs):
            raise ValueError(f"{image_name}: invalid tracks or candidate counts")

        track_rows = []
        track_ids = set()
        for track in tracks:
            track_id = track["track_id"]
            if track_id in track_ids:
                raise ValueError(f"{image_name}: duplicate track ID {track_id}")
            track_ids.add(track_id)
            try:
                features = track_features(track, epochs, late_epochs)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{image_name}, track {track_id}: {error}") from error
            track_rows.append((track, features))

        observed_counts = [
            sum(track["present_list"][epoch] for track in tracks)
            for epoch in range(epochs)
        ]
        if counts != observed_counts:
            raise ValueError(f"{image_name}: candidate counts do not match stored tracks")

        if track_rows:
            # Pick one location without using labels or TOPSIS results.
            chosen_track, chosen_features = min(
                track_rows,
                key=lambda item: (
                    -item[1]["all_confidence_evidence"],
                    -item[1]["late_confidence_evidence"],
                    item[0]["track_id"],
                ),
            )
            chosen_id = chosen_track["track_id"]
            chosen_box = chosen_track["representative_bbox"]
        else:
            chosen_features = {name: 0.0 for name in FEATURE_NAMES}
            chosen_id = None
            chosen_box = None

        features_by_image[image_name] = {
            "image_id": image["image_id"],
            "has_candidate": bool(track_rows),
            "candidate_track_count": len(track_rows),
            "representative_track_id": chosen_id,
            "representative_bbox": chosen_box,
            "features": chosen_features,
        }
    return features_by_image, epochs


def topsis_scores(feature_matrix):
    """Equal-weight topsispy scores; constant columns carry no ranking signal."""
    variable = np.ptp(feature_matrix, axis=0) > 0
    used_names = [name for name, use in zip(FEATURE_NAMES, variable) if use]
    if not used_names:
        return np.zeros(len(feature_matrix), dtype=np.float64), used_names

    data = feature_matrix[:, variable]
    weights = np.ones(data.shape[1], dtype=np.float64) / data.shape[1]
    signs = np.ones(data.shape[1], dtype=int)
    _, score_array = tp.topsis(data, weights, signs)
    scores = np.asarray(score_array, dtype=np.float64)
    if scores.shape != (len(feature_matrix),) or not np.all(np.isfinite(scores)):
        raise ValueError("topsispy returned invalid image scores")
    return scores, used_names


def build_ranking(features_by_image):
    names = sorted(features_by_image)
    matrix = np.asarray(
        [[features_by_image[name]["features"][feature] for feature in FEATURE_NAMES]
         for name in names],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Image features must be finite and in [0, 1]")
    scores, used_names = topsis_scores(matrix)
    order = sorted(range(len(names)), key=lambda index: (-scores[index], names[index]))

    ranking = []
    previous_score = None
    tied_rank = 0
    for position, index in enumerate(order, start=1):
        score = float(scores[index])
        if previous_score is None or score != previous_score:
            tied_rank = position
            previous_score = score
        name = names[index]
        image = features_by_image[name]
        ranking.append({
            "rank": tied_rank,
            "image_id": image["image_id"],
            "image_name": name,
            "topsis_score": score,
            "has_candidate": image["has_candidate"],
            "representative_track_id": image["representative_track_id"],
            "representative_bbox": image["representative_bbox"],
        })
    return ranking, used_names


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)
    features_by_image, epochs = build_image_features(metrics, LATE_EPOCHS)
    ranking, used_names = build_ranking(features_by_image)
    no_candidate_names = [name for name, image in features_by_image.items()
                          if not image["has_candidate"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "miss_process_features.json", {
        "epochs": epochs,
        "late_epochs": LATE_EPOCHS,
        "feature_names": FEATURE_NAMES,
        "images": features_by_image,
    })
    save_json(OUTPUT_DIR / "miss_image_ranking.json", {
        "method": "topsispy.topsis",
        "feature_names": FEATURE_NAMES,
        "used_feature_names": used_names,
        "excluded_constant_features": [name for name in FEATURE_NAMES if name not in used_names],
        "weights_for_used_features": [1.0 / len(used_names)] * len(used_names),
        "score_direction": "higher_is_more_suspicious",
        "no_candidate_images": no_candidate_names,
        "ranking": ranking,
    })
    print(f"Ranked {len(ranking)} images; results saved to {OUTPUT_DIR}")
    print(f"Images with no candidate track: {len(no_candidate_names)}")


if __name__ == "__main__":
    main()
