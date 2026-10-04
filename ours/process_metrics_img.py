"""Collect image-level evidence for potentially missing annotations.

Only annotations present in the training JSON are used. Saved post-NMS
predictions are filtered by geometry, then connected across epochs into
position-based tracks. Missing observations remain null in the output.

Edit the constants below and run from the repository root with
``python -m ours.process_metrics_img``.
"""

import json
import math
from pathlib import Path
from statistics import median


DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
ANNOTATIONS_PATH = DATA_ROOT / "fault_inject/0.1/voc/coco_format/annotations_no_miss.json"
PREDICTIONS_DIR = DATA_ROOT / (
    "collection_process_info/voc/yolov7/collected_predict_boxes/"
    "inject_0.1_repeat/repeat_1_1791094449"
)
OUTPUT_PATH = DATA_ROOT / (
    "process_metrics/voc/yolov7/0.1_repeat/"
    "repeat_1_1791094449/miss_process_metrics.json"
)
EPOCHS = 50

# The saved predictions were already filtered by NMS at confidence 0.25.
# Raising this threshold removes weak observations from their trajectories.
MIN_CANDIDATE_CONF = 0.25
MAX_ANNO_IOU = 0.2
MAX_ANNO_COVERAGE = 0.5  # Intersection area / prediction area.
DUPLICATE_IOU = 0.97
TRACK_IOU = 0.5


def validate_settings(epochs):
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    for name, value in (
        ("MIN_CANDIDATE_CONF", MIN_CANDIDATE_CONF),
        ("MAX_ANNO_IOU", MAX_ANNO_IOU),
        ("MAX_ANNO_COVERAGE", MAX_ANNO_COVERAGE),
        ("DUPLICATE_IOU", DUPLICATE_IOU),
        ("TRACK_IOU", TRACK_IOU),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if DUPLICATE_IOU <= TRACK_IOU:
        raise ValueError("DUPLICATE_IOU must exceed TRACK_IOU")


def validate_box(box, description):
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in box) or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError(f"Invalid xyxy box: {description}")
    return [float(value) for value in box]


def intersection_area(box_a, box_b):
    width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    return width * height


def box_iou(box_a, box_b):
    intersection = intersection_area(box_a, box_b)
    if intersection == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def load_visible_annotations(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    images = {}
    for image in data["images"]:
        name = image["file_name"]
        if name in images:
            raise ValueError(f"Duplicate image name: {name}")
        images[name] = {"image_id": image["id"], "annotation_boxes": []}
    if not images:
        raise ValueError("Annotation file contains no images")

    names_by_id = {item["image_id"]: name for name, item in images.items()}
    if len(names_by_id) != len(images):
        raise ValueError("Image IDs must be unique")
    for annotation in data["annotations"]:
        name = names_by_id[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        images[name]["annotation_boxes"].append(
            validate_box([x, y, x + width, y + height], f"anno {annotation['id']}")
        )
    return images


def deduplicate_predictions(predictions, image_name, epoch):
    """Merge almost identical multi-label NMS outputs, keeping max confidence."""
    normalized = []
    for prediction in predictions:
        box = validate_box(prediction["bbox"], f"{image_name}, epoch {epoch}")
        conf = prediction["conf"]
        if not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"Invalid confidence: {image_name}, epoch {epoch}")
        normalized.append({
            "predicted_box_id": prediction["predicted_box_id"],
            "predicted_cls": prediction["predicted_cls"],
            "conf": float(conf),
            "bbox": box,
        })

    kept = []
    for prediction in sorted(normalized, key=lambda item: -item["conf"]):
        if all(box_iou(prediction["bbox"], item["bbox"]) < DUPLICATE_IOU for item in kept):
            kept.append(prediction)
    return kept


def select_candidates(predictions, annotation_boxes):
    candidates = []
    for prediction in predictions:
        if prediction["conf"] < MIN_CANDIDATE_CONF:
            continue
        box = prediction["bbox"]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        max_iou = max((box_iou(box, anno) for anno in annotation_boxes), default=0.0)
        max_coverage = max(
            (intersection_area(box, anno) / box_area for anno in annotation_boxes),
            default=0.0,
        )
        if max_iou < MAX_ANNO_IOU and max_coverage < MAX_ANNO_COVERAGE:
            candidates.append({
                **prediction,
                "max_anno_iou": max_iou,
                "max_anno_coverage": max_coverage,
            })
    return candidates


def new_track(track_id, epochs):
    return {
        "track_id": track_id,
        "representative_bbox": None,
        "present_list": [False] * epochs,
        "conf_list": [None] * epochs,
        "bbox_list": [None] * epochs,
        "predicted_cls_list": [None] * epochs,
        "predicted_box_id_list": [None] * epochs,
        "max_anno_iou_list": [None] * epochs,
        "max_anno_coverage_list": [None] * epochs,
    }


def add_to_track(track, prediction, epoch):
    if track["present_list"][epoch]:
        raise ValueError(f"Track {track['track_id']} already has a box at epoch {epoch}")
    track["present_list"][epoch] = True
    track["conf_list"][epoch] = prediction["conf"]
    track["bbox_list"][epoch] = prediction["bbox"]
    track["predicted_cls_list"][epoch] = prediction["predicted_cls"]
    track["predicted_box_id_list"][epoch] = prediction["predicted_box_id"]
    track["max_anno_iou_list"][epoch] = prediction["max_anno_iou"]
    track["max_anno_coverage_list"][epoch] = prediction["max_anno_coverage"]
    observed_boxes = [box for box in track["bbox_list"] if box is not None]
    track["representative_bbox"] = [
        median(box[coordinate] for box in observed_boxes) for coordinate in range(4)
    ]


def connect_epoch(tracks, candidates, epoch, epochs):
    """Greedy maximum-IoU, one-to-one assignment to existing position tracks."""
    possible = []
    for track_index, track in enumerate(tracks):
        for candidate_index, candidate in enumerate(candidates):
            iou = box_iou(track["representative_bbox"], candidate["bbox"])
            if iou >= TRACK_IOU:
                possible.append((-iou, track_index, candidate_index))

    assigned_tracks = set()
    assigned_candidates = set()
    for _, track_index, candidate_index in sorted(possible):
        if track_index in assigned_tracks or candidate_index in assigned_candidates:
            continue
        add_to_track(tracks[track_index], candidates[candidate_index], epoch)
        assigned_tracks.add(track_index)
        assigned_candidates.add(candidate_index)

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index not in assigned_candidates:
            track = new_track(len(tracks), epochs)
            add_to_track(track, candidate, epoch)
            tracks.append(track)


def collect_metrics(annotation_path, predictions_dir, epochs):
    validate_settings(epochs)
    images = load_visible_annotations(annotation_path)
    result = {
        name: {
            "image_id": image["image_id"],
            "candidate_count_per_epoch": [],
            "tracks": [],
        }
        for name, image in sorted(images.items())
    }

    for epoch in range(epochs):
        path = predictions_dir / f"epoch_{epoch}_predicted_bboxs.json"
        with open(path, encoding="utf-8") as file:
            predictions_by_image = json.load(file)
        missing = result.keys() - predictions_by_image.keys()
        if missing:
            raise ValueError(f"Epoch {epoch} has no inference record for {len(missing)} images")

        for name, image in images.items():
            predictions = predictions_by_image[name]["predicted_bboxs"]
            if not isinstance(predictions, list):
                raise ValueError(f"Epoch {epoch}, {name}: predicted_bboxs must be a list")
            unique = deduplicate_predictions(predictions, name, epoch)
            candidates = select_candidates(unique, image["annotation_boxes"])
            image_result = result[name]
            image_result["candidate_count_per_epoch"].append(len(candidates))
            connect_epoch(image_result["tracks"], candidates, epoch, epochs)

    return {
        "epochs": epochs,
        "settings": {
            "min_candidate_conf": MIN_CANDIDATE_CONF,
            "max_anno_iou_exclusive": MAX_ANNO_IOU,
            "max_anno_coverage_exclusive": MAX_ANNO_COVERAGE,
            "duplicate_iou_inclusive": DUPLICATE_IOU,
            "track_iou_inclusive": TRACK_IOU,
        },
        "images": result,
    }


def main():
    metrics = collect_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, separators=(",", ":"))
    no_candidate = sum(not image["tracks"] for image in metrics["images"].values())
    print(f"Saved image process metrics for {len(metrics['images'])} images to {OUTPUT_PATH}")
    print(f"Images with no candidate track: {no_candidate}")


if __name__ == "__main__":
    main()
