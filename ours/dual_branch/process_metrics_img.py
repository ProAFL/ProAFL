"""收集漏标候选的位置轨迹，保持原图片分支的筛选和关联规则。"""

import math
from statistics import median

from ours.dual_branch.common import read_json, save_json
from ours.dual_branch.config import (
    ANNOTATIONS_PATH, PREDICTIONS_DIR, METRICS_DIR, EPOCHS,
    MIN_CANDIDATE_CONF, MAX_ANNO_IOU, MAX_ANNO_COVERAGE, DUPLICATE_IOU, TRACK_IOU,
)
from ours.dual_branch.process_metrics import box_iou


OUTPUT_PATH = METRICS_DIR / "miss_process_metrics.json"


def validate_box(box, description):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError(f"xyxy 框不合法：{description}")
    return [float(v) for v in box]


def intersection_area(box_a, box_b):
    return (max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
            * max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])))


def load_visible_annotations(path):
    data = read_json(path)
    images = {row["file_name"]: {"image_id": row["id"], "annotation_boxes": []}
              for row in data["images"]}
    names = {row["id"]: row["file_name"] for row in data["images"]}
    if not images or len(images) != len(data["images"]) or len(names) != len(images):
        raise ValueError("图片集合不能为空，图片 ID 和文件名必须唯一")
    for annotation in data["annotations"]:
        x, y, width, height = annotation["bbox"]
        box = validate_box([x, y, x + width, y + height], f"anno {annotation['id']}")
        images[names[annotation["image_id"]]]["annotation_boxes"].append(box)
    return images


def deduplicate_predictions(predictions, image_name, epoch):
    """同轮几乎重合的多类别预测只保留最高置信度，沿用原规则。"""
    normalized = []
    for prediction in predictions:
        box = [float(v) for v in prediction["bbox"]]
        # 保持原脚本对裁边后退化预测框的处理，避免改变图片轨迹。
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError(f"预测框不合法：epoch {epoch}, {image_name}")
        box[2] = box[2] if box[2] > box[0] else box[0] + 1
        box[3] = box[3] if box[3] > box[1] else box[1] + 1
        conf = prediction["conf"]
        if not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError(f"置信度不合法：epoch {epoch}, {image_name}")
        normalized.append({"bbox": box, "conf": float(conf),
                           "predicted_box_id": prediction["predicted_box_id"],
                           "predicted_cls": prediction["predicted_cls"]})
    kept = []
    for prediction in sorted(normalized, key=lambda row: -row["conf"]):
        if all(box_iou(prediction["bbox"], other["bbox"]) < DUPLICATE_IOU for other in kept):
            kept.append(prediction)
    return kept


def select_candidates(predictions, annotation_boxes):
    candidates = []
    for prediction in predictions:
        if prediction["conf"] < MIN_CANDIDATE_CONF:
            continue
        box = prediction["bbox"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        max_iou = max((box_iou(box, anno) for anno in annotation_boxes), default=0.0)
        max_coverage = max((intersection_area(box, anno) / area for anno in annotation_boxes), default=0.0)
        if max_iou < MAX_ANNO_IOU and max_coverage < MAX_ANNO_COVERAGE:
            candidates.append({**prediction, "max_anno_iou": max_iou,
                               "max_anno_coverage": max_coverage})
    return candidates


def new_track(track_id, epochs):
    return {"track_id": track_id, "representative_bbox": None,
            "present_list": [False] * epochs,
            **{key: [None] * epochs for key in (
                "conf_list", "bbox_list", "predicted_cls_list", "predicted_box_id_list",
                "max_anno_iou_list", "max_anno_coverage_list",
            )}}


def add_to_track(track, prediction, epoch):
    if track["present_list"][epoch]:
        raise ValueError("同一轨迹在同一轮不能接收两个预测框")
    track["present_list"][epoch] = True
    for key in ("conf", "bbox", "predicted_cls", "predicted_box_id", "max_anno_iou", "max_anno_coverage"):
        track[f"{key}_list"][epoch] = prediction[key]
    boxes = [box for box in track["bbox_list"] if box is not None]
    track["representative_bbox"] = [median(box[k] for box in boxes) for k in range(4)]


def connect_epoch(tracks, candidates, epoch, epochs):
    """按 IoU 从大到小贪心一对一关联；未关联候选创建新轨迹。"""
    pairs = []
    for track_index, track in enumerate(tracks):
        for candidate_index, candidate in enumerate(candidates):
            iou = box_iou(track["representative_bbox"], candidate["bbox"])
            if iou >= TRACK_IOU:
                pairs.append((-iou, track_index, candidate_index))
    used_tracks, used_candidates = set(), set()
    for _, track_index, candidate_index in sorted(pairs):
        if track_index not in used_tracks and candidate_index not in used_candidates:
            add_to_track(tracks[track_index], candidates[candidate_index], epoch)
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
    for index, candidate in enumerate(candidates):
        if index not in used_candidates:
            track = new_track(len(tracks), epochs)
            add_to_track(track, candidate, epoch)
            tracks.append(track)


def collect_metrics(annotation_path, predictions_dir, epochs):
    thresholds = (MIN_CANDIDATE_CONF, MAX_ANNO_IOU, MAX_ANNO_COVERAGE, DUPLICATE_IOU, TRACK_IOU)
    if epochs <= 0 or any(not 0 <= value <= 1 for value in thresholds) or DUPLICATE_IOU <= TRACK_IOU:
        raise ValueError("训练轮数或图片轨迹阈值不合法")
    images = load_visible_annotations(annotation_path)
    result = {name: {"image_id": row["image_id"], "candidate_count_per_epoch": [], "tracks": []}
              for name, row in sorted(images.items())}
    for epoch in range(epochs):
        print(f"收集图片轨迹：epoch {epoch}")
        predictions = read_json(predictions_dir / f"epoch_{epoch}_predicted_bboxs.json")
        for name, image in images.items():
            boxes = predictions.get(name, {}).get("predicted_bboxs", [])
            if not isinstance(boxes, list):
                raise ValueError(f"预测框列表不合法：epoch {epoch}, {name}")
            candidates = select_candidates(deduplicate_predictions(boxes, name, epoch), image["annotation_boxes"])
            result[name]["candidate_count_per_epoch"].append(len(candidates))
            connect_epoch(result[name]["tracks"], candidates, epoch, epochs)
    return {"epochs": epochs, "settings": {
        "min_candidate_conf": MIN_CANDIDATE_CONF,
        "max_anno_iou_exclusive": MAX_ANNO_IOU,
        "max_anno_coverage_exclusive": MAX_ANNO_COVERAGE,
        "duplicate_iou_inclusive": DUPLICATE_IOU, "track_iou_inclusive": TRACK_IOU,
    }, "images": result}


def main():
    metrics = collect_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    save_json(OUTPUT_PATH, metrics)
    print(f"已保存 {len(metrics['images'])} 张图片的过程度量：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
