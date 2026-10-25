"""提取图片六特征并进行 TOPSIS 排序，图片分支不做双分支改动。"""

import math

import numpy as np

from ours.dual_branch.common import read_json, save_json, topsis_scores, rank_rows
from ours.dual_branch.config import METRICS_DIR, FEATURES_DIR, LATE_EPOCHS
from ours.dual_branch.process_metrics_img import box_iou, validate_box


METRICS_PATH = METRICS_DIR / "miss_process_metrics.json"
OUTPUT_DIR = FEATURES_DIR
FEATURE_NAMES = (
    "all_presence_rate", "late_presence_rate",
    "all_confidence_evidence", "late_confidence_evidence",
    "longest_consecutive_presence_rate", "position_stability",
)


def track_features(track, epochs, late_epochs):
    present = track["present_list"]
    confidence = track["conf_list"]
    boxes = track["bbox_list"]
    sequences = [track[key] for key in (
        "present_list", "conf_list", "bbox_list", "predicted_cls_list",
        "predicted_box_id_list", "max_anno_iou_list", "max_anno_coverage_list",
    )]
    if any(not isinstance(values, list) or len(values) != epochs for values in sequences):
        raise ValueError("轨迹各字段必须覆盖全部训练轮次")
    observed_boxes, confidence_evidence = [], []
    longest_run = current_run = 0
    for epoch, appears in enumerate(present):
        if type(appears) is not bool:
            raise ValueError("present_list 必须为布尔值列表")
        if appears:
            score = confidence[epoch]
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"轨迹置信度不合法：epoch {epoch}")
            if any(values[epoch] is None for values in sequences[1:]):
                raise ValueError(f"已有观测的轨迹字段不能为 null：epoch {epoch}")
            observed_boxes.append(validate_box(boxes[epoch], f"track epoch {epoch}"))
            confidence_evidence.append(float(score))
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            if any(values[epoch] is not None for values in sequences[1:]):
                raise ValueError(f"缺失观测的轨迹字段必须为 null：epoch {epoch}")
            confidence_evidence.append(0.0)
            current_run = 0
    if not observed_boxes:
        raise ValueError("不能保存完全没有观测的轨迹")
    representative = validate_box(track["representative_bbox"], "representative_bbox")
    # 单次观测无法说明跨轮次稳定性，因此其稳定性设为 0。
    stability = (sum(box_iou(box, representative) for box in observed_boxes) / len(observed_boxes)
                 if len(observed_boxes) >= 2 else 0.0)
    return {
        "all_presence_rate": sum(present) / epochs,
        "late_presence_rate": sum(present[-late_epochs:]) / late_epochs,
        "all_confidence_evidence": sum(confidence_evidence) / epochs,
        "late_confidence_evidence": sum(confidence_evidence[-late_epochs:]) / late_epochs,
        "longest_consecutive_presence_rate": longest_run / epochs,
        "position_stability": stability,
    }


def build_image_features(metrics, late_epochs):
    epochs = metrics["epochs"]
    if type(epochs) is not int or not 1 <= late_epochs <= epochs:
        raise ValueError("训练轮数或后期窗口不合法")
    if not metrics["images"]:
        raise ValueError("图片过程度量不能为空")
    features_by_image = {}
    for name, image in sorted(metrics["images"].items()):
        tracks = image["tracks"]
        if len({track["track_id"] for track in tracks}) != len(tracks):
            raise ValueError(f"图片内轨迹 ID 重复：{name}")
        rows = [(track, track_features(track, epochs, late_epochs)) for track in tracks]
        observed_counts = [sum(track["present_list"][epoch] for track in tracks) for epoch in range(epochs)]
        if image["candidate_count_per_epoch"] != observed_counts:
            raise ValueError(f"候选框数量与轨迹不一致：{name}")
        if rows:
            # 沿用原规则：全程置信度证据最大者代表该图片；再比较后期证据、轨迹 ID。
            chosen, features = min(rows, key=lambda row: (
                -row[1]["all_confidence_evidence"], -row[1]["late_confidence_evidence"], row[0]["track_id"],
            ))
            chosen_id, chosen_box = chosen["track_id"], chosen["representative_bbox"]
        else:
            features = {key: 0.0 for key in FEATURE_NAMES}
            chosen_id, chosen_box = None, None
        features_by_image[name] = {
            "image_id": image["image_id"], "has_candidate": bool(rows),
            "candidate_track_count": len(rows), "representative_track_id": chosen_id,
            "representative_bbox": chosen_box, "features": features,
        }
    return features_by_image, epochs


def build_ranking(features_by_image):
    names = sorted(features_by_image)
    matrix = np.asarray([[features_by_image[name]["features"][key] for key in FEATURE_NAMES] for name in names])
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("图片特征必须是 [0,1] 内的有限值")
    # 常量列没有排序信息；全部为常量时，所有图片同分。
    variable = np.ptp(matrix, axis=0) > 0
    used_names = [key for key, use in zip(FEATURE_NAMES, variable) if use]
    scores = topsis_scores(matrix[:, variable]) if used_names else np.zeros(len(names))
    ranking = rank_rows([
        {"image_name": name, "image_id": features_by_image[name]["image_id"],
         "topsis_score": float(scores[index]), "has_candidate": features_by_image[name]["has_candidate"],
         "representative_track_id": features_by_image[name]["representative_track_id"],
         "representative_bbox": features_by_image[name]["representative_bbox"]}
        for index, name in enumerate(names)
    ], "image_name")
    return ranking, used_names


def save_results(features, epochs, ranking, used_names):
    """集中保存图片结果，并返回供合并使用的排名数据。"""
    save_json(OUTPUT_DIR / "miss_process_features.json", {
        "epochs": epochs, "late_epochs": LATE_EPOCHS, "feature_names": FEATURE_NAMES, "images": features,
    })
    source = {
        "method": "equal_weight_topsis", "feature_names": FEATURE_NAMES,
        "used_feature_names": used_names,
        "excluded_constant_features": [key for key in FEATURE_NAMES if key not in used_names],
        "weights_for_used_features": [1 / len(used_names)] * len(used_names) if used_names else [],
        "score_direction": "higher_is_more_suspicious",
        "no_candidate_images": [name for name, row in features.items() if not row["has_candidate"]],
        "ranking": ranking,
    }
    save_json(OUTPUT_DIR / "miss_image_ranking.json", source)
    print(f"已保存 {len(ranking)} 张图片的排名：{OUTPUT_DIR}")
    return source


def main():
    features, epochs = build_image_features(read_json(METRICS_PATH), LATE_EPOCHS)
    ranking, used_names = build_ranking(features)
    save_results(features, epochs, ranking, used_names)


if __name__ == "__main__":
    main()
