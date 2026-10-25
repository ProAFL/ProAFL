"""提取完整六特征，类别/几何各三项 TOPSIS，按百分位最大值融合。"""

import numpy as np

from ours.dual_branch.common import read_json, save_json, topsis_scores, suspicious_percentiles, rank_rows
from ours.dual_branch.config import (
    ANNOTATIONS_PATH, METRICS_DIR, FEATURES_DIR, EPOCHS,
    LATE_EPOCHS, CONF_THRESHOLD, IOU_THRESHOLD,
)


METRICS_PATH = METRICS_DIR / "process_metrics.json"
OUTPUT_DIR = FEATURES_DIR
FEATURE_NAMES = (
    "late_conf_deficit", "all_conf_deficit", "first_conf_reach",
    "late_iou_deficit", "all_iou_deficit", "first_iou_reach",
)


def first_reach_fraction(values, threshold):
    """首次严格超过阈值的轮次 / 总轮数，始终未达到则为 1。"""
    return next((epoch for epoch, value in enumerate(values) if value > threshold), len(values)) / len(values)


def extract_features(metric):
    conf = np.asarray(metric["conf_list"], dtype=np.float64)
    iou = np.asarray(metric["iou_list"], dtype=np.float64)
    overlap = metric["has_overlap_list"]
    if conf.shape != (EPOCHS,) or iou.shape != (EPOCHS,) or len(overlap) != EPOCHS:
        raise ValueError(f"每个标注必须包含 {EPOCHS} 轮轨迹")
    if not 1 <= LATE_EPOCHS <= EPOCHS or not all(type(v) is bool for v in overlap):
        raise ValueError("后期窗口或重叠标记不合法")
    if not 0 <= CONF_THRESHOLD <= 1 or not 0 <= IOU_THRESHOLD <= 1:
        raise ValueError("首次达到阈值必须位于 [0,1]")
    if any(not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)) for values in (conf, iou)):
        raise ValueError("置信度和 IoU 必须是 [0,1] 内的有限值")
    return {
        "late_conf_deficit": 1.0 - float(conf[-LATE_EPOCHS:].mean()),
        "all_conf_deficit": 1.0 - float(conf.mean()),
        "first_conf_reach": first_reach_fraction(conf, CONF_THRESHOLD),
        "late_iou_deficit": 1.0 - float(iou[-LATE_EPOCHS:].mean()),
        "all_iou_deficit": 1.0 - float(iou.mean()),
        "first_iou_reach": first_reach_fraction(iou, IOU_THRESHOLD),
    }, not any(overlap)


def build_results(annotations_path, metrics_path):
    # 只读取现存标注 ID；fault_type 和正确标注均不参与特征及排序。
    annotations = read_json(annotations_path)["annotations"]
    ids = sorted(annotation["id"] for annotation in annotations)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("标注 ID 必须非空且唯一")
    metrics = read_json(metrics_path)
    if set(metrics) != {str(i) for i in ids}:
        raise ValueError("标注 ID 与过程度量 ID 不一致")
    features, no_overlap = {}, []
    for annotation_id in ids:
        row, missing = extract_features(metrics[str(annotation_id)])
        features[str(annotation_id)] = row
        no_overlap.append(missing)
    matrix = np.asarray([[features[str(i)][name] for name in FEATURE_NAMES] for i in ids])
    no_overlap = np.asarray(no_overlap, dtype=bool)
    evidence = ~no_overlap

    # TOPSIS 在全部标注上计算；百分位只在至少有一次重叠的标注内计算。
    class_scores = topsis_scores(matrix[:, :3])  # 包含 first_conf_reach，不能删减。
    geometry_scores = topsis_scores(matrix[:, 3:])
    class_percentiles = suspicious_percentiles(class_scores, evidence)
    geometry_percentiles = suspicious_percentiles(geometry_scores, evidence)
    scores = np.maximum(class_percentiles, geometry_percentiles)

    ranking = rank_rows([
        {"anno_id": annotation_id, "topsis_score": float(scores[index]),
         "never_overlapped": bool(no_overlap[index])}
        for index, annotation_id in enumerate(ids)
    ], "anno_id")
    # 为兼容原 merge_rank，topsis_score 字段承载最终融合分数。
    # 两个分支的原始得分单独保存，不改变最终合并条目的字段。
    branches = {str(annotation_id): {
        "class_topsis_score": float(class_scores[index]),
        "geometry_topsis_score": float(geometry_scores[index]),
        "class_percentile": float(class_percentiles[index]),
        "geometry_percentile": float(geometry_percentiles[index]),
        "fused_score": float(scores[index]),
    } for index, annotation_id in enumerate(ids)}
    missing_ids = [annotation_id for index, annotation_id in enumerate(ids) if no_overlap[index]]
    no_overlap_report = {
        "definition": "has_overlap_list is false in every epoch",
        "count": len(missing_ids), "annotation_ids": missing_ids,
        "topsis_scores": {str(i): 0.0 for i in missing_ids},
        "note": "无重叠标注不参与百分位校准；合并时按原规则放在最前。",
    }
    return features, ranking, branches, no_overlap_report


def save_results(features, ranking, branches, no_overlap):
    """集中保存标注结果，并返回供合并使用的排名数据。"""
    save_json(OUTPUT_DIR / "process_features.json", features)
    source = {
        "feature_names": FEATURE_NAMES, "class_feature_names": FEATURE_NAMES[:3],
        "geometry_feature_names": FEATURE_NAMES[3:],
        "branch_weights": {"class": [1 / 3] * 3, "geometry": [1 / 3] * 3},
        "method": "dual_head_max_6features",
        "score_direction": "higher_is_more_suspicious",
        "epochs": EPOCHS, "late_epochs": LATE_EPOCHS,
        "conf_threshold_strictly_greater_than": CONF_THRESHOLD,
        "iou_threshold_strictly_greater_than": IOU_THRESHOLD,
        "ranking": ranking,
    }
    save_json(OUTPUT_DIR / "topsis_ranking.json", source)
    save_json(OUTPUT_DIR / "branch_scores.json", branches)
    save_json(OUTPUT_DIR / "never_overlapped.json", no_overlap)
    print(f"已保存 {len(ranking)} 个标注的六特征双分支排名：{OUTPUT_DIR}")
    return source


def main():
    save_results(*build_results(ANNOTATIONS_PATH, METRICS_PATH))


if __name__ == "__main__":
    main()
