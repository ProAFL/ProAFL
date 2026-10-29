"""支持原版 3+3 六维和扩展 4+4 八维排序，默认使用对称八维。

复用 process_metrics.json 的框匹配结果，回查原预测文件的完整类别得分。
输出字段和返回结构兼容原有合并与修复模块。
运行本模块会写入同一个 FEATURES_DIR；对照实验可通过 --output-dir 隔离。
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from ours.dual_branch.common import (
    rank_rows, read_json, save_json, suspicious_percentiles, topsis_scores,
)
from ours.dual_branch.config import (
    ANNOTATIONS_PATH, EPOCHS, FEATURES_DIR, METRICS_DIR, LATE_EPOCHS, PREDICTIONS_DIR,
    CONF_THRESHOLD, IOU_THRESHOLD,
)


BASE_FEATURE_NAMES = (
    "late_conf_deficit", "all_conf_deficit", "first_conf_reach",
    "late_iou_deficit", "all_iou_deficit", "first_iou_reach",
)
CLASS_TRAJECTORY_FEATURE_NAMES = ("relative_class_margin", "alternative_class_persistence")
CLASS_MATCH_IOU = 0.5
MIN_PREDICTION_CONF = 0.25
STRONG_CLASS_MARGIN = 0.5
METRICS_PATH = METRICS_DIR / "process_metrics.json"
OUTPUT_DIR = FEATURES_DIR
DEFAULT_GEOMETRY_WEIGHT = 1.0
DEFAULT_FEATURE_DIMENSIONS = 8
CLASS_PROCESS_FEATURE_NAMES = ("persistent_class_margin", "alternative_class_persistence")
FEATURE_NAMES = BASE_FEATURE_NAMES + CLASS_PROCESS_FEATURE_NAMES + ("persistent_iou_deficit",)
CLASS_FEATURE_NAMES = BASE_FEATURE_NAMES[:3] + ("persistent_class_margin",)
GEOMETRY_FEATURE_NAMES = BASE_FEATURE_NAMES[3:] + ("persistent_iou_deficit",)


def get_branch_feature_names(feature_dimensions):
    if type(feature_dimensions) is not int or feature_dimensions not in (6, 8):
        raise ValueError("feature_dimensions 只能为 6 或 8")
    if feature_dimensions == 6:
        return BASE_FEATURE_NAMES[:3], BASE_FEATURE_NAMES[3:]
    return CLASS_FEATURE_NAMES, GEOMETRY_FEATURE_NAMES


def first_reach_fraction(values, threshold):
    """首次严格超过阈值的轮次 / 总轮数，始终未达到则为 1。"""
    return next((epoch for epoch, value in enumerate(values) if value > threshold), len(values)) / len(values)


def extract_base_features(metric):
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


def relative_class_margin(probs, category_id):
    """相对差抵消共同 objectness；不是 softmax 概率或错误概率。"""
    if (not isinstance(probs, list) or len(probs) < 2
            or type(category_id) is not int or not 0 <= category_id < len(probs)
            or any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probs)):
        raise ValueError("完整类别得分或标注类别不合法")
    alternative = max((i for i in range(len(probs)) if i != category_id), key=probs.__getitem__)
    confidence = max(probs)
    margin = max(0.0, probs[alternative] - probs[category_id]) / max(confidence, 1e-12)
    return margin, alternative, confidence


def advance_run(current, previous, alternative, iou, margin, confidence):
    reliable = iou >= CLASS_MATCH_IOU and confidence >= MIN_PREDICTION_CONF
    if reliable and margin >= STRONG_CLASS_MARGIN:
        return (current + 1 if alternative == previous else 1), alternative
    return 0, -1


def collect_class_features(annotations, metrics, predictions_dir, epochs):
    """按现存预测 ID 回查，不重复执行 IoU 两两匹配；逐轮释放预测数据。"""
    predictions_dir = Path(predictions_dir)
    for epoch in range(epochs):
        path = predictions_dir / f"epoch_{epoch}_predicted_bboxs.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少完整类别预测文件：{path}；请指定 --predictions-dir")
    names = {image["id"]: image["file_name"] for image in annotations["images"]}
    ids = sorted(metrics, key=int)
    index = {aid: i for i, aid in enumerate(ids)}
    total_margin = np.zeros(len(ids))
    longest = np.zeros(len(ids), dtype=np.int32)
    current = np.zeros(len(ids), dtype=np.int32)
    previous = np.full(len(ids), -1, dtype=np.int32)
    by_image = defaultdict(list)
    for anno in annotations["annotations"]:
        aid, category = str(anno["id"]), anno["category_id"]
        if type(category) is not int or category < 0:
            raise ValueError(f"标注类别不合法：{aid}")
        row = metrics[aid]
        for key in ("selected_predicted_box_id_list", "iou_list", "conf_list", "has_overlap_list"):
            if len(row[key]) != epochs:
                raise ValueError(f"{aid} 的 {key} 必须包含 {epochs} 轮")
        by_image[names[anno["image_id"]]].append((index[aid], aid, category, row))
    for epoch in range(epochs):
        print(f"补充类别得分轨迹：epoch {epoch + 1}/{epochs}", flush=True)
        images = read_json(predictions_dir / f"epoch_{epoch}_predicted_bboxs.json")
        for name, rows in by_image.items():
            predictions = images.get(name, {}).get("predicted_bboxs", [])
            selected = {p["predicted_box_id"]: p for p in predictions}
            if len(selected) != len(predictions):
                raise ValueError(f"预测框 ID 重复：epoch {epoch}, {name}")
            for i, aid, category, row in rows:
                prediction_id = row["selected_predicted_box_id_list"][epoch]
                overlap, iou = row["has_overlap_list"][epoch], row["iou_list"][epoch]
                if (type(overlap) is not bool or overlap != (prediction_id is not None)
                        or not math.isfinite(iou) or not 0 <= iou <= 1):
                    raise ValueError(f"匹配轨迹不合法：{aid}, epoch {epoch}")
                if prediction_id is None:
                    current[i], previous[i] = 0, -1
                    continue
                prediction = selected.get(prediction_id)
                if prediction is None:
                    raise ValueError(f"预测目录与轨迹不匹配：{aid}, epoch {epoch}, ID {prediction_id}")
                probs = prediction.get("probs")
                margin, alternative, confidence = relative_class_margin(probs, category)
                # 检查实验来源，避免静默混用其他实验的相同预测 ID。
                if abs(probs[category] - row["conf_list"][epoch]) > 1e-6:
                    raise ValueError(f"预测类别得分与轨迹不一致：{aid}, epoch {epoch}")
                if iou >= CLASS_MATCH_IOU and confidence >= MIN_PREDICTION_CONF:
                    total_margin[i] += margin
                current[i], previous[i] = advance_run(
                    current[i], previous[i], alternative, iou, margin, confidence,
                )
                longest[i] = max(longest[i], current[i])
        del images
    return {aid: dict(zip(CLASS_TRAJECTORY_FEATURE_NAMES, (
        float(total_margin[i] / epochs), float(longest[i] / epochs),
    ))) for i, aid in enumerate(ids)}


def validate_geometry_weight(weight):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError("geometry_weight 必须是 [0,1] 内的有限数值")


def fuse_branch_percentiles(class_percentiles, geometry_percentiles,
                            geometry_weight=DEFAULT_GEOMETRY_WEIGHT):
    """保留分支内特征结构，通过分支优先级控制同预算下的修复取舍。"""
    validate_geometry_weight(geometry_weight)
    return np.maximum(class_percentiles, geometry_weight * np.asarray(geometry_percentiles))


def persistent_iou_deficit(metric):
    """平均可观测几何分歧 × 最长连续显著几何分歧占比。

    调用前由原六特征提取器检查轨迹；无匹配是证据缺失，不计作持续冲突。
    """
    total_deficit = 0.0
    longest = current = 0
    for iou, observed in zip(metric["iou_list"], metric["has_overlap_list"]):
        if observed:
            total_deficit += 1.0 - iou
        if observed and iou <= IOU_THRESHOLD:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return (total_deficit / EPOCHS) * (longest / EPOCHS)


def extract_features(metric, category_id, class_features):
    """保留原六特征，补充类别/几何持续冲突及类别连续支持辅助统计。"""
    features, never_overlapped = extract_base_features(metric)
    if type(category_id) is not int or category_id < 0:
        raise ValueError("标注类别必须是非负整数")
    for name in CLASS_TRAJECTORY_FEATURE_NAMES:
        value = class_features[name]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"类别过程特征必须位于 [0,1]：{name}")
        if never_overlapped and value != 0:
            raise ValueError("无匹配标注不能具有类别冲突证据")
    # 单纯的相对 margin 也会放大困难干净目标的类别混淆。
    # 需要同一个替代类别的连续强支持，才形成强的持续类别冲突证据。
    persistence = float(class_features["alternative_class_persistence"])
    features["persistent_class_margin"] = float(class_features["relative_class_margin"]) * persistence
    features["alternative_class_persistence"] = persistence
    features["persistent_iou_deficit"] = persistent_iou_deficit(metric)
    return features, never_overlapped


def build_results(annotations_path, metrics_path, predictions_dir=None, class_features=None,
                  geometry_weight=DEFAULT_GEOMETRY_WEIGHT,
                  feature_dimensions=DEFAULT_FEATURE_DIMENSIONS):
    """只使用当前标注类别和预测轨迹，不使用 fault_type 或正确标签。"""
    class_names, geometry_names = get_branch_feature_names(feature_dimensions)
    validate_geometry_weight(geometry_weight)
    annotation_data = read_json(annotations_path)
    annotations = annotation_data["annotations"]
    by_id = {annotation["id"]: annotation for annotation in annotations}
    ids = sorted(by_id)
    if not ids or len(ids) != len(annotations):
        raise ValueError("标注 ID 必须非空且唯一")
    metrics = read_json(metrics_path)
    if set(metrics) != {str(i) for i in ids}:
        raise ValueError("标注 ID 与过程度量 ID 不一致")
    if feature_dimensions == 8:
        if class_features is None:
            class_features = collect_class_features(
                annotation_data, metrics,
                PREDICTIONS_DIR if predictions_dir is None else predictions_dir, EPOCHS,
            )
        if set(class_features) != set(metrics):
            raise ValueError("类别过程特征 ID 与过程度量 ID 不一致")
    features, missing_flags = {}, []
    for annotation_id in ids:
        if feature_dimensions == 6:
            # 原六维只使用既有 conf/IoU 轨迹，不读取额外类别预测文件。
            row, missing = extract_base_features(metrics[str(annotation_id)])
        else:
            row, missing = extract_features(metrics[str(annotation_id)], by_id[annotation_id]["category_id"],
                                            class_features[str(annotation_id)])
        features[str(annotation_id)] = row
        missing_flags.append(missing)
    no_overlap = np.asarray(missing_flags, dtype=bool)
    evidence = ~no_overlap
    class_scores = topsis_scores(np.asarray([
        [features[str(i)][name] for name in class_names] for i in ids
    ]))
    geometry_scores = topsis_scores(np.asarray([
        [features[str(i)][name] for name in geometry_names] for i in ids
    ]))
    class_percentiles = suspicious_percentiles(class_scores, evidence)
    geometry_percentiles = suspicious_percentiles(geometry_scores, evidence)
    scores = fuse_branch_percentiles(class_percentiles, geometry_percentiles, geometry_weight)
    scores[no_overlap] = 0.0
    ranking = rank_rows([
        {"anno_id": annotation_id, "topsis_score": float(scores[index]),
         "never_overlapped": bool(no_overlap[index])}
        for index, annotation_id in enumerate(ids)
    ], "anno_id")
    branches = {str(annotation_id): {
        "class_topsis_score": float(class_scores[index]),
        "geometry_topsis_score": float(geometry_scores[index]),
        "class_percentile": float(class_percentiles[index]),
        "geometry_percentile": float(geometry_percentiles[index]),
        "weighted_geometry_percentile": float(geometry_weight * geometry_percentiles[index]),
        "fused_score": float(scores[index]),
    } for index, annotation_id in enumerate(ids)}
    missing_ids = [i for i, missing in zip(ids, missing_flags) if missing]
    no_overlap_report = {
        "definition": "has_overlap_list is false in every epoch",
        "count": len(missing_ids), "annotation_ids": missing_ids,
        "topsis_scores": {str(i): 0.0 for i in missing_ids},
        "note": "无重叠标注不参与百分位校准；合并时按原规则放在最前。",
    }
    return features, ranking, branches, no_overlap_report


def save_results(features, ranking, branches, no_overlap,
                 geometry_weight=DEFAULT_GEOMETRY_WEIGHT, output_dir=None,
                 feature_dimensions=DEFAULT_FEATURE_DIMENSIONS):
    """输出兼容 merge_rank、repair 以及流水线的内存对象传递。"""
    class_names, geometry_names = get_branch_feature_names(feature_dimensions)
    feature_names = BASE_FEATURE_NAMES if feature_dimensions == 6 else FEATURE_NAMES
    if any(set(row) != set(feature_names) for row in features.values()):
        raise ValueError("保存时的 feature_dimensions 与特征不一致，请与 build_results 使用相同维数")
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    validate_geometry_weight(geometry_weight)
    for row in branches.values():
        expected = max(row["class_percentile"], geometry_weight * row["geometry_percentile"])
        if abs(row["fused_score"] - expected) > 1e-12:
            raise ValueError("保存时的 geometry_weight 与排名不一致，请与 build_results 使用同一权重")
    source = {
        "feature_names": feature_names,
        "feature_dimensions": feature_dimensions,
        "feature_set": "six" if feature_dimensions == 6 else "symmetric",
        "ranking_feature_names": class_names + geometry_names,
        "auxiliary_feature_names": [name for name in feature_names
                                    if name not in class_names + geometry_names],
        "class_feature_names": class_names,
        "geometry_feature_names": geometry_names,
        "branch_weights": {"class": [1 / len(class_names)] * len(class_names),
                           "geometry": [1 / len(geometry_names)] * len(geometry_names)},
        "method": f"dual_head_{'max' if geometry_weight == 1 else 'weighted_max'}_{len(class_names) + len(geometry_names)}features",
        "fusion": {
            "formula": "max(class_percentile, geometry_weight * geometry_percentile)",
            "class_weight": 1.0, "geometry_weight": geometry_weight,
        },
        "score_direction": "higher_is_more_suspicious",
        "score_range": [0.0, 1.0],
        "epochs": EPOCHS, "late_epochs": LATE_EPOCHS,
        "conf_threshold_strictly_greater_than": CONF_THRESHOLD,
        "iou_threshold_strictly_greater_than": IOU_THRESHOLD,
        "class_process_features": {
            "minimum_match_iou": CLASS_MATCH_IOU,
            "minimum_prediction_confidence": MIN_PREDICTION_CONF,
            "strong_relative_margin": STRONG_CLASS_MARGIN,
            "margin_formula": "mean(reliable * max(0, max_other_score - label_score) / max_score)",
            "persistence_formula": "longest consecutive reliable strong-margin run for the same alternative / epochs",
            "persistent_class_margin_formula": "mean_relative_margin * alternative_class_persistence",
            "fusion": "max(class_percentile, geometry_weight * geometry_percentile)",
        },
        "geometry_process_features": {
            "conflict_iou_at_most": IOU_THRESHOLD,
            "persistent_iou_deficit_formula": "mean(observed * (1 - iou)) * longest_consecutive_observed_low_iou_run / epochs",
            "missing_observation_policy": "zero evidence and interrupt the conflict run",
        },
        "ranking": ranking,
    }
    if feature_dimensions == 6:
        source.pop("class_process_features")
        source.pop("geometry_process_features")
    for filename, value in (
        ("process_features.json", features), ("topsis_ranking.json", source),
        ("branch_scores.json", branches), ("never_overlapped.json", no_overlap),
    ):
        save_json(output_dir / filename, value)
    print(f"已保存 {len(ranking)} 个标注的 {feature_dimensions} 维双分支排名：{output_dir}")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--geometry-weight", type=float, default=DEFAULT_GEOMETRY_WEIGHT,
                        help="geometry branch priority in [0,1]; 1 restores equal-priority fusion")
    parser.add_argument("--feature-dimensions", type=int, choices=(6, 8), default=DEFAULT_FEATURE_DIMENSIONS)
    args = parser.parse_args(argv)
    save_results(*build_results(ANNOTATIONS_PATH, METRICS_PATH, args.predictions_dir,
                               geometry_weight=args.geometry_weight, feature_dimensions=args.feature_dimensions),
                 geometry_weight=args.geometry_weight, feature_dimensions=args.feature_dimensions,
                 output_dir=args.output_dir)


if __name__ == "__main__":
    main()
