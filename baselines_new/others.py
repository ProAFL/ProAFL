"""Entropy / DeepGini / Margin / Loss 标注排序。

从仓库根目录运行 python -m baselines_new.others。
读取 epoch 49 预测框，匹配规则复用 choose_prediction，最终只保存 rank.joblib。
可选嫁接 ObjectLab 漏标分数，以组内百分位混排 anno 和图片。
"""

import math
from pathlib import Path
from sys import float_info

import joblib

from ours.dual_branch.process_metrics import choose_prediction, load_annotations
from ours.small_utils import read_json, read_yaml

DATA_ROOT = Path(read_yaml("config.yaml")["exp_data_dir"])
DATASET_NAME = "visdrone"  # voc | kitti | visdrone
MODEL_NAME = "frcnn"     # yolov7 | rtdetr | frcnn 
FAULT_RATIO = 0.1
REPEAT_ID = 1
EPOCH = 49
BASELINES = ("entropy", "deepgini", "margin", "loss")
ENABLE_IMAGE_MIXING = True  # True：ObjectLab 图片分 + 百分位混排；False：图片全部置后
IMAGE_PERCENTILE_SCALE = 1.0 # 越小，图片越靠前；1.0 表示两组直接比较百分位

ANNOTATIONS_PATH = DATA_ROOT / (
    f"fault_inject/{FAULT_RATIO}/{DATASET_NAME}/coco_format/annotations_no_miss.json"
)
PREDICTIONS_PATH = DATA_ROOT / (
    f"collection_process_info/{DATASET_NAME}/{MODEL_NAME}/collected_predict_boxes/"
    f"inject_{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}/epoch_{EPOCH}_predicted_bboxs.json"
)
OUTPUT_ROOT = DATA_ROOT / "rank"


def calcu_entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def caclu_deepgini(probs):
    return 1.0 - sum(p * p for p in probs)


def caclu_margin(probs):
    first, second = sorted(probs, reverse=True)[:2]
    return (1.0 - (first - second)) ** 2


def caclu_loss(probs, category, prediction_box, annotation_box):
    """输入为类别概率及同一归一化 xyxy 坐标系的框；Smooth L1 的 beta=1。"""
    classification = -math.log(max(probs[category], 1e-12))
    differences = [abs(p - g) for p, g in zip(prediction_box, annotation_box)]
    localization = sum(0.5 * d * d if d < 1 else d - 0.5 for d in differences) / 4
    return classification + localization


def caclu_rank_score(baseline_name, probs, category, prediction_box, annotation_box):
    """分数越大，anno 越值得检查；probs 使用采集值，不重新 softmax。"""
    if baseline_name == "entropy":
        return calcu_entropy(probs)
    if baseline_name == "deepgini":
        return caclu_deepgini(probs)
    if baseline_name == "margin":
        return caclu_margin(probs)
    if baseline_name == "loss":
        return caclu_loss(probs, category, prediction_box, annotation_box)
    raise ValueError(f"不支持的 baseline：{baseline_name}")


def collect_image_scores(by_image, image_info, prediction_data):
    """直接复用 ObjectLab 的阈值、相似度和漏标分公式，包括全数据集 sim*。"""
    from baselines_new import objectLab

    if not 0 <= objectLab.LOW_CONF_THRESHOLD <= objectLab.HIGH_CONF_THRESHOLD <= 1:
        raise ValueError("ObjectLab 阈值须满足 0 <= low <= high <= 1")
    if (not 0 <= objectLab.SIM_ALPHA <= 1
            or not math.isfinite(objectLab.SIM_SIGMA) or objectLab.SIM_SIGMA <= 0):
        raise ValueError("ObjectLab 相似度参数不合法")
    predictions_by_image = {}
    min_similarity = 1.0
    for name, info in image_info.items():
        predictions = prediction_data.get(name, {}).get("predicted_bboxs", [])
        if not isinstance(predictions, list):
            raise ValueError(f"预测框列表不合法：{name}")
        for pred in predictions:
            conf = pred["conf"]
            if (isinstance(conf, bool) or not isinstance(conf, (int, float))
                    or not math.isfinite(conf) or not 0 <= conf <= 1):
                raise ValueError(f"预测置信度不合法：{name}")
        predictions = [pred for pred in predictions if pred["conf"] > objectLab.LOW_CONF_THRESHOLD]
        predictions_by_image[name] = predictions
        _, similarities = objectLab.box_matrices(
            by_image.get(name, []), predictions, (info["width"], info["height"])
        )
        if similarities.size:
            min_similarity = min(min_similarity, float(similarities.min()))
    min_similarity = max(min_similarity, float_info.min)
    scored = []
    for name, info in image_info.items():
        annos, predictions = by_image.get(name, []), predictions_by_image[name]
        ious, similarities = objectLab.box_matrices(annos, predictions, (info["width"], info["height"]))
        score = objectLab.image_miss_score(annos, predictions, ious, similarities, min_similarity)
        scored.append((name, score))
    return scored


def within_group_percentiles(scored, descending=False):
    """按可疑程度排序，同分取平均名次百分位；越小越靠前。"""
    scored = sorted(scored, key=lambda row: (-row[1] if descending else row[1], row[0]))
    result, start = [], 0
    while start < len(scored):
        end = start + 1
        while end < len(scored) and scored[end][1] == scored[start][1]:
            end += 1
        percentile = (start + end) / (2 * len(scored))
        result.extend((item, percentile) for item, _ in scored[start:end])
        start = end
    return result


def merge_rankings(unmatched, anno_scores, image_scores):
    """未匹配 anno 在前、有证据条目百分位混排、无漏标证据图片在最后。"""
    if (isinstance(IMAGE_PERCENTILE_SCALE, bool) or not math.isfinite(IMAGE_PERCENTILE_SCALE)
            or IMAGE_PERCENTILE_SCALE <= 0):
        raise ValueError("IMAGE_PERCENTILE_SCALE 必须为正数")
    candidates = [(name, score) for name, score in image_scores if score < 1.0]
    no_candidates = sorted(name for name, score in image_scores if score == 1.0)
    annos = [(percentile, 0, anno_id) for anno_id, percentile
             in within_group_percentiles(anno_scores, descending=True)]
    images = [(percentile * IMAGE_PERCENTILE_SCALE, 1, name) for name, percentile
              in within_group_percentiles(candidates)]
    # 跨组同百分位时 anno 在前；组内同分按 ID/图片名固定顺序。
    return sorted(unmatched) + [item for _, _, item in sorted(annos + images)] + no_candidates


def build_ranking(annotation_path, prediction_path, baseline_name, mix_images=None):
    """开关关闭时沿用原排名；开启时嫁接图片漏标分数和组内百分位。"""
    if mix_images is None:
        mix_images = ENABLE_IMAGE_MIXING
    if baseline_name not in BASELINES:
        raise ValueError(f"不支持的 baseline：{baseline_name}")
    data = read_json(annotation_path)
    by_image, _ = load_annotations(annotation_path)
    original_ids = {str(anno["id"]): anno["id"] for anno in data["annotations"]}
    image_info = {image["file_name"]: image for image in data["images"]}
    prediction_data = read_json(prediction_path)
    unmatched, scored = [], []
    for name, annos in by_image.items():
        predictions = prediction_data.get(name, {}).get("predicted_bboxs", [])
        if not isinstance(predictions, list):
            raise ValueError(f"预测框列表不合法：{name}")
        for annotation_id, category, box in annos:
            anno_id = original_ids[annotation_id]
            # 不做类别筛选或额外置信度过滤，完全沿用 choose_prediction 的匹配规则。
            prediction, _ = choose_prediction(box, predictions)
            if prediction is None:
                unmatched.append(anno_id)
                continue
            probs = prediction.get("probs")
            if (not isinstance(probs, list) or len(probs) < 2 or category >= len(probs)
                    or any(isinstance(p, bool) or not isinstance(p, (int, float))
                           or not math.isfinite(p) or not 0 <= p <= 1 for p in probs)):
                raise ValueError(f"类别概率不合法：anno {anno_id}, image {name}")
            prediction_box, annotation_box = prediction["bbox"], box
            if baseline_name == "loss":
                info = image_info[name]
                width, height = info["width"], info["height"]
                if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                    raise ValueError(f"图片宽高不合法：{name}")
                scale = (width, height, width, height)
                prediction_box = [v / s for v, s in zip(prediction_box, scale)]
                annotation_box = [v / s for v, s in zip(annotation_box, scale)]
            score = caclu_rank_score(baseline_name, probs, category, prediction_box, annotation_box)
            if not math.isfinite(score):
                raise ValueError(f"排序分数不合法：anno {anno_id}")
            scored.append((anno_id, score))
    scored.sort(key=lambda row: (-row[1], row[0]))
    if mix_images:
        image_scores = collect_image_scores(by_image, image_info, prediction_data)
        return merge_rankings(unmatched, scored, image_scores)
    # 使用 COCO images 覆盖零标注图片，无需扫描训练图片目录。
    return sorted(unmatched) + [anno_id for anno_id, _ in scored] + sorted(image_info)


def main():
    print(f"图片混排：{ENABLE_IMAGE_MIXING}，预测文件：{PREDICTIONS_PATH}")
    for baseline_name in BASELINES:
        rank = build_ranking(ANNOTATIONS_PATH, PREDICTIONS_PATH, baseline_name)
        output_path = OUTPUT_ROOT / (
            f"{baseline_name}/{DATASET_NAME}/{MODEL_NAME}/"
            f"{FAULT_RATIO}/repeat_{REPEAT_ID}/rank.joblib"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rank, output_path)
        print(f"{baseline_name}：已保存 {len(rank)} 个排名条目：{output_path}")


if __name__ == "__main__":
    main()
