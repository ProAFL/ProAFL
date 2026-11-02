"""ObjectLab 标注排序；从仓库根目录运行 python -m baselines_new.objectLab。

使用 epoch 49 的预测框。
公式参考 objectlab.pdf 第 3–4 页算法 1–4；适配说明见 objectLab.md。
anno 使用定位、类别质量的几何平均；image 使用漏标质量，混合升序排列。
打分使用全部有效预测框，最终仅保存 anno ID 和图片名组成的列表。
"""

import math
from pathlib import Path

import joblib
import numpy as np

from ours.dual_branch.common import read_json
from ours.dual_branch.process_metrics import load_annotations
from ours.small_utils import read_yaml


# 切换实验时修改这里；路径随这些配置自动生成，也可以直接修改下方路径。
DATA_ROOT = Path(read_yaml("config.yaml")["exp_data_dir"])
DATASET_NAME = "visdrone"  # voc | kitti | visdrone
MODEL_NAME = "frcnn"      # yolov7 | rtdetr | frcnn 
FAULT_RATIO = 0.1
REPEAT_ID = 1
EPOCH = 49
LOW_CONF_THRESHOLD = 0.25    # 有效预测须 conf > 0.25
HIGH_CONF_THRESHOLD = 0.5  # 类别、漏标证据须 conf > 0.5
SIM_ALPHA = 0.1            # sim = alpha * kernel + (1-alpha) * IoU
SIM_SIGMA = 0.1            # kernel = exp(-归一化 xyxy 欧氏距离 / sigma)

ANNOTATIONS_PATH = DATA_ROOT / (
    f"fault_inject/{FAULT_RATIO}/{DATASET_NAME}/coco_format/annotations_no_miss.json"
)
PREDICTIONS_DIR = DATA_ROOT / (
    f"collection_process_info/{DATASET_NAME}/{MODEL_NAME}/collected_predict_boxes/"
    f"inject_{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}"
)
PREDICTIONS_PATH = PREDICTIONS_DIR / f"epoch_{EPOCH}_predicted_bboxs.json"
OUTPUT_PATH = DATA_ROOT / (
    f"rank/objectlab/{DATASET_NAME}/{MODEL_NAME}/"
    f"{FAULT_RATIO}/repeat_{REPEAT_ID}/rank.joblib"
)

def box_matrices(annos, predictions, image_size):
    """返回 anno × prediction 的 IoU 和论文相似度矩阵；输入为像素 xyxy。"""
    width, height = image_size
    if not all(math.isfinite(value) and value > 0 for value in image_size):
        raise ValueError(f"图片宽高不合法：{image_size}")
    labels = np.asarray([box for _, _, box in annos], dtype=float).reshape(-1, 4)
    boxes = np.asarray([pred["bbox"] for pred in predictions], dtype=float).reshape(-1, 4)
    if not np.all(np.isfinite(labels)) or not np.all(np.isfinite(boxes)):
        raise ValueError("框坐标必须为有限数值")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("预测框宽高必须大于 0")
    overlap = np.maximum(0.0, np.minimum(labels[:, None, 2:], boxes[None, :, 2:])
                         - np.maximum(labels[:, None, :2], boxes[None, :, :2]))
    intersection = np.prod(overlap, axis=2)
    label_area = np.prod(labels[:, 2:] - labels[:, :2], axis=1)
    pred_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    union = label_area[:, None] + pred_area[None, :] - intersection
    ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    scale = np.asarray([width, height, width, height], dtype=float)
    distance = np.linalg.norm((labels[:, None, :] - boxes[None, :, :]) / scale, axis=2)
    similarities = SIM_ALPHA * np.exp(-distance / SIM_SIGMA) + (1.0 - SIM_ALPHA) * ious
    return ious, np.clip(similarities, 0.0, 1.0)


def objectlab_score(category, predictions, ious, similarities):
    """论文算法 2、3；anno 几何平均是本任务适配，越小越可疑。"""
    same_class = np.asarray([pred["predicted_cls"] == category for pred in predictions], dtype=bool)
    high_conf = np.asarray([pred["conf"] > HIGH_CONF_THRESHOLD for pred in predictions], dtype=bool)
    location_candidates = similarities[same_class & (ious > 0)]
    swap_candidates = similarities[~same_class & high_conf]
    badloc_score = float(location_candidates.max()) if location_candidates.size else 1.0
    # 0.95 是预测置信度阈值，不是 IoU 阈值。
    badcls_score = 1.0 - float(swap_candidates.max()) if swap_candidates.size else 1.0
    return math.sqrt(badloc_score * badcls_score)


def softmin(scores):
    """论文第 3 页：<q, softmax(1-q)>；无证据时质量为 1。"""
    scores = np.asarray(scores, dtype=float)
    if not scores.size:
        return 1.0
    logits = 1.0 - scores
    exp_logits = np.exp(logits - logits.max())
    weights = exp_logits / exp_logits.sum()
    return float(np.sum(weights * scores))


def image_miss_score(annos, predictions, ious, similarities, min_similarity):
    """算法 4 加作者实现的重叠保护：仅检查与任意标注均不重叠的预测。

    有同类标注时取最大同类 sim；否则取 sim* * (1-conf)。
    零 IoU 仍有距离核相似度，不再导致分数恒为零。
    """
    categories = np.asarray([category for _, category, _ in annos])
    scores = []
    for index, prediction in enumerate(predictions):
        if prediction["conf"] <= HIGH_CONF_THRESHOLD or np.any(ious[:, index] > 0):
            continue
        same_class = similarities[categories == prediction["predicted_cls"], index]
        quality = (float(same_class.max()) if same_class.size
                   else min_similarity * (1.0 - prediction["conf"]))
        scores.append(quality)
    return softmin(scores)


def build_ranking(annotation_path, prediction_path):
    """按质量分混合升序排序，直接返回 anno ID 和图片名列表。"""
    if not 0 <= LOW_CONF_THRESHOLD <= HIGH_CONF_THRESHOLD <= 1:
        raise ValueError("阈值须满足 0 <= low <= high <= 1")
    if not 0 <= SIM_ALPHA <= 1 or not math.isfinite(SIM_SIGMA) or SIM_SIGMA <= 0:
        raise ValueError("相似度参数须满足 0 <= alpha <= 1、sigma > 0")
    annotations = read_json(annotation_path)
    by_image, _ = load_annotations(annotation_path)
    # load_annotations 内部使用字符串 ID；输出恢复 COCO 原始 ID 类型。
    original_ids = {str(anno["id"]): anno["id"] for anno in annotations["annotations"]}
    raw_predictions = read_json(prediction_path)
    image_info = {image["file_name"]: image for image in annotations["images"]}
    # 旧版依次插入每张图的标注、图片；稳定排序保持同分条目的插入顺序。
    # 额外覆盖 COCO images 中没有剩余标注的图片，避免遗漏漏标检查条目。
    image_names = list(by_image) + [
        image["file_name"] for image in annotations["images"]
        if image["file_name"] not in by_image
    ]
    predictions_by_image = {}
    for name in image_names:
        predictions = raw_predictions.get(name, {}).get("predicted_bboxs", [])
        if not isinstance(predictions, list):
            raise ValueError(f"预测框列表不合法：{name}")
        for prediction in predictions:
            conf = prediction["conf"]
            if (isinstance(conf, bool) or not isinstance(conf, (int, float))
                    or not math.isfinite(conf) or not 0 <= conf <= 1):
                raise ValueError(f"预测置信度不合法：{name}")
        predictions_by_image[name] = [pred for pred in predictions if pred["conf"] > LOW_CONF_THRESHOLD]

    # 论文 sim*：整个数据集中同图 anno/prediction 配对的最小相似度。
    # 两遍计算，避免在内存中保留全部配对矩阵。
    min_similarity = 1.0
    for name in image_names:
        info = image_info[name]
        _, similarities = box_matrices(by_image.get(name, []), predictions_by_image[name],
                                       (info["width"], info["height"]))
        if similarities.size:
            min_similarity = min(min_similarity, float(similarities.min()))
    min_similarity = max(min_similarity, np.finfo(float).tiny)

    rows = []
    for name in image_names:
        annos = by_image.get(name, [])
        predictions = predictions_by_image[name]
        info = image_info[name]
        ious, similarities = box_matrices(annos, predictions, (info["width"], info["height"]))
        for index, (annotation_id, category, _) in enumerate(annos):
            score = objectlab_score(category, predictions, ious[index], similarities[index])
            rows.append((original_ids[annotation_id], score))
        score = image_miss_score(annos, predictions, ious, similarities, min_similarity)
        rows.append((name, score))

    # 仅按分数排序，同分保持原插入顺序。
    return [item for item, _ in sorted(rows, key=lambda row: row[1])]


def main():
    rank = build_ranking(ANNOTATIONS_PATH, PREDICTIONS_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rank, OUTPUT_PATH)
    print(f"预测文件：{PREDICTIONS_PATH}")
    print(f"已保存 {len(rank)} 个排名条目：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
