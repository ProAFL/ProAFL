"""收集标注框的类别得分、IoU 轨迹；从仓库根目录用 -m 方式启动。"""

from collections import defaultdict

from ours.dual_branch.common import read_json, save_json
from ours.dual_branch.config import ANNOTATIONS_PATH, PREDICTIONS_DIR, METRICS_DIR, EPOCHS


OUTPUT_PATH = METRICS_DIR / "process_metrics.json"


def box_iou(box_a, box_b):
    """输入均为 xyxy；无交集或退化框返回 0。"""
    width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def load_annotations(path):
    data = read_json(path)
    names = {image["id"]: image["file_name"] for image in data["images"]}
    if len(names) != len(data["images"]) or len(set(names.values())) != len(names):
        raise ValueError("图片 ID 和文件名必须唯一")
    by_image = defaultdict(list)
    metrics = {} 
    for annotation in data["annotations"]:
        annotation_id = str(annotation["id"])
        if annotation_id in metrics:
            raise ValueError(f"重复标注 ID：{annotation_id}")
        x, y, width, height = annotation["bbox"]
        category = annotation["category_id"]
        if width <= 0 or height <= 0 or type(category) is not int or category < 0:
            raise ValueError(f"标注框或类别不合法：{annotation_id}")
        by_image[names[annotation["image_id"]]].append(
            (annotation_id, category, [x, y, x + width, y + height])
        )
        metrics[annotation_id] = {key: [] for key in (
            "conf_list", "iou_list", "has_overlap_list",
            "selected_predicted_box_id_list", "selected_predicted_cls_list",
        )}
    return by_image, metrics


def choose_prediction(annotation_box, predictions):
    """不限制类别，优先选最大 IoU；同 IoU 再比较置信度、预测框 ID。"""
    best, best_key = None, (0.0, -1.0, float("-inf"))
    for prediction in predictions:
        iou = box_iou(annotation_box, prediction["bbox"])
        key = (iou, prediction["conf"], -prediction["predicted_box_id"])
        if iou > 0 and key > best_key:
            best, best_key = prediction, key
    return best, best_key[0]


def collect_process_metrics(annotation_path, predictions_dir, epochs):
    if epochs <= 0:
        raise ValueError("训练轮数必须大于 0")
    by_image, metrics = load_annotations(annotation_path)
    for epoch in range(epochs):
        print(f"收集标注轨迹：epoch {epoch}")
        images = read_json(predictions_dir / f"epoch_{epoch}_predicted_bboxs.json")
        for name, annotations in by_image.items():
            predictions = images.get(name, {}).get("predicted_bboxs", [])
            if not isinstance(predictions, list):
                raise ValueError(f"预测框列表不合法：epoch {epoch}, {name}")
            for annotation_id, category, box in annotations:
                prediction, iou = choose_prediction(box, predictions)
                conf, prediction_id, predicted_class = 0.0, None, None
                if prediction is not None:
                    probs = prediction.get("probs")
                    if not isinstance(probs, list) or category >= len(probs):
                        raise ValueError(f"缺少标注类别得分：epoch {epoch}, anno {annotation_id}")
                    # 使用标注类别的得分，不使用预测框自身的最高类别置信度。
                    conf = probs[category]
                    prediction_id = prediction["predicted_box_id"]
                    predicted_class = prediction["predicted_cls"]
                row = metrics[annotation_id]
                row["conf_list"].append(conf)
                row["iou_list"].append(iou)
                row["has_overlap_list"].append(prediction is not None)
                row["selected_predicted_box_id_list"].append(prediction_id)
                row["selected_predicted_cls_list"].append(predicted_class)
    return metrics


def main():
    metrics = collect_process_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    save_json(OUTPUT_PATH, metrics)
    print(f"已保存 {len(metrics)} 个标注的过程度量：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
