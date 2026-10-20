"""Build annotation confidence/IoU trajectories from saved post-NMS boxes.

For each annotation and epoch, choose the overlapping prediction with the
largest IoU, regardless of its predicted class. ``conf_list`` contains that
prediction's objectness-weighted score for the *annotation* class, rather
than the prediction's own ``conf``. An epoch without an overlapping box has
zero support and is identified by ``has_overlap_list``. Images absent from
an epoch's prediction records are treated as having no predicted boxes.

Edit the settings below, then run ``python ours/process_metrics.py``.
"""

import json
from collections import defaultdict
from pathlib import Path
from ours.small_utils import read_yaml

# Settings for the current VOC / YOLOv7 experiment. Edit these before running.
config = read_yaml("config.yaml")                           
exp_data_root_dir = config["exp_data_dir"]
DATA_ROOT = Path(exp_data_root_dir)
DATASETNAME = "voc" # voc|kitti|visdrone
MODEL_NAME = "rtdetr" # yolov7|rtdetr|frcnn
ANNOTATIONS_PATH = DATA_ROOT / f"fault_inject/0.1/{DATASETNAME}/coco_format/annotations_no_miss.json"
EXPID = 1792398239 # 其实是个时间戳 1792056154(kitti), 1791968956(visdrone)，1792398239(redetr-voc)
PREDICTIONS_DIR = DATA_ROOT / (
    f"collection_process_info/{DATASETNAME}/{MODEL_NAME}/collected_predict_boxes/"
    f"inject_0.1_repeat/repeat_1_{EXPID}"
)
OUTPUT_PATH = DATA_ROOT / f"process_metrics/{DATASETNAME}/{MODEL_NAME}/0.1_repeat/repeat_1_{EXPID}/process_metrics.json"
EPOCHS = 100 # redetr-voc 100 epochs,其他50


def box_iou(box_a, box_b):
    """IoU of two xyxy boxes; return zero for non-overlap or invalid boxes."""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def load_annotations(annotation_path):
    with open(annotation_path, encoding="utf-8") as file:
        data = json.load(file)

    image_names = {image["id"]: image["file_name"] for image in data["images"]}
    if len(image_names) != len(data["images"]) or len(set(image_names.values())) != len(image_names):
        raise ValueError("Annotation images must have unique IDs and file names")

    annotations_by_image = defaultdict(list)
    metrics = {}
    for annotation in data["annotations"]:
        annotation_id = str(annotation["id"])
        if annotation_id in metrics:
            raise ValueError(f"Duplicate annotation ID: {annotation_id}")
        image_name = image_names[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid annotation bbox: {annotation_id}")
        category_id = annotation["category_id"]
        if not isinstance(category_id, int) or category_id < 0:
            raise ValueError(f"Invalid annotation category: {annotation_id}")
        annotations_by_image[image_name].append(
            (annotation_id, category_id, (x, y, x + width, y + height))
        )
        metrics[annotation_id] = {
            "conf_list": [],
            "iou_list": [],
            "has_overlap_list": [],
            "selected_predicted_box_id_list": [],
            "selected_predicted_cls_list": [],
        }

    return set(image_names.values()), annotations_by_image, metrics


def choose_prediction(annotation_box, predictions):
    """Return (prediction, IoU), prioritizing geometry over class score."""
    best_prediction = None
    best_iou = 0.0
    best_conf = -1.0
    best_id = float("inf")
    for prediction in predictions:
        iou = box_iou(annotation_box, prediction["bbox"])
        if iou <= 0:
            continue
        conf = prediction["conf"]
        prediction_id = prediction["predicted_box_id"]
        if (iou > best_iou or
                (iou == best_iou and (conf > best_conf or
                                      (conf == best_conf and prediction_id < best_id)))):
            best_prediction = prediction
            best_iou = iou
            best_conf = conf
            best_id = prediction_id
    return best_prediction, best_iou


def collect_process_metrics(annotation_path, predictions_dir:Path, epochs=50):
    """Collect trajectories for every annotation, including never-overlapped ones."""
    # metrics[annotation_id] = {
    #         "conf_list": [],
    #         "iou_list": [],
    #         "has_overlap_list": [],
    #         "selected_predicted_box_id_list": [],
    #         "selected_predicted_cls_list": [],
    # }
    _, annotations_by_image, metrics = load_annotations(annotation_path)
    for epoch in range(epochs):
        print(f"Epoch:{epoch}")
        prediction_path = predictions_dir / f"epoch_{epoch}_predicted_bboxs.json"
        with open(prediction_path, encoding="utf-8") as file:
            predictions_by_image = json.load(file)
        for image_name, annotations in annotations_by_image.items():
            # The collector may omit images with no post-NMS predictions.
            predictions = (predictions_by_image[image_name]["predicted_bboxs"]
                           if image_name in predictions_by_image else [])
            if predictions is None:
                raise ValueError(f"Epoch {epoch}, {image_name}: prediction list is null")
            if predictions == []:
                print(f"Epoch {epoch}, {image_name}: no post-NMS predictions")
            for annotation_id, category_id, annotation_box in annotations: # category_id 是这个anno的类别id
                prediction, iou = choose_prediction(annotation_box, predictions) # 这张图下的标注框和预测框
                result = metrics[annotation_id]
                if prediction is None:
                    class_conf = 0.0
                    prediction_id = None
                    predicted_cls = None
                else:
                    probs = prediction.get("probs")
                    if not isinstance(probs, list) or category_id >= len(probs):
                        raise ValueError(
                            f"Epoch {epoch}, {image_name}, anno {annotation_id}: "
                            f"missing score for category {category_id}"
                        )
                    class_conf = probs[category_id]
                    prediction_id = prediction["predicted_box_id"]
                    predicted_cls = prediction["predicted_cls"]
                result["conf_list"].append(class_conf)
                result["iou_list"].append(iou)
                result["has_overlap_list"].append(prediction is not None)
                result["selected_predicted_box_id_list"].append(prediction_id)
                result["selected_predicted_cls_list"].append(predicted_cls)

    return metrics


def main():
    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    metrics = collect_process_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, separators=(",", ":"))

    never_overlapped = sum(not any(item["has_overlap_list"]) for item in metrics.values())
    print(f"Saved {len(metrics)} annotations × {EPOCHS} epochs to {OUTPUT_PATH}")
    print(f"Annotations without any overlapping post-NMS box: {never_overlapped}")

if __name__ == "__main__":
    main()
