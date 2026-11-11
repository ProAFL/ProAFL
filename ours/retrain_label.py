'''
For retraining after repair, split the train set into train and val.
'''
from pathlib import Path
from ours.small_utils import read_yaml
import json
import shutil

config = read_yaml("config.yaml")
exp_root_dir = Path(config["exp_data_dir"])
method_name = "ours"  # ours|datactive|objectlab|deepgini|entropy|margin|loss
dataset_name = "voc"  # voc|kitti|visdrone
model_name = "frcnn" # yolov7|rtdetr|frcnn
fault_ratio = 0.1
repeat_id = 1

def split_for_yolo():
    if method_name == "ours":
        labels_dir = (exp_root_dir / "repair" / "ours" / dataset_name / model_name /
                    f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                    "yolo_format" / "labels")
    elif method_name == "datactive":
        labels_dir = (exp_root_dir / "repair" / "datactive" / dataset_name /
                    f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                    "yolo_format" / "labels")
    elif method_name in ["objectlab","deepgini","entropy","margin","loss"]:
        labels_dir = (exp_root_dir / "repair" / method_name / dataset_name / model_name /
                    f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                    "yolo_format" / "labels")
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")

    split_dir = exp_root_dir / "dataset_split" / dataset_name
    for split in ("train", "val"):
        output_dir = labels_dir.parent / f"labels_{split}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
            label = labels_dir / f"{Path(image_name).stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            shutil.copy2(label, output_dir / label.name)

    print(f"Saved retrain labels to {labels_dir.parent}")

def split_for_coco():
    """按已有的 train/val 图片清单切分 COCO 标注，保留原始修复标注文件。"""
    # datactive 的目录不包含模型名，其余方法按方法名和模型名定位标注。
    if method_name in ["ours","objectlab","deepgini","entropy","margin","loss"] :
        repair_label_file = (exp_root_dir / "repair" / method_name / dataset_name / model_name /
                    f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                    "annotations.coco_repair.json")
    elif method_name == "datactive":
        repair_label_file = (exp_root_dir / "repair" / "datactive" / dataset_name /
                    f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" / "annotations.coco_repair.json")
    else:
        raise ValueError(f"Unsupported method_name: {method_name}")

    # 原始文件仅以只读方式打开；后续结果写入独立的 train/val 文件。
    with repair_label_file.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    split_dir = exp_root_dir / "dataset_split" / dataset_name
    # 使用不带路径和扩展名的图片名匹配，与 split_for_yolo 的规则一致。
    # 同名图片会导致匹配歧义，因此提前报错。
    images_by_stem = {}
    for image in coco_data["images"]:
        stem = Path(image["file_name"]).stem
        if stem in images_by_stem:
            raise ValueError(f"Duplicate image stem in {repair_label_file}: {stem}")
        images_by_stem[stem] = image

    split_data = {}
    for split in ("train", "val"):
        # 复用已有划分，不重新随机分配；忽略空行，并用集合去除重复项。
        image_stems = {
            Path(image_name.strip()).stem
            for image_name in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if image_name.strip()
        }
        # 清单中的图片必须存在于原始 COCO 数据中，避免静默遗漏样本。
        missing = image_stems - images_by_stem.keys()
        if missing:
            raise ValueError(f"Images missing from {repair_label_file} for {split}: {sorted(missing)}")

        # 保留原有图片顺序及 ID，通过 image_id 筛选对应标注。
        # 没有标注的图片也会保留在 images 中。
        images = [image for image in coco_data["images"] if Path(image["file_name"]).stem in image_stems]
        image_ids = {image["id"] for image in images}
        # 浅拷贝顶层字段，仅替换图片和标注列表，保留 categories、info 等元数据。
        split_data[split] = {
            **coco_data,
            "images": images,
            "annotations": [
                annotation for annotation in coco_data["annotations"]
                if annotation["image_id"] in image_ids
            ],
        }

    # 写文件前检查训练集和验证集是否重叠，防止同一图片进入两个集合。
    train_ids = {image["id"] for image in split_data["train"]["images"]}
    val_ids = {image["id"] for image in split_data["val"]["images"]}
    if train_ids & val_ids:
        raise ValueError("Train and val splits contain overlapping images")

    # 两个划分都通过检查后再输出；再次执行会覆盖这些输出文件，原始文件不受影响。
    for split, data in split_data.items():
        output_label = repair_label_file.parent / f"annotations_{split}.json"
        with output_label.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if model_name in ["yolov7", "rtdetr"]:
        split_for_yolo()
    else:
        # model_name==frcnn
        split_for_coco()

