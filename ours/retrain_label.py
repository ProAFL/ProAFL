'''
For retraining after repair, split the train set into train and val.
'''
from pathlib import Path
from ours.small_utils import read_yaml
import shutil

config = read_yaml("config.yaml")
exp_root_dir = Path(config["exp_data_dir"])
method_name = "datactive"  # ours|datactive
dataset_name = "visdrone"  # voc|visdrone|kitti
fault_ratio = 0.1
repeat_id = 1

split_dir = exp_root_dir / "dataset_split" / dataset_name
if method_name == "ours":
    labels_dir = (exp_root_dir / "repair" / "ours" / dataset_name / "yolov7" /
                  f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                  "yolo_format" / "labels")
elif method_name == "datactive":
    labels_dir = (exp_root_dir / "repair" / "datactive" / dataset_name /
                  f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" /
                  "yolo_format" / "labels")
else:
    raise ValueError(f"Unsupported method_name: {method_name}")

for split in ("train", "val"):
    output_dir = labels_dir.parent / f"labels_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
        label = labels_dir / f"{Path(image_name).stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        shutil.copy2(label, output_dir / label.name)

print(f"Saved retrain labels to {labels_dir.parent}")
