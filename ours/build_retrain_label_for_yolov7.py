'''
For retraining after repair, split the train set into train and val.
'''
import os
from pathlib import Path
from ours.small_utils import read_yaml
import shutil

config = read_yaml("config.yaml")
exp_root_dir = config["exp_data_dir"]
method_name = "ours" # ours|datactive
dataset_name = "visdrone" # voc|visdrone|kitti
fault_ratio = 0.1
repeat_id = 1
split_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/dataset_split/{dataset_name}")
if method_name == "ours":
    labels_dir = Path(os.path.join(exp_root_dir,"repair","ours",dataset_name,"yolov7",f"{fault_ratio}_repeat",f"repeat_{repeat_id}","yolo_format","labels"))
if method_name == "datactive":
    labels_dir = Path(os.path.join(exp_root_dir,"repair","datactive",f"{fault_ratio}_repeat",f"repeat_{repeat_id}","yolo_format","labels"))
for split in ("train", "val"):
    output_dir = labels_dir.parent / f"labels_{split}"
    output_dir.mkdir(exist_ok=True)
    for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
        label = labels_dir / f"{Path(image_name).stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        shutil.copy2(label, output_dir / label.name)