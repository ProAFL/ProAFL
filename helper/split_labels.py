import shutil
from pathlib import Path

def split_for_yolo(dataset_name,labels_dir:Path):
    split_dir = Path(f"{ROOT_DIR}/dataset_split/{dataset_name}")
    for split in ("train", "val"):
        output_dir = labels_dir.parent / f"labels_{split}"
        output_dir.mkdir(exist_ok=True)
        for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
            label = labels_dir / f"{Path(image_name).stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            shutil.copy2(label, output_dir / label.name)
    print(f"切分的数据集保存在{labels_dir.parent}")

if __name__ == "__main__":
    ROOT_DIR = "/data/mml/data_debugging_data/ProAFL_data"
    dataset_name = "kitti" # voc|kitti|visdrone
    inject_ratio = 0.1 # 0.01,0.05,0.1,0.15
    repeat_id = 1
    method_name = "ours"
    model_name = "yolov7"    
    labels_dir = Path(f"{ROOT_DIR}/ProAFL_data/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels")
    split_for_yolo(dataset_name,labels_dir)

