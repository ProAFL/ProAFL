from pathlib import Path
import shutil


dataset_name = "voc"
inject_ratio = 0.1 # 0.01,0.05,0.1,0.15
repeat_id = 10
method_name = "ours"
model_name = "yolov7"
split_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/dataset_split/{dataset_name}")
# labels_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/fault_inject/{inject_ratio}/{dataset_name}/yolo_fomat/labels")
labels_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/corrected_anno/{method_name}/{dataset_name}/{model_name}/{inject_ratio}_repeat/repeat_{repeat_id}/yolo_format/labels")
for split in ("train", "val"):
    output_dir = labels_dir.parent / f"labels_{split}"
    output_dir.mkdir(exist_ok=True)

    for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
        label = labels_dir / f"{Path(image_name).stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        shutil.copy2(label, output_dir / label.name)
