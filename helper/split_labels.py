from pathlib import Path
import shutil


dataset_name = "voc"
inject_ratio = 0.01

split_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/dataset_split/{dataset_name}")
labels_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/fault_inject/{inject_ratio}/{dataset_name}/yolo_fomat/labels")

for split in ("train", "val"):
    output_dir = labels_dir.parent / f"labels_{split}"
    output_dir.mkdir(exist_ok=True)

    for image_name in (split_dir / f"{split}.txt").read_text().splitlines():
        label = labels_dir / f"{Path(image_name).stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        shutil.copy2(label, output_dir / label.name)
