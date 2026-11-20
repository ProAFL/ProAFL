'''
过程特征的重要性讨论
'''
from pathlib import Path
from utils.common import read_yaml,read_json

def img_level_feature_importence():
    pass

def anno_level_feature_importance():
   feature_json = read_json(anno_process_features_json_path)
   pass

def main():
    anno_level_feature_importance()
    pass


if __name__ == "__main__":

    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "voc"
    model_name = "yolov7"
    fault_ratio = 0.1
    repeat_id = 1


    anno_process_features_json_path = (
        root_dir / "dual_branch" / "process_features" / dataset_name / model_name
        / f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" / "process_features.json"
    )

    main() 