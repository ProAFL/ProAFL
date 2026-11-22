'''
过程特征的重要性讨论
'''
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from utils.common import read_yaml,read_json

FEATURE_NAMES = (
    "all_presence_rate", "late_presence_rate", "all_confidence_evidence",
    "late_confidence_evidence", "longest_consecutive_presence_rate", "position_stability",
)


def get_missfault_imgname_set()->set[str]:
    # 带有.后缀名
    anno_json = read_json(anno_with_miss_json_path)
    images = anno_json["images"]
    imgId2imgname = {image["id"]:image["file_name"] for image in images}    
    annos = anno_json["annotations"]
    missfault_imgname_set = set()
    for anno in annos:
        if int(anno["fault_type"]) == 4:
            missfault_imgname_set.add(imgId2imgname[anno["image_id"]] )
    return missfault_imgname_set

    

def img_level_feature_importance():
    feature_json = read_json(img_process_features_json_path)
    fault_imgname_set = get_missfault_imgname_set()

    for feature_name in FEATURE_NAMES:
        isfault = []
        feature_values = []
        for imgname in feature_json["images"].keys():
            if imgname in fault_imgname_set:
                isfault.append(True)
            else:
                isfault.append(False)
            feature_values.append(feature_json["images"][imgname]["features"][feature_name])
        has_both_classes = len(set(isfault)) == 2
        if has_both_classes is False:
            raise Exception("必须是二类诊断问题")
        res = spearmanr(feature_values, isfault)
        auc = roc_auc_score(isfault, feature_values) if has_both_classes else float("nan")
        print(
            f"feature_name:{feature_name}|stat:{res.statistic:.4f}|pvalue:{res.pvalue:.4f}"
            f"|roc_auc:{auc:.6f}"
        )

def main():
    img_level_feature_importance()
    pass


if __name__ == "__main__":

    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "voc"
    model_name = "yolov7"
    fault_ratio = 0.1
    repeat_id = 1


    img_process_features_json_path = (
        root_dir / "dual_branch" / "process_features" / dataset_name / model_name
        / f"{fault_ratio}_repeat" / f"repeat_{repeat_id}" / "miss_process_features.json"
    )

    anno_with_miss_json_path = root_dir/"fault_inject"/str(fault_ratio)/dataset_name/"coco_format"/"annotations_with_miss.json"
    main()
