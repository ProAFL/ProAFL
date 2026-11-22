'''
过程特征的重要性讨论
'''
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from utils.common import read_yaml,read_json

BASE_FEATURE_NAMES = (
    "late_conf_deficit", "all_conf_deficit", "first_conf_reach",
    "late_iou_deficit", "all_iou_deficit", "first_iou_reach",
)

CLASS_FEATURE_NAMES = BASE_FEATURE_NAMES[:3] + ("persistent_class_margin",)
GEOMETRY_FEATURE_NAMES = BASE_FEATURE_NAMES[3:] + ("persistent_iou_deficit",)

FEATURE_NAMES = CLASS_FEATURE_NAMES + GEOMETRY_FEATURE_NAMES # 8个特征名称

def group_annoids():
    anno_json = read_json(anno_with_miss_json_path)
    annos = anno_json["annotations"]
    group = {
        "clean":[],
        "fault_with_miss":[],
        "fault_no_miss":[]
    }
    for anno in annos:
        aid = int(anno["id"])
        fault_type = int(anno["fault_type"])
        if fault_type == 0:
            group["clean"].append(aid)
        elif fault_type != 0:
            group["fault_with_miss"].append(aid)
            if fault_type in [1,2,3]:
                group["fault_no_miss"].append(aid)
    return group

def anno_level_feature_importance():
    feature_json = read_json(anno_process_features_json_path)
    aid_group = group_annoids()
    fault_aids = set(aid_group["fault_no_miss"])
    
    has_both_classes = len(set(isfault)) == 2
    for feature_name in FEATURE_NAMES:
        isfault = []
        feature_values = []
        for annoid_str in feature_json.keys():
            annoid = int(annoid_str)
            if annoid in fault_aids:
                isfault.append(True)
            else:
                isfault.append(False)
            feature_values.append(feature_json[annoid_str][feature_name]) 
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

    anno_with_miss_json_path = root_dir/"fault_inject"/str(fault_ratio)/dataset_name/"coco_format"/"annotations_with_miss.json"
    main()
