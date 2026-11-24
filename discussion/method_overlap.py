'''
讨论:ours和baselines的overlap
'''
import json
import joblib
from pathlib import Path
from utils.common import read_json,read_yaml

def get_missfault_imgname_set(anno_json)->set[str]:
    # 带有.后缀名
    images = anno_json["images"]
    imgId2imgname = {image["id"]:image["file_name"] for image in images}    
    annos = anno_json["annotations"]
    missfault_imgname_set = set()
    for anno in annos:
        if int(anno["fault_type"]) == 4:
            missfault_imgname_set.add(imgId2imgname[anno["image_id"]] )
    return missfault_imgname_set

def get_fault_items(anno_json) -> set:
    missfault_imgname_set = get_missfault_imgname_set(anno_json)
    annos = anno_json["annotations"]
    anno_fault_nomiss = set()
    for anno in annos:
        aid = int(anno["id"])
        fault_type = int(anno["fault_type"])
        if fault_type in [1,2,3]:
            anno_fault_nomiss.add(aid)
    fault_items = anno_fault_nomiss | missfault_imgname_set
    return fault_items

def get_ours_rank(model_name):
    rank_json_path = root_dir/"dual_branch"/"process_features"/dataset_name/model_name/f"{fault_ratio}_repeat"/f"repeat_{repeat_id}"/"merged_rank.json"
    with open(rank_json_path, encoding="utf-8") as file:
        ranking = json.load(file)["all_ranking"]
    ranked_items = []
    for item in ranking:
        if item["item_type"] == "anno":
            ranked_items.append(item["anno_id"])
        elif item["item_type"] == "image":
            ranked_items.append(item["image_name"])
        else:
            raise ValueError(f"Unsupported item_type: {item['item_type']}")
    ranked_set = set(ranked_items)
    if len(ranked_items) != len(ranked_set):
        raise ValueError("The total ranking contains duplicate items")
    return ranked_items

def get_rank_by_method(method_name,model_name=None,cut_off=None):
    rank = None
    if method_name == "datactive":
        rank_path = root_dir /"rank"/"datactive"/dataset_name/str(fault_ratio)/f"repeat_{repeat_id}"/"converted_rank.joblib"
        rank = joblib.load(rank_path)
    elif method_name == "ours":
        rank = get_ours_rank(model_name)
    elif method_name in ["objectlab","deepgini","entropy","margin","loss"]:
        rank_path = root_dir /"rank"/method_name/dataset_name/model_name/str(fault_ratio)/f"repeat_{repeat_id}"/"rank.joblib"
        rank = joblib.load(rank_path)
    if cut_off:
        cut = int(len(rank)*cut_off)
        return rank[:cut]
    return rank
    

def look_overlap_unique(tp_data:dict):
    ours_tp = tp_data["ours_tp"]
    baselines_tp = tp_data["baselines_tp"]
    baseline_combined_tp = tp_data["baseline_combined_tp"]

    ours_unique = ours_tp - baseline_combined_tp
    overlap = ours_tp & baseline_combined_tp
    baselines_combined_unique = baseline_combined_tp - ours_tp

    return ours_unique,overlap,baselines_combined_unique

def main():
    cut_off = 0.4
    datactive_rank = get_rank_by_method("datactive",cut_off=cut_off)
    model_name = "rtdetr"
    ours_rank = get_rank_by_method("ours",model_name,cut_off=cut_off)
    objectlab_rank = get_rank_by_method("objectlab",model_name,cut_off=cut_off)
    deepgini_rank = get_rank_by_method("deepgini",model_name,cut_off=cut_off)
    entropy_rank = get_rank_by_method("entropy",model_name,cut_off=cut_off)
    margin_rank = get_rank_by_method("margin",model_name,cut_off=cut_off)
    loss_rank = get_rank_by_method("loss",model_name,cut_off=cut_off)

    anno_withmiss_json = read_json(annotation_withmiss_json_path)
    fault_items = get_fault_items(anno_withmiss_json)

    ours_tp = set(ours_rank) & fault_items
    datactive_tp = set(datactive_rank) & fault_items
    objectlab_tp = set(objectlab_rank) & fault_items
    deepgini_tp = set(deepgini_rank) & fault_items  
    entropy_tp = set(entropy_rank) & fault_items
    margin_tp = set(margin_rank) & fault_items
    loss_tp = set(loss_rank) & fault_items

    baseline_combined_tp = set().union(datactive_tp, objectlab_tp, deepgini_tp, entropy_tp, margin_tp, loss_tp)
    all_method_tp = baseline_combined_tp.union(ours_tp)
    tp_data = {
        "ours_tp":ours_tp,
        "baselines_tp":{
            "datactive":datactive_tp,
            "objectlab":objectlab_tp,
            "deepgini":deepgini_tp,
            "entropy":entropy_tp,
            "margin":margin_tp,
            "loss":loss_tp
        },
        "baseline_combined_tp": baseline_combined_tp
    }

    ours_unique,overlap,baselines_combined_unique = look_overlap_unique(tp_data)
    ours_unique_percenct = len(ours_unique) / len(fault_items) * 100
    overlap_percenct = len(overlap) / len(fault_items) * 100
    baselines_combined_unique_percenct = len(baselines_combined_unique) / len(fault_items) * 100
    all_method_combine_percent = len(all_method_tp) /len(fault_items)  * 100
    print(f"数据集:{dataset_name}")
    print(f"ours unique:{len(ours_unique)}({ours_unique_percenct:.2f}%)")
    print(f"overlap:{len(overlap)}({overlap_percenct:.2f}%)")
    print(f"baselines combined unique:{len(baselines_combined_unique)}({baselines_combined_unique_percenct:.2f}%)")
    print(f"all method combine:{len(all_method_tp)}({all_method_combine_percent:.2f}%)")
    

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "voc" # voc|kitti|visdrone
    fault_ratio = 0.1
    repeat_id = 1
    
    annotation_withmiss_json_path = root_dir/f"fault_inject/{fault_ratio}/{dataset_name}/coco_format/annotations_with_miss.json"
    main()