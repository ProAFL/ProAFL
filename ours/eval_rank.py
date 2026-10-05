import json
from pathlib import Path
from ours.rank.analyse.common import compute_apfd, calc_fpr_fnr_f1, calc_top1,draw_rank_hot
from helper.data_organization_tools import get_all_explicit_fault_annoids,get_all_miss_error_img_name_set,get_all_error_imgset


data_root = Path("/data/mml/data_debugging_data/ProAFL_data")
rank_json_path = data_root / "process_features" / "voc" / "yolov7/0.1_repeat/repeat_1_1791094449/merged_rank.json"
anno_with_miss_fault_json_path = data_root / "fault_inject" / "0.1" / "voc" / "coco_format" / "annotations_with_miss.json"
rank_hot_path = data_root / "process_features" / "voc" / "yolov7/0.1_repeat/repeat_1_1791094449" / "rank_hot.png"
def main():
    """
    评估排名结果
    """
    with open(anno_with_miss_fault_json_path, encoding="utf-8") as file:
        anno_with_miss_json = json.load(file)
    all_explicit_fault_annoids = get_all_explicit_fault_annoids(anno_with_miss_json)
    print(f"Explicit Fault的anno数量有:{len(all_explicit_fault_annoids)}")
    missfault_imgset = get_all_miss_error_img_name_set(anno_with_miss_json)
    print(f"Miss Fault的图像数量有:{len(missfault_imgset)}")
    fault_set = set[int](all_explicit_fault_annoids) | set[str](missfault_imgset)
    print(f"Fault的元素数量有:{len(fault_set)}")

    fault_imgset = get_all_error_imgset(anno_with_miss_json)
    print(f"含有任何Fault的图像数量有:{len(fault_imgset)}")

    with open(rank_json_path, encoding="utf-8") as file:
        rank_json = json.load(file)
    ranking = rank_json["all_ranking"]
    rank_list = []
    for item in ranking:
        if item["item_type"] == "anno":
            rank_list.append(item["anno_id"])
        elif item["item_type"] == "image":
            rank_list.append(item["image_name"])
        else:
            raise ValueError(f"item item_type {item['item_type']} not supported")
    print(f"排序长度有:{len(rank_list)}")
    apfd = compute_apfd(fault_set, rank_list)
    fpr,fnr,f1 = calc_fpr_fnr_f1(rank_list,fault_set,cut_off=0.4)
    top1 = calc_top1(anno_with_miss_json,rank_list,fault_set,fault_imgset)
    print(f"APFD:{apfd}")
    print(f"FPR:{fpr}")
    print(f"FNR:{fnr}")
    print(f"F1:{f1}")
    print(f"Top1:{top1}")
    draw_rank_hot([ID in fault_set for ID in rank_list],rank_hot_path)
    print(f"排序热图结果已保存到:{rank_hot_path}")

if __name__ == "__main__":
    main()