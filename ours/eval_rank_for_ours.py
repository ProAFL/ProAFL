"""Evaluate the combined annotation/image ranking against injected faults."""

import json
from pathlib import Path

from helper.data_organization_tools import (
    get_all_error_imgset,
    get_all_explicit_fault_annoids,
    get_all_miss_error_img_name_set,
)
from ours.rank.analyse.common import (
    calc_fpr_fnr_f1,
    calc_top1,
    compute_apfd,
    calc_exam,
    draw_rank_hot,
)


DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
DATASETNAME = "visdrone" # voc|kitti|visdrone 
MODEL_NAME = "frcnn" # yolov7|rtdetr|frcnn|fcos
# yolov7-voc: 1791094449
# yolov7-kitti: 1792056154
# yolov7-visdrone: 1791968956
# rtdetr-voc: 1792398239
# rtdetr-visdrone: 1792504350
# rtdetr-kitti: 1792556872
# frcnn-voc: 1792570209
# frcnn-kitti: 1792642708
# frcnn-visdrone: 1792933936 # 1792748834,1792917048(500per),1792933936
# fcos-visdrone: 1793025492
EXPID = 1792917048
RANK_DIR = DATA_ROOT / "dual_branch" / f"process_features/{DATASETNAME}/{MODEL_NAME}/0.1_repeat/repeat_1_{EXPID}"
RANK_PATH = RANK_DIR / "merged_rank.json"
ANNOTATIONS_PATH = DATA_ROOT / f"fault_inject/0.1/{DATASETNAME}/coco_format/annotations_with_miss.json"
RANK_HOT_PATH = RANK_DIR / "rank_hot.png"
CUTOFF = 0.4

def main():
    with open(ANNOTATIONS_PATH, encoding="utf-8") as file:
        annotations = json.load(file)
    with open(RANK_PATH, encoding="utf-8") as file:
        ranking = json.load(file)["all_ranking"]

    explicit_faults = set(get_all_explicit_fault_annoids(annotations))
    miss_fault_images = get_all_miss_error_img_name_set(annotations)
    fault_items = explicit_faults | miss_fault_images
    fault_images = get_all_error_imgset(annotations)

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
    missing_faults = fault_items - ranked_set
    if missing_faults:
        raise ValueError(f"The total ranking omits {len(missing_faults)} fault items")

    apfd = compute_apfd(fault_items, ranked_items)
    fpr, fnr, f1 = calc_fpr_fnr_f1(ranked_items, fault_items, cut_off=CUTOFF)
    top1 = calc_top1(annotations, ranked_items, fault_items, fault_images)
    exam = calc_exam(annotations, ranked_items)

    print(f"Explicit fault annotations: {len(explicit_faults)}")
    print(f"Images with missing annotations: {len(miss_fault_images)}")
    print(f"Images with any fault: {len(fault_images)}")
    print(f"Ranking length: {len(ranked_items)}")
    print(f"APFD: {apfd}")
    print(f"FPR: {fpr}")
    print(f"FNR: {fnr}")
    print(f"F1: {f1}")
    print(f"Top1: {top1}")
    print(f"Exam: {exam}")
    draw_rank_hot([item in fault_items for item in ranked_items], RANK_HOT_PATH)
    print(f"Rank heatmap saved to: {RANK_HOT_PATH}")


if __name__ == "__main__":
    main()
