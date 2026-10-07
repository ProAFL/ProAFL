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
RANK_DIR = DATA_ROOT / "process_features/voc/yolov7/0.1_repeat/repeat_1_1791094449"
RANK_PATH = RANK_DIR / "merged_rank.json"
ANNOTATIONS_PATH = DATA_ROOT / "fault_inject/0.1/voc/coco_format/annotations_with_miss.json"
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
