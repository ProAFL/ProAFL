"""Evaluate the combined annotation/image ranking against injected faults."""

import json
import joblib
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
RANK_DIR = DATA_ROOT / "rank/datactive/voc/0.1_repeat/repeat_1"
RANK_PATH = RANK_DIR / "converted_rank.joblib"
ANNOTATIONS_PATH = DATA_ROOT / "fault_inject/0.1/voc/coco_format/annotations_with_miss.json"
RANK_HOT_PATH = RANK_DIR / "rank_hot.png"
CUTOFF = 0.4


def main():
    with open(ANNOTATIONS_PATH, encoding="utf-8") as file:
        annotations = json.load(file)
    ranking = joblib.load(RANK_PATH)

    explicit_faults = set(get_all_explicit_fault_annoids(annotations))
    miss_fault_images = get_all_miss_error_img_name_set(annotations)
    fault_items = explicit_faults | miss_fault_images
    fault_images = get_all_error_imgset(annotations)


    ranked_set = set(ranking)
    missing_faults = fault_items - ranked_set
    print(f"The total ranking omits {len(missing_faults)} fault items")

    apfd = compute_apfd(fault_items, ranking)
    fpr, fnr, f1 = calc_fpr_fnr_f1(ranking, fault_items, cut_off=CUTOFF)
    top1 = calc_top1(annotations, ranking, fault_items, fault_images)
    exam = calc_exam(annotations, ranking)

    print(f"Explicit fault annotations: {len(explicit_faults)}")
    print(f"Images with missing annotations: {len(miss_fault_images)}")
    print(f"Images with any fault: {len(fault_images)}")
    print(f"Ranking length: {len(ranking)}")
    print(f"APFD: {apfd}")
    print(f"FPR: {fpr}")
    print(f"FNR: {fnr}")
    print(f"F1: {f1}")
    print(f"Top1: {top1}")
    print(f"Exam: {exam}")

    draw_rank_hot([item in fault_items for item in ranking], RANK_HOT_PATH)
    print(f"Rank heatmap saved to: {RANK_HOT_PATH}")


if __name__ == "__main__":
    main()
