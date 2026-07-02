
import os
import joblib
from pycocotools.coco import COCO
from ours.base_data_manager import (exp_data_root_dir,get_collected_gt_box_json_path,
                                    get_error_ann_file_path,
                                    get_annotations_with_miss_json_path)

from ours.data_organization_tools import (conver_ours_rank, conver_datactive_rank,
                                          get_all_miss_error_img_name_set,
                                          get_all_annoids_detail,
                                          get_all_img_name)
from ours.small_utils import read_json


def _analyse(ours_rank,falut_type):

    def caclu_recall(fault_set, positive_set, negative_set):
        tp = len(positive_set & fault_set)
        fn = len(set(negative_set) & fault_set)
        recall = tp / (tp+fn)
        return recall

    cut = int(len(ours_rank)*0.5)
    positive_rank = ours_rank[:cut]
    negative_rank = ours_rank[cut:]
    miss_fault_imgset = get_all_miss_error_img_name_set(anno_with_miss_error_path)
    detail = get_all_annoids_detail(anno_error_with_miss)
    if falut_type == "class_fault":
        fault_set = set(detail[falut_type])
        p_set = set(positive_rank)
        n_set = set(negative_rank)
        recall = caclu_recall(fault_set, p_set, n_set)
    elif falut_type == "loc_fault":
        fault_set = set(detail[falut_type])
        p_set = set(positive_rank)
        n_set = set(negative_rank)
        recall = caclu_recall(fault_set, p_set, n_set)
    elif falut_type == "redun_fault":
        fault_set = set(detail[falut_type])
        p_set = set(positive_rank)
        n_set = set(negative_rank)
        recall = caclu_recall(fault_set, p_set, n_set)
    elif falut_type == "miss_fault":
        fault_set = miss_fault_imgset
        p_set = set(positive_rank)
        n_set = set(negative_rank)
        recall = caclu_recall(fault_set, p_set, n_set)
    else:
        raise Exception("fault type 错误")
    return round(recall,4)

def main():
    ours_rank = joblib.load(ours_rank_path)
    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)
    for fault_type in ["class_fault","loc_fault","redun_fault","miss_fault"]:
        print(f"fault_type: {fault_type}")
        recall = _analyse(converted_ours_rank,fault_type)
        print(f"recall: {recall}")
        print()





if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_error_path = get_annotations_with_miss_json_path(dataset_name)
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split",dataset_name,"images","origin")
    anno_error_with_miss = read_json(anno_with_miss_error_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)

    ours_rank_path = os.path.join(exp_data_root_dir,"Results","ours",
                                  dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")

    main()