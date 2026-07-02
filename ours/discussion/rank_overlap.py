
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




def caclu_setiou(list_1,list_2,fault_type):
    attention_set = set()
    if fault_type == "miss_fault":
        miss_fault_imgset = get_all_miss_error_img_name_set(anno_with_miss_error_path)
        attention_set = set(miss_fault_imgset)
    elif fault_type in ["class_fault","loc_fault","redun_fault"]:
        detail = get_all_annoids_detail(anno_error_with_miss)
        attention_set = set(detail[fault_type])
    elif fault_type == "clean":
        all_imgname_list = get_all_img_name(imgs_dir)
        miss_fault_imgset = get_all_miss_error_img_name_set(anno_with_miss_error_path)
        no_miss_fault_imgset = set(all_imgname_list) - miss_fault_imgset
        detail = get_all_annoids_detail(anno_error_with_miss)
        attention_set = set(detail[fault_type]) | set(no_miss_fault_imgset)
    else:
        raise Exception("fault type 错误")

    set_1 = set(list_1)
    set_2 = set(list_2)
    attention_join = set_1 & set_2 & attention_set
    join = set_1 & set_2
    iou = round(len(attention_join) / len(join),4)
    return iou

def overlap_analyse(ours_rank, datactive_rank, entropy_rank, loss_rank, deepgini_rank, margin_rank, falut_type):
    cut = int(len(ours_rank)*0.5)
    ours_rank = ours_rank[:cut]
    datactive_rank = datactive_rank[:cut]
    entropy_rank = entropy_rank[:cut]
    loss_rank = loss_rank[:cut]
    deepgini_rank = deepgini_rank[:cut]
    margin_rank = margin_rank[:cut]

    ours_datactive_iou = caclu_setiou(ours_rank,datactive_rank,falut_type)
    ours_entropy_iou = caclu_setiou(ours_rank,entropy_rank,falut_type)
    ours_loss_iou = caclu_setiou(ours_rank,loss_rank,falut_type)
    ours_deepgini_iou = caclu_setiou(ours_rank,deepgini_rank,falut_type)
    ours_margin_iou = caclu_setiou(ours_rank,margin_rank,falut_type)

    print("ours_datactive_iou:",ours_datactive_iou)
    print("ours_entropy_iou:",ours_entropy_iou)
    print("ours_loss_iou:",ours_loss_iou)
    print("ours_deepgini_iou:",ours_deepgini_iou)
    print("ours_margin_iou:",ours_margin_iou)


def main():
    ours_rank = joblib.load(ours_rank_path)
    datactive_rank = joblib.load(datactive_rank_path)
    entropy_rank = joblib.load(entropy_rank_path)
    loss_rank = joblib.load(loss_rank_path)
    deepgini_rank = joblib.load(deepgini_rank_path)
    margin_rank = joblib.load(margin_rank_path)

    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)
    converted_entropy_rank = conver_ours_rank(entropy_rank,g_boxes_json, anno_error)
    converted_loss_rank = conver_ours_rank(loss_rank,g_boxes_json, anno_error)
    converted_deepgini_rank = conver_ours_rank(deepgini_rank,g_boxes_json, anno_error)
    converted_margin_rank = conver_ours_rank(margin_rank,g_boxes_json, anno_error)

    
    coco = COCO(anno_error_path)
    bg_catId = coco.getCatIds()[-1]+1
    converted_datactive_rank = conver_datactive_rank(datactive_rank, bg_catId)
    print(f"datactive rank数量:{len(converted_datactive_rank)}")

    overlap_analyse(converted_ours_rank, converted_datactive_rank, 
                    converted_entropy_rank,converted_loss_rank,
                    converted_deepgini_rank,converted_margin_rank,
                    "clean") # class_fault|loc_fault|redun_fault|miss_fault|clean


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
    
    datactive_rank_path = os.path.join(exp_data_root_dir,"Results","datactive",
                                       dataset_name,"YOLOv7", # 与模型无关
                                  "exp_01","rank","rank.joblib")

    
    entropy_rank_path = os.path.join(exp_data_root_dir,"Results","other_baselines","entropy",
                                     dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")
    
    loss_rank_path = os.path.join(exp_data_root_dir,"Results","other_baselines","loss",
                                  dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")
    
    deepgini_rank_path = os.path.join(exp_data_root_dir,"Results","other_baselines","deepgini",
                                dataset_name,model_name,
                                "exp_01","rank","rank.joblib")
    
    margin_rank_path = os.path.join(exp_data_root_dir,"Results","other_baselines","margin",
                                dataset_name,model_name,
                                "exp_01","rank","rank.joblib")
    main()