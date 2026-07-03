'''
统计分析ours和datactive,排序结果的中前40%的TP的交并比
'''
import os
import joblib
from helper.base_data_manager import exp_data_root_dir,get_collected_gt_box_json_path
from ours.small_utils import read_json
from pycocotools.coco import COCO
from helper.data_organization_tools import (get_gid_to_anno_id,get_error_annoid_set,get_all_miss_img_name_list,
                                          get_imgid_to_imgname,get_all_error_clean_set, conver_ours_rank, conver_datactive_rank)



def calcu_jaccard(ours_list:list,datactive_list:list)->float:
    ours_set = set(ours_list)
    datactive_set = set(datactive_list)
    union = ours_set | datactive_set
    intersection = ours_set & datactive_set
    jaccard = round(len(intersection) / len(union),4)
    return jaccard

def analyse_jaccard(ours_list:list,datactive_list:list,anno_error_with_miss:dict) -> dict:
    ours_set = set(ours_list)
    datactive_set = set(datactive_list)
    intersection = ours_set & datactive_set
    total_commmon_nums = len(intersection)
    all_error_clean_set = get_all_error_clean_set(anno_error_with_miss)
    miss_set = all_error_clean_set["miss_set"]
    cls_set = all_error_clean_set["cls_set"]
    loc_set = all_error_clean_set["loc_set"]
    redun_set  = all_error_clean_set["redun_set"]
    clean_set = all_error_clean_set["clean_set"]

    common_tp_mis_list = []
    common_tp_cls_list = []
    common_tp_loc_list = []
    common_tp_redun_list = []
    common_fp_list = []
    for idd in intersection:
        if type(idd) is str:
            img_name = idd
            if img_name in miss_set:
                common_tp_mis_list.append(img_name)
            else:
                common_fp_list.append(img_name)
        else:
            if idd in cls_set:
                common_tp_cls_list.append(idd)
            elif idd in loc_set:
                common_tp_loc_list.append(idd)
            elif idd in redun_set:
                common_tp_redun_list.append(idd)
            elif idd in clean_set:
                common_fp_list.append(idd)
    common_tp_mis_rate = round(len(common_tp_mis_list) / total_commmon_nums,4)
    common_tp_cls_rate = round(len(common_tp_cls_list) / total_commmon_nums,4)
    common_tp_loc_rate = round(len(common_tp_loc_list) / total_commmon_nums,4)
    common_tp_redun_rate = round(len(common_tp_redun_list) / total_commmon_nums,4)
    common_fp_rate = round(len(common_fp_list) / total_commmon_nums,4)

    res = {
        "common_tp_mis_rate":common_tp_mis_rate,
        "common_tp_cls_rate":common_tp_cls_rate,
        "common_tp_loc_rate":common_tp_loc_rate,
        "common_tp_redun_rate":common_tp_redun_rate,
        "common_fp_rate":common_fp_rate
    }
    return res




def main():
    # 得到转换后的ours rank list
    ours_rank = joblib.load(ours_rank_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)
    anno_error_with_miss = read_json(anno_with_miss_error_path)
    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)

    # 得到转换后的datactive rank list
    datactive_rank = joblib.load(datactive_rank_path)
    coco = COCO(anno_error_path)
    bg_catId = coco.getCatIds()[-1]+1
    converted_datactive_rank = conver_datactive_rank(datactive_rank,bg_catId)

    # 比较cut off中两个方法的交并比
    cut_off_rate = 0.4
    cut_off_point = int(len(converted_ours_rank) * cut_off_rate)
    ours_cut_off = converted_ours_rank[:cut_off_point]
    datactive_cut_off = converted_datactive_rank[:cut_off_point]
    jaccard = calcu_jaccard(ours_cut_off, datactive_cut_off)
    print(f"jaccard: {jaccard}")

    analysed_jaccard_res = analyse_jaccard(ours_cut_off, datactive_cut_off, anno_error_with_miss)
    print(analysed_jaccard_res)


if __name__ == "__main__":
    dataset_name = "VisDrone"
    model_name = "YOLOv7"
    ours_rank_path = os.path.join(exp_data_root_dir,"final_res","ours",dataset_name,model_name,"rank_res",
                             "alpha=1.5","rank_topsis.joblib")
    datactive_rank_path = os.path.join(exp_data_root_dir,"final_res","datactive",dataset_name,"ranked_result",
                            "ranked_list.joblib")
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",
                            "_annotations.coco_error.json")
    anno_with_miss_error_path = os.path.join(exp_data_root_dir,"error_anno",dataset_name,"coco_format",
                            "annotations_with_miss.json")
    main()