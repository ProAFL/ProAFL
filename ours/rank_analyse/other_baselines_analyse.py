import os
import joblib
from ours.base_data_manager import (get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path,get_error_ann_file_path
                                    )
from ours.small_utils import read_json
from ours.rank_analyse.common import compute_apfd,look_total_rank,calc_fpr_fnr_f1,calc_top1,calc_exam
from ours.data_organization_tools import (get_all_errored_g_box_id_set,get_all_miss_error_img_name_set,
                                          get_img_name_to_missed_annids,get_all_error_annoids,get_annoId_to_anno,
                                          conver_ours_rank,get_all_error_imgset,get_all_error_idd_set)


from ours.repair.repair_analyse import count_repair_rate

def get_gid_rank(rank_res):
    gid_rank = []
    for idd in rank_res:
        if type(idd) is int:
            gid_rank.append(idd)
    return gid_rank


def analyse_rank(dataset_name, gt_json_path:str, rank_res:list, annos_with_miss_json_path:str, vis:bool=False):
    '''
    rank_res: 我们方法获得的排序结果（idd:img_name or gid）
    '''
    g_boxes_json = read_json(gt_json_path)
    # 得到错误的gid_set
    all_errored_g_box_id_set = get_all_errored_g_box_id_set(g_boxes_json)
    # 得到missed_error_img_name_set
    all_miss_error_img_name_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)
    # rank_res = get_gid_rank(rank_res)
    # 计算APFD,FPR和FNR
    error_set = all_errored_g_box_id_set | all_miss_error_img_name_set
    APFD = compute_apfd(error_set, rank_res)
    FPR,FNR,F1 =calc_fpr_fnr_f1(rank_res, error_set, cut_off=0.5)

    print(f"排序总长度:{len(rank_res)}")
    print(f"APFD:{APFD},FPR:{FPR},FNR:{FNR},F1:{F1}")
    annos_with_miss_json = read_json(annos_with_miss_json_path)
    error_imgset = get_all_error_imgset(annos_with_miss_json)
    converted_rank = conver_ours_rank(rank_res, g_boxes_json, anno_error)
    error_idd_set = get_all_error_idd_set(annos_with_miss_json)
    top1 = calc_top1(annos_with_miss_json,converted_rank,error_idd_set,error_imgset)
    exam=calc_exam(annos_with_miss_json,converted_rank)
    print(f"top1:{top1},exam:{exam}")

    # 统计该rank的修复率
    # anno_with_miss_error = read_json(annos_with_miss_json_path)
    # imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error) 
    # all_error_annoids = get_all_error_annoids(anno_with_miss_error)
    # annoId_to_anno = get_annoId_to_anno(anno_with_miss_error)
    # imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error)
    # anno_error_path = get_error_ann_file_path(dataset_name)
    # anno_error = read_json(anno_error_path)
    # # idd的转换
    # converted_rank = conver_ours_rank(rank_res, g_boxes_json, anno_error)
    # repaired_box_count,repair_rate = count_repair_rate(converted_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate=0.4)
    # print(f"预计修复数量: {repaired_box_count}, 预计修复率: {repair_rate}")

    if vis:
        vis_rank(rank_res,all_errored_g_box_id_set,all_miss_error_img_name_set)

def vis_rank(rank_res,errored_gid_set, miss_img_set):
    ranked_gid_list = []
    ranked_image_name_list = []
    for idd in rank_res:
        if type(idd) == str:
            ranked_image_name_list.append(idd)
        else:
            ranked_gid_list.append(idd)
    assert len(ranked_gid_list) + len(ranked_image_name_list) == len(rank_res), "数量不对"
    look_total_rank(rank_res,errored_gid_set,miss_img_set)


if __name__ == "__main__":
    
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012" # VOC2012|KITTI_8|VisDrone
    model_name = "FRCNN" # YOLOv7|FRCNN|SSD|rtdetr
    baseline_name = "clod" # entropy|loss|deepgini|margin|objectlab|clod
    rank_path = os.path.join(exp_root_dir,"Results",
                             "other_baselines",baseline_name,dataset_name,model_name,
                             "exp_01","rank","rank_temp.joblib")
    rank = joblib.load(rank_path)

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_error = read_json(anno_error_path)
    analyse_rank(dataset_name,gt_json_path,rank,annos_with_miss_json_path)