'''
分析ours,datactive修复box数量
'''

import os
import joblib
from ours.base_data_manager import (exp_data_root_dir,get_collected_gt_box_json_path,get_error_ann_file_path,
                                    get_annotations_with_miss_json_path)
from ours.small_utils import read_json
from pycocotools.coco import COCO
from ours.data_organization_tools import (get_gid_to_anno_id,get_error_annoid_set,
                                          get_all_miss_img_name_list,
                                          get_all_miss_error_img_name_set,
                                          get_imgid_to_imgname,get_all_error_clean_set, 
                                          conver_ours_rank, 
                                          conver_datactive_rank,
                                          get_img_name_to_missed_annids,
                                          get_annoId_to_anno,
                                          get_all_error_annoids)


def count_repair_rate(rank:list,imgname_to_missed_annids:dict,all_error_annoids:list[int],annoId_to_anno:dict,cut_off_rate:float=0.4):
    cut_off =  int(len(rank)*cut_off_rate)
    cut_off_rank = rank[:cut_off]
    # 修复的miss count
    repaired_miss_box_count = 0
    # 修复的other count
    repaired_other_box_count = 0
    for idd in cut_off_rank:
        if type(idd) is str:
            img_name = idd
            repaired_miss_box_count += len(imgname_to_missed_annids[img_name])
        else:
            anno_id = idd
            anno = annoId_to_anno[anno_id]
            if anno["fault_type"] != 0:
                repaired_other_box_count += 1
    repaired_box_count = repaired_miss_box_count + repaired_other_box_count
    repair_rate = round(repaired_box_count/len(all_error_annoids),4)
    print(f"全部错误的annids的数量(包括miss fault):{len(all_error_annoids)}")
    return repaired_box_count,repair_rate
    





def hard_case(converted_rank_res):
    cut = int(len(converted_rank_res)*0.4)
    rank = converted_rank_res[cut:]
    anno_with_miss_error = read_json(anno_with_miss_error_path)
    annos = anno_with_miss_error["annotations"]
    cls_fault_gids = set()
    loc_fault_gids = set()
    red_fault_gids = set()
    miss_fault_imgnames = get_all_miss_error_img_name_set(anno_with_miss_error_path)

    for anno in annos:
        if anno["fault_type"] == 1: # cls fault
            cls_fault_gids.add(anno["id"])
        elif anno["fault_type"] == 2: # loc fault
            loc_fault_gids.add(anno["id"])
        elif anno["fault_type"] == 3: # red fault
            red_fault_gids.add(anno["id"])
    
    hard_cls_nums = 0
    hard_loc_nums = 0
    hard_red_nums = 0
    hard_miss_nums = 0
    for idd in rank:
        if idd in cls_fault_gids:
            hard_cls_nums += 1
        elif idd in loc_fault_gids:
            hard_loc_nums += 1
        elif idd in red_fault_gids:
            hard_red_nums += 1
        elif idd in miss_fault_imgnames:
            hard_miss_nums += 1
    print("hard_cls_nums:",hard_cls_nums)
    print("hard_loc_nums:",hard_loc_nums)
    print("hard_red_nums:",hard_red_nums)
    print("hard_miss_nums:",hard_miss_nums)



def main():
    # 得到我们方法的排序
    ours_rank = joblib.load(ours_rank_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)
    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)
    print(f"ours rank数量:{len(converted_ours_rank)}")

    # 得到datactive方法的排序
    datactive_rank = joblib.load(datactive_rank_path)
    coco = COCO(anno_error_path)
    bg_catId = coco.getCatIds()[-1]+1
    converted_datactive_rank = conver_datactive_rank(datactive_rank, bg_catId)
    print(f"datactive rank数量:{len(converted_datactive_rank)}")

    # 得到entropy rank
    entropy_rank = joblib.load(entropy_rank_path)
    converted_entropy_rank = conver_ours_rank(entropy_rank, g_boxes_json, anno_error)
    print(f"entropy rank数量:{len(converted_entropy_rank)}")

    # 得到loss rank
    loss_rank = joblib.load(loss_rank_path)
    converted_loss_rank = conver_ours_rank(loss_rank, g_boxes_json, anno_error)
    print(f"loss rank数量:{len(converted_loss_rank)}")

    # 得到deepgini rank
    deepgini_rank = joblib.load(deepgini_rank_path)
    converted_deepgini_rank = conver_ours_rank(deepgini_rank, g_boxes_json, anno_error)
    print(f"deepgini rank数量:{len(converted_deepgini_rank)}")

    # 得到margin rank
    margin_rank = joblib.load(margin_rank_path)
    converted_margin_rank = conver_ours_rank(margin_rank, g_boxes_json, anno_error)
    print(f"margin rank数量:{len(converted_margin_rank)}")

    anno_with_miss_error = read_json(anno_with_miss_error_path)
    all_error_annoids = get_all_error_annoids(anno_with_miss_error)
    annoId_to_anno = get_annoId_to_anno(anno_with_miss_error)
    imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error) 
    
    print(f"总共有错误的box数量（包括miss error）:{len(all_error_annoids)}")
    cut_off_rate = 0.4

    repaired_box_count,repair_rate = count_repair_rate(converted_ours_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"ours修复数量: {repaired_box_count}, 修复率: {repair_rate}")

    repaired_box_count,repair_rate = count_repair_rate(converted_datactive_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"datactive修复数量: {repaired_box_count}, 修复率: {repair_rate}")

    repaired_box_count,repair_rate = count_repair_rate(converted_entropy_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"entropy修复数量: {repaired_box_count}, 修复率: {repair_rate}")

    repaired_box_count,repair_rate = count_repair_rate(converted_loss_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"loss修复数量: {repaired_box_count}, 修复率: {repair_rate}")

    
    # overlap_analyse(converted_ours_rank, converted_datactive_rank, converted_entropy_rank, 
    #                 converted_loss_rank, converted_deepgini_rank, converted_margin_rank)
    # hard_case(converted_ours_rank)

if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7" # YOLOv7|FRCNN|SSD

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_error_path = get_annotations_with_miss_json_path(dataset_name)
    
    
    ours_rank_path = os.path.join(exp_data_root_dir,"Results","ours",
                                  dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")
    
    datactive_rank_path = os.path.join(exp_data_root_dir,"Results","datactive",
                                       dataset_name,"YOLOv7", # 与模型无关
                                  "exp_02","rank","rank.joblib")

    
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