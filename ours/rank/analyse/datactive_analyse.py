'''
分析DataDetective的rank
'''
import os
import json
import joblib
from pycocotools.coco import COCO
from helper.base_data_manager import (exp_data_root_dir, get_datactive_rank_res_path,
                                    get_error_ann_file_path,get_annotations_with_miss_json_path)
from ours.rank.analyse.common import *
from helper.data_organization_tools import conver_datactive_rank,get_all_error_imgset,get_all_error_idd_set

from ours.small_utils import read_json

def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name


def get_missed_img_name_set(annotations_with_miss_json):
    miss_img_name_set = set()
    imgId_to_imgName = get_image_id_to_image_name_for_coco(annotations_with_miss_json)
    annos = annotations_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgId_to_imgName[anno["image_id"]]
            miss_img_name_set.add(img_name)
    return miss_img_name_set

def get_error_ann_id_set(coco:COCO):
    anns = coco.loadAnns(coco.getAnnIds())
    error_ann_id_set = set()
    for ann in anns:
        if ann["fault_type"] in [1,2,3]: # cls,loc,red
            error_ann_id_set.add(ann["id"])
    return error_ann_id_set


def look_annid_rank(ranked_gid_list:list[int], all_errored_g_box_id_set:set[int]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "annid_rank.png")
    error_flag_list = []
    for gid in ranked_gid_list:
        if gid in all_errored_g_box_id_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")


def vis_rank(rank_res,errored_annoid_set, miss_img_set, pic_save_path):
    ranked_annoid_list = []
    ranked_image_name_list = []
    for idd in rank_res:
        if type(idd) == str:
            ranked_image_name_list.append(idd)
        else:
            ranked_annoid_list.append(idd)
    assert len(ranked_annoid_list) + len(ranked_image_name_list) == len(rank_res), "数量不对"
    look_total_rank(rank_res,errored_annoid_set,miss_img_set,pic_save_path)

def main():
    coco = COCO(anno_coco_error_json_path)
    catIds = coco.getCatIds()
    bg_id = catIds[-1]+1
    converted_rank_list = conver_datactive_rank(ranked_list,bg_id)
    print(f"rank list 长度:{len(converted_rank_list)}")
    ranked_annid_list = []
    ranked_img_name_list = []
    for idd in converted_rank_list:
        if type(idd) is str:
            ranked_img_name_list.append(idd)
        else:
            ranked_annid_list.append(idd)
    error_ann_id_set = get_error_ann_id_set(coco)
    
    with open(annotations_with_miss_json_path,'r') as f:
        annotations_with_miss_json = json.load(f)
    missed_img_name_set =  get_missed_img_name_set(annotations_with_miss_json)
    # look_annid_rank(ranked_annid_list, error_ann_id_set)
    # look_img_rank(ranked_img_name_list, missed_img_name_set)
    # look_total_rank(converted_rank_list,error_ann_id_set,missed_img_name_set)
    total_error_set = error_ann_id_set | missed_img_name_set

    # 计算APFD,FPR和FNR
    APFD = compute_apfd(total_error_set, converted_rank_list)
    FPR,FNR,F1 =calc_fpr_fnr_f1(converted_rank_list, total_error_set, cut_off=0.5)
    print(f"APFD:{APFD},FPR:{FPR},FNR:{FNR},F1:{F1}")
    annos_with_miss_json = read_json(annotations_with_miss_json_path)
    error_idd_set = get_all_error_idd_set(annos_with_miss_json)
    error_imgset = get_all_error_imgset(annos_with_miss_json)
    top1 = calc_top1(annos_with_miss_json,converted_rank_list,error_idd_set,error_imgset)
    exam=calc_exam(annos_with_miss_json,converted_rank_list)
    print(f"top1:{top1},exam:{exam}")
    # 可视化全排序
    # pic_save_dir = os.path.join(exp_data_root_dir,"temp","total_rank")
    # os.makedirs(pic_save_dir,exist_ok=True)
    # pic_save_file_name = "datactive.png"
    # pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)

    # vis_rank(converted_rank_list,error_ann_id_set, missed_img_name_set, pic_save_path)



def xiufu_rank_res():
    '''
    这是个一次性方法
    '''
    coco = COCO(anno_coco_error_json_path)
    catIds = coco.getCatIds()
    bg_id = catIds[-1]+1
    converted_rank = conver_datactive_rank(ranked_list, bg_id)
    # assert 129286 in converted_rank, "不通过"
    assert 61921 in converted_rank, "不通过"
    removed_idx_list = []
    for idx, instance in enumerate(ranked_list):
        if instance["anno_id"] == 129286:
            removed_idx_list.append(idx)
        if instance["anno_id"] == 61921:
            removed_idx_list.append(idx)
    for idx in removed_idx_list:
        del ranked_list[idx]
    # joblib.dump(ranked_list,f"{exp_data_root_dir}/Results/datactive/{dataset_name}/YOLOv7/{exp_id}/rank/rank_new.joblib")

if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    exp_id = "exp_02"
    # datactive 排序的idd
    ranked_list = joblib.load(f"{exp_data_root_dir}/Results/datactive/{dataset_name}/YOLOv7/{exp_id}/rank/rank.joblib")
    anno_coco_error_json_path = get_error_ann_file_path(dataset_name)
    annotations_with_miss_json_path =get_annotations_with_miss_json_path(dataset_name)
    main()
    # xiufu_rank_res()





