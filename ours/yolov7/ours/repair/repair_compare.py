'''
检查一下repair结果是否正确
'''
import os
import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ours.base_data_manager import (exp_data_root_dir,get_correct_ann_file_path,get_error_ann_file_path,
                                    get_annotations_with_miss_json_path,get_collected_gt_box_json_path)
from ours.small_utils import read_json
from ours.data_organization_tools import (get_g_id_to_g_box, get_gid_to_anno_id,
                                          get_imgname_to_imgid, get_cls_id_to_name, get_annoid_to_imgname,
                                          get_all_miss_img_name_list,conver_ours_rank,conver_datactive_rank,
                                          get_img_name_to_missed_annids,get_img_name_to_ann_ids,get_annoId_to_anno)


def get_repaired_info(rank_res:list,g_boxes_json:dict, gid_to_annoid:dict, anno_with_miss_json:dict):
    repair_cutoff_rate = 0.4
    repair_cutoff_point = int(len(rank_res) * repair_cutoff_rate)
    repair_idds = rank_res[:repair_cutoff_point]
    gid_to_gbox = get_g_id_to_g_box(g_boxes_json)
    all_miss_img_name_list = get_all_miss_img_name_list(anno_with_miss_json)
    repaired_info = {
        "no_fault":[],
        "miss_fault":[],
        "cls_fault":[],
        "loc_falut":[],
        "redun_fault":[]
    }
    for idd in repair_idds:
        if type(idd) is str:
            img_name = idd
            if img_name in all_miss_img_name_list:
                repaired_info["miss_fault"].append(img_name)
        else:
            gid = idd
            gbox = gid_to_gbox[gid]
            fault_type = gbox["fault_type"]
            annoid = gid_to_annoid[gid]
            if fault_type == 1:
                repaired_info["cls_fault"].append(annoid)
            elif fault_type == 2:
                repaired_info["loc_falut"].append(annoid)
            elif fault_type == 3:
                repaired_info["redun_fault"].append(annoid)
            elif fault_type == 0:
                repaired_info["no_fault"].append(annoid)
            else:
                raise Exception("fault_type 错误")
    return repaired_info

def repair_info(anno_correct_json,anno_error_json,anno_repair_json):
    '''
    仔细比对repair anno json与correct anno json异同
    '''
    repair_annoId_to_anno = get_annoId_to_anno(anno_repair_json)
    correct_annoId_to_anno =  get_annoId_to_anno(anno_correct_json)
    correct_id_list = []
    repair_id_list = []
    for anno in anno_correct_json["annotations"]:
        correct_id_list.append(anno["id"])
    for anno in anno_repair_json["annotations"]:
        repair_id_list.append(anno["id"])
    common_id_set = set(correct_id_list) & set(repair_id_list)

    residue_miss_id_set = set(correct_id_list) - set(repair_id_list) # 残留的miss fault anno id set
    repaired_miss_id_set = set() # 修复的miss fault anno id set

    repaired_cls_id_set = set() # 修复的cls fault anno id set
    residue_cls_id_set = set() # 残留的cls fault anno id set

    repaired_loc_id_set = set() # 修复的loc fault anno id set
    residue_loc_id_set = set() # 残留的loc fault anno id set

    residue_redunc_id_set = set(repair_id_list) - set(correct_id_list) # 残留的redunc fault anno id set
    redunc_fault_id_set = set()
    for error_anno in anno_error_json["annotations"]:
        if error_anno["fault_type"] == 3:
            redunc_fault_id_set.add(error_anno["id"])
    repaired_redunc_id_set = redunc_fault_id_set - residue_redunc_id_set # 修复的redunc fault anno id set


    irrelevant_id_set = set() # 无关的anno id set
    for c_id in common_id_set:
        repair_anno = repair_annoId_to_anno[c_id]
        correct_anno = correct_annoId_to_anno[c_id]
        if "fault_type" in repair_anno:
            fault_type = repair_anno["fault_type"]
            if fault_type == 0:
                # 没被篡改过的anno
                irrelevant_id_set.add(c_id)
            elif fault_type == 1:
                # cls fault anno, 接着判断是否修复
                correct_label = correct_anno["category_id"]
                repair_label = repair_anno["category_id"]
                if repair_label == correct_label:
                    # cls 被修复了
                    repaired_cls_id_set.add(c_id)
                else:
                    residue_cls_id_set.add(c_id)
            elif fault_type == 2:
                # loc fault anno, 接着判断是否修复
                correct_bbox = correct_anno["bbox"]
                repair_bbox = repair_anno["bbox"]
                if correct_bbox == repair_bbox:
                    # loc 被修复了
                    repaired_loc_id_set.add(c_id)
                else:
                    residue_loc_id_set.add(c_id)
        else:
            # 从 correct anno json迁移过来的anno,即修复了的miss fault anno
            repaired_miss_id_set.add(c_id)

    _data = {
        "无关的anno_id_set":irrelevant_id_set,
        "cls_fault":{
            "修复的": repaired_cls_id_set,
            "残留的": residue_cls_id_set
        },
        "loc_fault":{
            "修复的": repaired_loc_id_set,
            "残留的": residue_loc_id_set
        },
        "redunc_fault":{
            "修复的": repaired_redunc_id_set,
            "残留的": residue_redunc_id_set
        },
        "miss_fault":{
            "修复的": repaired_miss_id_set,
            "残留的": residue_miss_id_set
        }
    }

    print("无关的anno数量:", len(irrelevant_id_set))
    print("修复的cls fault数量:", len(repaired_cls_id_set))
    print("残留的cls fault数量:", len(residue_cls_id_set))
    print("修复的loc fault数量:", len(repaired_loc_id_set))
    print("残留的loc fault数量:", len(residue_loc_id_set))
    print("修复的redunc fault数量:", len(repaired_redunc_id_set))
    print("残留的redunc fault数量:", len(residue_redunc_id_set))
    print("修复的miss fault数量:", len(repaired_miss_id_set))
    print("残留的miss fault数量:", len(residue_miss_id_set))
    print("==="*30)

    return _data

def repair_compare(ours_repair_info, datactive_repair_info, aid_to_imaname):
    assert ours_repair_info["无关的anno_id_set"] == datactive_repair_info["无关的anno_id_set"], "数据错误"
    cls_fault_id_set = ours_repair_info["cls_fault"]["修复的"] & datactive_repair_info["cls_fault"]["残留的"]
    imgname_list = []
    for aid in cls_fault_id_set:
        imgname = aid_to_imaname[aid]
        imgname_list.append(imgname)
    print(imgname_list)

def main():
    anno_correct_json = read_json(anno_correct_json_path)
    anno_error_json = read_json(anno_error_json_path)
    ours_anno_repair_json = read_json(ours_anno_repair_json_path)
    datactive_anno_repair_json = read_json(datactive_anno_repair_json_path)
    anno_with_miss_json = read_json(anno_with_miss_json_path)
    aid_to_imaname = get_annoid_to_imgname(anno_with_miss_json)
    ours_repair_info = repair_info(anno_correct_json,anno_error_json,ours_anno_repair_json)
    datactive_repair_info = repair_info(anno_correct_json,anno_error_json,datactive_anno_repair_json)
    # repair_compare(ours_repair_info, datactive_repair_info, aid_to_imaname)

if __name__ == '__main__':
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    exp_id = "exp_02"

    anno_correct_json_path = get_correct_ann_file_path(dataset_name,"train")
    anno_error_json_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    
    # repaired anno json path
    ours_anno_repair_json_path = os.path.join(exp_data_root_dir,"Results","ours",dataset_name,model_name,exp_id,
                                         "repair","_annotations.coco_repair.json")
    datactive_anno_repair_json_path = os.path.join(exp_data_root_dir,"Results","datactive",dataset_name,model_name,exp_id,
                                         "repair","_annotations.coco_repair.json")

    main()
