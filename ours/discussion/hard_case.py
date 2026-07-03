'''
hard case discussion
'''

import numpy as np
import os
import joblib
from pycocotools.coco import COCO
from helper.base_data_manager import (exp_data_root_dir,get_collected_gt_box_json_path,
                                    get_error_ann_file_path,
                                    get_annotations_with_miss_json_path,
                                    get_correct_ann_file_path)

from helper.data_organization_tools import (conver_ours_rank, conver_datactive_rank,
                                          get_all_miss_error_img_name_set,
                                          get_all_annoids_detail,
                                          get_all_img_name,get_gid_to_anno_id,
                                          get_annoid_to_imgname,get_annoId_to_anno,
                                          get_cls_id_to_name,get_img_name_to_ann_ids)
from ours.small_utils import read_json

from ours.rank.box_rank import box_rank
from ours.rank.img_rank import img_rank_2

import matplotlib.pyplot as plt
import cv2

def draw_bbox(imgfilepath,error_anno,correct_anno,cls_id_to_name:dict,isfault,save_path):
                                                           
    img = cv2.imread(imgfilepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    

                                   
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.set_title('Correct Annotation', fontsize=14, color='green')
    ax1.axis('off')
    
    for _anno in correct_anno:
        x, y, w, h = _anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = _anno['category_id']
        class_name = cls_id_to_name[class_id]
                           
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax1.add_patch(rect)
                         
        ax1.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax1.axis('off')

                                 
    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title('Error Annotation', fontsize=14, color='red')
    ax2.axis('off')
    if error_anno is not None:
        x, y, w, h = error_anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = error_anno['category_id']
        class_name = cls_id_to_name[class_id]
                           
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="red", facecolor='none')
        ax2.add_patch(rect)
                         
        ax2.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="red", 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax2.axis('off')

    plt.savefig(save_path)

def draw_bbox_for_missFault(imgfilepath,correct_annos,missed_annoidset,cls_id_to_name:dict,save_path):
                                                           
    img = cv2.imread(imgfilepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    

                                   
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.set_title('Correct Annotation', fontsize=14, color='green')
    ax1.axis('off')
    
    for _anno in correct_annos:
        x, y, w, h = _anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = _anno['category_id']
        class_name = cls_id_to_name[class_id]
                           
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax1.add_patch(rect)
                         
        ax1.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax1.axis('off')

                                 
    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title('Error Annotation', fontsize=14, color='red')
    ax2.axis('off')

    for _anno in correct_annos:
        if _anno['id'] in missed_annoidset:
            x, y, w, h = _anno["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = _anno['category_id']
            class_name = cls_id_to_name[class_id]
                               
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2,linestyle='--',edgecolor="red", facecolor='none')
            ax2.add_patch(rect)
                             
            ax2.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="red", 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax2.axis('off')
        else:
            x, y, w, h = _anno["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = _anno['category_id']
            class_name = cls_id_to_name[class_id]
                               
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
            ax2.add_patch(rect)
                             
            ax2.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax2.axis('off')

    plt.savefig(save_path)


def _analyse_miss_fault(ours_rank):
    cut = int(len(ours_rank)*0.5)
    positive_rank = ours_rank[:cut]
    negative_rank = ours_rank[cut:]
    miss_fault_imgset = get_all_miss_error_img_name_set(anno_with_miss_error_path)
    hard_img_list = []
    for idd in negative_rank[::-1]:
        if idd in miss_fault_imgset:
            hard_img_list.append(idd)
    hard_img = hard_img_list[5]

                
    rank_res = img_rank_2(img_to_nomatched_pboxs_json_path)
    ranked_imgs = rank_res["ranked_imgs"]
    X =  rank_res["feature_data"]
    feature_names = rank_res["feature_names"]
    idx = ranked_imgs.index(hard_img)
    feature =  X[idx]
    rows_to_use = np.delete(X, idx, axis=0)                                                         
    mean_values = np.mean(rows_to_use, axis=0)
    print(f"imgname: {hard_img}")
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        cur_val = float(feature[i])
        mean_val = float(mean_values[i])
        print(f"{feature_name}|cur_val:{cur_val}|mean:{mean_val}")


                         
    imgfilepath = os.path.join(imgs_dir,hard_img)
               
    correct_annos = []
    imgname_to_annoids = get_img_name_to_ann_ids(anno_correct)
    correct_annoids = imgname_to_annoids[hard_img]
    annoid_to_anno = get_annoId_to_anno(anno_correct)
    for correct_annoid in correct_annoids:
        correct_annos.append(annoid_to_anno[correct_annoid])

    imgname_to_annoids = get_img_name_to_ann_ids(anno_error)
    error_annoids = imgname_to_annoids[hard_img]
    missed_anno_id_set = set(correct_annoids) - set(error_annoids)
                         
    cls_id_to_name = get_cls_id_to_name(anno_error)
    save_path = os.path.join(exp_data_root_dir,"temp","bbox_img","missfault.png")
    draw_bbox_for_missFault(imgfilepath,correct_annos,missed_anno_id_set,cls_id_to_name,save_path)


def _analyse(ours_rank,fault_type):
    cut = int(len(ours_rank)*0.5)
    positive_rank = ours_rank[:cut]
    negative_rank = ours_rank[cut:]
    miss_fault_imgset = get_all_miss_error_img_name_set(anno_with_miss_error_path)
    detail = get_all_annoids_detail(anno_error_with_miss)
    fault_annoid_set = set(detail[fault_type])

    hard_idd_list = []
    for idd in negative_rank[::-1]:
        if idd in fault_annoid_set:
            hard_idd_list.append(idd)

                                      
    hard_idd = hard_idd_list[0]
    gid_to_annoid = get_gid_to_anno_id(g_boxes_json,anno_error)
    annoid_to_gid = dict(zip(gid_to_annoid.values(), gid_to_annoid.keys()))
    hard_gid = annoid_to_gid[hard_idd]

    rank_res = box_rank(gt_json_path,metric_json_path)
    ranked_gids =  rank_res["ranked_gids"]
    X =  rank_res["feature_data"]
    feature_names = rank_res["feature_names"]
    feature_signs = rank_res["sign_list"]
    idx = ranked_gids.index(hard_gid)
    feature =  X[idx]
    rows_to_use = np.delete(X, idx, axis=0)                                                         
    mean_values = np.mean(rows_to_use, axis=0)
    print(f"gid: {hard_gid}")
    print(f"annoid: {hard_idd}")
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        feature_sign = feature_signs[i]
        cur_val = float(feature[i])
        mean_val = float(mean_values[i])
        print(f"{feature_name}|{feature_sign}|cur_val:{cur_val}|mean:{mean_val}")


                 
    annoid_to_imgname = get_annoid_to_imgname(anno_error)
    imgname = annoid_to_imgname[hard_idd]
    imgfilepath = os.path.join(imgs_dir,imgname)


    annoid_to_anno = get_annoId_to_anno(anno_error)
    error_anno = annoid_to_anno[hard_idd]


    if fault_type == "redun_fault":
        correct_annos = []
        imgname_to_annoids = get_img_name_to_ann_ids(anno_correct)
        annoids = imgname_to_annoids[imgname]
        annoid_to_anno = get_annoId_to_anno(anno_correct)
        for annoid in annoids:
            anno = annoid_to_anno[annoid]
            correct_annos.append(anno)
        correct_anno = correct_annos                    
    else:
        annoid_to_anno = get_annoId_to_anno(anno_correct)
        correct_anno = annoid_to_anno[hard_idd]                   

                         
    cls_id_to_name = get_cls_id_to_name(anno_error)
    save_path = os.path.join(exp_data_root_dir,"temp","bbox_img","1.png")
    draw_bbox(imgfilepath,error_anno,correct_anno,cls_id_to_name,isfault=True,save_path=save_path)
    print()

def main():
    ours_rank = joblib.load(ours_rank_path)
    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)
                                                                                     
                                              
    _analyse_miss_fault(converted_ours_rank)


if __name__ == "__main__":
    dataset_name = "VOC2012"                           
    model_name = "YOLOv7"

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_error_path = get_annotations_with_miss_json_path(dataset_name)
    correct_anno_path = get_correct_ann_file_path(dataset_name,"train")

    
    match_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,
                                   "gp_box_match","match_v2.json")
    metric_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,
                                   "collection_metric","collection_metrics_v2.json")
    img_to_nomatched_pboxs_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",
                dataset_name,model_name,"img_to_nomatched_pboxs.json")
    
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split",dataset_name,"images","origin")
    anno_error_with_miss = read_json(anno_with_miss_error_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)
    anno_correct = read_json(correct_anno_path)
    match_json = read_json(match_json_path)
    metric_json = read_json(metric_json_path)

    ours_rank_path = os.path.join(exp_data_root_dir,"Results","ours",
                                  dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")

    main()