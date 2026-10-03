import os
import json
import time
from datetime import datetime
from collections import defaultdict
from ours.small_utils import get_cost_time,get_formatted_time
import numpy as np
from tqdm import tqdm
from PIL import Image

from pycocotools.coco import COCO
from ours.small_utils import read_json
from helper.base_data_manager import (
    exp_data_root_dir,
    get_annotations_no_miss_json_path
    )

def get_epoch_to_pboxs(predicted_bboxs_dir) -> dict:
    _dict = {}
    for epoch in range(epochs):
        epoch_predicted_bboxs_json_path = os.path.join(predicted_bboxs_dir,f"epoch_{epoch}_predicted_bboxs.json")
        with open(epoch_predicted_bboxs_json_path,"r") as f:
            epoch_predicted_bboxs_dict = json.load(f)
        _dict[epoch] = epoch_predicted_bboxs_dict
    return _dict


def pretty_print(content,count,col_nums=10):
    print(content, end=' ')
    if count % col_nums == 0:                                      
        print()                   

def get_img_path_by_img_name(img_name,style):
    if style == "yolo":
        image_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-yolo","origin","train","images",img_name)
    elif style == "coco":
        image_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",img_name)
    return image_path

def xcycwh_to_x1y1x2y2(bbox,W,H):
    xc = bbox[0]
    yc = bbox[1]
    w = bbox[2]
    h = bbox[3]

                            
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

                                          
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

                              
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]

def offset_p_label(p_box_list):
    for box in p_box_list:
        box["predicted_cls"] -= 1
    return p_box_list


def calu_iou(gt_bbox,predicted_bbox):
    x1_min, y1_min, x1_max, y1_max = gt_bbox
    x2_min, y2_min, x2_max, y2_max = predicted_bbox

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def get_iou_matrix_PG(p_box_list,gt_box_list):
    P = len(p_box_list)
    G = len(gt_box_list)
    iou_matrix = np.zeros((P,G))
    for i,p_box in enumerate(p_box_list):
        for j,g_box in enumerate(gt_box_list):
            p_bbox = p_box["bbox"]
            g_bbox = g_box["gt_bbox"]
            iou = calu_iou(g_bbox,p_bbox)
            iou_matrix[i][j] = iou
    return iou_matrix


def get_iou_matrix_GP(gt_box_list, p_box_list):
    
    G = len(gt_box_list)
    P = len(p_box_list)
    iou_matrix = np.zeros((G,P))
    for i,g_box in enumerate(gt_box_list):
        for j,p_box in enumerate(p_box_list):
            p_bbox = p_box["bbox"]
            g_bbox = g_box["gt_bbox"]
            iou = calu_iou(g_bbox,p_bbox)
            iou_matrix[i][j] = iou
    return iou_matrix

def search_match_GP(gt_box_list, predicted_box_list, iou_thre=0.5):

                                      
    predicted_box_list = sorted(predicted_box_list, key=lambda x: x["conf"], reverse=True)

    matches = []

    cls_set = set(gt["cls"] for gt in gt_box_list)

    for cls in cls_set:

        cur_gt = [g for g in gt_box_list if g["cls"] == cls]
        cur_pred = [p for p in predicted_box_list if p["predicted_cls"] == cls]

        if not cur_gt or not cur_pred:
            continue

        iou_matrix = get_iou_matrix_GP(cur_gt, cur_pred)

                             
        assert iou_matrix.shape == (len(cur_gt), len(cur_pred)), "Invalid shape"

        used_pred = set()

        for g_idx in range(len(cur_gt)):

                          
            best_pred_idx = iou_matrix[g_idx].argmax().item()
            best_iou = float(iou_matrix[g_idx, best_pred_idx])

            if best_iou < iou_thre:
                continue

            if best_pred_idx in used_pred:
                continue

            used_pred.add(best_pred_idx)
            matches.append((cur_gt[g_idx], cur_pred[best_pred_idx], best_iou))

    return matches

def gt_best_match(predicted_box_list, gt_box_list):
    matches = []
    for gt in gt_box_list:
        best_p = None
        best_iou = 0.0
        for p in predicted_box_list:
            if p["predicted_cls"] != gt["cls"]:
                continue
            iou = calu_iou(p["bbox"], gt["gt_bbox"])
            if iou > best_iou:
                best_iou = iou
                best_p = p
        if best_p:
            matches.append((gt,best_p,best_iou))
    return matches


def search_match_PG(predicted_box_list, gt_box_list, iou_thre=0.5):
    '''
    Match gt boxes and predicted boxes for one image.
    args:
        gt_box_list: all g_boxes of this image x1y1x2y2
        predicted_box_list: p_boxes of this image at a given epoch
        iou_thre: pboxpbox and gbox are matched only when IoU is above this threshold
    '''
                                                   
    predicted_box_list.sort(key=lambda x: x["conf"], reverse=True)
                        
    P = len(predicted_box_list)
                 
    G = len(gt_box_list)
                             
    used_gt = set()
                            
    matches = []
                 
    cls_set = set([gt_box["cls"] for gt_box in gt_box_list])
                                         
    for cls in cls_set:
                                       
        cur_cls_gt_box_list = [box for box in gt_box_list if box["cls"] == cls]
                                                                               
        cur_cls_p_box_list = [box for box in predicted_box_list if box["predicted_cls"] == cls]
        if len(cur_cls_gt_box_list) == 0 or len(cur_cls_p_box_list) == 0:
            continue
                                                                                           
        iou_matrix = get_iou_matrix_PG(cur_cls_p_box_list,cur_cls_gt_box_list,)
        assert iou_matrix.shape == (len(cur_cls_p_box_list), len(cur_cls_gt_box_list))
                                               
        best_gt_box_id_list = iou_matrix.argmax(axis=1)
                                                                
        best_iou_list = iou_matrix.max(axis=1)
        for r_i,iou_val in enumerate(best_iou_list):
            iou_val = iou_val.item()
            if iou_val < iou_thre:
                                                                      
                continue
                                        
            best_gt_id = best_gt_box_id_list[r_i]
                                                      
            matched_gt_box = cur_cls_gt_box_list[best_gt_id]
            if matched_gt_box["box_id"] in used_gt:
                                                                                                                                   
                continue
            used_gt.add(matched_gt_box["box_id"])
            p_box = cur_cls_p_box_list[r_i]
            matches.append((matched_gt_box,p_box,iou_val))
    return matches


def search_match_PG2(cur_epoch_p_boxs,anns,iou_thre=0.5):
    """Match predicted boxes to COCO annotations for one image.
    这里面都是某一张图片预测框和真实框
    Annotations use ``category_id`` and COCO ``[x, y, width, height]``
    boxes. Return ``(annotation, prediction, IoU)`` tuples, as in
    ``search_match_PG``.
    这里真实框使用的是coco标注框
    """
    cur_epoch_p_boxs.sort(key=lambda box: box["conf"], reverse=True)
    matches = []
    used_ann_ids = set()

    for cls in {ann["category_id"] for ann in anns}: # ground_truth_classid set
        cls_anns = [ann for ann in anns if ann["category_id"] == cls] # 标注框
        cls_preds = [box for box in cur_epoch_p_boxs if box["predicted_cls"] == cls] # 预测框
        if not cls_preds:
            # 没有该cls的预测框，则直接去处理下个cls
            continue

        for p_box in cls_preds:
            best_ann = None # 该预测框匹配到的最好ann
            best_iou = -1.0 # 该预测框匹配到的最好的ann的iou
            for ann in cls_anns:
                x, y, width, height = ann["bbox"]
                ann_bbox = [x, y, x + width, y + height] # x1,y1,x2,y2
                iou = calu_iou(ann_bbox, p_box["bbox"])
                if iou > best_iou:
                    best_ann = ann
                    best_iou = iou

            if best_ann is None or best_iou < iou_thre:
                # 没有最好的ann匹配或者匹配上了iou也没达标(>0.5)，则这个p_box没匹配上任何的anno
                continue
            if best_ann["id"] in used_ann_ids:
                # 匹配到的标注框之前已经被匹配到了，则该p_box也就落空。
                continue

            used_ann_ids.add(best_ann["id"]) # 该标注框匹配到预测框了
            matches.append((best_ann, p_box, best_iou))

    return matches

def match(ann_file:str, epoch_to_p_boxs:dict, offset:bool, save_path):
    start_time = time.time()
    matched_anno_dict = defaultdict(list)
    coco = COCO(ann_file)
    # 遍历所有图片id
    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids,desc="遍历图像"):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        # 获取该图片所有标注id
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for epoch in range(epochs):
            p_boxs_dict = epoch_to_p_boxs[epoch]
            if img_name not in p_boxs_dict:
                continue # 当前轮次该imgname没有任何一个预测框则跳过当前轮次。
            cur_epoch_p_boxs = p_boxs_dict[img_name]["predicted_bboxs"]
            if cur_epoch_p_boxs == None:
                continue
            if offset:
                cur_epoch_p_boxs = offset_p_label(cur_epoch_p_boxs)
            # matches = search_match_PG(cur_epoch_p_boxs,g_boxs,iou_thre=0.5)
            matches = search_match_PG2(cur_epoch_p_boxs,anns,iou_thre=0.5) # 这里是某一张图片某个轮次预测框和真实框的匹配结果
            for match in matches:
                matched_anno = match[0]
                p_box = match[1]
                iou_val = match[2]
                anno_id = matched_anno["id"]
                matched_anno_dict[anno_id].append({"epoch":epoch,"p_box":p_box,"iou_val":iou_val})

    print(f"图像数量: {len(img_ids)}")
    print(f"anno数量: {len(coco.anns)}")
    print(f"匹配到的anno数量: {len(matched_anno_dict)}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(matched_anno_dict, f, indent=4)
    print(f"matched_anno_dict is saved in {save_path}")
    end_time = time.time()
    cost_time = end_time - start_time
    cost_time = get_cost_time(cost_time)
    print(f"match操作消耗时间:{cost_time}")
    print(f"完成时间:{get_formatted_time()}")
    return matched_anno_dict

def collect_metrics_for_annos(match_json:dict, save_path:str):
    '''
    textgt_boxtext cross epochtextconftextiou

    Parameters:
    ---
    match_json: dict
        Data format:
        {
            annoid:[{"epoch":epoch,"p_box":p_box},...],
            ...
        }
    '''
    start_time = time.time()
    annoId2metrics = {}
    for anno_id_str in match_json.keys():
        matched_info_over_epoch = match_json[anno_id_str]
        anno_id = int(anno_id_str)
        annoId2metrics[anno_id] = {"conf_list":[],"iou_list":[]}
        temp_dict = {}
        for matched_info in matched_info_over_epoch:
            epoch = matched_info["epoch"]
            temp_dict[epoch] = {
                "p_box":matched_info["p_box"],
                "iou_val":matched_info["iou_val"]
            }
        for epoch in range(epochs):
            matched_info = temp_dict.get(epoch)
            if matched_info is None:
                conf = 0
                iou = 0
            else:
                conf = matched_info["p_box"]["conf"]
                iou = matched_info["iou_val"]
            annoId2metrics[anno_id]["conf_list"].append(conf)
            annoId2metrics[anno_id]["iou_list"].append(iou)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(annoId2metrics, f, indent=4)
    print(f"annoId2metrics is saved in {save_path}")

    end_time = time.time()
    cost_time = end_time - start_time
    cost_time = get_cost_time(cost_time)
    print(f"metrics操作消耗时间:{cost_time}")
    print(f"完成时刻:{get_formatted_time()}")

def main():
    mode = 0
    if model_name in ["yolov7","rtdetr"]:
        offset = False
    elif model_name == "frcnn":
        offset = True
    else:
        raise Exception("model_name有误")
    if mode == 0 or mode == 1:
        print("Match Start")
        save_dir = os.path.join(exp_data_root_dir,"match_metrics",dataset_name,model_name,f"{str(inject_ratio)}_repeat",f"repeat_{repeat_id}_1791014401")
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir,"match.json")
        match(anno_no_miss_json_path, epoch_to_p_boxs, offset, save_path)
        print("Match End")
    if mode == 0 or mode == 2:
        print("Metrics Start")
        match_json_path = os.path.join(exp_data_root_dir,"match_metrics",dataset_name,model_name,f"{str(inject_ratio)}_repeat",f"repeat_{repeat_id}_1791014401", "match.json")
        with open(match_json_path, "r") as f:
            matched_anno_dict = json.load(f)
        save_path = os.path.join(exp_data_root_dir,"match_metrics",dataset_name,model_name,f"{str(inject_ratio)}_repeat",f"repeat_{repeat_id}_1791014401","metrics.json")
        collect_metrics_for_annos(matched_anno_dict, save_path)
        print("Metrics End")

if __name__ == "__main__":
    pid = os.getpid()
    print(f"pid:{pid}")
    dataset_name = "voc"
    model_name = "yolov7"
    inject_ratio = 0.1 # 0.01,0.05,0.1,0.15
    repeat_id = 1
    epochs = 50 if model_name != "rtdetr" else 100
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_process_info",
                                       dataset_name,model_name,"collected_predict_boxes",
                                       f"inject_{inject_ratio}_repeat",f"repeat_{repeat_id}_1791014401")
    epoch_to_p_boxs = get_epoch_to_pboxs(predicted_bboxs_dir)
    # match时肯定使用的是anno no miss json
    # 此时当前的注错率下还只是一次注入，并没有重复注入
    anno_no_miss_json_path = get_annotations_no_miss_json_path(dataset_name,inject_ratio)
    main()