
'''
Collecting p_box information and matching it with g_box is the basis for other YOLOv7 baselines on three datasets.
'''
import os
import time
import json
import yaml
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from utils.datasets import create_dataloader
from utils.general import (colorstr,non_max_suppression,
                           non_max_suppression_with_probs,
                           scale_coords,xyxy2xywh)
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from yolov7.models.yolo import Model
from helper.base_data_manager import (
                            exp_data_root_dir,
                            get_collected_gt_box_json_path,
                            get_error_train_model_weight_file_path, 
                            get_nc_by_datasetname, 
                            get_error_ann_file_path)
from custom_module.small_utils import read_yaml,read_json


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

def search_match(gt_box_list, predicted_box_list, iou_thre=0.5):
    '''
    Match gt boxes and predicted boxes for one image.
    args:
        gt_box_list: all g_boxes of this image
        predicted_box_list: p_boxes of this image at a given epoch
        iou_thre: pboxpbox and gbox are matched only when IoU is above this threshold
    '''
                                                   
    predicted_box_list.sort(key=lambda x: x["conf"], reverse=True)
                 
    G = len(gt_box_list)
                        
    P = len(predicted_box_list)
                             
    used_gt = set()
                            
    matches = []
                 
    cls_set = set([gt_box["cls"] for gt_box in gt_box_list])
                                         
    for cls in cls_set:
                                       
        cur_cls_gt_box_list = [box for box in gt_box_list if box["cls"] == cls]
                                                                               
        cur_cls_p_box_list = [box for box in predicted_box_list if box["predicted_cls"] == cls]
        if len(cur_cls_gt_box_list) == 0 or len(cur_cls_p_box_list) == 0:
            continue
                                                                                           
        iou_matrix = get_iou_matrix_PG(cur_cls_p_box_list,cur_cls_gt_box_list)
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

def search_match_v2(gt_box_list, predicted_box_list):
    '''
    text"datactive: For each bounding box in the original annotation set, we identify the
    predicted box with the highest overlap to compute the prediction loss."
    textGT boxtextIoUtext
    '''
    matches = []
    for gt_box in gt_box_list:
        gt_bbox = gt_box["gt_bbox"]
        gt_cls = gt_box["cls"]

              
        same_cls_preds = [p for p in predicted_box_list if p["predicted_cls"] == gt_cls]
        if not same_cls_preds:
            continue

                     
        max_iou = 0.0
        best_pred = None
        for pred in same_cls_preds:
            iou = calu_iou(gt_bbox, pred["bbox"])
            if iou > max_iou:
                max_iou = iou
                best_pred = pred

        if best_pred and max_iou > 0:
            matches.append((gt_box, best_pred, max_iou))

    return matches

def offset_p_label(p_box_list):
                        
    for box in p_box_list:
        box["predicted_cls"] -= 1
    return p_box_list

def pretty_print(content,count,col_nums=10):
    print(content, end=' ')
    if count % col_nums == 0:                                      
        print()                   

def get_img_path_by_img_name(img_name,style):
    if style == "yolo":
        image_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-yolo","origin","train","images",img_name)
    elif style == "coco":
        image_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-coco","train",img_name)
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


def model_load_weight(model:nn.Module,device,weight_path:str):
                        
    
    if weight_path.endswith("last.pt"):
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
        state_dict = state_dict['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    return model

def collectprobs_one_epoch(model,dataloader,conf_thres=0.25,iou_thres=0.65):
    '''
    iou_threstextNMS
    '''
    predicted_box_dict = {}
    predicted_box_id = 0
               
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        '''
        shapes: list
            textbatchsize,textshape
        paths: list
            textbatchsize,text
        '''
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape                                     
        with torch.no_grad():
                                               
            outs, train_outs = model(imgs, augment=False)                                  
            '''
            outs: list
                textbatchsize
                outs[i][0]textitextOutputtext(p_box_nums,6),
                outs[i][0]textitextOutputtext(p_box_nums,20)
            '''
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)             
                                                                       
            lb = []
                                                      
                                                                         
                                                             
            outs = non_max_suppression_with_probs(outs, conf_thres=conf_thres, 
                                                  iou_thres=iou_thres,labels=lb, multi_label=True)
                           
            for loc_i, (pred,prob) in enumerate(outs):
                                                        
                pred = pred.clone()            
                path = Path(paths[loc_i])
                img_name = path.name
                scale_coords(imgs[loc_i].shape[1:], pred[:, :4], shapes[loc_i][0], shapes[loc_i][1])                     
                predicted_bbox_list = []
                               
                for box_i, (*xyxy, conf, cls) in enumerate(pred.tolist()):
                    predicted_box = {
                        "predicted_box_id":predicted_box_id,
                        "img_name":img_name,
                        "predicted_cls":int(cls),
                        "conf":conf,
                        "bbox":xyxy,
                        "prob":prob[box_i].tolist()
                    }
                    predicted_box_id += 1
                    predicted_bbox_list.append(predicted_box)
                predicted_box_dict[img_name] = {
                        "predicted_bboxs":predicted_bbox_list,
                        "height":shapes[loc_i][0][0],
                        "weight":shapes[loc_i][0][1]
                }
    save_dir = collect_p_box_dir
    os.makedirs(save_dir,exist_ok=True)
    save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(predicted_box_dict, f, indent=4)
    print(f"Data saved at:{save_json_path}")



def match(g_json_path,p_json_path,offset):
                                                                    
    g_json = read_json(g_json_path)
    p_json = read_json(p_json_path)

    '''
    Collect matches between dataset g_boxes and p_boxes for each epoch.
    '''
    start_time = time.time()                     
                   
                                          
    gt_box_match = {}
                           
    with_gtboxed_img_count = 0
                                               
    for img_name,g_boxs in g_json.items():
        with_gtboxed_img_count += 1
        pretty_print(img_name,with_gtboxed_img_count,col_nums=10)
                    
        image_path = get_img_path_by_img_name(img_name,"yolo")
                                
        image = Image.open(image_path)
        width, height = image.size
                                                       
        for g_box in g_boxs:
            g_box["gt_bbox"] = xcycwh_to_x1y1x2y2(g_box["gt_bbox"],width,height)
                   
        
                                                         
        epoch_predicted_bboxs_dict = p_json
        if img_name not in epoch_predicted_bboxs_dict:
                                                                                   
            continue
                                                                  
        cur_epoch_p_boxs = epoch_predicted_bboxs_dict[img_name]["predicted_bboxs"]
        if cur_epoch_p_boxs == None:
                                                                                                          
            continue
                                                                             
        if offset:
            cur_epoch_p_boxs = offset_p_label(cur_epoch_p_boxs)
        matches = search_match_v2(g_boxs,cur_epoch_p_boxs)
        for match in matches:
            matched_g_box = match[0]
            p_box = match[1]
            iou_val = match[2]
            g_box_id = matched_g_box["box_id"]
            gt_box_match[g_box_id] = {"g_box":matched_g_box, "p_box":p_box,"iou_val":iou_val}

    with open(match_save_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_match, f, indent=4)
    print(f"\ngt_box_match is saved in {match_save_path}")
    end_time = time.time()                   
    elapsed_time = end_time - start_time                                     
    hours = int(elapsed_time // 3600)                   
    minutes = int((elapsed_time % 3600) // 60)                     
    seconds = elapsed_time % 60                               
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

def collect_p():
          
    model_weight_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch)
                    
    nc = get_nc_by_datasetname(dataset_name)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    
    model = model_load_weight(model,device,model_weight_path)
    model.eval()

                   
                         
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)                          
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
                 
    dataloader = create_dataloader(data["origin_train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"Total image count:{imgs_num}")
    collectprobs_one_epoch(model,dataloader,conf_thres=0.25,iou_thres=0.65)


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_root_dir = config["exp_data_dir"]
    dataset_name = "voc"                     
    model_name = "yolov7"                      
    epoch = 49
    if model_name == "rtdetr":
        epoch = 99
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")
    error_anno_file_path = get_error_ann_file_path(dataset_name)
    
    collect_p_box_dir = os.path.join(exp_data_root_dir,"collection_process_info",
                                     dataset_name,model_name,"for_baselines",
                                     "collected_predict_boxes_withprobs")
    os.makedirs(collect_p_box_dir,exist_ok=True)
             
                 

    g_json_path = get_collected_gt_box_json_path(dataset_name)
    p_json_path = os.path.join(collect_p_box_dir,
        f"epoch_{epoch}_predicted_bboxs.json"
    )
    offset = (model_name not in ["yolov7","rtdetr"] )                 
    match_save_dir = os.path.join(exp_root_dir,"collection_process_info",
                                  dataset_name,model_name,"for_baselines")
    os.makedirs(match_save_dir,exist_ok=True)
    match_save_path = os.path.join(match_save_dir,"match.json")
           
                                           
