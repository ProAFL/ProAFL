'''
Collect gt_box and p_box information.
'''
import os
import argparse
import torch
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from yolov7.models.yolo import Model
import yaml
import json
from utils.general import colorstr,non_max_suppression,scale_coords,xyxy2xywh
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd

from custom_module.base_data_manager import get_error_train_model_weight_file_path,get_error_ann_file_path
from custom_module.small_utils import read_yaml

def get_nc(dataset_name)->int:
    if dataset_name == "VOC2012":
        nc = 20
    elif dataset_name == "KITTI_8":
        nc = 8
    elif dataset_name == "KITTI":
        nc = 9
    elif dataset_name == "VisDrone":
        nc = 10
    else:
        raise Exception("Invalid dataset parameters")
    return nc


def collect_one_epoch(model,dataloader,epoch, conf_thres=0.25,iou_thres=0.65):
    predicted_box_dict = {}
    predicted_box_id = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0                        
                                      
                                                             
                                                                     
                                      
                            
                                                     
        targets = targets.to(device)
                 
        nb, _, height, width = img.shape                                       
        with torch.no_grad():
            out, train_out = model(img, augment=False)
            lb = []                     
                                                                                                              
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True)                                  
                                  
            for si, pred in enumerate(out):
                if len(pred) == 0:
                               
                    continue
                img_name = paths[si].split("/")[-1]
                predn = pred.clone()
                                         
                                                           
                                             
                                     
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                                                        
                          
                predicted_bbox_list = []
                for *xyxy, conf, cls in predn.tolist():
                    predicted_box = {
                        "predicted_box_id":predicted_box_id,
                        "img_name":img_name,
                        "predicted_cls":int(cls),
                        "conf":conf,
                        "bbox":xyxy
                    }
                    predicted_box_id += 1
                    predicted_bbox_list.append(predicted_box)
                predicted_box_dict[img_name] = {
                        "predicted_bboxs":predicted_bbox_list,
                        "height":shapes[si][0][0],
                        "weight":shapes[si][0][1]
                }
                '''
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    item = {}
                    item["img_name"] = path.stem
                    item["bbox"] = list(xywh)
                    item["conf"] = conf
                    item["predicted_cls"] = int(cls)
                '''

    save_dir = collect_p_box_dir
    os.makedirs(save_dir,exist_ok=True)
    save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(predicted_box_dict, f, indent=4)
    print(f"Data saved at:{save_json_path}")

def collect_predicted_box(conf_thres=0.25,iou_thres=0.65):
                        
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)                          
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
                 
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"Total image count:{imgs_num}")

    for epoch in range(epochs):
                          
                                                                                                                          
        weights_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch)
        state_dict = torch.load(weights_path, map_location=device)                   
                             
        model.load_state_dict(state_dict, strict=True)
                          
        model.eval()
        collect_one_epoch(model,dataloader,epoch,conf_thres,iou_thres)

def collect_gt_box():
    with open(error_annotations_path, 'r') as f:
        error_annotations = json.load(f)
    images_list = error_annotations["images"]
    gt_box_dict  = defaultdict(list)
    box_id = 0
    no_anno_count = 0
    for image in images_list:
        img_id = image["id"]
                                
        annos_of_img = search_annotations_by_img_id(img_id,error_annotations)
        img_name = image["file_name"]
        imge_name_no_ext = img_name.split(".")[0]
                           
        txt_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-yolo","train","labels",f"{imge_name_no_ext}.txt")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            no_anno_count += 1                           
        assert len(lines) == len(annos_of_img), "Annotation mismatch"
        for l_id, line in enumerate(lines):
            box_line = line.split()
            cls = int(box_line[0])
            x_center = float(box_line[1])
            y_center = float(box_line[2])
            width = float(box_line[3])
            height = float(box_line[4])
            fault_type = annos_of_img[l_id]["fault_type"]
            box = {
                "box_id":box_id,
                "img_name":img_name,
                "cls":cls,
                "gt_bbox":[x_center,y_center,width,height],
                "fault_type":fault_type
            }
            box_id += 1
            gt_box_dict[img_name].append(box)
    save_dir = collect_gt_box_dir
    save_json_file_name = "gt_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_dict, f, indent=4)
    print(f"collect_gt_boxtext, Saved at:{save_json_path}")

def search_annotations_by_img_id(img_id,annotations_no_miss):
    annos_of_img = []
    annotations = annotations_no_miss["annotations"]
               
    for anno in annotations:
        if anno["image_id"] == img_id:
            annos_of_img.append(anno)
              
    return annos_of_img

def read_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return 

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    dataset_name = "voc"                     
    nc = get_nc(dataset_name)
    model_name = "yolov7"                     
                
    device = select_device('0')
                       
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)

    pbox_confi_thres = 0.25
    iou_thres = 0.65
    epochs = 50

          
    collect_p_box_dir = os.path.join(exp_data_root,
                                     "collection_process_info",dataset_name,model_name,"collected_predicted_box")
    os.makedirs(collect_p_box_dir,exist_ok=True)
    collect_predicted_box(conf_thres=pbox_confi_thres,iou_thres=iou_thres)

    
    error_annotations_path = get_error_ann_file_path(dataset_name)
    collect_gt_box_dir = os.path.join(exp_data_root,"collection_process_info",dataset_name)
    collect_gt_box()