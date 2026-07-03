
'''
收集p_box信息并与g_box进行匹配是YOLOv7在3个数据集上的其他基线方法的基础
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
from ours.small_utils import read_json


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
    一张图像的gt boxs与predicted boxs匹配函数.
    args:
        gt_box_list: 该图像的所有g_boxes
        predicted_box_list: 该图像在某个轮次的p_boxes
        iou_thre: pbox与gbox的iou只有大于这个阈值才算match
    '''
    # 将预测框按照conf从大到小进行排序
    predicted_box_list.sort(key=lambda x: x["conf"], reverse=True)
    # GT box数量
    G = len(gt_box_list)
    # predicted box数量
    P = len(predicted_box_list)
    # 记录已经匹配成功的GT box
    used_gt = set()
    # 匹配结果容器
    matches = []
    # 所有的cls
    cls_set = set([gt_box["cls"] for gt_box in gt_box_list])
    # 分类下的match，遍历cls_set
    for cls in cls_set:
        # 当前类别的gt boxs
        cur_cls_gt_box_list = [box for box in gt_box_list if box["cls"] == cls]
        # 当前类别的p boxs，（已经按照conf从大到小）
        cur_cls_p_box_list = [box for box in predicted_box_list if box["predicted_cls"] == cls]
        if len(cur_cls_gt_box_list) == 0 or len(cur_cls_p_box_list) == 0:
            continue
        # 获得p_boxs与g_boxs的iou矩阵，shape:(len(p_boxs),len(g_boxs))
        iou_matrix = get_iou_matrix_PG(cur_cls_p_box_list,cur_cls_gt_box_list)
        assert iou_matrix.shape == (len(cur_cls_p_box_list), len(cur_cls_gt_box_list))
        # 每个p_box(行)匹配最好的g_box
        best_gt_box_id_list = iou_matrix.argmax(axis=1)
        # 每个p_box(行)匹配最好的g_box对应的iou值
        best_iou_list = iou_matrix.max(axis=1)
        for r_i,iou_val in enumerate(best_iou_list):
            iou_val = iou_val.item()
            if iou_val < iou_thre:
                # 说明这个p_box与所有的g_box的iou都没达到阈值以上
                continue
            # 这个p_box匹配上了一个g_box
            best_gt_id = best_gt_box_id_list[r_i]
            # 这个g_box被p_box[i]匹配上了
            matched_gt_box = cur_cls_gt_box_list[best_gt_id]
            if matched_gt_box["box_id"] in used_gt:
                # p_box看中的g_box已经被conf 更大的p_box占有了，就不管你（当前p_box[r_i]）了!!
                continue
            used_gt.add(matched_gt_box["box_id"])
            p_box = cur_cls_p_box_list[r_i]
            matches.append((matched_gt_box,p_box,iou_val))
    return matches

def search_match_v2(gt_box_list, predicted_box_list):
    '''
    基于"datactive: For each bounding box in the original annotation set, we identify the
    predicted box with the highest overlap to compute the prediction loss."
    对每个GT box找IoU最高的预测框
    '''
    matches = []
    for gt_box in gt_box_list:
        gt_bbox = gt_box["gt_bbox"]
        gt_cls = gt_box["cls"]

        # 筛选同类别预测框
        same_cls_preds = [p for p in predicted_box_list if p["predicted_cls"] == gt_cls]
        if not same_cls_preds:
            continue

        # 找IoU最高的预测框
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
    # predicted box数量
    for box in p_box_list:
        box["predicted_cls"] -= 1
    return p_box_list

def pretty_print(content,count,col_nums=10):
    print(content, end=' ')
    if count % col_nums == 0:  # 如果计数器是10的倍数
        print()  # 打印换行符

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

    # 1. 归一化 -> 像素
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

    # 2. 中心 -> 左上 / 右下
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

    # 3. 转 int + 裁剪
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]


def model_load_weight(model:nn.Module,device,weight_path:str):
    # 加载模型权重
    
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
    iou_thres用于NMS
    '''
    predicted_box_dict = {}
    predicted_box_id = 0
    # 将数据集喂给model
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        '''
        shapes: list
            长度等于batchsize,该批次图像数据增强前的shape
        paths: list
            长度等于batchsize,该批次图像的文件路径
        '''
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, wid
        with torch.no_grad():
            # out:shape:(bs,anchors*grids,nc+5)
            outs, train_outs = model(imgs, augment=False)  # inference and training outputs
            '''
            outs: list
                长度等于batchsize
                outs[i][0]为该批次第i个样本输出的检测数据(p_box_nums,6),
                outs[i][0]为该批次第i个样本输出的检测数据(p_box_nums,20)
            '''
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            lb = []
            # len(outs) eq batch_size, type(outs):list
            # outs[0][0].shape eq torch.Size([82, 6]): dets xyxy,conf,cls
            # outs[0][1].shape eq torch.Size([82, 20]): probs
            outs = non_max_suppression_with_probs(outs, conf_thres=conf_thres, 
                                                  iou_thres=iou_thres,labels=lb, multi_label=True)
            # 遍历每个input的检测信息
            for loc_i, (pred,prob) in enumerate(outs):
                # loc_i: 这批图像的局部索引
                pred = pred.clone() # 拷贝一份这张图像的预测(det)
                path = Path(paths[loc_i])
                img_name = path.name
                scale_coords(imgs[loc_i].shape[1:], pred[:, :4], shapes[loc_i][0], shapes[loc_i][1])  # native-space pred
                predicted_bbox_list = []
                # 遍历每个p_box的预测信息
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
    print(f"数据保存在:{save_json_path}")



def match(g_json_path,p_json_path,offset):
    # 加载g_box json,bbox 坐标还是归一的xcycwh,no anno的img_name是不存在这个json中的
    g_json = read_json(g_json_path)
    p_json = read_json(p_json_path)

    '''
    收集数据集中g_boxs与每个epoch的p_box的匹配关系
    '''
    start_time = time.time()  # 记录开始时间
    # 收集每个g_box的匹配信息
    # {g_id:{"g_box":g_box,"p_box":p_box}}
    gt_box_match = {}
    # 至少含有一个g_box的img数量
    with_gtboxed_img_count = 0
    # 遍历所有的img name和该图像的g_boxes
    for img_name,g_boxs in g_json.items():
        with_gtboxed_img_count += 1
        pretty_print(img_name,with_gtboxed_img_count,col_nums=10)
        # 图像路径
        image_path = get_img_path_by_img_name(img_name,"yolo")
        # 图像的width,height
        image = Image.open(image_path)
        width, height = image.size
        # 当前图像的g_boxs的bbox格式进行转换
        for g_box in g_boxs:
            g_box["gt_bbox"] = xcycwh_to_x1y1x2y2(g_box["gt_bbox"],width,height)
        # 在该图像下，遍历预测结果
        
        # 当前epoch下的图像->predicted_boxes
        epoch_predicted_bboxs_dict = p_json
        if img_name not in epoch_predicted_bboxs_dict:
            # 图像在当前epoch下没有预测结果,则直接跳过当前epoch
            continue
        # 得到当前epoch该图像的预测p_boxs
        cur_epoch_p_boxs = epoch_predicted_bboxs_dict[img_name]["predicted_bboxs"]
        if cur_epoch_p_boxs == None:
            # 图像在当前epoch下没有预测结果,则直接跳过当前epoch,此处可能是多余
            continue
        # 获得当前图像g_boxs与当前epoch的p_boxs的匹配关系
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
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数
    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")

def collect_p():
    # 加载最后的模型
    model_weight_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch)
    # 加载模型结构
    nc = get_nc_by_datasetname(dataset_name)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    
    model = model_load_weight(model,device,model_weight_path)
    model.eval()

    # 加载error数据集
    # 读取data yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    dataloader = create_dataloader(data["origin_train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"总共图像数量:{imgs_num}")
    collectprobs_one_epoch(model,dataloader,conf_thres=0.25,iou_thres=0.65)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "rtdetr" # YOLOv7|FRCNN|SSD|rtdetr
    epoch = 49
    if model_name == "rtdetr":
        epoch = 99
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")
    error_anno_file_path = get_error_ann_file_path(dataset_name)
    collect_p_box_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                     dataset_name,model_name,"other_baselines",
                                     "predicted_bbox_withprobs")
    os.makedirs(collect_p_box_dir,exist_ok=True)
    # collect_p()

    g_json_path = get_collected_gt_box_json_path(dataset_name)
    p_json_path = os.path.join(collect_p_box_dir,
        f"epoch_{epoch}_predicted_bboxs.json"
    )
    offset = (model_name not in ["YOLOv7","rtdetr"] ) # 是否会对预测标签进行offset(-1)
    match_save_dir = os.path.join(exp_root_dir,"collection_bbox_level",
                                  dataset_name,model_name,"other_baselines")
    os.makedirs(match_save_dir,exist_ok=True)
    match_save_path = os.path.join(match_save_dir,"match.json")
    match(g_json_path,p_json_path,offset)
