

import joblib
import math
from datetime import datetime

import os
import json
from PIL import Image
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import pandas as pd
import topsispy as tp
from helper.base_data_manager import (get_ours_gt_box_metric_path,
                                    get_ours_match_path,get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path
                                    )
from ours.small_utils import get_formatted_time
from helper.data_organization_tools import get_all_img_name

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

def search_match_for_gt(gt_box_list, predicted_box_list, iou_thre=0.5):

    # 仍然按 conf 排序，但不再在 pred 维度贪心
    predicted_box_list = sorted(predicted_box_list, key=lambda x: x["conf"], reverse=True)

    matches = []

    cls_set = set(gt["cls"] for gt in gt_box_list)

    for cls in cls_set:

        cur_gt = [g for g in gt_box_list if g["cls"] == cls]
        cur_pred = [p for p in predicted_box_list if p["predicted_cls"] == cls]

        if not cur_gt or not cur_pred:
            continue

        iou_matrix = get_iou_matrix_GP(cur_gt, cur_pred)

        # 行: gt，列: pred
        assert iou_matrix.shape == (len(cur_gt), len(cur_pred)), "形状不对"

        used_pred = set()

        for g_idx in range(len(cur_gt)):

            # 当前 GT 找最佳预测框
            best_pred_idx = iou_matrix[g_idx].argmax().item()
            best_iou = float(iou_matrix[g_idx, best_pred_idx])

            if best_iou < iou_thre:
                continue

            if best_pred_idx in used_pred:
                continue

            used_pred.add(best_pred_idx)
            matches.append((cur_gt[g_idx], cur_pred[best_pred_idx], best_iou))

    return matches

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
        iou_matrix = get_iou_matrix_PG(cur_cls_gt_box_list,cur_cls_p_box_list)
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

def pretty_print(content,count,col_nums=10):
    print(content, end=' ')
    if count % col_nums == 0:  # 如果计数器是10的倍数
        print()  # 打印换行符

def get_all_epoch(predicted_bboxs_dir):
    _dict = {}
    for epoch in range(epochs):
        epoch_predicted_bboxs_json_path = os.path.join(predicted_bboxs_dir,f"epoch_{epoch}_predicted_bboxs.json")
        with open(epoch_predicted_bboxs_json_path,"r") as f:
            epoch_predicted_bboxs_dict = json.load(f)
        _dict[epoch] = epoch_predicted_bboxs_dict
    return _dict

def get_gt_boxs():
    with open(gt_json_path,"r") as file:
        gt_json = json.load(file)
    return gt_json

def get_img_path_by_img_name(img_name,style):
    if style == "yolo":
        image_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-yolo","origin","train","images",img_name)
    elif style == "coco":
        image_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-coco","train",img_name)
    return image_path

def offset_p_label(p_box_list):
    # predicted box数量
    for box in p_box_list:
        box["predicted_cls"] -= 1
    return p_box_list

def match(match_save_path, offset):
    '''
    收集数据集中g_boxs与每个epoch的p_box的匹配关系
    '''
    start_time = time.time()  # 记录开始时间
    # 加载g_box json, no anno的img_name是不存在这个json中的
    # bbox 坐标还是归一的xcycwh
    gt_json = get_gt_boxs()
    # 收集每个g_box在所有轮次中的匹配信息
    # {g_id:[{"epoch":epoch,"g_box":g_box,"p_box":p_box}]}
    gt_box_match = defaultdict(list)
    
    # 获得epoh -> predicts
    epoch_to_predicts = get_all_epoch(predicted_bboxs_dir)
    # 至少含有一个g_box的img数量
    with_gtboxed_img_count = 0
    # 遍历所有的img name和该图像的g_boxes
    for img_name,g_boxs in gt_json.items():
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
        # 在该图像下，遍历所有的epoch预测结果
        for epoch in range(epochs):
            # 当前epoch下的图像->predicted_boxes
            epoch_predicted_bboxs_dict = epoch_to_predicts[epoch]
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
            # matches = search_match(g_boxs,cur_epoch_p_boxs,iou_thre=0.5)
            matches = search_match_for_gt(g_boxs,cur_epoch_p_boxs,iou_thre=0.5)
            for match in matches:
                matched_g_box = match[0]
                p_box = match[1]
                iou_val = match[2]
                g_box_id = matched_g_box["box_id"]
                gt_box_match[g_box_id].append({"epoch":epoch, "g_box":matched_g_box, "p_box":p_box,"iou_val":iou_val})

    with open(match_save_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_match, f, indent=4)
    print(f"\ngt_box_match is saved in {match_save_path}")
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数
    print(f"耗时: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

def gt_box_metric_collection(match_json_path, metric_save_path):
    '''
    收集所有gt_box在over epoch上的预测conf和iou
    '''
    start_time = time.time()  # 记录开始时间
    # 加载每个g_box在所有轮次中的匹配信息
    # {g_id:[{"epoch":epoch,"g_box":g_box,"p_box":p_box}]}
    with open(match_json_path, 'r') as f:
        gt_box_match = json.load(f)

    # 结果容器
    collect = []
    # g_boxs数量
    count = 0
    for g_box_id in gt_box_match.keys():
        count += 1
        pretty_print(g_box_id,count,col_nums=50)
        # 该g_box的match info [{"epoch":epoch,"g_box":g_box,"p_box":p_box},{}]
        matched_info_over_epoch = gt_box_match[g_box_id]
        instance = {
            "g_box_id":int(g_box_id),
            "conf_list":[],
            "iou_list":[]
        }
        # epoch => gbox,pbox,iou
        temp_dict = {}
        for matched_info in matched_info_over_epoch:
            epoch = matched_info["epoch"]
            temp_dict[epoch] = {
                "g_box":matched_info["g_box"],
                "p_box":matched_info["p_box"],
                "iou_val":matched_info["iou_val"]
            }
        # 遍历所有的epoch
        for epoch in range(epochs):
            matched_info = temp_dict.get(epoch)
            if matched_info is None:
                # 当前epoch，该g_box没有p_box匹配
                conf = 0
                iou = 0
            else:
                conf = matched_info["p_box"]["conf"]
                iou = matched_info["iou_val"]
            instance["conf_list"].append(conf)
            instance["iou_list"].append(iou)
        collect.append(instance)
    
    
    with open(metric_save_path, "w", encoding="utf-8") as f:
        json.dump(collect, f, indent=4)
    print(f"\ncollection_metrics is saved in {metric_save_path}")
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数

    print(f"耗时: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

def get_all_gids():
    g_boxs_dict = get_gt_boxs()
    all_g_box_id_list = []
    for img_name, g_box_list in g_boxs_dict.items():
        for g_box in g_box_list:
            all_g_box_id_list.append(g_box["box_id"])
    return all_g_box_id_list


def get_g_id_to_metric(metric_json_path):
    
    with open(metric_json_path, "r", encoding="utf-8") as f:
        gt_box_metric_collection_list = json.load(f)
    print(f"matched gt_box数量:{len(gt_box_metric_collection_list)}")

    g_box_id_to_metric = {}

    for collection in gt_box_metric_collection_list:
        g_box_id = collection["g_box_id"]
        conf_list = collection["conf_list"]
        iou_list = collection["iou_list"]
        g_box_id_to_metric[g_box_id] = {
            "conf_list":conf_list,
            "iou_list":iou_list,
        }
    return g_box_id_to_metric


def draw_scatter(data_correct,data_cls_fault,data_loc_fault,data_redun_fault,metric_name:str):
    plt.figure(figsize=(8, 5))
    _,epoch_num = data_correct.shape
    epoch_list = range(1, 51)
    for i in range(epoch_num):
        plt.scatter(epoch_list, data_correct[i], s=8, alpha=0.5,c="green")
        # plt.scatter(epoch_list, data_cls_fault[i], s=8, alpha=0.5,c="red")
        # plt.scatter(epoch_list, data_loc_fault[i], s=8, alpha=0.5,c="blue")
        # plt.scatter(epoch_list, data_redun_fault[i], s=8, alpha=0.5,c="black")

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} across epochs")
    plt.grid(True)
    plt.tight_layout()
    save_dir = os.path.join(exp_root_dir,"imgs","correct_vs_error_box",f"{metric_name}_scatter")
    os.makedirs(save_dir,exist_ok=True)
    suffix = get_formatted_time()
    save_path = os.path.join(save_dir,f"{dataset_name}_{model_name}_{suffix}.png")
    plt.savefig(save_path)
    print("correct_vs_error is saved in",save_path)

def draw_line(fault_to_metric_list,metric_name,stat_name):
     # 准备 x 轴 epoch
    no_fault_list = fault_to_metric_list[0][f"{metric_name}_{stat_name}"]
    cls_fault_list = fault_to_metric_list[1][f"{metric_name}_{stat_name}"]
    loc_fault_list = fault_to_metric_list[2][f"{metric_name}_{stat_name}"]
    redundancy_fault_list = fault_to_metric_list[3][f"{metric_name}_{stat_name}"]

    epoch_list = range(1, 51)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_list, no_fault_list, label="no fault", marker='o', color = "green")
    plt.plot(epoch_list, cls_fault_list, label="cls fault", marker='o', color = "red")
    plt.plot(epoch_list, loc_fault_list, label="loc fault", marker='o', color = "blue")
    plt.plot(epoch_list, redundancy_fault_list, label="redundancy fault", marker='o', color = "black")

    plt.xlabel("Epoch")
    plt.ylabel(f"{stat_name} {metric_name.upper()}")
    plt.title(f"{stat_name} {metric_name.upper()} Over 50 Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_dir = os.path.join(exp_root_dir,"imgs","correct_vs_error_box",f"{metric_name}_{stat_name}")
    os.makedirs(save_dir,exist_ok=True)
    suffix = get_formatted_time()
    save_path = os.path.join(save_dir,f"{dataset_name}_{model_name}_{suffix}.png")
    plt.savefig(save_path)
    print("correct_vs_error is saved in",save_path)

def correct_vs_fault(metric_json_path):
    
    g_box_id_to_metric = get_g_id_to_metric(metric_json_path)
    gt_bboxs_dict = get_gt_boxs()
    
    g_box_id_to_info = {}
    for img_name in gt_bboxs_dict.keys():
        g_box_list = gt_bboxs_dict[img_name]
        for g_box in g_box_list:
            g_box_id = g_box["box_id"]
            g_box_id_to_info[g_box_id] = g_box

    # gt_box按照fault_type分组
    '''
    {
        0:[item,]
        1:[item,...]
    }
    '''
    group_dict = defaultdict(list)
    for g_id in g_box_id_to_info.keys():
        if g_id in g_box_id_to_metric:
            metric_dict = g_box_id_to_metric[g_id]
            conf_list = metric_dict["conf_list"]
            iou_list = metric_dict["iou_list"]
            box_info = g_box_id_to_info[g_id]
            fault_type = box_info["fault_type"]
            item = {
                "g_id":g_id,
                "img_name":box_info["img_name"],
                "cls":box_info["cls"],
                "bbox":box_info["gt_bbox"],
                "conf_list":conf_list,
                "iou_list":iou_list,
                "fault_type":fault_type
            }
        else:
            box_info = g_box_id_to_info[g_id]
            fault_type = box_info["fault_type"]
            item = {
                "g_id":g_id,
                "img_name":box_info["img_name"],
                "cls":box_info["cls"],
                "bbox":box_info["gt_bbox"],
                "conf_list":[0]*epochs,
                "iou_list":[0]*epochs,
                "fault_type":fault_type
            }
        group_dict[fault_type].append(item)

    fault_to_metric = {}
    for fault_type in group_dict.keys():
        # 该错误下的g_box收集的confi_list和iou_list
        item_list = group_dict[fault_type]
        conf_list_list = []
        iou_list_list = []
        for item in item_list:
            conf_list_list.append(item["conf_list"])
            iou_list_list.append(item["iou_list"])
            
        conf_2darray = np.array(conf_list_list)
        iou_2darray = np.array(iou_list_list)


        conf_avg = np.mean(conf_2darray,axis = 0)
        iou_avg = np.mean(iou_2darray,axis = 0)
        conf_mid = np.median(conf_2darray,axis = 0)
        iou_mid = np.median(iou_2darray,axis = 0)
        fault_to_metric[fault_type] = {
            "conf_avg":conf_avg.tolist(),
            "iou_avg":iou_avg.tolist(),
            "conf_mid":conf_mid.tolist(),
            "iou_mid":iou_mid.tolist(),
            "conf_2d_array": conf_2darray,
            "iou_2d_array": iou_2darray
        }
    draw_line(fault_to_metric,metric_name="conf", stat_name="avg")
    draw_line(fault_to_metric,metric_name="iou", stat_name="avg")
    

    data_correct = fault_to_metric[0]["conf_2d_array"]
    data_cls_fault = fault_to_metric[1]["conf_2d_array"]
    data_loc_fault = fault_to_metric[2]["conf_2d_array"]
    data_redun_fault = fault_to_metric[3]["conf_2d_array"]
    draw_scatter(data_correct,data_cls_fault,data_loc_fault,data_redun_fault,metric_name="Conf")

    data_correct = fault_to_metric[0]["iou_2d_array"]
    data_cls_fault = fault_to_metric[1]["iou_2d_array"]
    data_loc_fault = fault_to_metric[2]["iou_2d_array"]
    data_redun_fault = fault_to_metric[3]["iou_2d_array"]
    draw_scatter(data_correct,data_cls_fault,data_loc_fault,data_redun_fault,metric_name="IOU")

def add_path_value(d, keys, value):
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur.setdefault(keys[-1], []).append(value)

def find(x,parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b, parent, rank):
    ra, rb = find(a,parent), find(b,parent)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1

def clusing(box_list,thre):
    N = len(box_list)
    parent = list(range(N))
    rank = [0]*N
    for i in range(N):
        for j in range(i+1,N):
            i_bbox  = box_list[i]["bbox"]
            j_bbox = box_list[j]["bbox"]
            if calu_iou(i_bbox,j_bbox) > thre:
                union(i,j,parent,rank)

    clusters = defaultdict(list)
    for i in range(N):
        r = find(i,parent)
        clusters[r].append(i)
    cluster_list = list(clusters.values())
    return cluster_list

def get_epoch_to_matched_p_boxs(gt_match_dict):
    # 每个epoch中所有被匹配上的p_box
    epoch_to_match_info = {}
    # 遍历所有的g_box
    for g_box_id in gt_match_dict.keys():
        # 当前g_box的匹配信息
        match_info_list = gt_match_dict[g_box_id]
        for match_info in match_info_list:
            epoch = match_info["epoch"]
            p_box = match_info["p_box"]
            p_box_id = p_box["predicted_box_id"]
            if epoch in epoch_to_match_info:
                epoch_to_match_info[epoch][p_box_id] = p_box
            else:
                epoch_to_match_info[epoch] = {p_box_id:p_box}
    return epoch_to_match_info

def get_img_to_p_box_list(img_name_to_no_match_p):
    img_to_p_list = defaultdict(list)
    for img_name in img_name_to_no_match_p.keys():
        for epoch in img_name_to_no_match_p[img_name].keys():
            for p_box in img_name_to_no_match_p[img_name][epoch]:
                p_box["epoch"] = epoch
                img_to_p_list[img_name].append(p_box)
    return img_to_p_list

def get_img_to_clusters(img_to_p_box_list,iou_thre=0.8):
    img_to_clusters = defaultdict(list)
    for img_name,p_box_list in img_to_p_box_list.items():
        cluster_list = clusing(p_box_list,thre=iou_thre)
        for cluster in cluster_list:
            cur_cluster_p_box_list = []
            for id in cluster:
                p_box = p_box_list[id]
                cur_cluster_p_box_list.append(p_box)
            img_to_clusters[img_name].append(cur_cluster_p_box_list)
    return img_to_clusters

def stability_pairwise_mean_iou(boxes):
    n = len(boxes)
    if n <= 1:
        return 1.0
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            i_bbox = boxes[i]["bbox"]
            j_bbox =  boxes[j]["bbox"]
            total += calu_iou(i_bbox,j_bbox)
            cnt += 1
    return total / max(1, cnt)

def conf_score(boxes):
    conf_sum = 0
    for p_box in boxes:
        conf_sum += p_box["conf"]
    return conf_sum / len(boxes)

def cls_consis_score(boxes):
    counter = defaultdict(int)
    
    for p_box in boxes:
        counter[p_box["predicted_cls"]] += 1
        
    max_count = -1
    max_cls = -1
    for cls,count in counter.items():
        if count > max_count:
            max_count = count
            max_cls = cls
    return max_count/len(boxes)

def epoch_freq(boxes,last_epoch):
    epoch_cover = set()
    for p_box in boxes:
        epoch_cover.add(p_box["epoch"])
    return len(epoch_cover) / last_epoch


def caclu_cluster_score(cluster,last_epoch): 
    '''簇权分'''
    conf = conf_score(cluster) # [0,1]
    stab = stability_pairwise_mean_iou(cluster) # [0,1]
    cls_consis = cls_consis_score(cluster) # [0,1]
    e_freq = epoch_freq(cluster,last_epoch) # [0,1]
    
    score=0.30*conf+0.20*stab+0.20*cls_consis+0.30*e_freq
    return score

def get_cluster_feaure(cluster,last_epoch):
    conf = conf_score(cluster) # [0,1] 
    stab = stability_pairwise_mean_iou(cluster) # [0,1]
    cls_consis = cls_consis_score(cluster) # [0,1]
    e_freq = epoch_freq(cluster,last_epoch) # [0,1]
    sign = [1,1,1,1]
    feature = [conf,stab,cls_consis,e_freq]
    
    return feature, sign

def get_img_to_topsis_score(img_to_clusters,last_epoch):
    clusters_feature = []
    feature_sign = []
    img_name_to_c_ids = defaultdict(list)
    c_id = 0
    for img_name,clusters in img_to_clusters.items():
        for cluster in clusters:
            feature,sign = get_cluster_feaure(cluster,last_epoch)
            clusters_feature.append(feature)
            feature_sign = sign
            img_name_to_c_ids[img_name].append(c_id)
            c_id += 1
    data_array = np.array(clusters_feature)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(feature_sign), "数据有误"
    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, feature_sign)
    # 从大到小排序并返回索引
    # sorted_cluster_id = np.argsort(score_array)[::-1]
    img_name_to_max_score = {}
    for img_name,c_ids in img_name_to_c_ids.items():
        max_score = 0
        for c_id in c_ids:
            if score_array[c_id] > max_score:
                max_score = score_array[c_id]
        img_name_to_max_score[img_name] = max_score
    return img_name_to_max_score

def sort_cluster_by_weight_score(img_to_clusters,last_epoch):
    cluster_list = []
    for img_name,clusters in img_to_clusters.items():
        for cluster in clusters:
            s = caclu_cluster_score(cluster,last_epoch)
            cluster_list.append({
                "cluster":cluster,
                "img_name":img_name,
                "score":s
            })
    sorted_cluster_list = sorted(cluster_list, key=lambda x: x['score'], reverse=True)
    return sorted_cluster_list

def get_img_epoch_to_unmatched_p_boxs(epoch_to_matched_p_boxs,last_epoch,conf_threshold=0.6):
    '''
    epoch_to_matched_p_boxs:记录了每个epoch对应的被gt_box匹配到的p_box
    last_epoch:从倒数第几个轮次开始记录
    threshold:只有大于这个阈值的p_box才需要被考虑匹不匹配的问题
    '''
    img_name_to_no_match_p = {}
    # 只关心最后5个epoch的预测情况
    for epoch in range(epochs-last_epoch,epochs):
        # 加载当前epoch的预测结果
        predicted_epoch_json_path = os.path.join(predicted_bboxs_dir, f"epoch_{epoch}_predicted_bboxs.json")
        with open(predicted_epoch_json_path,mode="r") as f:
            predicted_epoch_dict = json.load(f)
        # 统计所有图像中没被gt_box匹配到的高置信度预测box
        for img_name in predicted_epoch_dict.keys():
            # img_name在该epoch下的所有预测框
            p_box_list = predicted_epoch_dict[img_name]["predicted_bboxs"]
            # 遍历预测框
            for p_box in p_box_list:
                p_id = p_box["predicted_box_id"]
                if p_id not in epoch_to_matched_p_boxs[epoch] and p_box["conf"] > conf_threshold:
                    add_path_value(img_name_to_no_match_p,keys=[img_name,epoch],value=p_box)
    return img_name_to_no_match_p

def sort_img(sorted_clusters):
    '''
    sorted_clusters:根据簇得分排序后的簇
    '''
    img_name_to_score = defaultdict(float)
    for cluster in sorted_clusters:
        img_name = cluster['img_name']
        score = cluster["score"]
        if score > img_name_to_score[img_name]:
            img_name_to_score[img_name] = score
    # [(img_name,max_cluster_score),...]
    sorted_imgs = sorted(img_name_to_score.items(), key=lambda item: item[1], reverse=True)
    return sorted_imgs

def filter_imgs(sorted_imgs,threshold_score=0.6):
    filterd_imgs = []
    for img_name,score in sorted_imgs:
        if score > threshold_score:
            filterd_imgs.append(img_name)
    return filterd_imgs

def get_fault_imgs_by_type(fault_type_list):
    '''
    fault_type:0(no)|1(cls)|2(loc)|3(red)|4(mis)
    '''
    img_id_to_img_name = {}
    for image in annos_with_miss_json["images"]:
        img_id_to_img_name[image["id"]] = image["file_name"]

    annos = annos_with_miss_json['annotations']
    fault_img_name_set = set()
    for anno in annos:
        if anno["fault_type"] in fault_type_list:
            img_name = img_id_to_img_name[anno["image_id"]]
            fault_img_name_set.add(img_name)
    return fault_img_name_set



def misimg_detect_by_topsis(match_json_path, last_epoch=5):
    all_img_name_list = get_all_img_name(imgs_dir)
    # 读取所有gt_box的匹配信息
    with open(match_json_path,"r") as f:
        gt_match_dict = json.load(f)
    epoch_to_matched_p_boxs = get_epoch_to_matched_p_boxs(gt_match_dict)

    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    # {img__name:{epoch:[] # no_matched_p_boxs }}
    img_name_to_no_match_p = get_img_epoch_to_unmatched_p_boxs(epoch_to_matched_p_boxs,last_epoch,conf_threshold=0.6)

    # 把每个epoch未匹配到的p_box拉平
    # {img__name:[] # no_matched_p_boxs}
    img_to_p_box_list  = get_img_to_p_box_list(img_name_to_no_match_p)

    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters(img_to_p_box_list,iou_thre=0.6)
    img_name_to_topsis_score = get_img_to_topsis_score(img_to_clusters,last_epoch)
    
    no_clusters_image_name_set = set(all_img_name_list) - set(img_name_to_topsis_score.keys())
    for img_name in sorted(no_clusters_image_name_set):
        img_name_to_topsis_score[img_name] = 0
    return img_name_to_topsis_score

def misimg_detect_by_weight_score(match_json_path, last_epoch=5):
    all_img_name_list = get_all_img_name(imgs_dir)
    # 读取所有gt_box的匹配信息
    
    with open(match_json_path,mode="r") as f:
        gt_match_dict = json.load(f)
    epoch_to_matched_p_boxs = get_epoch_to_matched_p_boxs(gt_match_dict)

    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    # {img__name:{epoch:[] # no_matched_p_boxs }}
    img_name_to_no_match_p = get_img_epoch_to_unmatched_p_boxs(epoch_to_matched_p_boxs,last_epoch,conf_threshold=0.6)

    # 把每个epoch未匹配到的p_box拉平
    # {img__name:[] # no_matched_p_boxs}
    img_to_p_box_list  = get_img_to_p_box_list(img_name_to_no_match_p)

    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters(img_to_p_box_list,iou_thre=0.6)
    # 对簇进行打分且排序
    # [{"cluster":c,"img_name":img_name,"score":s},..,]
    get_img_to_topsis_score(img_to_clusters,last_epoch)
    sorted_clusters = sort_cluster_by_weight_score(img_to_clusters,last_epoch)
    sorted_img_name_list = sort_img(sorted_clusters)
    detected_mis_img_name_list = filter_imgs(sorted_img_name_list,threshold_score=-1)

    ranked_img_list = []
    for detected_img_name in detected_mis_img_name_list:
        ranked_img_list.append(detected_img_name)

    for img_name in all_img_name_list:
        if img_name not in ranked_img_list:
            ranked_img_list.append(img_name)
    
    return ranked_img_list,detected_mis_img_name_list


    # save_dir = os.path.join(exp_root_dir,"Ours",dataset_name,model_name)
    # save_file_name = "detected_mis_imgs.joblib"
    # save_path = os.path.join(save_dir,save_file_name)
    # joblib.dump(detected_mis_img_name_list,save_path)

    # print(f"detected_mis_imgs被保存在:{save_path}")
    # fault_img_name_set = get_fault_imgs_by_type(fault_type_list=[2,4])


    # fn = len(fault_img_name_set - set(filterd_img_name_list))
    # tp = len(fault_img_name_set & set(filterd_img_name_list))
    # fp = len(set(filterd_img_name_list) - fault_img_name_set)

    # precision = tp / (tp+fp)
    # recall = tp / (tp+fn)
    # f1 = 2*precision*recall / (precision+recall)

    # print("precision:",precision)
    # print("recall:",recall)
    # print("f1:",f1)

def gt_box_features_build(metric_json_path):
    g_box_id_to_metric = get_g_id_to_metric(metric_json_path)
    
    g_id_to_features = {}
    for g_id in g_box_id_to_metric.keys():
        conf_list = g_box_id_to_metric[g_id]["conf_list"]
        iou_list = g_box_id_to_metric[g_id]["iou_list"]
        epochs = len(conf_list)
        W_e = int(0.2*epochs)
        W_l = int(0.2*epochs)
        # 早期置信度均值，越小越可疑
        early_conf_mean = np.mean(conf_list[0:W_e])
        # 后期置信度均值，越小越可疑
        lastly_conf_mean = np.mean(conf_list[-W_l:])
        # 早期iou均值，越小越可疑
        early_iou_mean = np.mean(iou_list[0:W_e])
        # 后期iou均值，越小越可疑
        lastly_iou_mean = np.mean(iou_list[-W_l:])

        # 全局均值，越小越可疑
        conf_mean = np.mean(conf_list)
        iou_mean = np.mean(iou_list)

        conf_threshold = 0.5*lastly_conf_mean
        iou_threshold = 0.5*lastly_iou_mean

        min_e_conf = 0
        min_e_iou = 0
        for e in range(epochs):
            if conf_list[e] > conf_threshold:
                min_e_conf = e
                break
        for e in range(epochs):
            if iou_list[e] > iou_threshold:
                min_e_iou = e
                break
        # 起量延迟（显式刻画“涨得晚”）
        # 越大越可疑
        D_conf = min_e_conf / epochs
        D_iou = min_e_iou / epochs

        g_id_to_features[g_id] = {
            "early_conf_mean":early_conf_mean,
            "early_iou_mean":early_iou_mean,
            "lastly_conf_mean":lastly_conf_mean,
            "lastly_iou_mean":lastly_iou_mean,
            "conf_mean":conf_mean,
            "iou_mean":iou_mean,
            "D_conf":D_conf,
            "D_iou":D_iou,
        }
    feature_name_to_sign = {
        "early_conf_mean":-1, # 越小越可疑
        "early_iou_mean":-1,
        "lastly_conf_mean":-1,
        "lastly_iou_mean":-1,
        "conf_mean":-1,
        "iou_mean":-1,
        "D_conf":1,
        "D_iou":1
    }
    all_gids =  get_all_gids()
    print(f"all gbox数量:{len(all_gids)}")
    print(f"matched gbox数量:{len(g_id_to_features)}")
    
    for g_id in all_gids:
        if g_id not in g_id_to_features:
            g_id_to_features[g_id] = {
                "early_conf_mean":0,
                "early_iou_mean":0,
                "lastly_conf_mean":0,
                "lastly_iou_mean":0,
                "conf_mean":0,
                "iou_mean":0,
                "D_conf":1,
                "D_iou":1, 
            }
    return g_id_to_features,feature_name_to_sign

def rank_gid(g_id_to_features,feature_name_to_sign):
    '''
    g_id_to_features:{g_id:{attr:(value,flag),},}
    '''
    g_id_list = list(g_id_to_features.keys())
    g_id_list.sort() # 升序
    data = []
    id_to_gid ={}
    id = 0
    feature_name_list = [
        "early_conf_mean",
        "early_iou_mean",
        "lastly_conf_mean",
        "lastly_iou_mean",
        "conf_mean",
        "iou_mean",
        "D_conf",
        "D_iou",
    ]
    sign_list = []
    for feature_name in feature_name_list:
        sign_list.append(feature_name_to_sign[feature_name])

    for g_id in g_id_list:
        feature_dict = g_id_to_features[g_id]
        feature_list = [feature_dict[name] for name in feature_name_list]
        data.append(feature_list)
        id_to_gid[id]= g_id
        id += 1
    for id,gid in id_to_gid.items():
        assert id == gid, "数据有误"
    assert len(sign_list) > 0, "数据有误"
    data_array = np.array(data)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(sign_list), "数据有误"
    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, sign_list)
    # 从大到小排序并返回索引
    sorted_gt_id = np.argsort(score_array, kind="mergesort")[::-1]

    ranked_gid_list = [int(g_id) for g_id in sorted_gt_id]
    topsis_score_list = []
    for gid in ranked_gid_list:
        topsis_score_list.append(score_array[gid])
    return ranked_gid_list, topsis_score_list

def total_rank_by_topsis_score(ranked_gid_list,topsis_score_list,img_name_to_topsis_score,alpha:float):
    
    _map = {}
    for gid,score in zip(ranked_gid_list,topsis_score_list):
        _map[gid] = score
    for img_name,score in img_name_to_topsis_score.items():
        _map[img_name] = alpha*score

    sorted_items = sorted(
        _map.items(),
        key=lambda kv: (-float(kv[1]), str(kv[0]))  # 先按分数降序，同分按ID字典序
    )
    rank_res = [k for k, _ in sorted_items]
    return rank_res

def total_rank_by_loc(ranked_gid_list,ranked_img_list):
    gid_num = len(ranked_gid_list)
    img_num = len(ranked_img_list)
    all_img_name_list = get_all_img_name(imgs_dir)
    all_gid_list = get_all_gids()
    assert gid_num == len(all_gid_list), "gid数量错误"
    assert img_num == len(all_img_name_list), "img数量错误"

    data_list = []
    for rank,gid in enumerate(ranked_gid_list):
        score = rank / gid_num
        data_list.append((gid,score))
    for rank,img_name in enumerate(ranked_img_list):
        score = rank / img_num
        data_list.append((img_name,score))
    # 根据二元组的第二个元素进行排序
    data_list.sort(key=lambda x: x[1])
    res = [ ID for ID,score in data_list]
    assert len(res) == (gid_num+img_num)
    return res

def compute_apfd(fault_set:set, rankded_list):
    """
    list_A: set/list, 真实错误图像路径
    list_B: list, 按可疑度排序的图像路径
    """
    n = len(rankded_list)
    
    TF_positions = []

    # 遍历 list_B 找到真实错误的位置
    for idx, ID in enumerate(rankded_list, start=1):  # 从1开始计数
        if ID in fault_set:
            TF_positions.append(idx)

    m = len(fault_set)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    return apfd

def eval_apfd(rank_res):
    g_box_dict = get_gt_boxs()
    fault_g_id_set = set()
    for img_name,g_box_list in g_box_dict.items():
        for g_box in g_box_list:
            g_id = g_box["box_id"]
            fault_type = g_box["fault_type"]
            if fault_type != 0:
                fault_g_id_set.add(g_id)
    img_id_to_img_name = {}
    for image in annos_with_miss_json["images"]:
        img_id_to_img_name[image["id"]] = image["file_name"]

    annos = annos_with_miss_json['annotations']
    mis_img_name_set = set()
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = img_id_to_img_name[anno["image_id"]]
            mis_img_name_set.add(img_name)
    fault_set = fault_g_id_set.union(mis_img_name_set)
    apfd = compute_apfd(fault_set, rank_res)
    apfd = round(apfd,4)
    print(f"APFD:{apfd}")




if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    '''
    mode: 0:全流程;1:match操作;2:metric操作
    '''
    start_time = get_formatted_time()

    mode = 0
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "SSD" # YOLOv7|FRCNN|SSD
    label_offset = False # 是否包含bg class idx,包含的话 1 label => 0 classname
    epochs = 200 # 200,新SSD设置成200
    save_dir = os.path.join(exp_root_dir,"collection_bbox_level",
                                    dataset_name,model_name)
    imgs_dir = os.path.join(exp_root_dir,"datasets",dataset_name,"images","train")
    
    os.makedirs(save_dir,exist_ok=True)
    settings_info = {
        "mode":0,
        "start_time":start_time,
        "dataset_name":dataset_name,
        "model_name":model_name,
        "label_offset": label_offset,
        "epochs":epochs,
        "save_dir":save_dir
    }

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    '''
    弃用(2026/05/04 09:15)
    predicted_bboxs_dir = os.path.join(exp_root_dir,"collection_indicator_bbox_level",
                                        dataset_name,model_name,"collected_predicted_box","v2")
    '''
    predicted_bboxs_dir = os.path.join(save_dir,"predicted_bbox")

    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)

    with open(annos_with_miss_json_path, 'r') as f:
        annos_with_miss_json = json.load(f)

    if mode == 0 or mode == 1:
        # match 操作
        print("matching...")
        # 弃用(2026/05/04 09:15) match_json_save_path = os.path.join(match_save_dir,"match_v21.json")
        match_json_save_path = os.path.join(save_dir,"match.json")
        match(match_json_save_path,offset=label_offset)
        print("match END")
    if mode == 0 or mode == 2:
        if mode == 2:
            match_json_save_path = os.path.join(save_dir,"match.json")
        # metirc 操作
        print("metirc...")
        collection_metric_save_dir = os.path.join(exp_root_dir,"collection_bbox_level",
                                                dataset_name,model_name)
        os.makedirs(collection_metric_save_dir,exist_ok=True)
        # 弃用(2026/05/04 09:15) metric_save_path = os.path.join(collection_metric_save_dir,"collection_metrics_v21.json")
        metric_save_path = os.path.join(save_dir,"metrics.json")
        gt_box_metric_collection(match_json_save_path,metric_save_path)
        print("metric END")
    if mode == 0 or mode == 3:
        # 排序
        print("Ranking...")
        if mode == 3:
            match_json_save_path = os.path.join(save_dir,"match.json") # mode=1
            metric_save_path = os.path.join(save_dir,"metrics.json") # mode=2
        g_id_to_features,feature_name_to_sign = gt_box_features_build(metric_save_path)
        print("gid ranking...")
        ranked_gid_list, topsis_score_list = rank_gid(g_id_to_features,feature_name_to_sign)
        print("image ranking...")
        img_name_to_topsis_score = misimg_detect_by_topsis(match_json_save_path)

        alpha = 1.5 # img_score = alpha * img_score, default=1.5
        rank_res = total_rank_by_topsis_score(ranked_gid_list,topsis_score_list,
                                            img_name_to_topsis_score,alpha)
        print(f"rank的长度:{len(rank_res)}")
        eval_apfd(rank_res)
        # 保存排序结果
        rank_res_save_file_name = "rank.joblib"
        rank_res_save_path = os.path.join(save_dir, rank_res_save_file_name)
        joblib.dump(rank_res, rank_res_save_path)
        print(f"rank is saved in {rank_res_save_path}")
    # 3: 绘制metric line
    # correct_vs_fault(metric_save_path)
    

    '''
    # 权排序
    ranked_img_list,detected_mis_img_name_list = misimg_detect_by_weight_score(match_json_save_path)
    # gid与img name两个序列合并成一个序列
    rank_res = total_rank_by_loc(ranked_gid_list,ranked_img_list)
    '''
