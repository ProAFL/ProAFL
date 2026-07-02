
'''
收集的标注框和所有轮次的预测框的匹配和度量(confi/iou)统计
'''
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from ours.small_utils import get_cost_time,get_formatted_time
import numpy as np
from tqdm import tqdm
from PIL import Image

from ours.base_data_manager import (
    exp_data_root_dir,
    get_collected_gt_box_json_path
    )

def get_json(json_path:str) -> dict:
    with open(json_path,"r") as file:
        _json = json.load(file)
    return _json

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
    if count % col_nums == 0:  # 如果计数器是10的倍数
        print()  # 打印换行符

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

def offset_p_label(p_box_list):
    # predicted box数量
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
    一张图像的gt boxs与predicted boxs匹配函数.
    args:
        gt_box_list: 该图像的所有g_boxes x1y1x2y2
        predicted_box_list: 该图像在某个轮次的p_boxes
        iou_thre: pbox与gbox的iou只有大于这个阈值才算match
    '''
    # 将预测框按照conf从大到小进行排序
    predicted_box_list.sort(key=lambda x: x["conf"], reverse=True)
    # predicted box数量
    P = len(predicted_box_list)
    # GT box数量
    G = len(gt_box_list)
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
        iou_matrix = get_iou_matrix_PG(cur_cls_p_box_list,cur_cls_gt_box_list,)
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


def match(gt_json:dict, epoch_to_p_boxs:dict, offset:bool, save_path):
    '''
    收集数据集中g_boxs与每个epoch的p_box的匹配关系
    参数
    ---
    gt_json : dict
        数据格式：
        {
            img_name:[g_box1,...], 
            ...
        }
        注意：gbox其中的bbox坐标是归一的xcycwh, 并且没有任何标注的img的img name是不会出现在gt_json中的
    
    '''
    start_time = time.time()  # 记录开始时间
    # 收集每个g_box在所有轮次中的匹配信息
    # {g_id:[{"epoch":epoch,"g_box":g_box,"p_box":p_box}]}
    matched_gbox = defaultdict(list)
    total_gbox_num = 0
    # 遍历所有的img name和该图像的g_boxes
    for img_name,g_boxs in tqdm(gt_json.items(),total=len(gt_json)):
        # 图像路径
        image_path = get_img_path_by_img_name(img_name,"yolo")
        # 图像的width,height
        image = Image.open(image_path)
        width, height = image.size
        # 当前图像的g_boxs的bbox格式进行转换
        for g_box in g_boxs:
            total_gbox_num += 1
            g_box["gt_bbox"] = xcycwh_to_x1y1x2y2(g_box["gt_bbox"],width,height)
        # 在该图像下，遍历所有的epoch预测结果
        for epoch in range(epochs):
            # 当前epoch下的图像->predicted_boxes
            p_boxs_dict = epoch_to_p_boxs[epoch]
            if img_name not in p_boxs_dict:
                # 图像在当前epoch下没有预测结果,则直接跳过当前epoch
                continue
            # 得到当前epoch该图像的预测p_boxs
            cur_epoch_p_boxs = p_boxs_dict[img_name]["predicted_bboxs"]
            if cur_epoch_p_boxs == None:
                # 图像在当前epoch下没有预测结果,则直接跳过当前epoch,此处可能是多余
                continue
            # 获得当前图像g_boxs与当前epoch的p_boxs的匹配关系
            if offset:
                cur_epoch_p_boxs = offset_p_label(cur_epoch_p_boxs)
            # matches = gt_best_match(cur_epoch_p_boxs,g_boxs)
            matches = search_match_PG(cur_epoch_p_boxs,g_boxs,iou_thre=0.5)
            # matches = search_match_GP(g_boxs,cur_epoch_p_boxs,iou_thre=0.5)
            for match in matches:
                matched_g_box = match[0]
                p_box = match[1]
                iou_val = match[2]
                g_box_id = matched_g_box["box_id"]
                matched_gbox[g_box_id].append({"epoch":epoch, "g_box":matched_g_box, "p_box":p_box,"iou_val":iou_val})
    print(f"带有标注的img数量: {len(gt_json)}")
    print(f"总共的gbox数量: {total_gbox_num}")
    print(f"matched gbox数量: {len(matched_gbox)}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(matched_gbox, f, indent=4)
    print(f"matched gbox is saved in {save_path}")
    end_time = time.time()  # 记录结束时间
    elapsed_timestamp = end_time - start_time  # 计算运行时间（秒）
    cost_time = get_cost_time(elapsed_timestamp)
    print(f"运行时间:{cost_time}")
    print(f"实验结束时刻:{get_formatted_time()}")
    return matched_gbox

def collect_metrics_for_gboxs(match_json:dict, save_path:str):
    '''
    收集所有gt_box在 cross epoch上的预测conf和iou

    参数：
    ---
    match_json: dict
        数据格式：
        {
            g_id:[{"epoch":epoch,"g_box":g_box,"p_box":p_box},...],
            ...
        }
    '''
    start_time = time.time()  # 记录开始时间
    # 结果容器
    collect = []
    for g_box_id in tqdm(match_json.keys(),total=len(match_json)):
        # 该g_box的match info [{"epoch":epoch,"g_box":g_box,"p_box":p_box},{}]
        matched_info_over_epoch = match_json[g_box_id]
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

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collect, f, indent=4)
    print(f"\ncollection_metrics is saved in {save_path}")
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数

    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"实验结束时刻: {now_str}")

def main():
    mode = 0
    print("装载gt框json数据...")
    gt_json = get_json(gt_json_path)
    print("装载每个epoch的预测框json数据...")
    epoch_to_p_boxs = get_epoch_to_pboxs(predicted_bboxs_dir)
    offset = False
    if model_name not in ["YOLOv7","rtdetr"]: # FRCNN是需要offset的0（bg）, 1->0..,10->9
        offset = True
    if mode == 0 or mode == 1:
        print("match START")
        # PG match
        save_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,"match.json")
        matched_gbox = match(gt_json, epoch_to_p_boxs, offset, save_path)
        print("match END")
    if mode == 0 or mode == 2:
        print("metrics START")
        # collect metrics
        if mode == 2:
            match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,"match.json")
            with open(match_json_path, "r") as f:
                matched_gbox = json.load(f)
        save_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,"metrics.json")
        collect_metrics_for_gboxs(matched_gbox, save_path)
        print("metrics END")

if __name__ == "__main__":
    pid = os.getpid()
    print(f"pid:{pid}")
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "rtdetr" # YOLOv7|FRCNN|SSD|rtdetr
    epochs = 50
    if model_name == "rtdetr":
        epochs = 100
    # 需要的数据文件路径
    # gt box数据
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    # 预测 box数据
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                       dataset_name,model_name,
                                       "predicted_bbox")
    main()