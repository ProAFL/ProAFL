
'''
我们方法序的详细分析脚本
'''
import os
import json
import joblib
from datetime import datetime
from collections import defaultdict
import numpy as np
import topsispy as tp
import matplotlib.pyplot as plt
from common import *
from ours.base_data_manager import (exp_data_root_dir,
                                    get_ours_gt_box_metric_path,
                                    get_ours_match_path,get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path,
                                    get_error_ann_file_path
                                    )

from ours.data_organization_tools import (get_all_errored_g_box_id_set,get_all_miss_error_img_name_set,
                                          get_img_name_to_missed_annids,get_all_error_annoids,get_annoId_to_anno,
                                          conver_ours_rank,get_all_error_idd_set,get_all_error_imgset)
from ours.small_utils import read_json
from ours.repair.repair_analyse import count_repair_rate

def draw_violin(data,save_path):
    # 假设 data 已经存在，形状为 (10000, 50)
    # data = np.random.rand(10000, 50)  # 示例

    num_epochs = data.shape[1]
    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(16, 6))

    # violinplot 期望的输入是“每个分布一列”的序列
    # 这里每个 epoch 对应 data[:, i]
    violin_parts = plt.violinplot(
        [data[:, i] for i in range(num_epochs)],
        positions=epochs,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Confidence score")
    plt.title("Confidence distribution per epoch (violin plot)")

    # x 轴刻度：每个 epoch 一个刻度，或者隔几个画一个防止太密
    plt.xticks(epochs)  # 如果太密，可以改成 plt.xticks(epochs[::5])

    # 如果你的 confidence 在 [0, 1] 区间，可以固定 y 轴范围
    plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    print(f"图像保存在：{save_path}")

def draw_box(data,save_path):
    num_epochs = data.shape[1]
    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(16, 6))

    # 箱线图：每列是一个 epoch 上的 10000 个 confidence
    # matplotlib 的 boxplot 默认是“每一列一个箱子”，所以直接转置即可
    box = plt.boxplot(
        data,                # 形状 (10000, 50)，每列一个箱线
        positions=epochs,    # 对应的 epoch 位置
        showfliers=False,    # 是否显示离群点，可按需改 True/False
        patch_artist=True    # 允许填充颜色，方便后面美化
    )

    plt.xlabel("Epoch")
    plt.ylabel("Confidence score")
    plt.title("Confidence distribution per epoch (boxplot)")

    # x 轴刻度：如果太密，可以改成 epochs[::5]
    plt.xticks(epochs)

    # 如果 confidence 在 [0, 1]，固定 y 轴范围
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    print(f"图像保存在：{save_path}")

def draw_hot(data,save_path):
    num_epochs = data.shape[1]
    epochs = np.arange(1, num_epochs + 1)

    # 1. 为 confidence 定义若干 bin
    num_bins = 10  # 纵向分成 50 个格子，可按需调整
    conf_min, conf_max = 0.0, 1.0  # 如果不是 [0,1]，可以改为 data.min(), data.max()
    bins = np.linspace(conf_min, conf_max, num_bins + 1)

    # 2. 统计每个 epoch 在各个 bin 内的样本数
    #    density[i, j] 表示第 j 个 epoch 在第 i 个 bin 的样本数量
    density = np.zeros((num_bins, num_epochs), dtype=float)

    for j in range(num_epochs):
        hist, _ = np.histogram(data[:, j], bins=bins)
        density[:, j] = hist

    # 也可以转成概率密度（按每个 epoch 共 10000 个样本归一化）
    density = density / density.sum(axis=0, keepdims=True)

    # 3. 画成热力图
    plt.figure(figsize=(14, 6))

    # imshow 的 extent: [x_min, x_max, y_min, y_max]
    # 注意 origin='lower'，让低 confidence 在下，高的在上
    im = plt.imshow(
        density,
        aspect='auto',
        origin='lower',
        extent=[1, num_epochs, conf_min, conf_max],
    )

    plt.colorbar(im, label="Density")

    plt.xlabel("Epoch")
    plt.ylabel("Confidence")
    plt.title("Confidence density over epochs")

    # 可选：只在 x 轴上每隔几个 epoch 标一下，防止太挤
    plt.xticks(np.linspace(1, num_epochs, 11, dtype=int))  # 例如 1,6,11,...,50

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    print(f"图像保存在：{save_path}")

def get_all_gids(gt_json:dict) -> list[int]:
    '''
    得到所有的g_box_id_list
    
    参数
    ----
    gt_json : dict
        数据格式：
        {
            image_name:[g_box_1,g_box_2],
            ...
        }
    返回
    ---
    all_g_box_id_list : list[int]
        提取出的所有的g_box_id_list
    '''
    all_g_box_id_list = []
    for img_name, g_box_list in gt_json.items():
        for g_box in g_box_list:
            all_g_box_id_list.append(g_box["box_id"])
    return all_g_box_id_list


def get_g_id_to_metric(metric_json_path):
    '''
    提供每个gid对应的metric(conf_list和iou_list)
    '''
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

def get_formatted_time():
    """返回当前时间的格式化字符串（YYYY-MM-DD HH:MM:SS）"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")

def add_path_value(d:dict, keys:list, value):
    '''
    多层级字典，最后指向[]
    '''
    cur = d
    # 遍历所有层级的key
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

def clusing(box_list:list,iou_thre:float=0.6) -> list[list[int]]:
    '''
    使用并查集对这些box进行分簇
    box_list : list
        数据结构示例:
        [box_1,box_2,...]
    iou_thre : float, default = 0.6
        元素归并的条件阈值
    返回:
    ---
    cluster_list : list[list[int]]
        数据结构示例
        [[loc_idx1,loc_idx3,...],...]
    '''
    N = len(box_list)
    parent = list(range(N))
    rank = [0]*N
    for i in range(N):
        for j in range(i+1,N):
            i_bbox  = box_list[i]["bbox"]
            j_bbox = box_list[j]["bbox"]
            if calu_iou(i_bbox,j_bbox) > iou_thre:
                union(i,j,parent,rank)
    clusters = defaultdict(list)
    for i in range(N):
        r = find(i,parent)
        clusters[r].append(i)
    cluster_list = list(clusters.values())
    return cluster_list

def get_img_name_to_no_matched_p_boxs(img_name_to_epoch_to_no_match_p_boxs:dict) -> dict:
    '''
    展平img_name to no matched_p_boxs
    '''
    img_to_p_boxs = defaultdict(list)
    for img_name in img_name_to_epoch_to_no_match_p_boxs.keys():
        for epoch in img_name_to_epoch_to_no_match_p_boxs[img_name].keys():
            for p_box in img_name_to_epoch_to_no_match_p_boxs[img_name][epoch]:
                p_box["epoch"] = epoch
                img_to_p_boxs[img_name].append(p_box)
    return img_to_p_boxs

def get_img_to_clusters(img_to_p_boxs:dict,iou_thre:float=0.6):
    '''
    构建img_name -> p_box的分簇
    参数
    ---
    img_to_p_boxs : dict
        数据格式：
        {image_name:[p_box1,p_box2]}
    iou_thre : float,default=0.6
        分簇的iou条件
    
    返回：
    ---
    img_to_clusters : dict
        数据格式示例：
        {
            img_name:[[p_box1,p_box2,...],...],
            ...
        }
    '''
    img_to_clusters = defaultdict(list)
    for img_name,p_box_list in img_to_p_boxs.items():
        # [[loc_id1,loc_id2...],...]
        cluster_list = clusing(p_box_list,iou_thre)
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

def get_img_to_topsis_score(img_to_clusters:dict,last_epoch:int):
    '''
    获得img_name -> topsis score
    
    参数：
    ---
    img_to_clusters : dict
        数据格式
        {img_name:[[pbox1,pbox3],...]}
    last_epoch : int
    '''
    # 存放每个cluster的features
    clusters_features = []
    # 存放features对应的政府指标符号
    features_signs = []

    # 存放每个img对应的cluster的idx list
    img_name_to_cluster_ids = defaultdict(list)

    # 追踪每个cluster的索引
    cluster_idx = 0
    # 遍历每个图像与其对应的簇群
    for img_name,clusters in img_to_clusters.items():
        # 遍历该图像的所有簇
        for cluster in clusters:
            # 得到该簇的特征数据和特征对应的正负符号
            features,signs = get_cluster_feaure(cluster,last_epoch)
            clusters_features.append(features)
            features_signs = signs
            img_name_to_cluster_ids[img_name].append(cluster_idx)
            cluster_idx += 1

    # 构建特征数据集
    data_array = np.array(clusters_features)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(features_signs), "数据有误"
    weights = np.ones(n_features) / n_features
    # 基于topsis获得clusters的score
    best_cluster_id, score_array = tp.topsis(data_array, weights, features_signs)
    score_array = np.nan_to_num(score_array, nan=0.0, posinf=1.0, neginf=0.0)
    # 从大到小排序并返回索引
    # sorted_cluster_id = np.argsort(score_array)[::-1]
    # 将img_name中得分最高的cluster的得分作为该img的score
    img_name_to_max_score = {}
    for img_name,cluster_ids in img_name_to_cluster_ids.items():
        max_score = 0
        for cluster_id in cluster_ids:
            if score_array[cluster_id] > max_score:
                max_score = score_array[cluster_id]
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

def get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids:dict,last_epoch: int=5, conf_threshold: float=0.6):
    '''
    得到图像在后面几个epoch中未得到匹配的高置信度p_box
    参数：
    ---
    epoch_to_matched_p_ids : dict
        每个epoch下的被匹配的p_ids
    last_epoch : int, default=5
    conf_threshold: float, default=0.6

    返回：
    ---
    img_name_to_epoch_to_no_match_p_boxs : dict
        数据结构示例：
        {
            img_name:{
                epoch:[p_box1,p_box2],
                ...
            },
            ...
        }
    '''
    img_name_to_no_match_p = {}
    # 只关心最后5个epoch的预测情况
    for epoch in range(epochs-last_epoch,epochs):
        # 加载当前epoch的预测结果
        predicted_epoch_json_path = os.path.join(predicted_bboxs_dir, f"epoch_{epoch}_predicted_bboxs.json")
        with open(predicted_epoch_json_path,mode="r") as f:
            predicted_epoch_dict = json.load(f)
        # 统计所有图像中没被gt_box匹配到的高置信度预测box
        for img_name in sorted(predicted_epoch_dict.keys()):
            # img_name在该epoch下的所有预测框
            p_box_list = predicted_epoch_dict[img_name]["predicted_bboxs"]
            # 遍历预测框
            for p_box in p_box_list:
                p_id = p_box["predicted_box_id"]
                if p_id not in epoch_to_matched_p_ids[epoch] and p_box["conf"] > conf_threshold:
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

def get_fault_img_name_set(fault_type_list, annos_with_miss_json:dict) -> set[str]:
    '''
    参数：
    ----
    fault_type_list : list[int]
        [1,2,3,4]:
        1: cls fault
        2: loc fault
        3: redundancy_fault
        4: missing_fault
    返回：
    ---
    fault_img_name_set : set[str]
        返回所有的包含错误anno的img_name set
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

def get_all_img_name(imgs_dir:str) -> list[str]:
    img_name_list = []
    for filename in sorted(os.listdir(imgs_dir)):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isfile(filepath):
            img_name_list.append(filename)
    return img_name_list

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

def rank_img_name(all_img_name_list:list[str], gt_match_json:dict, last_epoch=5, conf_threshold=0.6):
    epoch_to_matched_p_ids = get_epoch_to_matched_p_boxs(gt_match_json)

    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    img_name_to_epoch_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids,last_epoch,conf_threshold)

    img_name_to_no_matched_p_boxs  = get_img_name_to_no_matched_p_boxs(img_name_to_epoch_no_match_p_boxs)

    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters(img_name_to_no_matched_p_boxs,iou_thre=0.6)
    img_name_to_topsis_score = get_img_to_topsis_score(img_to_clusters,last_epoch)
    no_clusters_image_name_set = sorted(set(all_img_name_list) - set(img_name_to_topsis_score.keys()))
    print(f"没有预测簇的图像数量:{len(no_clusters_image_name_set)}")
    for img_name in no_clusters_image_name_set:
        img_name_to_topsis_score[img_name] = 0.0
    sorted_items = sorted(img_name_to_topsis_score.items(),key=lambda x: (-float(x[1]), x[0]))   # 同分按文件名稳定排序
    ranked_image_name_list = []
    ranked_score_list = []
    for image_name,score in sorted_items:
        ranked_image_name_list.append(image_name)
        ranked_score_list.append(score)
    return ranked_image_name_list, ranked_score_list

def build_feature_beta(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
    """
    根据每个 ground truth box 的跨 epoch 过程度量（conf_list 与 iou_list），
    为其构建特征向量。

    参数
    ----------
    g_box_id_to_metric : dict
        以 g_box 的唯一 ID 为键的字典，格式示例：
        {
            g_id: {
                "conf_list": [c_1, c_2, ..., c_T],  # 该 g_box 在各个 epoch 上的置信度序列
                "iou_list":  [i_1, i_2, ..., i_T],  # 该 g_box 在各个 epoch 上的 IoU 序列
            },
            ...
        }
    K : float
        前期后期epoch的分界点。前 K*100% 的epoch区间为前期，后 K*100% 的epoch区间为后期

    返回
    ----------
    (g_id_to_features,feature_name_to_sign): tuple
    g_id_to_features: dict
        以 g_box 的唯一 ID 为键的字典，格式示例：
        {
            g_id: {
                "early_conf_mean": v1,
                "early_conf_mean":0,
                "early_iou_mean":0,
                "lastly_conf_mean":0,
                "lastly_iou_mean":0,
                "conf_mean":0,
                "iou_mean":0,
                "D_conf":1,
                "D_iou":1, 
            },
            ...
        }
    feature_name_to_sign: dict
        以特征名称为 key 的字典，格式示例：
        {
            "early_conf_mean":-1, # 越小越可疑，topsis分越高
            "early_iou_mean":-1,
            "lastly_conf_mean":-1,
            "lastly_iou_mean":-1,
            "conf_mean":-1,
            "iou_mean":-1,
            "D_conf":1, # 越大越可疑，topsis分越高
            "D_iou":1
        }
    """
    g_id_to_features = {}

    def tong_ji(conf_list, iou_list):
        conf_mean = np.mean(conf_list)
        iou_mean = np.mean(iou_list)
        match_count = len(iou_list) - iou_list.count(0)
        match_rate = match_count / len(iou_list)
        return conf_mean, iou_mean, match_rate
    
    def tong_ji_remove_0(conf_list, iou_list):
        new_conf_list = []
        for conf in conf_list:
            if conf != 0:
                new_conf_list.append(conf)

        new_iou_list = []
        for iou in iou_list:
            if iou != 0:
                new_iou_list.append(iou)
        if new_conf_list != []:
            conf_mean = np.mean(new_conf_list)
        else:
            conf_mean = 0
        if new_iou_list != []:
            iou_mean = np.mean(new_iou_list)
        else:
            iou_mean = 0

        match_count = len(iou_list) - iou_list.count(0)
        match_rate = match_count / len(iou_list)
        return conf_mean, iou_mean, match_rate


    for g_id in g_box_id_to_metric.keys():
        conf_list = g_box_id_to_metric[g_id]["conf_list"]
        iou_list = g_box_id_to_metric[g_id]["iou_list"]
        epochs = len(conf_list)
        W_e = int(K*epochs)
        W_l = int(K*epochs)

        early_conf_list = conf_list[0:W_e]
        lastly_conf_list = conf_list[-W_l:]
        early_iou_list = iou_list[0:W_e]
        lastly_iou_list = iou_list[-W_l:]

        # 总体：
        conf_mean, iou_mean, match_rate = tong_ji_remove_0(conf_list,iou_list)
        
        # 前期
        early_conf_mean, early_iou_mean, early_match_rate = tong_ji_remove_0(early_conf_list,early_iou_list)
        # early_conf_mean, early_iou_mean, early_match_rate = help(early_conf_list,early_iou_list)

        # 后期
        lastly_conf_mean, lastly_iou_mean, lastly_match_rate = tong_ji_remove_0(lastly_conf_list,lastly_iou_list)
        # lastly_conf_mean, lastly_iou_mean, lastly_match_rate = help(lastly_conf_list,lastly_iou_list)

        # # 提升
        improve_conf_mean = lastly_conf_mean - early_conf_mean
        improve_iou_mean = lastly_iou_mean - early_iou_mean
 

        '''
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
        '''

        g_id_to_features[g_id] = {
            "conf_mean":conf_mean,
            "iou_mean":iou_mean,
            "match_rate":match_rate,
            "early_conf_mean":early_conf_mean,
            "early_iou_mean":early_iou_mean,
            "early_match_rate":early_match_rate,
            "lastly_conf_mean":lastly_conf_mean,
            "lastly_iou_mean":lastly_iou_mean,
            "lastly_match_rate":lastly_match_rate,
            "improve_conf_mean":improve_conf_mean,
            "improve_iou_mean":improve_iou_mean
        }

    print(f"all gbox数量:{len(all_gids)}")
    print(f"matched gbox数量:{len(g_id_to_features)}")
    
    # for g_id in all_gids:
    #     if g_id not in g_id_to_features:
    #         g_id_to_features[g_id] = {
    #             "conf_mean":0
    #         }
    

    feature_name_to_sign = {
        "conf_mean":-1,
        "iou_mean":-1,
        "match_rate":-1,
        "early_conf_mean":-1,
        "early_iou_mean":-1,
        "early_match_rate":-1,
        "lastly_conf_mean":-1,
        "lastly_iou_mean":-1,
        "lastly_match_rate":-1,
        "improve_conf_mean":-1,
        "improve_iou_mean":-1
    }
    return (g_id_to_features,feature_name_to_sign)

def build_feature_orignal(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
    g_id_to_features = {}
    for g_id in g_box_id_to_metric.keys():
        conf_list = g_box_id_to_metric[g_id]["conf_list"]
        iou_list = g_box_id_to_metric[g_id]["iou_list"]
        epochs = len(conf_list)
        W_e = int(K*epochs)
        W_l = int(K*epochs)
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
            "early_conf_mean":early_conf_mean, # 早期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "early_iou_mean":early_iou_mean, # 早期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_conf_mean":lastly_conf_mean, # 后期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_iou_mean":lastly_iou_mean, # 后期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "conf_mean":conf_mean, # 全期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "iou_mean":iou_mean, # 全期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "D_conf":D_conf, # 起量延迟 conf，越大越可疑 -> topsis分数越高 -> 1
            "D_iou":D_iou, # 起量延迟 iou，越大越可疑 -> topsis分数越高 -> 1
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

    print(f"all gbox数量:{len(all_gids)}")
    print(f"matched gbox数量:{len(g_id_to_features)}")
    
    for g_id in all_gids:
        if g_id not in g_id_to_features:
            # 没有匹配上的gid都是最可疑的
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
    return (g_id_to_features,feature_name_to_sign)

def rank_gid_beta(g_id_to_features, feature_name_to_sign: dict):
    """
    g_id_to_features: {g_id: {attr: (value, flag),},}
    """
    g_id_list = sorted(g_id_to_features.keys())  # 升序
    data = []
    id_to_gid = {}

    # sign 顺序要与特征顺序一致
    sign_list = [sign for _, sign in feature_name_to_sign.items()]

    id = 0
    for g_id in g_id_list:
        feature_dict = g_id_to_features[g_id]

        feature_list = []
        for feature_name in feature_name_to_sign.keys():
            value = feature_dict[feature_name]
            feature_list.append(value)

        data.append(feature_list)
        id_to_gid[id] = g_id
        id += 1

    assert len(sign_list) > 0, "数据有误"

    data_array = np.array(data)
    n_features = data_array.shape[1]
    assert n_features == len(sign_list), "数据有误"

    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, sign_list)

    # 返回排序后的“内部 id”
    sorted_internal_ids = np.argsort(score_array)[::-1]

    # ★ 正确映射回 g_id
    ranked_gid_list = [id_to_gid[i] for i in sorted_internal_ids]

    ranked_score_list = [float(score_array[i]) for i in sorted_internal_ids]

    return ranked_gid_list, ranked_score_list

def rank_gid_original(g_id_to_features,feature_name_to_sign:dict):
    '''
    g_id_to_features:{g_id:{attr:(value,flag),},}
    '''
    g_id_list = list(g_id_to_features.keys())
    g_id_list.sort() # 升序
    data = []
    id_to_gid ={}
    id = 0
    sign_list = []
    feature_name_list = []
    for feature_name,sign in feature_name_to_sign.items():
        sign_list.append(sign)
        feature_name_list.append(feature_name)
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
    ranked_score_list = []
    for gid in ranked_gid_list:
        ranked_score_list.append(score_array[gid])
    return ranked_gid_list, ranked_score_list

def get_all_errored_g_box_id_set(gt_json:dict) -> set[int]:
    '''
    基于我们收集的g_boxs，获得fault g box id set
    '''

    all_errored_g_box_id_set = set()
    for img_name,g_boxs in gt_json.items():
        for g_box in g_boxs:
            if g_box["fault_type"] != 0:
                all_errored_g_box_id_set.add(g_box["box_id"])
    return all_errored_g_box_id_set

def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name



def get_all_miss_error_img_name_set(annos_with_miss_json_path:str) -> set[str]:
    '''
    获得所有具有miss fault的 img name set
    '''
    with open(annos_with_miss_json_path, "r") as f:
        annos_with_miss_json = json.load(f)
    imageid_2_imagename = get_image_id_to_image_name_for_coco(annos_with_miss_json)
    print(f"总图像数:{len(list(imageid_2_imagename.keys()))}")
    anns = annos_with_miss_json["annotations"]
    all_miss_error_img_name_set = set()
    for ann in anns:
        if ann["fault_type"] == 4:
            image_name = imageid_2_imagename[ann["image_id"]]
            all_miss_error_img_name_set.add(image_name)
    print(f"miss error 图像数量:{len(all_miss_error_img_name_set)}")
    return all_miss_error_img_name_set

def look_gid_rank(ranked_gid_list:list[int], all_errored_g_box_id_set:set[int]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "gid_rank.png")
    error_flag_list = []
    for gid in ranked_gid_list:
        if gid in all_errored_g_box_id_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")

def look_metirc(g_box_id_to_metric, all_gids, gt_json:dict):
    for g_id in all_gids:
        if g_id not in g_box_id_to_metric:
            g_box_id_to_metric[g_id] = {
                "conf_list":[0] * epochs,
                "iou_list":[0] * epochs
            } 

    error_group = defaultdict(list[int])
    for img_name,g_boxs in gt_json.items():
        for g_box in g_boxs:
            error_group[g_box["fault_type"]].append(g_box["box_id"])


    fault_to_metric = {}
    for fault_type, g_ids in error_group.items():
        conf_list_list = []
        iou_list_list = []
        for g_id in g_ids:
            conf_list_list.append(g_box_id_to_metric[g_id]["conf_list"])
            iou_list_list.append(g_box_id_to_metric[g_id]["iou_list"])
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
    # 0: correct, 1:cls, 2:loc, 3:redun
    conf_2darray = fault_to_metric[0]["conf_2d_array"]
    save_path = "/data/mml/data_debugging_data/temp/correct_confi_dist_hot.png"
    # draw_violin(conf_2darray, save_path)
    # draw_box(conf_2darray, save_path)
    draw_hot(conf_2darray, save_path)
    

def get_gid_level_rank(gt_json:dict,g_box_metrics_json_path:str):
    '''
    ours方法的gid rank
    '''
    # 获得所有的gids
    all_gids = get_all_gids(gt_json)
    # 获得每个gid的metrics
    g_box_id_to_metric = get_g_id_to_metric(g_box_metrics_json_path)
    # 获得每个gid对应的features和signs
    g_id_to_features,feature_name_to_sign = build_feature_orignal(all_gids,g_box_id_to_metric)
    # 获得gid rank和score（降序）
    ranked_gid_list, ranked_gid_score_list = rank_gid_original(g_id_to_features,feature_name_to_sign)
    # direct_erro_gid_set = set(all_gids) - set(g_box_id_to_metric.keys())
    # look_metirc(g_box_id_to_metric, all_gids, gt_json)
    # g_id_to_features,feature_name_to_sign = build_feature_beta(all_gids,g_box_id_to_metric)
    # ranked_gid_list, ranked_gid_score_list = rank_gid_beta(g_id_to_features,feature_name_to_sign)
    '''
    new_ranked_gid_list = []
    new_ranked_gid_score_list = []
    for gid in direct_erro_gid_set:
        new_ranked_gid_list.append(gid)
        new_ranked_gid_score_list.append(1)
    new_ranked_gid_list.extend(ranked_gid_list)
    new_ranked_gid_score_list.extend(ranked_gid_score_list)
    '''
    return ranked_gid_list, ranked_gid_score_list

def get_img_level_rank(imgs_dir:str,match_json_path:str):
    '''得到ours方法的img的排序'''
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    ranked_image_name_list,ranked_img_score_list = rank_img_name(all_img_name_list, gt_match_json)
    return ranked_image_name_list,ranked_img_score_list


def merge_rank(ranked_gid_list,ranked_gid_score_list,ranked_image_name_list,ranked_img_score_list,
               alpha:float=1.5) -> list:
    merged_rank = []
    idd_to_score = {}
    for gid,score in zip(ranked_gid_list,ranked_gid_score_list):
        idd_to_score[gid] = score
    for img_name,score in zip(ranked_image_name_list,ranked_img_score_list):
        idd_to_score[img_name] = alpha*score
    
    sorted_items = sorted(idd_to_score.items(),key=lambda x: (-float(x[1]), str(x[0])))  # 同分按ID稳定排序
    for idd,score in sorted_items:
        merged_rank.append(idd)
    return merged_rank

def extract_gid(rank_res)->list[int]:
    gid_rank = []
    for idd in rank_res:
        if type(idd) is str:
            continue
        gid_rank.append(idd)
    return gid_rank

def extract_img(rank_res)->list[str]:
    img_rank = []
    for idd in rank_res:
        if type(idd) is str:
            img_rank.append(idd)
    return img_rank

def vis_rank(rank_res,errored_gid_set, miss_img_set, pic_save_path):
    ranked_gid_list = []
    ranked_image_name_list = []
    for idd in rank_res:
        if type(idd) == str:
            ranked_image_name_list.append(idd)
        else:
            ranked_gid_list.append(idd)
    assert len(ranked_gid_list) + len(ranked_image_name_list) == len(rank_res), "数量不对"
    look_total_rank(rank_res,errored_gid_set,miss_img_set,pic_save_path)

def analyse_rank(gt_json_path:str, annos_with_miss_json_path:str, rank_res:list, only_gid:bool=False, only_img:bool=False):
    '''
    rank_res: 我们方法获得的排序结果（idd:img_name or gid）
    '''
    g_boxes_json = read_json(gt_json_path)
    # 得到错误的gid_set
    all_errored_g_box_id_set = get_all_errored_g_box_id_set(g_boxes_json)
    # 得到missed_error_img_name_set
    all_miss_error_img_name_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)

    if only_gid:
        error_set = all_errored_g_box_id_set
        rank_res = extract_gid(rank_res)
    elif only_img:
        error_set = all_miss_error_img_name_set
        rank_res = extract_img(rank_res)
    else:
        error_set = all_errored_g_box_id_set | all_miss_error_img_name_set
    # 计算APFD,FPR和FNR
    APFD = compute_apfd(error_set, rank_res)
    FPR,FNR,F1 =calc_fpr_fnr_f1(rank_res, error_set, cut_off=0.5)
    print(f"排序总长度:{len(rank_res)}")
    print(f"APFD:{APFD},FPR:{FPR},FNR:{FNR},F1:{F1}")
    anno_error_json = read_json(anno_error_json_path)
    converted_rank_list = conver_ours_rank(rank_res,g_boxes_json,anno_error_json)
    annos_with_miss_json = read_json(annos_with_miss_json_path)
    error_idd_set = get_all_error_idd_set(annos_with_miss_json)
    error_imgset = get_all_error_imgset(annos_with_miss_json)
    top1 = calc_top1(annos_with_miss_json,converted_rank_list,error_idd_set,error_imgset)
    exam=calc_exam(annos_with_miss_json,converted_rank_list)
    print(f"top1:{top1},exam:{exam}")
    

    # 统计该rank的修复率
    # anno_with_miss_error = read_json(annos_with_miss_json_path)
    # imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error) 
    # all_error_annoids = get_all_error_annoids(anno_with_miss_error)
    # annoId_to_anno = get_annoId_to_anno(anno_with_miss_error)
    # imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error)
    # anno_error_path = get_error_ann_file_path(dataset_name)
    # anno_error = read_json(anno_error_path)
    # # idd的转换
    # converted_rank = conver_ours_rank(rank_res, g_boxes_json, anno_error)
    # repaired_box_count,repair_rate = count_repair_rate(converted_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate=0.4)
    # print(f"预计修复数量: {repaired_box_count}, 预计修复率: {repair_rate}")

    # 可视化全排序
    # pic_save_dir = os.path.join(exp_data_root_dir,"temp","total_rank")
    # os.makedirs(pic_save_dir,exist_ok=True)
    # pic_save_file_name = "1.png"
    # pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)
    # vis_rank(rank_res,all_errored_g_box_id_set, all_miss_error_img_name_set, pic_save_path)


if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7" # YOLOv7|FRCNN|rtdetr|SSD
    epochs = 50
    if model_name == "rtdetr":
        epochs = 100
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                       dataset_name,model_name,"predicted_bbox")
    # 收集的gboxs
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    anno_error_json_path = get_error_ann_file_path(dataset_name)
    # 我们的序
    # exp_03(cluster)
    # rank_res = joblib.load(os.path.join(exp_data_root_dir,"Results","ours",dataset_name,model_name,
    #                                     "exp_01","rank","rank.joblib"))

    rank_res = joblib.load(os.path.join(exp_data_root_dir,"Discussion_Results",dataset_name,model_name,
                                        "exp_01","rank","alpha=2","rank.joblib"))

    
    # rank_res = joblib.load(os.path.join(exp_data_root_dir,"Discussion_Results",dataset_name,model_name,
    #                                      "exp_01","topsis_feature","img_level","e_freq", "rank.joblib"))

    # 序分析
    analyse_rank(gt_json_path, annos_with_miss_json_path,rank_res)
