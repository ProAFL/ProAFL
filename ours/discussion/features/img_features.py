'''
对排序使用的features进行讨论
'''
import os
import csv
import pandas as pd
import numpy as np
from ours.data_organization_tools import (get_all_gids,get_g_id_to_metric,
                                          get_all_errored_g_box_id_set,get_all_correct_g_box_id_set,
                                          get_all_miss_error_img_name_set,split_img_miss_no_miss)
from ours.base_data_manager import (get_collected_gt_box_json_path,exp_data_root_dir,get_annotations_with_miss_json_path)
from ours.small_utils import read_json,save_json_file
import matplotlib.pyplot as plt
import json
from scipy import stats
import seaborn as sns
import topsispy as tp
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif



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


def get_cluster_feaure(cluster,last_epoch):
    conf = conf_score(cluster) # [0,1] 
    stab = stability_pairwise_mean_iou(cluster) # [0,1]
    cls_consis = cls_consis_score(cluster) # [0,1]
    e_freq = epoch_freq(cluster,last_epoch) # [0,1]
    sign = [1,1,1,1]
    feature = [conf,stab,cls_consis,e_freq]
    return feature, sign

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

def add_path_value(d:dict, keys:list, value):
    '''
    多层级字典，最后指向[]
    '''
    cur = d
    # 遍历所有层级的key
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur.setdefault(keys[-1], []).append(value)

def get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids:dict,
                                              last_epoch: int=5, conf_threshold: float=0.6):
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
                # 没被匹配的和conf大于一定阈值的pid
                if p_id not in epoch_to_matched_p_ids[epoch] and p_box["conf"] > conf_threshold:
                    add_path_value(img_name_to_no_match_p,keys=[img_name,epoch],value=p_box)
    return img_name_to_no_match_p

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



def get_img_to_feature(img_to_clusters: dict, last_epoch: int,
                       persist_freq_threshold: float = 0.4):
    '''
    计算图像级特征并用 TOPSIS 排序。

    思路：先用 get_img_to_scoreAndFeature 获得各簇的 TOPSIS 分（best_cluster_topsis），
    再在图像粒度聚合多个判别力更强的指标，最终对图像级特征再跑一次 TOPSIS。

    图像级特征（全部 sign=+1，越大越可疑）：
    ┌─────────────────────────────┬──────────────────────────────────────────────────────┐
    │ num_clusters                │ 未匹配高置信度簇总数，越多说明可疑区域越多           │
    │ persistent_cluster_count    │ epoch_freq ≥ threshold 的簇数，最直接的判别信号      │
    │ max_epoch_freq              │ 最持久簇的跨 epoch 覆盖率，真实漏标对象接近 1.0      │
    │ max_conf                    │ 置信度最高的簇的均值置信度                           │
    │ max_cluster_size_ratio      │ max(len(cluster)/last_epoch)，越大越稳定             │
    │ best_cluster_topsis         │ 单簇 TOPSIS 分最大值（来自 get_img_to_scoreAndFeature）│
    └─────────────────────────────┴──────────────────────────────────────────────────────┘

    参数：
    ---
    img_to_clusters : dict  {img_name: [[pbox,...], ...]}
    last_epoch : int
    persist_freq_threshold : float, default=0.4

    返回：
    ---
    img_name_to_feature : dict
        {img_name: {"topsis_score": float, "img_features": {feature_name: value}}}
    feature_names : list[str]
    img_level_signs : list[int]
    '''
    # ── Step 1: 复用 get_img_to_scoreAndFeature 获取单簇 TOPSIS 分 ────────
    img_name_to_cluster_score = get_img_to_scoreAndFeature(img_to_clusters, last_epoch)

    # ── Step 2: 图像级特征聚合 ────────────────────────────────────────────
    feature_names = [
        "num_clusters",
        "max_clusterConfi",
        "min_clusterConfi",
        "max_Confi",
        "min_Confi",
        "max_clusterSize",
        "min_clusterSize",
        "max_clusterTopsis",
        "min_clusterTopsis",
        "max_epoch_cross",
        "min_epoch_cross",
        "persistent_cluster_count"
    ]
    img_level_signs = [1]*len(feature_names)

    img_names_ordered = list(img_to_clusters.keys())
    img_level_feature_list = []

    for img_name in img_names_ordered:
        clusters = img_to_clusters[img_name]
        e_freqs     = [epoch_freq(c, last_epoch) for c in clusters] # 每个cluster的epoch跨度(归一了)
        confs       = [conf_score(c)             for c in clusters] # 每个cluster的平均conf
        cluster_size_list = [len(c) for c in clusters] # 每个cluster中包含的p_box的数量
        max_clusterTopsis = img_name_to_cluster_score[img_name]["max_score"]
        min_clusterTopsis = img_name_to_cluster_score[img_name]["min_score"]


        maxConfi = 0
        minConfi = 10
        for cluster in clusters:
            for p_box in cluster:
                if p_box["conf"] > maxConfi:
                    maxConfi = p_box["conf"]
                if p_box["conf"] < minConfi:
                    minConfi = p_box["conf"]
        
        img_level_feature_list.append([
            len(clusters),# num_clusters
            max(confs), # max_clusterConfi
            min(confs), # min_clusterConfi
            maxConfi, # maxConfi
            minConfi, # minConfi
            max(cluster_size_list), # max_clusterSize
            min(cluster_size_list), # min_clusterSize
            max_clusterTopsis, # max_clusterTopsis
            min_clusterTopsis, # min_clusterTopsis
            max(e_freqs), # max_epoch_cross
            min(e_freqs), # min_epoch_cross
            sum(1 for ef in e_freqs if ef >= persist_freq_threshold)  # persistent_cluster_count
        ])

    # ── Step 3: 图像级 TOPSIS ────────────────────────────────────────────
    img_data_array = np.array(img_level_feature_list)
    n_img_feats = img_data_array.shape[1]
    img_weights = np.ones(n_img_feats) / n_img_feats
    _, img_score_array = tp.topsis(img_data_array, img_weights, img_level_signs)
    img_score_array = np.nan_to_num(img_score_array, nan=0.0, posinf=1.0, neginf=0.0)

    # ── Step 4: 构建返回结果 ──────────────────────────────────────────────
    img_name_to_feature = {}
    for idx, img_name in enumerate(img_names_ordered):
        feats = img_level_feature_list[idx]
        img_name_to_feature[img_name] = {
            "topsis_score": float(img_score_array[idx]),
            "img_features": {fname: feats[i] for i, fname in enumerate(feature_names)},
        }
    return img_name_to_feature, feature_names, img_level_signs


def get_img_to_scoreAndFeature(img_to_clusters:dict,last_epoch:int):
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
    img_name_to_scoreAndFeature = {}
    for img_name,cluster_ids in img_name_to_cluster_ids.items():
        img_name_to_scoreAndFeature[img_name] = {
            "max_score":None,
            "avg_score":None,
            "best_feature":{}
        }
        max_score = 0
        min_score = 100
        best_feaure = None
        score_list = []
        for cluster_id in cluster_ids:
            cur_cluster_score = score_array[cluster_id]
            if cur_cluster_score > max_score:
                max_score = cur_cluster_score
                best_feaure = data_array[cluster_id]
            if cur_cluster_score < min_score:
                min_score = cur_cluster_score
            score_list.append(cur_cluster_score)
        avg_score = np.mean(score_list) 
        img_name_to_scoreAndFeature[img_name]["max_score"] = float(max_score)
        img_name_to_scoreAndFeature[img_name]["min_score"] = float(min_score)
        img_name_to_scoreAndFeature[img_name]["avg_score"] = float(avg_score)
        img_name_to_scoreAndFeature[img_name]["best_feature"] = {
            "conf":float(best_feaure[0]),
            "stab":float(best_feaure[1]),
            "cls":float(best_feaure[2]),
            "epoch_cross":float(best_feaure[3])
        }
    return img_name_to_scoreAndFeature


def get_img_to_no_matched_pboxs(all_img_name_list, gt_match_json:dict)->dict:
    last_epoch_nums = 5
    conf_threshold = 0.6
    epoch_to_matched_p_ids = get_epoch_to_matched_p_boxs(gt_match_json)
    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    img_name_to_epoch_to_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(
        epoch_to_matched_p_ids,last_epoch_nums,conf_threshold)
    # 划分出带有miss fault的img set和不带有miss fault的img set
    with_miss_fault_img_set,no_miss_fault_img_set = split_img_miss_no_miss(imgs_dir,annos_with_miss_json_path)
    # 展平epoch key
    img_to_p_boxs = {}

    for img_name in all_img_name_list:
        img_to_p_boxs[img_name] = {}
        if img_name in with_miss_fault_img_set:
            img_to_p_boxs[img_name]["with_miss_fault_flag"] = 1 # 记录该imgname包含miss fault标识
        else:
            img_to_p_boxs[img_name]["with_miss_fault_flag"] = 0 # 记录该imgname不含miss fault标识

        img_to_p_boxs[img_name]["No_matched_p_box_list"] = [] # 记录该imgname中没有得到匹配的预测box list（conf>0.6 and last 5 epochs）
        if img_name in img_name_to_epoch_to_no_match_p_boxs.keys():
            for epoch in img_name_to_epoch_to_no_match_p_boxs[img_name].keys():
                for p_box in img_name_to_epoch_to_no_match_p_boxs[img_name][epoch]:
                    p_box["epoch"] = epoch
                    img_to_p_boxs[img_name]["No_matched_p_box_list"].append(p_box)
    return img_to_p_boxs


def build_img_feature(all_img_name_list:list[str], gt_match_json:dict, last_epoch=5, conf_threshold=0.6):
    epoch_to_matched_p_ids = get_epoch_to_matched_p_boxs(gt_match_json)

    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    img_name_to_epoch_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(
        epoch_to_matched_p_ids,last_epoch,conf_threshold)

    img_name_to_no_matched_p_boxs  = get_img_name_to_no_matched_p_boxs(img_name_to_epoch_no_match_p_boxs)

    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters(img_name_to_no_matched_p_boxs,iou_thre=0.6)
    img_name_to_feature, feature_names, img_level_signs = get_img_to_feature(img_to_clusters, last_epoch)
    no_clusters_image_name_set = sorted(set(all_img_name_list) - set(img_name_to_feature.keys()))
    print(f"没有预测簇的图像数量:{len(no_clusters_image_name_set)}")
    for img_name in no_clusters_image_name_set:
        img_name_to_feature[img_name] = {
            "topsis_score": 0.0,
            "img_features": {fname: 0.0 for fname in feature_names},
        }
    feature_to_sign = {fname: sign for fname, sign in zip(feature_names, img_level_signs)}
    return img_name_to_feature, feature_to_sign

def hypothesis_testing(list_1:list[float],list_2:list[float],alternative:str="two-sided"):
    def mannwhitneyu_effect_size(u_stat, n1, n2):
        """
        计算Mann-Whitney U检验的效应量r（正确版本）
        参数：
            u_stat: mannwhitneyu返回的U统计量
            n1: 第一组数据的样本量
            n2: 第二组数据的样本量
        返回：
            效应量r（绝对值），越大表示差异越明显
        """
        # 步骤1：计算U统计量的均值（零假设下的期望U值）
        mean_u = (n1 * n2) / 2
        # 步骤2：计算U统计量的标准差
        std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        # 步骤3：将U值转换为Z分数（标准化）
        z = (u_stat - mean_u) / std_u
        # 步骤4：计算效应量r（Cohen's r）
        r = abs(z) / np.sqrt(n1 + n2)
        return r
    
    u_stat, u_p = stats.mannwhitneyu(list_1, list_2, alternative=alternative)
    
    print(f"Mann-Whitney U检验：U值={u_stat:.3f}, p值={u_p:.3f}")
    # 计算效应量r
    r = mannwhitneyu_effect_size(u_stat, len(list_1), len(list_2))
    print(f"效应量r：{r:.3f}")

def visualization(correct_list,error_list,save_file_name:str):
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 箱线图
    ax1.boxplot([correct_list, error_list], labels=['correct', 'error'])
    ax1.set_title('Box plot: Data distribution comparison')
    ax1.set_ylabel('Numerical value')
    # 直方图+核密度估计（KDE）
    # 1. 可视化：箱线图（看分布位置、离散程度）+ 直方图（看分布形态）
    
    sns.histplot(correct_list, kde=True, ax=ax2, label='correct', alpha=0.5)
    sns.histplot(error_list, kde=True, ax=ax2, label='error', alpha=0.5)
    ax2.set_title('Histogram +KDE: Shape of distribution')
    ax2.legend()
    plt.savefig(f"/data/mml/data_debugging_data/temp/{save_file_name}.png")


def main():
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    '''得到ours方法的img的排序'''
    img_to_feature,feature_to_sign = build_img_feature(all_img_name_list, gt_match_json)
    with_miss_fault_img_set,no_miss_fault_img_set = split_img_miss_no_miss(imgs_dir,annos_with_miss_json_path)

    # correct_data_list = []
    # error_data_list = []
    # for img_name in no_miss_fault_img_set:
    #     correct_data_list.append(img_to_feature[img_name]["topsis_score"])
    # for img_name in with_miss_fault_img_set:
    #     error_data_list.append(img_to_feature[img_name]["topsis_score"])
    
    # # visualization(correct_data_list,error_data_list,"topsis_score")
    
    # hypothesis_testing(correct_data_list,error_data_list,"less")
    
    for feature_name,sign in feature_to_sign.items():

        img_to_feature[img_name]["img_features"][feature_name]

        correct_data_list = []
        error_data_list = []
        for img_name in no_miss_fault_img_set:
            correct_data_list.append(img_to_feature[img_name]["img_features"][feature_name])
        for img_name in with_miss_fault_img_set:
            error_data_list.append(img_to_feature[img_name]["img_features"][feature_name])
        
        # visualization(correct_data_list,error_data_list,feature_name)
        if feature_to_sign[feature_name] == 1:
            # 我们直觉认为 error data list > correct data list, 因为sign == -1, 说明越小越可疑，排名越靠前。
            # 单侧检验是否 correct < error
            hypothesis_testing(correct_data_list,error_data_list,"less")


def main_2():
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    with_miss_fault_img_set,no_miss_fault_img_set = split_img_miss_no_miss(imgs_dir,annos_with_miss_json_path)
    mode = 0 # 0:全程贯通,1:基于csv进行特征重要性分析
    if mode == 0: 
        csv_path = build_feature_csv(all_img_name_list, gt_match_json, with_miss_fault_img_set, last_epoch=5, conf_threshold=0.6)
    if mode == 0 or mode == 1:
        if mode == 1:
            csv_path = os.path.join(RESULT_DIR,f"box_feature_table_{dataset_name}_{model_name}.csv")
        df = pd.read_csv(csv_path)
        run_feature_importance_analysis(df)

def build_feature_csv(all_img_name_list:list[str], gt_match_json:dict, missfaultimg_set:dict, last_epoch=5, conf_threshold=0.6):
    '''
    构建出每个img的4个feature
    '''
    epoch_to_matched_p_ids = get_epoch_to_matched_p_boxs(gt_match_json)

    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    img_name_to_epoch_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(
        epoch_to_matched_p_ids,last_epoch,conf_threshold)

    img_name_to_no_matched_p_boxs  = get_img_name_to_no_matched_p_boxs(img_name_to_epoch_no_match_p_boxs)

    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters(img_name_to_no_matched_p_boxs,iou_thre=0.6)
    img_name_to_scoreAndFeature = get_img_to_scoreAndFeature(img_to_clusters, last_epoch)
    # 没有可疑预测簇的图像
    no_clusters_image_name_set = sorted(set(all_img_name_list) - set(img_name_to_scoreAndFeature.keys()))
    print(f"没有预测簇的图像数量:{len(no_clusters_image_name_set)}")

    rows = []
    for img_name in img_name_to_scoreAndFeature.keys():
        is_missfault = False
        if img_name in missfaultimg_set:
            is_missfault = True
        feature = img_name_to_scoreAndFeature[img_name]["best_feature"]
        conf = feature["conf"]
        stab = feature["stab"]
        cls = feature["cls"]
        epoch_cross = feature["epoch_cross"]
        rows.append({
            "img_name":img_name,
            "conf":conf,
            "stab":stab,
            "cls":cls, # 做多数量类的占比（类的一致性）
            "epoch_cross":epoch_cross,
            "conf_sign":1,
            "stab_sign":1,
            "cls_sign":1,
            "epoch_cross_sign":1,
            "is_missfault":is_missfault,
            "hasCluster":True
        })
    for img_name in no_clusters_image_name_set:
        is_missfault = False
        if img_name in missfaultimg_set:
            is_missfault = True
        rows.append({
            "img_name":img_name,
            "conf":0,
            "stab":0,
            "cls":0,
            "epoch_cross":0,
            "conf_sign":1,
            "stab_sign":1,
            "cls_sign":1,
            "epoch_cross_sign":1,
            "is_missfault":is_missfault,
            "hasCluster":False
        })

    # csv 保存路径
    save_file_name = f"img_feature_table_{dataset_name}_{model_name}.csv"
    output_path = os.path.join(RESULT_DIR, save_file_name)
    fieldnames = ["img_name","conf","stab","cls","epoch_cross","conf_sign","stab_sign","cls_sign","epoch_cross_sign", "is_missfault", "hasCluster"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"img feature csv saved: {output_path}")
    return output_path

def get_feature_names_from_df(df:pd.DataFrame):
    required_columns = {"is_missfault"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV缺少必要列: {missing_columns}")

    feature_names = [fn for fn in FEATURE_NAME_TO_SIGN.keys() if fn in df.columns]
    if len(feature_names) != len(FEATURE_NAME_TO_SIGN):
        missing_features = sorted(set(FEATURE_NAME_TO_SIGN.keys()) - set(feature_names))
        raise ValueError(f"CSV缺少过程特征列: {missing_features}")

    missing_sign_columns = [
        f"{feature_name}_sign"
        for feature_name in feature_names
        if f"{feature_name}_sign" not in df.columns
    ]
    if missing_sign_columns:
        raise ValueError(f"CSV缺少feature sign列: {missing_sign_columns}")
    return feature_names

def _series_to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool).to_numpy()
    true_values = {"true", "1", "yes", "y", "t"}
    return series.astype(str).str.strip().str.lower().isin(true_values).to_numpy()

def _safe_corr(func, x, y):
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    stat, p_value = func(x, y)
    return float(stat), float(p_value)

def correlation_importance_analysis(df, feature_names):
    labels = _series_to_bool(df["is_missfault"]).astype(int)
    rows = []
    for feature_name in feature_names:
        raw_values = df[feature_name].astype(float).to_numpy()
        sign = int(df[f"{feature_name}_sign"].iloc[0])
        suspicious_values = sign * raw_values
        pearson_r, pearson_p = _safe_corr(stats.pearsonr, suspicious_values, labels)
        spearman_r, spearman_p = _safe_corr(stats.spearmanr, suspicious_values, labels)
        point_biserial_r, point_biserial_p = _safe_corr(stats.pointbiserialr, raw_values, labels)
        rows.append({
            "feature": feature_name,
            "n": len(labels),
            "n_error": int(labels.sum()),
            "n_correct": int(len(labels) - labels.sum()),
            "pearson_r_suspicious": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r_suspicious": spearman_r,
            "spearman_p": spearman_p,
            "point_biserial_r_raw": point_biserial_r,
            "point_biserial_p": point_biserial_p,
            "abs_spearman_r": abs(spearman_r) if not np.isnan(spearman_r) else float("nan"),
        })
    rows.sort(key=lambda row: -row["abs_spearman_r"] if not np.isnan(row["abs_spearman_r"]) else float("-inf"))
    return rows

def _discretize_by_quantile(values, n_bins=10):
    """
    用分位数离散化连续特征，便于计算信息熵/互信息。
    重复取值很多时，实际 bin 数可能小于 n_bins。
    """
    values = np.asarray(values, dtype=float)
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        return np.zeros(len(values), dtype=int)
    quantiles = np.linspace(0, 1, min(n_bins, len(unique_values)) + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) == 0:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges, right=False)

def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))

def _mutual_information_discrete(feature_bins, labels):
    feature_bins = np.asarray(feature_bins, dtype=int)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    feature_values = np.unique(feature_bins)
    label_values = np.unique(labels)
    joint_counts = np.zeros((len(feature_values), len(label_values)), dtype=float)
    f_to_i = {v: i for i, v in enumerate(feature_values)}
    y_to_i = {v: i for i, v in enumerate(label_values)}
    for f, y in zip(feature_bins, labels):
        joint_counts[f_to_i[f], y_to_i[y]] += 1

    pxy = joint_counts / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

    h_feature = _entropy_from_counts(joint_counts.sum(axis=1))
    h_label = _entropy_from_counts(joint_counts.sum(axis=0))
    mi_over_h_label = mi / h_label if h_label > 0 else float("nan")
    mi_over_h_feature = mi / h_feature if h_feature > 0 else float("nan")
    return float(mi), float(h_feature), float(h_label), float(mi_over_h_label), float(mi_over_h_feature)


def mutual_information_importance_analysis(df, feature_names, n_bins=10):
    labels = _series_to_bool(df["is_missfault"]).astype(int)
    rows = []
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return rows

    suspicious_matrix = np.column_stack([
        int(df[f"{feature_name}_sign"].iloc[0]) * df[feature_name].astype(float).to_numpy()
        for feature_name in feature_names
    ])
    n_neighbors = max(1, min(3, len(labels) - 1))
    mi_continuous = mutual_info_classif(
        suspicious_matrix, labels, discrete_features=False,
        n_neighbors=n_neighbors, random_state=0
    )
    for idx, feature_name in enumerate(feature_names):
        suspicious_values = suspicious_matrix[:, idx]
        feature_bins = _discretize_by_quantile(suspicious_values, n_bins=n_bins)
        mi_bits, h_feature, h_label, mi_over_h_label, mi_over_h_feature = (
            _mutual_information_discrete(feature_bins, labels)
        )
        rows.append({
            "feature": feature_name,
            "n": len(labels),
            "n_error": int(labels.sum()),
            "n_correct": int(len(labels) - labels.sum()),
            "bins": int(len(np.unique(feature_bins))),
            "mi_bits_quantile": mi_bits,
            "h_feature_bits": h_feature,
            "h_label_bits": h_label,
            "mi_over_h_label": mi_over_h_label,
            "mi_over_h_feature": mi_over_h_feature,
            "mi_knn_nats": float(mi_continuous[idx]),
        })
    rows.sort(key=lambda row: -row["mi_over_h_label"] if not np.isnan(row["mi_over_h_label"]) else float("-inf"))
    return rows

def print_importance_rows(title, rows, metric_name):
    print("="*60)
    print(title)
    for row in rows:
        value = row[metric_name]
        if np.isnan(value):
            value_text = "nan"
        else:
            value_text = f"{value:.4f}"
        print(f"  {row['feature']:20s}  {metric_name}={value_text}")

def plot_importance_bar(rows, metric_name, title, output_path):
    if not rows:
        return
    feature_names = [row["feature"] for row in rows]
    values = [row[metric_name] for row in rows]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=values, y=feature_names, orient="h")
    plt.xlabel(metric_name)
    plt.ylabel("feature")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_feature_correlation_heatmap(df, feature_names, output_path):
    matrix = np.column_stack([
        int(df[f"{feature_name}_sign"].iloc[0]) * df[feature_name].astype(float).to_numpy()
        for feature_name in feature_names
    ])
    if matrix.shape[0] < 2:
        return
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title(f"Feature Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def write_rows_to_csv(rows, output_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def run_feature_importance_analysis(df:pd.DataFrame):
    feature_names = get_feature_names_from_df(df)
    result_dir = RESULT_DIR
    output_prefix = f"img_feature_importance_{dataset_name}_{model_name}"
    
    all_corr_rows = [] # 相关性
    all_mi_rows = [] # 互信息
    
    labels = _series_to_bool(df["is_missfault"]).astype(int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise Exception("样本为空或只包含单一类别，无法计算重要性。")
    corr_rows = correlation_importance_analysis(df, feature_names)
    mi_rows = mutual_information_importance_analysis(df, feature_names)
    all_corr_rows.extend(corr_rows)
    all_mi_rows.extend(mi_rows)

    print_importance_rows(
        f"相关性重要性排名 (按 |Spearman r|)",
        corr_rows,
        "abs_spearman_r",
    )
    print_importance_rows(
        f"信息熵/互信息重要性排名 (按 MI/H(label))",
        mi_rows,
        "mi_over_h_label",
    )

    plot_importance_bar(
        corr_rows,
        "abs_spearman_r",
        f"Correlation Importance",
        f"{result_dir}/{output_prefix}_correlation_importance.png",
    )
    plot_importance_bar(
        mi_rows,
        "mi_over_h_label",
        f"Mutual Information Importance",
        f"{result_dir}/{output_prefix}_mi_importance.png",
    )
    plot_feature_correlation_heatmap(
        df,
        feature_names,
        f"{result_dir}/{output_prefix}_feature_corr_heatmap.png",
    )

    write_rows_to_csv(all_corr_rows, f"{result_dir}/{output_prefix}_correlation_importance.csv")
    write_rows_to_csv(all_mi_rows, f"{result_dir}/{output_prefix}_mutual_information_importance.csv")
    return all_corr_rows, all_mi_rows



if __name__ == "__main__":
    
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7" # YOLOv7|FRCNN|SSD
    epochs = 50
    gt_json_path = get_collected_gt_box_json_path(dataset_name)

    # match json
    match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name, "gp_box_match",
                                   "match_v3.json") # v3!
    if not os.path.exists(match_json_path):
        # 使用了新路径
        match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,
                                   "match.json")
    # metric json
    g_box_metrics_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,"collection_metric",
                                           "collection_metrics_v3.json") # v3!
    if not os.path.exists(g_box_metrics_json_path):
        # 使用了新路径
        g_box_metrics_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,
                                            "metrics.json")

    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                    dataset_name,model_name,"collected_predicted_box","v2")
    # 一定要是全量的trainset的imgsdir
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split", dataset_name,
                             "images", "origin")
    
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    img_to_p_boxs = get_img_to_no_matched_pboxs(all_img_name_list, gt_match_json)
    
    save_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                            dataset_name,model_name)
    save_file_name = "img_to_nomatched_pboxs.json"
    save_path = os.path.join(save_dir,save_file_name)
    save_json_file(img_to_p_boxs,save_path)
    RESULT_DIR = "/data/mml/data_debugging_data/discussion/"
    FEATURE_NAME_TO_SIGN = {
        "conf":1,
        "stab":1,
        "cls":1,
        "epoch_cross":1,
    }
    main_2()
    # main()