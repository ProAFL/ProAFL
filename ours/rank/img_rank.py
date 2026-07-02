

import os
import math
import numpy as np
import topsispy as tp
from collections import defaultdict, Counter
from ours.rank_analyse.common import draw_rank_hot
from ours.small_utils import read_json,save_json_file
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ks_2samp

def box_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def cluster_boxes_by_iou(p_boxes, iou_threshold=0.6, require_same_cls=False):
    n = len(p_boxes)
    if n == 0:
        return []

    graph = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            iou = box_iou(p_boxes[i]["bbox"], p_boxes[j]["bbox"])
            if iou < iou_threshold:
                continue
            if require_same_cls and p_boxes[i]["predicted_cls"] != p_boxes[j]["predicted_cls"]:
                continue
            graph[i].append(j)
            graph[j].append(i)

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(p_boxes[cur])
            for nei in graph[cur]:
                if not visited[nei]:
                    visited[nei] = True
                    stack.append(nei)
        clusters.append(comp)

    return clusters


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def cluster_features(cluster, total_epoch_num=5):
    confs = [b["conf"] for b in cluster]
    clses = [b["predicted_cls"] for b in cluster]
    epochs = [b["epoch"] for b in cluster]

    size = len(cluster)
    mean_conf = float(np.mean(confs))
    max_conf = float(np.max(confs))

    epoch_count = len(set(epochs))
    epoch_coverage = epoch_count / total_epoch_num

    most_common_cls_count = Counter(clses).most_common(1)[0][1]
    class_consistency = most_common_cls_count / size

    pairwise_ious = []
    for i in range(size):
        for j in range(i + 1, size):
            pairwise_ious.append(box_iou(cluster[i]["bbox"], cluster[j]["bbox"]))
    mean_iou = float(np.mean(pairwise_ious)) if pairwise_ious else 0.0

    centers = np.array([box_center(b["bbox"]) for b in cluster], dtype=float)
    if len(centers) <= 1:
        center_std = 1e6   # 单点簇视为稳定性很弱
    else:
        std_x = np.std(centers[:, 0])
        std_y = np.std(centers[:, 1])
        center_std = float(math.sqrt(std_x**2 + std_y**2))

    
    center_stability = 1.0 / (1.0 + center_std)

    normalized_cluster_size = min(size / total_epoch_num, 1.0)
    
    return {
        "cluster_size": size,
        "normalized_cluster_size": normalized_cluster_size,
        "epoch_coverage": epoch_coverage,
        "mean_conf": mean_conf,
        "max_conf": max_conf,
        "class_consistency": class_consistency,
        "mean_iou": mean_iou,
        "center_std": center_std,
        "center_stability": center_stability
    }

def score_cluster(feat):
    """
    输入: 单个cluster的特征字典
    输出: 该cluster的综合分数，越大表示越像潜在miss annotation区域
    """
    score = (
        0.28 * feat["mean_conf"]
        + 0.22 * feat["epoch_coverage"]
        + 0.18 * feat["normalized_cluster_size"]
        + 0.14 * feat["class_consistency"]
        + 0.12 * feat["mean_iou"]
        + 0.06 * feat["center_stability"]
    )
    return float(score)

def image_features_from_clusters(clusters, total_epoch_num=5):
    feature_names = [
        "cluster_count", # F1
        "max_cluster_size", # F2
        "max_epoch_coverage", # F3
        "max_mean_conf", # # F4
        "max_conf", # F5
        "max_class_consistency", # F6
        "max_mean_iou", # F7
        "best_center_stability" # F8
    ]

    if len(clusters) == 0:
        return [0.0] * len(feature_names), feature_names

    feats = [cluster_features(c, total_epoch_num=total_epoch_num) for c in clusters]
    # cluster_scores = [score_cluster(f) for f in feats] F9

    F1 = float(len(clusters)) # 该图像包含的簇数量
    F2 = float(max(f["cluster_size"] for f in feats)) # 簇中最大size簇
    F3 = float(max(f["epoch_coverage"] for f in feats)) # 簇中最大的epoch覆盖值
    F4 = float(max(f["mean_conf"] for f in feats)) # 簇中最大mean conf
    F5 = float(max(f["max_conf"] for f in feats)) # 簇中最大max max conf
    F6 = float(max(f["class_consistency"] for f in feats))
    F7 = float(max(f["mean_iou"] for f in feats))
    F8 = float(max(f["center_stability"] for f in feats)) # 越稳定越好
    # F9 = float(max(cluster_scores))   # strongest_cluster_score
    return [F1, F2, F3, F4, F5, F6, F7, F8],feature_names



def compute_apfd(fault_set:set, rankded_list):
    """
    fault_set: set/list, 真实错误idd(box_id/anno_id|img_name)
    rankded_list: list, 按可疑度排序的图像路径
    """
    # n: 排序总量
    n = len(rankded_list)
    
    TF_positions = []

    # 遍历 rankded_list 找到真实错误的位置
    for idx, ID in enumerate(rankded_list, start=1):  # 从1开始计数
        if ID in fault_set:
            TF_positions.append(idx)

    # m:错误总量
    m = len(fault_set)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    apfd = round(apfd,4)
    return apfd



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

def clusing_by_unifind(box_list:list,iou_thre:float=0.6) -> list[list[int]]:
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

def get_img_to_clusters_by_unifind(img_to_p_boxs:dict,iou_thre:float=0.6):
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
        cluster_list = clusing_by_unifind(p_box_list,iou_thre)
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


def get_cluster_feaure(cluster,last_epoch):
    conf = conf_score(cluster) # [0,1] 
    stab = stability_pairwise_mean_iou(cluster) # [0,1]
    cls_consis = cls_consis_score(cluster) # [0,1]
    e_freq = epoch_freq(cluster,last_epoch) # [0,1]
    sign = [1,1,1,1]
    feature = [conf,stab,cls_consis,e_freq]
    feature_names = ["conf","stab","cls_consis","e_freq"]
    return feature, sign, feature_names

def get_img_to_features_and_score(img_to_clusters:dict,no_clusters_image_name_set:set,last_epoch:int):
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
    feature_names = []
    # 存放每个img对应的cluster的idx list
    img_name_to_cluster_ids = defaultdict(list)

    # 追踪每个cluster的索引
    cluster_idx = 0
    # 遍历每个图像与其对应的簇群
    for img_name,clusters in img_to_clusters.items():
        # 遍历该图像的所有簇
        for cluster in clusters:
            # 得到该簇的特征数据和特征对应的正负符号
            features,signs,names = get_cluster_feaure(cluster,last_epoch)
            clusters_features.append(features)
            features_signs = signs
            feature_names = names
            img_name_to_cluster_ids[img_name].append(cluster_idx)
            cluster_idx += 1

    # 构建特征数据集
    data_array = np.array(clusters_features)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(features_signs), "数据有误"
    
    # weights = entropy_weight(data_array)
    weights = np.ones(n_features) / n_features
    # 基于topsis获得clusters的score
    best_cluster_id, score_array = tp.topsis(data_array, weights, features_signs)
    score_array = np.nan_to_num(score_array, nan=0.0, posinf=1.0, neginf=0.0)
    # 从大到小排序并返回索引
    # sorted_cluster_id = np.argsort(score_array)[::-1]
    # 将img_name中得分最高的cluster的得分作为该img的score

    res = {}
    for img_name,cluster_ids in img_name_to_cluster_ids.items():
        res[img_name] = {}
        feature_data = None
        max_score = 0
        feature_data = None
        best_cluster_id = -1
        for cluster_id in cluster_ids:
            if score_array[cluster_id] > max_score:
                max_score = score_array[cluster_id]
                best_cluster_id = cluster_id
        max_score = score_array[best_cluster_id]
        feature_data = data_array[best_cluster_id].tolist()
        res[img_name]["max_topsis_score"] = max_score
        res[img_name]["feature_data"] = feature_data

    for img_name in no_clusters_image_name_set:
        res[img_name] = {}
        res[img_name]["max_topsis_score"] = 0.0
        res[img_name]["feature_data"] = [0.0]*len(signs)
    return res,feature_names


def img_rank_2(img_to_nomatched_pboxs_json_path):
    '''
    对img的unmatched pboxs进行聚类后，提取各个簇的特征并将最大的那个簇的特征作为该img的特征
    '''
    json_data = read_json(img_to_nomatched_pboxs_json_path)
    img_to_p_boxs = defaultdict(list)
    all_img_name_set = set()
    no_clusters_image_name_set = set()
    for img_name,info in json_data.items():
        all_img_name_set.add(img_name)
        if info["No_matched_p_box_list"] == []:
            no_clusters_image_name_set.add(img_name)
        img_to_p_boxs[img_name] = info["No_matched_p_box_list"]
    # 采用并查集算法将该img这些高置信度未匹配p_box进行分簇，一个簇其实就是一个统一的p_box
    img_to_clusters = get_img_to_clusters_by_unifind(img_to_p_boxs,iou_thre=0.6)
    img_name_to_features_and_score,feature_names = get_img_to_features_and_score(img_to_clusters,
                                                    no_clusters_image_name_set,last_epoch=5)
    ranked_img_names = sorted(img_name_to_features_and_score.keys(), 
                              key=lambda x: img_name_to_features_and_score[x]["max_topsis_score"], 
                              reverse=True)
    ranked_imgs = list(ranked_img_names)
    ranked_score_list = []
    ranked_flag_list = []
    fault_set = set()
    X = []
    Y = []
    for img_name in ranked_imgs:
        ranked_score_list.append(img_name_to_features_and_score[img_name]["max_topsis_score"])
        ranked_flag_list.append(json_data[img_name]["with_miss_fault_flag"])
        if json_data[img_name]["with_miss_fault_flag"] == 1:
            fault_set.add(img_name)
        X.append(img_name_to_features_and_score[img_name]["feature_data"])
        Y.append(json_data[img_name]["with_miss_fault_flag"])
        
    X = np.array(X)
    Y = np.array(Y)
    
    
    res = {
        "ranked_imgs":ranked_imgs,
        "ranked_scores": ranked_score_list,
        "ranked_isFault_list": ranked_flag_list,
        "fault_imgset": fault_set,
        "feature_names": feature_names,
        "feature_data": X,
        "label":Y
    }
    return res


def img_rank(img_to_nomatched_pboxs_json_path):
    '''
    对img的unmatched pboxs进行聚类后，直接提取图像层面的特征
    '''
    json_data = read_json(img_to_nomatched_pboxs_json_path)
    img_names = []
    X = []
    Y = []
    total_epoch_num = 5 # 最后5个轮次
    for img_name, info in json_data.items():
        # 该图像下的p_boxs
        p_boxes = info["No_matched_p_box_list"]
        # 对这些p_boxs的位置关系进行聚类
        clusters = cluster_boxes_by_iou(
            p_boxes,
            iou_threshold=0.6,
            require_same_cls=False
        )
        # 基于clustered pboxs对image进行特征构建
        # feats: 特征向量; feature_names: 特征名称
        feats, feature_names = image_features_from_clusters(clusters, total_epoch_num=total_epoch_num)
        img_names.append(img_name)
        X.append(feats)
        Y.append(info["with_miss_fault_flag"])
    X = np.array(X)
    Y = np.array(Y)
    weights = np.ones(X.shape[1]) / X.shape[1] # 特征权重
    signs = [1] * X.shape[1] # 特征方向: 1表示越大topsis分数越高
     # 基于topsis获得clusters的score
    best_id, score_array = tp.topsis(X, weights, signs)
    # 从大到小排序并返回索引
    sorted_id = np.argsort(score_array, kind="mergesort")[::-1]
    ranked_id_list = [int(id) for id in sorted_id]
    ranked_score_list = []
    ranked_imgs = []
    ranked_flag_list = []
    fault_set = set()
    for id in ranked_id_list:
        img_name = img_names[id]
        ranked_score_list.append(float(score_array[id]))
        ranked_imgs.append(img_name)
        ranked_flag_list.append(json_data[img_name]["with_miss_fault_flag"])
        if json_data[img_name]["with_miss_fault_flag"] == 1:
            fault_set.add(img_name)
    res = {
        "ranked_imgs":ranked_imgs,
        "ranked_scores": ranked_score_list,
        "ranked_isFault_list": ranked_flag_list,
        "fault_imgset": fault_set,
        "feature_names": feature_names,
        "feature_data": X,
        "label":Y,
        "ranked_ids":ranked_id_list
    }
    return res



def cohens_d(x1, x0):

    mean1 = np.mean(x1)
    mean0 = np.mean(x0)

    std1 = np.std(x1, ddof=1)
    std0 = np.std(x0, ddof=1)

    n1 = len(x1)
    n0 = len(x0)

    pooled_std = np.sqrt(
        ((n1-1)*std1**2 + (n0-1)*std0**2) / (n1+n0-2)
    )

    if pooled_std == 0:
        return 0

    return (mean1 - mean0) / pooled_std

def visualization(correct_list,error_list,save_file_name:str):
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 箱线图
    ax1.boxplot([correct_list, error_list], tick_labels=['correct', 'error'])
    ax1.set_title('Box plot: Data distribution comparison')
    ax1.set_ylabel('Numerical value')
    # 直方图+核密度估计（KDE）
    # 1. 可视化：箱线图（看分布位置、离散程度）+ 直方图（看分布形态）
    
    sns.histplot(correct_list, kde=True, ax=ax2, label='correct', alpha=0.5)
    sns.histplot(error_list, kde=True, ax=ax2, label='error', alpha=0.5)
    ax2.set_title('Histogram +KDE: Shape of distribution')
    ax2.legend()
    plt.savefig(f"/data/mml/data_debugging_data/temp/img_rank/{save_file_name}.png")


def analyze_feature_importance(X, y, feature_names):

    results = []

    for i, name in enumerate(feature_names):

        x = X[:, i]

        x1 = x[y == 1]
        x0 = x[y == 0]

        # AUC
        try:
            auc = roc_auc_score(y, x)
            if auc < 0.5:
                auc = 1-auc
        except:
            auc = 0.5

        auc_importance = abs(auc - 0.5)

        # KS
        ks_stat, _ = ks_2samp(x1, x0)

        # Cohen's d
        d = cohens_d(x1, x0)

        results.append({
            "feature": name,
            "AUC": round(auc,3),
            "AUC_importance": round(auc_importance,3),
            "KS": round(ks_stat,3),
            "Cohen_d": round(d,3)
        })

    df = pd.DataFrame(results)

    # mutual information
    mi = mutual_info_classif(X, y)

    df["MutualInfo"] = mi

    return df.sort_values("AUC_importance", ascending=False)


def rank_analyse(rank_res):
    ranked_imgs = rank_res["ranked_imgs"]
    fault_set = rank_res["fault_imgset"]
    feature_names = rank_res["feature_names"]

    print(f"总共的img数量:{len(ranked_imgs)}")
    print(f"包含miss fault的img数量:{len(fault_set)}")
    apfd = compute_apfd(fault_set,ranked_imgs)
    print(f"apfd:{apfd}")

    X = rank_res["feature_data"]
    Y = rank_res["label"]
    importance_df = analyze_feature_importance(X, Y, feature_names)
    print(importance_df)

def rank_vis(rank_res,save_file_path):
    ranked_flag_list = rank_res["ranked_isFault_list"]
    draw_rank_hot(ranked_flag_list,save_file_path)
    X = rank_res["feature_data"]
    Y = rank_res["label"]
    feature_names = rank_res["feature_names"]

    # for i, name in enumerate(feature_names):
    #     x = X[:, i]
    #     x1 = x[Y == 1]
    #     x0 = x[Y == 0]
    #     visualization(x0,x1,name)

def compare():
    rank_imgLevelFeature = img_rank(img_to_nomatched_pboxs_json_path) # img level feature
    rank_clusterLevelFeature = img_rank_2(img_to_nomatched_pboxs_json_path) # cluster level feature
    print("imgLevel")
    rank_analyse(rank_imgLevelFeature)
    print("clusterLevel")
    rank_analyse(rank_clusterLevelFeature)

    pic_save_dir = os.path.join(exp_root_dir,"temp","img_rank")
    pic_save_file_name = "imgLevel.png"
    pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)
    rank_vis(rank_imgLevelFeature,pic_save_path)

    pic_save_file_name = "clusterLevel.png"
    pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)
    rank_vis(rank_clusterLevelFeature,pic_save_path)

    scores_1 = rank_imgLevelFeature["ranked_scores"]
    scores_2 = rank_clusterLevelFeature["ranked_scores"]
    data= [scores_1,scores_2]
    labels = ["imgLevelFeature","clusterLevelFeature"]
    plt.boxplot(data, tick_labels=labels)

    plt.title("Boxplot Comparison")
    plt.ylabel("Value")
    plt.savefig("/data/mml/data_debugging_data/temp/img_rank/compare.png")




def main():
    # 数据
    # rank_res = img_rank(img_to_nomatched_pboxs_json_path) # img level feature
    rank_res = img_rank_2(img_to_nomatched_pboxs_json_path) # cluster level feature
    # 分析
    # rank_analyse(rank_res)
    # 可视化
    # pic_save_dir = os.path.join(exp_root_dir,"img_rank","max")
    # pic_save_file_name = "rank.png"
    # pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)
    # rank_vis(rank_res,pic_save_path)



if __name__ == "__main__":
    exp_root_dir= "/data/mml/data_debugging_data"
    dataset_name = "VOC2012" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    epochs = 50
    img_to_nomatched_pboxs_json_path = os.path.join(
        exp_root_dir,"collection_bbox_level",
        dataset_name,model_name,"img_to_nomatched_pboxs.json")
    main()
    compare()