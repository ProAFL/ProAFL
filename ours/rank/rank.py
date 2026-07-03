
'''
Get ranking results of our method based on collected data
'''
import os
import json
import joblib
import pprint

from collections import defaultdict
import numpy as np
import topsispy as tp
from helper.base_data_manager import (exp_data_root_dir,
                                    get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path,
                                    get_ours_gt_box_metric_path,
                                    get_ours_match_path,
                                    get_collected_predict_boxes_dir,
                                    get_all_trainimgs_dir,
                                    get_img_to_nomatched_pboxs_json_path
                                    )
from ours.small_utils import read_json,read_yaml
from ours.rank.img_rank import img_rank


def get_all_gids(gt_json:dict) -> list[int]:
    '''
    Get all g_box_id_list values
    
    Parameters
    ----
    gt_json : dict
        Data format:
        {
            image_name:[g_box_1,g_box_2],
            ...
        }
    Returns
    ---
    all_g_box_id_list : list[int]
        All extracted g_box_id_list values
    '''
    all_g_box_id_list = []
    for img_name, g_box_list in gt_json.items():
        for g_box in g_box_list:
            all_g_box_id_list.append(g_box["box_id"])
    return all_g_box_id_list


def get_g_id_to_metric(metric_json_path):
    '''
    Provide metrics (conf_list and iou_list) for each gid
    '''
    with open(metric_json_path, "r", encoding="utf-8") as f:
        gt_box_metric_collection_list = json.load(f)
    print(f"matched gt_boxCount:{len(gt_box_metric_collection_list)}")

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



def add_path_value(d:dict, keys:list, value):
    '''
    Nested dictionary whose final value is [].
    '''
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
    Use union-find to cluster these boxes
    box_list : list
        Data structure example:
        [box_1,box_2,...]
    iou_thre : float, default = 0.6
        threshold condition for merging elements
    Returns:
    ---
    cluster_list : list[list[int]]
        Data structure example
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
    Flatten img_name to unmatched p_boxs
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
    Build img_name -> p_boxclusters
    Parameters
    ---
    img_to_p_boxs : dict
        Data format:
        {image_name:[p_box1,p_box2]}
    iou_thre : float,default=0.6
        IoU condition for clustering
    
    Returns:
    ---
    img_to_clusters : dict
        Data format example:
        {
            img_name:[[p_box1,p_box2,...],...],
            ...
        }
    '''
    img_to_clusters = defaultdict(list)
    for img_name,p_box_list in img_to_p_boxs.items():
                                    
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


def get_cluster_feaure(cluster,last_epoch):
    conf = conf_score(cluster)         
    stab = stability_pairwise_mean_iou(cluster)        
    cls_consis = cls_consis_score(cluster)        
    e_freq = epoch_freq(cluster,last_epoch)        
    sign = [1,1,1,1]
    feature = [conf,stab,cls_consis,e_freq]
    return feature, sign

def get_img_to_topsis_score(img_to_clusters:dict,last_epoch:int):
    '''
    Get img_name -> topsis score
    
    Parameters:
    ---
    img_to_clusters : dict
        Data format
        {img_name:[[pbox1,pbox3],...]}
    last_epoch : int
    '''
                                     
    clusters_features = []
                                   
    features_signs = []

                                             
    img_name_to_cluster_ids = defaultdict(list)

                                     
    cluster_idx = 0
                                         
    for img_name,clusters in img_to_clusters.items():
                                            
        for cluster in clusters:
                                                                           
            features,signs = get_cluster_feaure(cluster,last_epoch)
            clusters_features.append(features)
            features_signs = signs
            img_name_to_cluster_ids[img_name].append(cluster_idx)
            cluster_idx += 1

                           
    data_array = np.array(clusters_features)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(features_signs), "Invalid data"
    
    weights = entropy_weight(data_array)
    weights = np.ones(n_features) / n_features
                                        
    best_cluster_id, score_array = tp.topsis(data_array, weights, features_signs)
    score_array = np.nan_to_num(score_array, nan=0.0, posinf=1.0, neginf=0.0)
                                        
                                                       
                                                                   
    img_name_to_max_score = {}
    for img_name,cluster_ids in img_name_to_cluster_ids.items():
        max_score = 0
        for cluster_id in cluster_ids:
            if score_array[cluster_id] > max_score:
                max_score = score_array[cluster_id]
        img_name_to_max_score[img_name] = max_score
    return img_name_to_max_score


def get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids:dict,last_epoch: int=5, conf_threshold: float=0.6):
    '''
    Get high-confidence unmatched p_boxes in later epochs for each image.
    Parameters:
    ---
    epoch_to_matched_p_ids : dict
        matched p_ids under each epoch
    last_epoch : int, default=5
    conf_threshold: float, default=0.6

    Returns:
    ---
    img_name_to_epoch_to_no_match_p_boxs : dict
        Data structure example:
        {
            img_name:{
                epoch:[p_box1,p_box2],
                ...
            },
            ...
        }
    '''
    img_name_to_no_match_p = {}
                                                    
    for epoch in range(epochs-last_epoch,epochs):
                                                      
        predicted_epoch_json_path = os.path.join(predicted_bboxs_dir, f"epoch_{epoch}_predicted_bboxs.json")
        with open(predicted_epoch_json_path,mode="r") as f:
            predicted_epoch_dict = json.load(f)
                                                                                     
        for img_name in sorted(predicted_epoch_dict.keys()):
                                                          
            p_box_list = predicted_epoch_dict[img_name]["predicted_bboxs"]
                                     
            for p_box in p_box_list:
                p_id = p_box["predicted_box_id"]
                if p_id not in epoch_to_matched_p_ids[epoch] and p_box["conf"] > conf_threshold:
                    add_path_value(img_name_to_no_match_p,keys=[img_name,epoch],value=p_box)
    return img_name_to_no_match_p




def get_all_img_name(imgs_dir:str) -> list[str]:
    img_name_list = []
    for filename in sorted(os.listdir(imgs_dir)):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isfile(filepath):
            img_name_list.append(filename)
    return img_name_list

def get_epoch_to_matched_p_boxs(gt_match_dict):
                                       
    epoch_to_match_info = {}
                         
    for g_box_id in gt_match_dict.keys():
                                                
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

                                                                                        
    img_name_to_epoch_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids,last_epoch,conf_threshold)
    img_name_to_no_matched_p_boxs  = get_img_name_to_no_matched_p_boxs(img_name_to_epoch_no_match_p_boxs)

                                                                                                                         
    img_to_clusters = get_img_to_clusters(img_name_to_no_matched_p_boxs,iou_thre=0.6)
    img_name_to_topsis_score = get_img_to_topsis_score(img_to_clusters,last_epoch)
    no_clusters_image_name_set = sorted(set(all_img_name_list) - set(img_name_to_topsis_score.keys()))
    print(f"Number of images without predicted clusters:{len(no_clusters_image_name_set)}")
    for img_name in no_clusters_image_name_set:
        img_name_to_topsis_score[img_name] = 0.0
    sorted_items = sorted(img_name_to_topsis_score.items(),key=lambda x: (-float(x[1]), x[0]))                
    ranked_image_name_list = []
    ranked_score_list = []
    for image_name,score in sorted_items:
        ranked_image_name_list.append(image_name)
        ranked_score_list.append(score)
    return ranked_image_name_list, ranked_score_list


def build_feature_beta(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
    """
    text ground truth box text epoch text(conf_list text iou_list),
    text.

    Parameters
    ----------
    g_box_id_to_metric : dict
        Dictionary keyed by the unique g_box ID; format example:
        {
            g_id: {
                "conf_list": [c_1, c_2, ..., c_T],  # text g_box text epoch text
                "iou_list":  [i_1, i_2, ..., i_T],  # text g_box text epoch text IoU text
            },
            ...
        }
    K : float
        textepochtext.text K*100% textepochtext,text K*100% textepochtext

    Returns
    ----------
    (g_id_to_features,feature_name_to_sign): tuple
    g_id_to_features: dict
        Dictionary keyed by the unique g_box ID; format example:
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
        text key text,text:
        {
            "early_conf_mean":-1, # smaller means more suspicious and gives a higher TOPSIS score
            "early_iou_mean":-1,
            "lastly_conf_mean":-1,
            "lastly_iou_mean":-1,
            "conf_mean":-1,
            "iou_mean":-1,
            "D_conf":1, # larger means more suspicious and gives a higher TOPSIS score
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

               
        conf_mean, iou_mean, match_rate = tong_ji_remove_0(conf_list,iou_list)
        
              
        early_conf_mean, early_iou_mean, early_match_rate = tong_ji_remove_0(early_conf_list,early_iou_list)
                                                                                                  

              
        lastly_conf_mean, lastly_iou_mean, lastly_match_rate = tong_ji_remove_0(lastly_conf_list,lastly_iou_list)
                                                                                                       

                
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
        # rise delay, explicitly modeling late increase
        # larger means more suspicious
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

    print(f"all gboxCount:{len(all_gids)}")
    print(f"matched gboxCount:{len(g_id_to_features)}")
    
                           
                                          
                                        
                               
               
    

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
                                                              
        early_conf_mean = np.mean(conf_list[0:W_e])
                                                             
        lastly_conf_mean = np.mean(conf_list[-W_l:])
                                                       
        early_iou_mean = np.mean(iou_list[0:W_e])
                                                      
        lastly_iou_mean = np.mean(iou_list[-W_l:])

                                                    
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
        "early_conf_mean":-1,                                
        "early_iou_mean":-1,
        "lastly_conf_mean":-1,
        "lastly_iou_mean":-1,
        "conf_mean":-1,
        "iou_mean":-1,
        "D_conf":1,
        "D_iou":1
    }

    print(f"all gboxCount:{len(all_gids)}")
    print(f"matched gboxCount:{len(g_id_to_features)}")
    
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
    return (g_id_to_features,feature_name_to_sign)

def rank_gid_beta(g_id_to_features, feature_name_to_sign: dict):
    """
    g_id_to_features: {g_id: {attr: (value, flag),},}
    """
    g_id_list = sorted(g_id_to_features.keys())             
    data = []
    id_to_gid = {}

               
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

    assert len(sign_list) > 0, "Invalid data"

    data_array = np.array(data)
    n_features = data_array.shape[1]
    assert n_features == len(sign_list), "Invalid data"

    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, sign_list)

                                 
    sorted_internal_ids = np.argsort(score_array)[::-1]

                 
    ranked_gid_list = [id_to_gid[i] for i in sorted_internal_ids]

    ranked_score_list = [float(score_array[i]) for i in sorted_internal_ids]

    return ranked_gid_list, ranked_score_list

def entropy_weight(data):
               
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    prob_matrix = data_normalized / data_normalized.sum(axis=0)
                           
    epsilon = 1e-9
    entropy = -np.sum(prob_matrix * np.log(prob_matrix + epsilon), axis=0) / np.log(len(data))
    diff_coeff = 1 - entropy / np.log(len(data))
    weights = diff_coeff / diff_coeff.sum()
    return weights



def rank_gid_original(g_id_to_features,feature_name_to_sign:dict):
    '''
    g_id_to_features:{g_id:{attr:(value,flag),},}
    '''
    g_id_list = list(g_id_to_features.keys())
    g_id_list.sort()            
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
        assert id == gid, "Invalid data"
    
    assert len(sign_list) > 0, "Invalid data"

    data_array = np.array(data)
    n_features = data_array.shape[1]
    assert data_array.shape[1] == len(sign_list), "Invalid data"

          
                                          
    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, sign_list)
                                        
    sorted_gt_id = np.argsort(score_array, kind="mergesort")[::-1]

    ranked_gid_list = [int(g_id) for g_id in sorted_gt_id]
    ranked_score_list = []
    for gid in ranked_gid_list:
        ranked_score_list.append(score_array[gid])
    return ranked_gid_list, ranked_score_list

def get_gid_level_rank(gt_json:dict,g_box_metrics_json_path:str):
    '''
    ourstextgid rank
    '''
              
    all_gids = get_all_gids(gt_json)
                        
    g_box_id_to_metric = get_g_id_to_metric(g_box_metrics_json_path)
                                  
    g_id_to_features,feature_name_to_sign = build_feature_orignal(all_gids,g_box_id_to_metric)
                                 
    ranked_gid_list, ranked_gid_score_list = rank_gid_original(g_id_to_features,feature_name_to_sign)
                                                                          
                                                        
                                                                                             
                                                                                                   
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
    '''clustertext'''
    '''Get image ranking of our method'''
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
    
    sorted_items = sorted(idd_to_score.items(),key=lambda x: (-float(x[1]), str(x[0])))                     
    for idd,score in sorted_items:
        merged_rank.append(idd)
    return merged_rank


def rank()->list:
              
    gt_json = read_json(gt_json_path)
                    
    ranked_gid_list,ranked_gid_score_list = get_gid_level_rank(gt_json,g_box_metrics_json_path)
    if _args["feature_level"] == "imgLevel":
                        
        img_rank_res = img_rank(img_to_nomatched_pboxs_json_path)                                       
        ranked_image_name_list = img_rank_res["ranked_imgs"]
        ranked_img_score_list = img_rank_res["ranked_scores"]
    elif _args["feature_level"] == "clusterLevel": 
        ranked_image_name_list,ranked_img_score_list = get_img_level_rank(imgs_dir,match_json_path)          
    else:
        raise Exception("textimg ranktextlevel feature")
                 
    alpha = _args["alpha"]
    total_rank = merge_rank(ranked_gid_list,ranked_gid_score_list,ranked_image_name_list,
                            ranked_img_score_list,alpha)
    return total_rank


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root_dir = config["exp_data_dir"]
                    
    _args = {
        "dataset_name":"voc",                     
        "model_name":"frcnn",                     
        "alpha":1.5,                              
    }
    _args["epochs"] = 50
    if _args["model_name"] == "rtdetr":
        _args["epochs"] = 100
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]
    epochs = _args["epochs"]

                     
    _args["save_dir"] = os.path.join(exp_data_root_dir,"ours",dataset_name,model_name,"rank")
    os.makedirs(_args["save_dir"],exist_ok=True)

    pprint.pprint(_args)
    
          

                              
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
                
    match_json_path = get_ours_gt_box_metric_path(dataset_name,model_name)
                  
    g_box_metrics_json_path = get_ours_match_path(dataset_name,model_name)

    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    predicted_bboxs_dir = get_collected_predict_boxes_dir(dataset_name,model_name)
                          
    img_to_nomatched_pboxs_json_path = get_img_to_nomatched_pboxs_json_path(dataset_name,model_name)

    imgs_dir = get_all_trainimgs_dir(dataset_name)

          
    total_rank = rank()
    print(f"Total ranking length:{len(total_rank)}")
    save_path = os.path.join(_args["save_dir"], "rank.joblib")
    joblib.dump(total_rank,save_path)
    print(f"rankingResult saved at:{save_path}")