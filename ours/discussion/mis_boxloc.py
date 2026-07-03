'''
textmiss fault boxtextlocationtext
'''


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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from helper.base_data_manager import (exp_data_root_dir,
                                    get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path
                                    )
from helper.data_organization_tools import get_imgid_to_imgname
from ours.small_utils import read_json


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

def bbox_iou(box1, box2):
    """
    textIoU.
    box1: [x_min, y_min, x_max, y_max]
    box2: [x_min, y_min, x_max, y_max]
    """
          
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
          
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    if inter_area == 0:
        return 0.0
    
          
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
          
    union_area = box1_area + box2_area - inter_area
    
             
    iou = inter_area / union_area
    return iou

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


def missed_box_to_xyxy(missed_box:dict) -> list:
    x, y, w, h = missed_box["bbox"]
    return [int(x), int(y), int(x + w), int(y + h)]


def cluster_locates_any_missed_box(cluster:list, missed_boxs:list, iou_threshold:float):
    '''
    textcandidatesclustertextmissed box.

    Returns:
    ---
    hit_flag: bool
        clustertextp_boxtextmissed boxtextIoUtext
    max_iou: float
        clustertextp_boxtextmissed boxtextIoU
    '''
    max_iou = 0.0
    for missed_box in missed_boxs:
        missed_bbox = missed_box_to_xyxy(missed_box)                          
        for p_box in cluster:
            iou = calu_iou(missed_bbox, p_box["bbox"])
            max_iou = max(max_iou, iou)
            if iou > iou_threshold:
                return True, max_iou
    return False, max_iou


def evaluate_cluster_precision_at_iou(img_to_clusters:dict,           
                                      imgname_to_missed_boxs:dict,
                                      iou_threshold:float) -> dict:
    '''
    textIoUtextprecision.

    textTPtext:
    clustertext1textp_boxtextmiss boxtextIoU > iou_threshold,
    textclustertextTP;textFP.
    '''
    tp = 0
    fp = 0
    max_iou_list = []
    total_missed_box_count = sum(len(v) for v in imgname_to_missed_boxs.values())
    located_missed_box_set = set()
    for img_name, clusters in img_to_clusters.items():
        missed_boxs = imgname_to_missed_boxs.get(img_name, [])
        for cluster in clusters:
            hit_flag, max_iou = cluster_locates_any_missed_box(
                cluster, missed_boxs, iou_threshold)
            max_iou_list.append(max_iou)
            if hit_flag:
                tp += 1
                for missed_idx, missed_box in enumerate(missed_boxs):
                    missed_bbox = missed_box_to_xyxy(missed_box)
                    for p_box in cluster:
                        if calu_iou(missed_bbox, p_box["bbox"]) > iou_threshold:
                            located_missed_box_set.add((img_name, missed_idx))
                            break
            else:
                fp += 1

    total_pred_clusters = tp + fp
    precision = tp / total_pred_clusters if total_pred_clusters > 0 else 0.0
    recall = (len(located_missed_box_set) / total_missed_box_count
              if total_missed_box_count > 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    result = {
        "iou_threshold": float(iou_threshold),
        "pred_cluster_count": total_pred_clusters,
        "tp_cluster_count": tp,
        "fp_cluster_count": fp,
        "total_missed_box_count": total_missed_box_count,
        "located_missed_box_count": len(located_missed_box_set),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "max_iou_mean": float(np.mean(max_iou_list)) if max_iou_list else 0.0,
        "max_iou_median": float(np.median(max_iou_list)) if max_iou_list else 0.0,
    }

    print("\n[Cluster Precision @ IoU]")
    pprint.pprint({
        "iou_threshold": result["iou_threshold"],
        "pred_cluster_count": result["pred_cluster_count"],
        "tp_cluster_count": result["tp_cluster_count"],
        "fp_cluster_count": result["fp_cluster_count"],
        "total_missed_box_count": result["total_missed_box_count"],
        "located_missed_box_count": result["located_missed_box_count"],
        "precision": round(result["precision"], 4),
        "recall": round(result["recall"], 4),
        "f1": round(result["f1"], 4),
        "max_iou_mean": round(result["max_iou_mean"], 4),
        "max_iou_median": round(result["max_iou_median"], 4),
    })
    return result


def build_cluster_rank_records(img_to_clusters:dict,
                               imgname_to_missed_boxs:dict,
                               last_epoch:int,
                               iou_threshold:float) -> list[dict]:
    '''
    textclustertextrankingtext.

    textclustertext:
    - score: clustertextTOPSIStext,larger means more suspicious
    - label: textclustertextmissed box
    - max_iou_to_missed: textmissed boxestextIoU
    '''
    cluster_features = []
    cluster_signs = []
    cluster_meta = []

    for img_name, clusters in img_to_clusters.items():
        missed_boxs = imgname_to_missed_boxs.get(img_name, [])
        for cluster_id, cluster in enumerate(clusters):
            features, signs = get_cluster_feaure(cluster, last_epoch)
            hit_flag, max_iou = cluster_locates_any_missed_box(
                cluster, missed_boxs, iou_threshold)
            cluster_features.append(features)
            cluster_signs = signs
            cluster_meta.append({
                "img_name": img_name,
                "cluster_id": cluster_id,
                "label": int(hit_flag),
                "max_iou_to_missed": float(max_iou),
                "cluster_size": len(cluster),
            })

    if not cluster_features:
        return []

    data_array = np.array(cluster_features)
    weights = np.ones(data_array.shape[1]) / data_array.shape[1]
    _, score_array = tp.topsis(data_array, weights, cluster_signs)
    score_array = np.nan_to_num(score_array, nan=0.0, posinf=1.0, neginf=0.0)

    records = []
    for meta, features, score in zip(cluster_meta, cluster_features, score_array):
        record = dict(meta)
        record["score"] = float(score)
        record["features"] = {
            "conf": float(features[0]),
            "stab": float(features[1]),
            "cls_consis": float(features[2]),
            "e_freq": float(features[3]),
        }
        records.append(record)
    records.sort(key=lambda x: (-x["score"], x["img_name"], x["cluster_id"]))
    return records


def best_f1_from_scores(y_true:np.ndarray, y_score:np.ndarray) -> dict:
    '''
    textscoretext,ReturnstextF1textprecision/recall/threshold.
    text: score >= threshold textmiss faulttextcluster.
    '''
    best = {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": int(np.sum(y_true == 1)),
    }
    for threshold in sorted(set(y_score.tolist()), reverse=True):
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best


def print_topk_cluster_metrics(records:list[dict], positive_count:int):
    topk_specs = [
        ("top1", 1),
        ("top5", 5),
        ("top10", 10),
        ("top1%", max(1, int(len(records) * 0.01))),
        ("top5%", max(1, int(len(records) * 0.05))),
        ("top10%", max(1, int(len(records) * 0.10))),
        ("top20%", max(1, int(len(records) * 0.20))),
    ]

    print("\n[Cluster Top-K text]")
    print(f"{'scope':<10}{'k':>8}{'hits':>8}{'precision':>12}"
          f"{'recall':>12}{'f1':>12}")
    for name, k in topk_specs:
        k = min(k, len(records))
        selected = records[:k]
        hits = sum(r["label"] for r in selected)
        precision = hits / k if k > 0 else 0.0
        recall = hits / positive_count if positive_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        print(f"{name:<10}{k:>8}{hits:>8}{precision:>12.4f}"
              f"{recall:>12.4f}{f1:>12.4f}")


def evaluate_cluster_ranking(img_to_clusters:dict,
                             imgname_to_missed_boxs:dict,
                             last_epoch:int,
                             iou_threshold:float,
                             save_dir:str):
    '''
    text"textclustertextmiss fault".

    label=1: clustertextp_boxtextmissed boxtextIoU > iou_threshold
    score: cluster TOPSIStext
    '''
    records = build_cluster_rank_records(
        img_to_clusters, imgname_to_missed_boxs, last_epoch, iou_threshold)
    if not records:
        print("\n[Cluster Ranking] textcandidatescluster,text.")
        return

    y_true = np.array([r["label"] for r in records], dtype=int)
    y_score = np.array([r["score"] for r in records], dtype=float)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))

    print("\n[Cluster Ranking ROC/AUC/F1]")
    print(f"clustertext: {len(records)}")
    print(f"textmiss faulttextclusterCount: {positive_count}")
    print(f"textmiss faulttextclusterCount: {negative_count}")

    if positive_count == 0 or negative_count == 0:
        print("text,textROC/AUCtextF1text.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    print(f"cluster_score_auc: {roc_auc:.4f}")

    best_f1 = best_f1_from_scores(y_true, y_score)
    print("best_f1_threshold_eval:")
    pprint.pprint({
        "threshold": round(best_f1["threshold"], 6),
        "precision": round(best_f1["precision"], 4),
        "recall": round(best_f1["recall"], 4),
        "f1": round(best_f1["f1"], 4),
        "tp": best_f1["tp"],
        "fp": best_f1["fp"],
        "fn": best_f1["fn"],
    })

    print_topk_cluster_metrics(records, positive_count)

    os.makedirs(save_dir, exist_ok=True)
    roc_save_path = os.path.join(
        save_dir, f"cluster_loc_roc_iou_{iou_threshold}.png")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cluster ranking ROC for miss-fault localization")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_save_path, dpi=150)
    plt.close()
    print(f"ROCtextSave path: {roc_save_path}")


def get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids:dict,
                                              last_epoch: int=5, conf_threshold: float=0.6):
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

                                                                                        
    img_name_to_epoch_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(
        epoch_to_matched_p_ids,last_epoch,conf_threshold)

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
    return ranked_image_name_list, ranked_score_list, img_to_clusters


def img_name_to_missed_box_list(annos_with_miss_json:dict)->dict:
    imgname_to_missed_box = defaultdict(list)
                      
    imgId_to_imgName = get_imgid_to_imgname(annos_with_miss_json)
    annos = annos_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgId_to_imgName[anno["image_id"]]
            imgname_to_missed_box[img_name].append(anno)                                    
    return imgname_to_missed_box

def main():
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    '''Get image ranking of our method'''
    ranked_image_name_list,ranked_img_score_list, img_to_clusters = rank_img_name(all_img_name_list, gt_match_json)
    imgname_to_missed_boxs = img_name_to_missed_box_list(annos_with_miss_json)

    r = evaluate_cluster_precision_at_iou(
        img_to_clusters, imgname_to_missed_boxs, iou_threshold)
    print()

    '''
    print("textmiss faulttextimgCount:", len(imgname_to_missed_boxs.keys()))
    for img_name, missed_boxs in imgname_to_missed_boxs.items():
        for missed_box in missed_boxs:
            missed_box["success_loc_flag"] = False
            missed_box["loc_p_box_list"] = []

    for img_name, missed_boxs in imgname_to_missed_boxs.items():
        clusters = img_to_clusters[img_name]
        # # textimgtextmissed boxs
        for missed_box in missed_boxs:
            missed_bbox = missed_box_to_xyxy(missed_box)
            # textimgtext
            for cluster in clusters:
                for p_box in cluster:
                    px1,py1,px2,py2 = p_box["bbox"]
                    p_bbox = [px1,py1,px2,py2]
                    iou = bbox_iou(missed_bbox,p_bbox)
                    iou2 = calu_iou(missed_bbox,p_bbox)
                    assert iou == iou2, "ioutext"
                    if iou > iou_threshold:
                        missed_box["success_loc_flag"] = True
                        missed_box["loc_p_box_list"].append(p_box)
                        break
                if missed_box["success_loc_flag"] is True:
                    break

    total_missed_box_nums = 0
    loced_nums = 0
    for imgname, missed_boxs in imgname_to_missed_boxs.items():
        total_missed_box_nums += len(missed_boxs)
        for missed_box in missed_boxs:
            if missed_box["success_loc_flag"] is True:
                loced_nums += 1
    print("textmissed_boxCount:",total_missed_box_nums)
    print("loctextCount:",loced_nums)
    loc_success_rate = round(loced_nums / total_missed_box_nums,4)
    print("misloc_recall_rate:",loc_success_rate)
    evaluate_cluster_precision_at_iou(
        img_to_clusters, imgname_to_missed_boxs, iou_threshold)

    save_dir = os.path.join(os.path.dirname(__file__), "features", "results",
                            "mis_loc", dataset_name, model_name)
    evaluate_cluster_ranking(img_to_clusters, imgname_to_missed_boxs,
                             last_epoch=5,
                             iou_threshold=iou_threshold,
                             save_dir=save_dir)
    '''


if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    iou_threshold = 0.5                        

                    
    _args = {
        "dataset_name":"VOC2012",                           
        "model_name":"YOLOv7",                  
        "epochs":50,
        "loc_iou_threshold": iou_threshold
    }
    
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]
    epochs = _args["epochs"]

    pprint.pprint(_args)


                              
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
                
    if dataset_name == "VisDrone":
        match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name, "gp_box_match",
                                   "match_v3.json")                  
    else:
        match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name, "gp_box_match",
                                   "match_v2.json")
    if not os.path.exists(match_json_path):
              
        match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,model_name,
                                   "match.json")
    
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    annos_with_miss_json = read_json(annos_with_miss_json_path)
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                       dataset_name,model_name,"collected_predicted_box","v2")
                                                
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split", dataset_name,
                             "images", "origin")
          
    main()
