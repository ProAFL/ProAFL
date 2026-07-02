
import os
import numpy as np
from ours.base_data_manager import (exp_data_root_dir,
                                    get_collected_gt_box_json_path
                                    )
from ours.small_utils import read_json
from ours.data_organization_tools import (get_all_gids,get_g_id_to_metric,
                                          get_all_errored_g_box_id_set,get_all_correct_g_box_id_set)
import topsispy as tp

from ours.rank.img_rank import analyze_feature_importance
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

def split_gid_clean_error(gt_json):
    error_gid_set = get_all_errored_g_box_id_set(gt_json)
    correct_gid_set = get_all_correct_g_box_id_set(gt_json)
    return correct_gid_set,error_gid_set

def rank_gid(g_id_to_features,feature_name_to_sign:dict):
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

    # 熵权法
    # weights = entropy_weight(data_array)
    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(data_array, weights, sign_list)
    # 从大到小排序并返回索引
    sorted_gt_id = np.argsort(score_array, kind="mergesort")[::-1]

    ranked_gid_list = [int(g_id) for g_id in sorted_gt_id]
    ranked_score_list = []
    for gid in ranked_gid_list:
        ranked_score_list.append(score_array[gid])
    return ranked_gid_list, ranked_score_list

def build_gid_feature(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
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

        min_e_conf = 0 # epochs(按理说这初始值更为合理)
        min_e_iou = 0 # epochs(按理说这初始值更为合理)
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



def rank_analyse(rank_res):
    ranked_gids = rank_res["ranked_gids"]
    fault_gidset = rank_res["fault_gidset"]
    feature_names = rank_res["feature_names"]

    print(f"总共的gid数量:{len(ranked_gids)}")
    print(f"包含fault的gid数量:{len(fault_gidset)}")
    apfd = compute_apfd(fault_gidset,ranked_gids)
    print(f"apfd:{apfd}")

    X = rank_res["feature_data"]
    Y = rank_res["label"]
    importance_df = analyze_feature_importance(X, Y, feature_names)
    print(importance_df)

def box_rank(gt_json_path,metrics_json_path):
    # 读取gbox
    gt_json = read_json(gt_json_path)
    # 获得所有的gids
    all_gids = get_all_gids(gt_json)
    # 获得每个gid的metrics
    g_box_id_to_metric = get_g_id_to_metric(metrics_json_path)
    # 获得每个gid对应的features和signs
    g_id_to_features,feature_name_to_sign = build_gid_feature(all_gids,g_box_id_to_metric)
    ranked_gid_list, ranked_gid_score_list = rank_gid(g_id_to_features,feature_name_to_sign)
    error_gid_set = get_all_errored_g_box_id_set(gt_json)

    ranked_isFault_list = []
    for gid in ranked_gid_list:
        if gid in error_gid_set:
            ranked_isFault_list.append(1)
        else:
            ranked_isFault_list.append(0)


    feature_names = []
    sign_list = []
    for feature_name, sign in feature_name_to_sign.items():
        feature_names.append(feature_name)
        sign_list.append(sign)


    X = []
    Y = []
    for gid in ranked_gid_list:
        features = []
        for feature_name in feature_names:
            features.append(g_id_to_features[gid][feature_name])
        X.append(features)
        if gid in error_gid_set:
            Y.append(1)
        else:
            Y.append(0)
    X = np.array(X)
    Y = np.array(Y)
    rank_res = {
        "ranked_gids":ranked_gid_list,
        "ranked_scores": ranked_gid_score_list,
        "ranked_isFault_list": ranked_isFault_list,
        "fault_gidset": error_gid_set,
        "feature_names": feature_names,
        "sign_list":sign_list,
        "feature_data": X,
        "label":Y
    }

    return rank_res


def main():
    
    rank_res = box_rank(gt_json_path,metrics_json_path)
    # 分析
    rank_analyse(rank_res)
    # 可视化
    # pic_save_dir = os.path.join(exp_root_dir,"img_rank","max")
    # pic_save_file_name = "rank.png"
    # pic_save_path = os.path.join(pic_save_dir,pic_save_file_name)
    # rank_vis(rank_res,pic_save_path)

if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    epochs = 50
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    match_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,
                                   "gp_box_match","match_v2.json")
    metrics_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"collection_metric",
                                           "collection_metrics_v2.json")
    main()