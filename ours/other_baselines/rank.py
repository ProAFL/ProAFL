
import os
import math
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from ours.small_utils import read_json
from queue import PriorityQueue
from ours.data_organization_tools import get_all_gids,get_all_errored_g_box_id_set
from ours.base_data_manager import get_collected_gt_box_json_path,get_all_img_name,get_annotations_with_miss_json_path
from ours.rank_analyse.other_baselines_analyse import analyse_rank as other_baseline_analyse_rank

def calcu_entropy(prob_list):
    entropy = 0.0
    for p in prob_list:
        entropy -= p * math.log(p)
    return entropy

def caclu_deepgini(p_list):
    _sum = 0.0
    for p in p_list:
        _sum += p*p
    deep_gini = 1 - _sum
    return deep_gini


def caclu_margin(p_list):
    margin = 0.0
    p_list.sort()
    max_p = p_list[-1]
    second_max_p = p_list[-2]
    margin = (1 - (max_p-second_max_p)) ** 2
    return margin

def caclu_loss(p_list,g_label,p_loc,g_loc):
    ce_loss = F.cross_entropy(torch.tensor([p_list],dtype=torch.float32),torch.tensor([g_label]))
    smooth_l1_loss = nn.SmoothL1Loss()
    sl1_loss =  smooth_l1_loss(torch.tensor(p_loc),torch.tensor(g_loc))
    loss = ce_loss + sl1_loss
    loss = loss.item()
    return loss




def caclu_rank_score(baseline_name, p_list,g_label,p_loc,g_loc):
    '''
    score 越大越可疑
    '''
    if baseline_name == "entropy":
        score = calcu_entropy(p_list)
    elif baseline_name == "deepgini":
        score = caclu_deepgini(p_list)
    elif baseline_name == "margin":
        score = caclu_margin(p_list)
    elif baseline_name == "loss":
        score = caclu_loss(p_list,g_label,p_loc,g_loc)
    else:
        raise Exception("baseline name error")
    return score

def main(g_json_path,match_json_path,baseline_name:str):
    g_json = read_json(g_json_path)
    match_json = read_json(match_json_path)
    all_gids = get_all_gids(g_json)
    priority_queue = PriorityQueue() # 越小优先级越高
    matchid_gid_set = set() # 存储匹配到p_box的g_box的id
    for gid_str in match_json.keys():
        # gid:'1'
        p_box = match_json[gid_str]["p_box"]
        g_box = match_json[gid_str]["g_box"]
        gid = g_box["box_id"]
        p_list = p_box["prob"]
        g_label = g_box["cls"]
        p_loc = p_box["bbox"]
        g_loc = g_box["gt_bbox"]
        # 分数越大越可疑，优先级越高，越排在队头
        score = caclu_rank_score(baseline_name,p_list,g_label,p_loc,g_loc)
        priority_queue.put((-score,gid))# entropy越大优先级越高
        matchid_gid_set.add(gid)

    print(f"all gid数量:{len(all_gids)}")
    print(f"匹配上的gid数量:{len(matchid_gid_set)}")
    print(f"没匹配上的gid数量:{len(all_gids) - len(matchid_gid_set)}")
    erro_gids = get_all_errored_g_box_id_set(g_json)
    print(f"fault gid数量:{len(erro_gids)}")
    no_matched_gid_set = set(all_gids) - matchid_gid_set
    bad_gid_set = set(erro_gids) & no_matched_gid_set
    print(f"没匹配上的gid中的错误数量:{len(bad_gid_set)}/{len(erro_gids)}")

    for g_id in all_gids:
        if g_id not in matchid_gid_set:
            priority_queue.put((-100,g_id))

    # 获取并弹出优先级最高的元素
    gid_rank = []
    while not priority_queue.empty():
        priority, g_id = priority_queue.get()
        gid_rank.append(g_id)

    rank = []
    all_img_name_list = get_all_img_name(all_train_img_dir)
    rank.extend(gid_rank)
    rank.extend(all_img_name_list)
    
    for idd in rank[-len(all_img_name_list):]:
        if type(idd) is int:
            raise Exception("图片位置放错了")
    
    return rank


if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "rtdetr" # YOLOv7|FRCNN|SSD|rtdetr
    exp_id = "01"
    baseline_name = "margin" # entropy|loss|deepgini|margin|
    g_json_path = get_collected_gt_box_json_path(dataset_name)
    match_json_path = os.path.join(exp_root_dir, "collection_bbox_level",
                              dataset_name,model_name,"other_baselines","match.json")
    all_train_img_dir = os.path.join(exp_root_dir,"retrain_dataset_split", dataset_name,
                                    "images", "origin")
    rank = main(g_json_path,match_json_path,baseline_name)
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)

    # 保存rank数据    
    save_dir = os.path.join(exp_root_dir,"Results","other_baselines",baseline_name,
                            dataset_name,model_name,f"exp_{exp_id}","rank")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "rank.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(rank,save_path)
    print(f"rank长度为:{len(rank)}")
    print(f'rank结果保存在:{save_path}')
    
    # 对排序结果进行一个简单性能分析
    other_baseline_analyse_rank(dataset_name,g_json_path,rank,annos_with_miss_json_path,vis=False)
