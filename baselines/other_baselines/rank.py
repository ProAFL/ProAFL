
import os
import math
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from queue import PriorityQueue
from helper.data_organization_tools import get_all_gids,get_all_errored_g_box_id_set
from helper.base_data_manager import get_collected_gt_box_json_path,get_all_img_name,get_annotations_with_miss_json_path,get_all_trainimgs_dir
from baselines.other_baselines.custom_module.small_utils import read_json,read_yaml

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
    score larger means more suspicious
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
    priority_queue = PriorityQueue()       
    matchid_gid_set = set()                           
    for gid_str in match_json.keys():
                 
        p_box = match_json[gid_str]["p_box"]
        g_box = match_json[gid_str]["g_box"]
        gid = g_box["box_id"]
        p_list = p_box["prob"]
        g_label = g_box["cls"]
        p_loc = p_box["bbox"]
        g_loc = g_box["gt_bbox"]
                                                    
        score = caclu_rank_score(baseline_name,p_list,g_label,p_loc,g_loc)
        priority_queue.put((-score,gid))             
        matchid_gid_set.add(gid)

    print(f"all gidCount:{len(all_gids)}")
    print(f"textgidCount:{len(matchid_gid_set)}")
    print(f"textgidCount:{len(all_gids) - len(matchid_gid_set)}")
    erro_gids = get_all_errored_g_box_id_set(g_json)
    print(f"fault gidCount:{len(erro_gids)}")
    no_matched_gid_set = set(all_gids) - matchid_gid_set
    bad_gid_set = set(erro_gids) & no_matched_gid_set
    print(f"textgidtextCount:{len(bad_gid_set)}/{len(erro_gids)}")

    for g_id in all_gids:
        if g_id not in matchid_gid_set:
            priority_queue.put((-100,g_id))

          
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
            raise Exception("Image path is incorrect")
    
    return rank


if __name__ == "__main__":
    config = read_yaml["config.yaml"]
    PID = os.getpid()
    print("PID:",PID)
    exp_root_dir = config["exp_data_dir"]
    dataset_name = "voc"                     
    model_name = "yolov7"                      
    baseline_name = "entropy"                                
    g_json_path = get_collected_gt_box_json_path(dataset_name)
    
    match_json_path = os.path.join(exp_root_dir, "collection_process_info",
                              dataset_name,model_name,"for_baselines","match.json")
    all_train_img_dir = get_all_trainimgs_dir(dataset_name)
    rank = main(g_json_path,match_json_path,baseline_name)
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)

                  
    save_dir = os.path.join(exp_root_dir,"baselines",baseline_name,
                            dataset_name,model_name,"rank")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "rank.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(rank,save_path)
    print(f"ranktext:{len(rank)}")
    print(f'rankResult saved at:{save_path}')
