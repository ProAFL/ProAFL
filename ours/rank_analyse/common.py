
import os
import scienceplots # sci绘图包
import matplotlib
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from ours.base_data_manager import exp_data_root_dir
from ours.data_organization_tools import (get_imgid_to_imgname,get_annoid_to_imgname,
                                          get_all_miss_error_img_name_set,get_all_annoids_detail)

def draw_rank_hot(isError_list,save_path):
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isError_list]
    # 绘制热力图
    # 创建图形时设置较小的高度
    plt.style.use(['science','ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })
    plt.figure(figsize=(3, 0.5))  # 宽度为10，高度为2（可根据需要调整）
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    # plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking',fontsize='3')
    # 调整横轴刻度字号
    plt.xticks(fontsize=3)  # 明确设置横轴刻度字号为6pt
    # plt.colorbar()
    plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()

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
    apfd = round(apfd,3)
    return apfd

def calc_fpr_fnr_f1(rank_list,error_set,cut_off=0.4):
    cut_point = int(len(rank_list) * cut_off)
    P_list = rank_list[:cut_point] # 预测为P
    N_list = rank_list[cut_point:] # 预测为N
    fp = 0
    fn = 0
    tp = 0
    correct_set = set(rank_list) - error_set
    for idd in P_list:
        if idd not in error_set:
            fp += 1 # 错阳
        else:
            tp += 1 # 正确阳
    for idd in N_list:
        if idd in error_set:
            fn += 1 # 错阴
    fpr = fp / len(correct_set)
    fnr = fn / len(error_set)
    fpr = round(fpr,3)
    fnr = round(fnr,3)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision + recall)
    f1 = round(f1,3)
    return fpr,fnr,f1


def calc_top1(annos_with_miss_json:dict,rank_list,error_set,error_imageset):
    annoid2imgname = get_annoid_to_imgname(annos_with_miss_json)
    imgname2rankedcompoents = get_imgname_to_ranked_components(rank_list,annoid2imgname)

    mingzhong_count = 0
    img_nums = 0
    for imgname in error_imageset:
        rankedcomponents = imgname2rankedcompoents[imgname]
        img_nums += 1
        if len(rankedcomponents) == 0:
            # 这张图像没有排序组件
            continue
        if rankedcomponents[0] in error_set:
            mingzhong_count+=1         
    return round(mingzhong_count/img_nums,3)

def calc_exam(annos_with_miss_json:dict,rank_list):
    imgs_group = get_imgs_group_by_fault(annos_with_miss_json)
    annoid2imgname = get_annoid_to_imgname(annos_with_miss_json)
    imgname2rankedcompoents = get_imgname_to_ranked_components(rank_list,annoid2imgname)

    '''
    fault2annoids = {
        "class_fault":[],
        "loc_fault":[],
        "redun_fault":[],
        "miss_fault":[],
        "clean":[]
    }
    '''
    fault2annoids = get_all_annoids_detail(annos_with_miss_json)
    missfault_imgname_set = get_all_miss_error_img_name_set(annos_with_miss_json)
    faultid2faultname = {
        0:"clean",
        1:"class_fault",
        2:"loc_fault",
        3:"redun_fault",
        4:"miss_fault"
    }
    exam_list = []
    for fault_id,imgset in imgs_group.items(): # fault_id:[1,2,3,4]
        faultset = None
        if fault_id != 4:
            fault_name = faultid2faultname[fault_id]
            annoids = fault2annoids[fault_name]
            faultset = set(annoids)
        else:
            faultset = missfault_imgname_set
        exam_one_fault = exam_by_one_fault(imgset,imgname2rankedcompoents,faultset)
        exam_list.append(exam_one_fault)
    exam = round(sum(exam_list)/len(exam_list),3)
    return exam

def get_imgs_group_by_fault(annos_with_miss_json):
    imgid2imgname = get_imgid_to_imgname(annos_with_miss_json)
    group = defaultdict(set[str])
    annos = annos_with_miss_json["annotations"]
    for anno in annos:
        imgname = imgid2imgname[anno["image_id"]]
        if anno["fault_type"] != 0:
            group[anno["fault_type"]].add(imgname)
    return group

def get_imgname_to_ranked_components(rank_list,annoid2imgname:dict):
    imgname_to_ranked_components = defaultdict(list)
    for idd in rank_list:
        imgname = None
        if type(idd) is str:
            imgname = idd
        else:
            annoid = idd
            imgname = annoid2imgname[annoid]
        imgname_to_ranked_components[imgname].append(idd)
    return imgname_to_ranked_components

def exam_by_one_fault(imgname_set:set,imgname2rankedcompoents,faultset):
    exam_list = []
    for imgname in imgname_set:
        ranked_components = imgname2rankedcompoents[imgname]
        if len(ranked_components) == 0:
            exam_list.append(0)
            continue
        exam_count = len(ranked_components)
        for idx,component in enumerate(ranked_components):
            if component in faultset:
                exam_count = idx+1
                break
        exam_list.append(exam_count/len(ranked_components))
    return sum(exam_list)/len(exam_list)

def draw_total_rank(error_flag_list, save_path):
    # error_flag_list: 包含 0/1/2 的列表

    distribution = list(error_flag_list)
    n = len(distribution)
    split_idx = int(n * 0.4)  # 40% 位置对应的列索引

    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })

    # 自定义 colormap：0 -> 白色；1 -> 红色；2 -> 蓝色
    cmap = ListedColormap(['white', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(3, 0.5))
    plt.imshow(
        [distribution],
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )

    # 在 40% 处画一条黑色分割线（画在像素列 split_idx 的左边界）
    ax = plt.gca()
    ax.axvline(x=split_idx - 0.5, color='black', linewidth=0.5)

    plt.xlabel('ranking', fontsize=3)
    plt.xticks(fontsize=3)
    plt.yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.close()

def look_total_rank(total_rank,all_errored_g_box_id_set,all_miss_error_img_name_set, pic_save_path):
    total_error_set = all_errored_g_box_id_set | all_miss_error_img_name_set
    error_flags = []
    for idd in total_rank:
        if idd in total_error_set:
            if type(idd) is int:
                error_flags.append(1) # red, box id
            else:
                error_flags.append(2) # blue, img
        else:
            error_flags.append(0)
    draw_total_rank(error_flags, pic_save_path)
    print(f"图片保存在：{pic_save_path}")

def look_img_rank(ranked_img_name_list:list[str], all_miss_error_img_name_set:set[str]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "image_name_rank.png")
    error_flag_list = []
    for img_name in ranked_img_name_list:
        if img_name in all_miss_error_img_name_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")
