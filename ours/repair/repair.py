

'''
基于方法(ours/datactive/other baselines)的排序结果，对label进行修复(repair) coco_style
'''
import os
import joblib
import json

import time
from datetime import datetime

from helper.base_data_manager import (get_collected_gt_box_json_path,
                                      get_error_ann_file_path,get_correct_ann_file_path,
                                      get_annotations_with_miss_json_path,
                                      get_rank_data_path)

from helper.data_organization_tools import (conver_ours_rank,conver_datactive_rank,
                                          get_img_name_to_ann_ids,get_annoId_to_anno,
                                          get_all_error_annoids,get_annoid_to_imgname)
from helper.conver_label_style import coco2yolo
from ours.small_utils import read_json,read_yaml
import pprint
from pycocotools.coco import COCO

def get_gid_to_img_and_line():
    res = {}
    with open(gt_json_path,mode='r') as f:
        gt_box = json.load(f)
    for img_name,g_box_list in gt_box.items():
        line_no = 0
        for g_box in g_box_list:
            gid = g_box["box_id"]
            res[gid] = {
                "img_name":img_name,
                "line_no":line_no
            }
            line_no += 1
    return res

def get_repair_info(converted_rank:list,anno_correct_json:dict, anno_error_json:dict,cut_off_rate:float)->dict:
    repair_info = {
        "miss":{}, # {imgname:[missed_annos]}
        "cls":{}, # {anno_id:correct_anno}
        "loc":{}, # {anno_id:correct_anno}
        "redun":{} # {redun_anno_id:anno}
    }


    cut_off = int(cut_off_rate*len(converted_rank))
    cutted_converted_rank = converted_rank[:cut_off]

    correct_imgname_to_annoids = get_img_name_to_ann_ids(anno_correct_json)
    error_imgname_to_annoids = get_img_name_to_ann_ids(anno_error_json)
    
    correct_annoId_to_anno = get_annoId_to_anno(anno_correct_json)
    error_annoId_to_anno = get_annoId_to_anno(anno_error_json)
    

    for idd in cutted_converted_rank:
        if type(idd) is str:
            image_name = idd
            missd_annos = [] # 用于存放该image的真正missed annos
            correct_anno_ids = correct_imgname_to_annoids[image_name]
            cur_anno_ids = error_imgname_to_annoids[image_name]
            # 正确的有，当前没有
            missed_anno_id_set = set(correct_anno_ids) - set(cur_anno_ids)
            missed_anno_id_list = list(missed_anno_id_set)
            for missed_anno_id in missed_anno_id_list:
                missed_anno = correct_annoId_to_anno[missed_anno_id]
                missd_annos.append(missed_anno)
            repair_info["miss"][image_name] = missd_annos
        else:
            anno_id = idd
            cur_anno = error_annoId_to_anno[anno_id]
            if cur_anno["fault_type"] == 1:
                 # cls fault
                 correct_anno = correct_annoId_to_anno[anno_id]
                 correct_anno["repair_ops"] = "repair_cls"
                 repair_info["cls"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 2:
                # loc fault
                correct_anno = correct_annoId_to_anno[anno_id]
                correct_anno["repair_ops"] = "repair_loc"
                repair_info["loc"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 3:
                # redun fault
                repair_info["redun"].append(anno_id)
    return repair_info
    

def repair_anno_json(cur_anno_json:dict,repair_info:dict)->dict:
    '''
    Args: 
    repair_info = {
        "miss":{}, # {imgname:[missed_annos]}
        "cls":{}, # {anno_id:correct_anno}
        "loc":{}, # {anno_id:correct_anno}
        "redun":[] # [redun_anno_ids]
    }
    '''
    miss_info = repair_info["miss"]
    cls_info = repair_info["cls"]
    loc_info = repair_info["loc"]
    redun_anno_id_list = list(repair_info["redun"].keys())

    annos = cur_anno_json["annotations"]
    # 修复 cls
    for anno in annos:
        anno_id = anno["id"]
        if anno_id in cls_info:
            correct_anno = cls_info[anno_id]
            anno["category_id"] = correct_anno["category_id"]
    # 修复 loc
    for anno in annos:
        anno_id = anno["id"]
        if anno_id in loc_info:
            correct_anno = loc_info[anno_id]
            anno["bbox"] = correct_anno["bbox"]

    # 修复mis
    for img_name,missed_annos in miss_info.items():
        annos.extend(missed_annos)

    # 修复redun(最后删除修复冗余annoid)
    new_annos = [anno for anno in annos if anno["id"] not in redun_anno_id_list]
    cur_anno_json["annotations"] = new_annos
    return cur_anno_json

def count_repair_info(repair_info:dict,anno_with_miss_error_json:dict,anno_error_json:dict):
    all_error_annoids = get_all_error_annoids(anno_with_miss_error_json) # 包括 miss fault
    annoid_to_imgname = get_annoid_to_imgname(anno_error_json)
    miss_info = repair_info["miss"]
    cls_info = repair_info["cls"]
    loc_info = repair_info["loc"]
    redun_info = repair_info["redun"]

    repaired_miss_img_name_set = set()
    repaired_cls_img_name_set = set()
    repaired_loc_img_name_set = set()
    repaired_redun_img_name_set = set()

    
    redun_anno_id_list = list(repair_info["redun"].keys())

    repaired_miss_box_count = 0 # 修复的miss fault box数量

    repaired_miss_img_name_set = set()
    for img_name, missed_annos in miss_info.items():
        repaired_miss_box_count += len(missed_annos)
        repaired_miss_img_name_set.add(img_name)

    for annid,anno in cls_info.items():
        img_name = annoid_to_imgname[annid]
        repaired_cls_img_name_set.add(img_name)

    for annid,anno in loc_info.items():
        img_name = annoid_to_imgname[annid]
        repaired_loc_img_name_set.add(img_name)

    for annid,anno in redun_info.items():
        img_name = annoid_to_imgname[annid]
        repaired_redun_img_name_set.add(img_name)

    repaired_imgname_set = repaired_miss_img_name_set | repaired_cls_img_name_set | repaired_loc_img_name_set | repaired_redun_img_name_set
    
    repaired_cls_count = len(cls_info)
    repaired_loc_count = len(loc_info)
    repaired_redun_count = len(redun_anno_id_list)

    total_repair_num = repaired_miss_box_count + repaired_cls_count + repaired_loc_count + repaired_redun_count
    repair_rate = round(total_repair_num / len(all_error_annoids),4)

    detail_info = {
        "repaired_cls_count":repaired_cls_count,
        "repaired_loc_count":repaired_loc_count,
        "repaired_redun_count":repaired_redun_count,
        "repaired_miss_box_count":repaired_miss_box_count,
        "repaired_cls_imgcount": len(repaired_cls_img_name_set),
        "repaired_loc_imgcount": len(repaired_loc_img_name_set),
        "repaired_redun_imgcount": len(repaired_redun_img_name_set),
        "repaired_miss_imgcount": len(repaired_miss_img_name_set),
        "repaired_imgcount": len(repaired_imgname_set)
    }

    count_info = {
        "all_fault_num": len(all_error_annoids),
        "total_repair_num": total_repair_num,
        "repair_rate": repair_rate,
        "detail_info":detail_info
    }
    return count_info


def repair_kit(converted_rank:list, anno_correct_json:dict, anno_error_json:dict, cut_off_rate:float) -> dict:
    repair_info = get_repair_info(
        converted_rank,
        anno_correct_json,
        anno_error_json,
        cut_off_rate)
    
    # 统计一下修复信息
    anno_with_miss_error_json = read_json(anno_with_miss_error_path)
    count_info = count_repair_info(repair_info,anno_with_miss_error_json)
    pprint.pprint(count_info,indent=4,sort_dicts=False)
    # 修复anno
    new_annos = repair_anno_json(anno_error_json,repair_info)

    return new_annos
    

def repair_kit_strict_cost(converted_rank:list, anno_correct_json:dict, anno_error_json:dict,
                 cut_off_rate:float=0.4):
    '''严格控制cost'''
    # 获得annonums
    total_imgnums = len(anno_error_json["images"])
    total_annoLen = len(anno_error_json["annotations"])
    total_len  = total_imgnums + total_annoLen
    cost = int(total_len * cut_off_rate)

    print(f"cost:{cost}")
    repair_info = {
        "miss":{}, # {imgname:[missed_annos]}
        "cls":{}, # {anno_id:correct_anno}
        "loc":{}, # {anno_id:correct_anno}
        "redun":{} # {anno_id:anno}
    }

    correct_imgname_to_annoids = get_img_name_to_ann_ids(anno_correct_json)
    error_imgname_to_annoids = get_img_name_to_ann_ids(anno_error_json)
    correct_annoId_to_anno = get_annoId_to_anno(anno_correct_json)
    error_annoId_to_anno = get_annoId_to_anno(anno_error_json)


    for idd in converted_rank:
        if cost <= 0: # 在取下个元素前先看看你还有没有cost
            # cost <= 0, 直接结束取取循环中的元素了
            break
        # 取出一个元素
        if type(idd) is str:
            # 遇到img
            image_name = idd
            correct_anno_ids = correct_imgname_to_annoids[image_name] # 该图像正确标注情况下所有的anno ids
            cur_anno_ids = error_imgname_to_annoids[image_name] # # 该图像含错标注情况下的所有正确的anno ids
            # 正确的有，当前没有
            missed_anno_id_set = set(correct_anno_ids) - set(cur_anno_ids)
            missed_anno_id_list = list(missed_anno_id_set)

            if len(missed_anno_id_list) == 0:
                # 这张图像不含miss fault
                cost -= 1
            else:
                missd_annos = [] # 用于存放该image的真正missed annos
                # 这张图像含miss fault
                for missed_anno_id in missed_anno_id_list:
                    if cost <= 0: # 在取下个元素前先看看你还有没有cost
                        break # # cost <= 0, 直接结束取取循环中的元素了
                    missed_anno = correct_annoId_to_anno[missed_anno_id]
                    missd_annos.append(missed_anno)
                    cost -= 1 # 每修复一个miss fault cost -= 1
                repair_info["miss"][image_name] = missd_annos
        else:
            # 遇到 anno id
            anno_id = idd
            cost -= 1
            cur_anno = error_annoId_to_anno[anno_id]
            if cur_anno["fault_type"] == 1:
                 # 如果该anno 是 cls fault
                 correct_anno = correct_annoId_to_anno[anno_id]
                 correct_anno["repair_ops"] = "repair_cls"
                 repair_info["cls"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 2:
                # 如果该anno 是 loc fault
                correct_anno = correct_annoId_to_anno[anno_id]
                correct_anno["repair_ops"] = "repair_loc"
                repair_info["loc"][anno_id] = correct_anno
                
            elif cur_anno["fault_type"] == 3:
                # 如果该anno 是 redunc fault
                repair_info["redun"][anno_id] = cur_anno
    print(f"remain cost:{cost}")
    # 统计一下修复信息
    anno_with_miss_error_json = read_json(anno_with_miss_error_path)
    count_info = count_repair_info(repair_info,anno_with_miss_error_json,anno_error_json)
    pprint.pprint(count_info,indent=4,sort_dicts=False)
    # 修复anno
    new_annos = repair_anno_json(anno_error_json,repair_info)
    return new_annos


def main():
    start_time = time.time()  # 记录开始时间

    # 正确标注json和错误标注json
    anno_error_json = read_json(anno_error_path)
    anno_correct_json = read_json(anno_correct_path)

    # 排序数据转换
    ours_and_otherbaseline = []
    ours_and_otherbaseline.append("ours")
    ours_and_otherbaseline.extend(other_baselines_list)
    if _args['rank_method'] in ours_and_otherbaseline:
        g_boxes_json = read_json(gt_json_path)
        converted_rank = conver_ours_rank(rank,g_boxes_json,anno_error_json)
    elif _args['rank_method'] == "datactive":
        coco = COCO(anno_error_path)
        bg_catId = coco.getCatIds()[-1]+1
        converted_rank = conver_datactive_rank(rank,bg_catId)
    else:
        raise Exception("rank method 参数错误")

    # 修复的标注json
    if _args["strict_cost"]:
        anno_repaired_json = repair_kit_strict_cost(converted_rank, anno_correct_json, anno_error_json, 
                                    cut_off_rate=_args["cut_off_rate"])
    else:
        anno_repaired_json = repair_kit(converted_rank, anno_correct_json, anno_error_json, 
                                     cut_off_rate=_args["cut_off_rate"])
    

    # 结果保存与计时
    if _args["is_save"]:
        with open(repair_anno_save_path,"w") as f:
            json.dump(anno_repaired_json,f)
        print(f"anno_repaired_json保存在: {repair_anno_save_path}")

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数
    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")
    now_timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"实验结束时刻: {now_timestr}")


if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    config = read_yaml("config.yaml")
    exp_root_dir = config['exp_data_dir']

    all_methods = config["all_methods"]
    _args = {
        "dataset_name":"VisDrone", # voc|kitti|visdrone
        "model_name":"YOLOv7", # yolov7|frcnn|rtdetr
        "rank_method":"ours", # ours|datactive|entropy|loss|deepgini|margin|objectlab
        "cut_off_rate": 0.5,
        "strict_cost":True, # box level的cost
        "is_save":True # 修复结果保存
    }
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]

    if _args["rank_method"] == "ours":
        _args["save_dir"] = os.path.join(exp_root_dir,"ours",dataset_name,model_name,"repair")
        _args["rank_data_path"] = get_rank_data_path(dataset_name,"ours", model_name)
    elif _args["rank_method"] in config["baselines"]:
        if _args["rank_method"] == "datactive":
            _args["save_dir"] = os.path.join(exp_root_dir,"baselines","datactive",dataset_name)
            _args["rank_data_path"] = get_rank_data_path(dataset_name,"datactive")
        else:
            _args["save_dir"] = os.path.join(exp_root_dir,"baselines",_args["rank_method"],dataset_name,model_name)
            _args["rank_data_path"] = get_rank_data_path(dataset_name,_args["rank_method"],model_name)
    else:
        raise Exception(f"{_args["rank_method"]} set error")

    pprint.pprint(_args)
    # 获得公共数据
    # 收集的gboxs
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    # anno_error
    anno_error_path = get_error_ann_file_path(dataset_name)
    # anno_error_withmiss
    anno_with_miss_error_path = get_annotations_with_miss_json_path(dataset_name)
    # anno correct
    anno_correct_path = get_correct_ann_file_path(dataset_name,"train")
    rank = joblib.load(_args["rank_data_path"])
    if _args["is_save"]:
        os.makedirs(_args["save_dir"],exist_ok=True)
        repair_anno_save_path = os.path.join(_args["save_dir"],"_annotations.coco_repair.json")
    main()
    # repaired anno json -> yolo format
    yolo_output_dir = os.path.join(_args["save_dir"],"yolo_format")
    os.makedirs(yolo_output_dir)
    coco2yolo(repair_anno_save_path,yolo_output_dir,"train")

