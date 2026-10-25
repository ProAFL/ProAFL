'''
repair coco_style
'''
from copy import deepcopy
import os
import joblib
import json
from pathlib import Path

import time
from datetime import datetime

from helper.base_data_manager import (get_collected_gt_box_json_path,
                                      get_error_ann_file_path,get_correct_ann_file_path,
                                      get_annotations_with_miss_json_path,
                                      get_rank_data_path)

from helper.data_organization_tools import (conver_ours_rank,conver_datactive_rank,
                                          get_img_name_to_ann_ids,get_annoId_to_anno,
                                          get_all_error_annoids,get_annoid_to_imgname)
from helper.convert_labels import coco2yolo
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
        "miss":{},                           
        "cls":{},                         
        "loc":{},                         
        "redun":{}                       
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
            missd_annos = []                            
            correct_anno_ids = correct_imgname_to_annoids[image_name]
            cur_anno_ids = error_imgname_to_annoids[image_name]
                       
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
                            
                 correct_anno = correct_annoId_to_anno[anno_id]
                 correct_anno["repair_ops"] = "repair_cls"
                 repair_info["cls"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 2:
                           
                correct_anno = correct_annoId_to_anno[anno_id]
                correct_anno["repair_ops"] = "repair_loc"
                repair_info["loc"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 3:
                             
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
              
    for anno in annos:
        anno_id = anno["id"]
        if anno_id in cls_info:
            correct_anno = cls_info[anno_id]
            anno["category_id"] = correct_anno["category_id"]
              
    for anno in annos:
        anno_id = anno["id"]
        if anno_id in loc_info:
            correct_anno = loc_info[anno_id]
            anno["bbox"] = correct_anno["bbox"]

             
    for img_name,missed_annos in miss_info.items():
        annos.extend(missed_annos)

                           
    new_annos = [anno for anno in annos if anno["id"] not in redun_anno_id_list]
    cur_anno_json["annotations"] = new_annos
    return cur_anno_json

def count_repair_info(repair_info:dict,anno_with_miss_error_json:dict,anno_error_json:dict):
    all_error_annoids = get_all_error_annoids(anno_with_miss_error_json)                  
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

    repaired_miss_box_count = 0                          

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
    
          
    anno_with_miss_error_json = read_json(anno_with_miss_error_path)
    count_info = count_repair_info(repair_info,anno_with_miss_error_json)
    pprint.pprint(count_info,indent=4,sort_dicts=False)
              
    new_annos = repair_anno_json(anno_error_json,repair_info)

    return new_annos
    

def repair_kit_strict_cost(converted_rank:list, anno_correct_json:dict, anno_error_json:dict,
                 cut_off_rate:float=0.4):
    '''textcost'''
                  
    total_imgnums = len(anno_error_json["images"])
    total_annoLen = len(anno_error_json["annotations"])
    total_len  = total_imgnums + total_annoLen
    cost = int(total_len * cut_off_rate)

    print(f"cost:{cost}")
    repair_info = {
        "miss":{},                           
        "cls":{},                         
        "loc":{},                         
        "redun":{}                 
    }

    correct_imgname_to_annoids = get_img_name_to_ann_ids(anno_correct_json)
    error_imgname_to_annoids = get_img_name_to_ann_ids(anno_error_json)
    correct_annoId_to_anno = get_annoId_to_anno(anno_correct_json)
    error_annoId_to_anno = get_annoId_to_anno(anno_error_json)


    for idd in converted_rank:
        if cost <= 0:           
                             
            break
              
        if type(idd) is str:
                     
            image_name = idd
            correct_anno_ids = correct_imgname_to_annoids[image_name]               
            cur_anno_ids = error_imgname_to_annoids[image_name]                 
                       
            missed_anno_id_set = set(correct_anno_ids) - set(cur_anno_ids)
            missed_anno_id_list = list(missed_anno_id_set)

            if len(missed_anno_id_list) == 0:
                                
                cost -= 1
            else:
                missd_annos = []                            
                                
                for missed_anno_id in missed_anno_id_list:
                    if cost <= 0:           
                        break                    
                    missed_anno = correct_annoId_to_anno[missed_anno_id]
                    missd_annos.append(missed_anno)
                    cost -= 1                           
                repair_info["miss"][image_name] = missd_annos
        else:
                          
            anno_id = idd
            cost -= 1
            cur_anno = error_annoId_to_anno[anno_id]
            if cur_anno["fault_type"] == 1:
                                          
                 correct_anno = correct_annoId_to_anno[anno_id]
                 correct_anno["repair_ops"] = "repair_cls"
                 repair_info["cls"][anno_id] = correct_anno
            elif cur_anno["fault_type"] == 2:
                                         
                correct_anno = correct_annoId_to_anno[anno_id]
                correct_anno["repair_ops"] = "repair_loc"
                repair_info["loc"][anno_id] = correct_anno
                
            elif cur_anno["fault_type"] == 3:
                                            
                repair_info["redun"][anno_id] = cur_anno
    print(f"remain cost:{cost}")
          
    anno_with_miss_error_json = read_json(anno_with_miss_error_path)
    count_info = count_repair_info(repair_info,anno_with_miss_error_json,anno_error_json)
    pprint.pprint(count_info,indent=4,sort_dicts=False)
              
    new_annos = repair_anno_json(anno_error_json,repair_info)
    return new_annos




def main():
    start_time = time.time()
    # load rank
    if METHOD == "ours":
        with open(RANK_PATH, encoding="utf-8") as file:
            ranking = json.load(file)["all_ranking"]
        ranked_items = []
        for item in ranking:
            if item["item_type"] == "anno":
                ranked_items.append(item["anno_id"])
            elif item["item_type"] == "image":
                ranked_items.append(item["image_name"])
            else:
                raise ValueError(f"Unsupported item_type: {item['item_type']}")
    else:
        ranked_items = joblib.load(RANK_PATH)

    anno_fault_with_miss_json = read_json(ANNOTATIONS_WITHMISS_PATH)
    anno_fault_json = read_json(ANNOTATIONS_FAULT_PATH)
    anno_correct_json = read_json(ANNOTATIONS_CORRECT_PATH)

    
    # budget = int(CUTOFF * (len(anno_fault_json["images"]) +
    #                        len(anno_fault_json["annotations"])))
    # budget = int(len(ranked_items) * CUTOFF)

    budget = int(len(anno_correct_json["annotations"])*FAULT_RATIO*4)
    cost = budget
    anno_json = deepcopy(anno_fault_json) # copy一份故障标注文件，以防修改原始文件
    correct_imgname_to_annoids = get_img_name_to_ann_ids(anno_correct_json)
    current_imgname_to_annoids = get_img_name_to_ann_ids(anno_json)
    correct_annoid_to_anno = get_annoId_to_anno(anno_correct_json)
    current_annoid_to_anno = get_annoId_to_anno(anno_json)
    image_names = {image["file_name"] for image in anno_json["images"]}
    repair_info = {"miss": {}, "cls": {}, "loc": {}, "redun": {}}
    seen_items = set()

    # 按排名顺序处理图片或标注，严格限制总预算。
    for ranked_item in ranked_items:
        if cost <= 0:
            break
        # 跳过重复条目，避免重复修复或重复扣费。
        if ranked_item in seen_items:
            continue
        seen_items.add(ranked_item)
        # 字符串表示图片名，需要检查该图片是否存在漏标框。
        if isinstance(ranked_item, str):
            if ranked_item not in image_names:
                raise ValueError(f"Unknown image name: {ranked_item}")
            current_ids = set(current_imgname_to_annoids.get(ranked_item, []))
            # 正确标注中存在、当前标注中不存在的 ID 即为漏标；排序保证修复顺序可复现。
            missed_ids = sorted(set(correct_imgname_to_annoids.get(ranked_item, [])) -
                                current_ids)
            if not missed_ids:
                # 即使没有漏标，检查一张图片也消耗 1 个 cost。
                cost -= 1
                continue
            repaired_missing = []
            for missed_id in missed_ids:
                # 预算可能在图片内部耗尽，只补回预算允许修复的漏标框。
                if cost <= 0:
                    break
                annotation = deepcopy(correct_annoid_to_anno[missed_id])
                annotation.update(fault_type=0, repair_ops="repair_miss")
                repaired_missing.append(annotation)
                # strict cost：每补回一个漏标框扣除 1，而不是整张图片只扣除 1。
                cost -= 1
            # 记录已选中的漏标框，后续由 repair_anno_json 统一补入输出标注。
            repair_info["miss"][ranked_item] = repaired_missing
        else:
            # 非字符串表示标注 ID；检查一个标注消耗 1 个 cost，无故障也扣费。
            annotation = current_annoid_to_anno[ranked_item]
            cost -= 1
            fault_type = annotation["fault_type"]
            if fault_type == 1:
                # 分类错误：记录正确类别的来源，并将当前标注标记为已修复。
                repair_info["cls"][ranked_item] = correct_annoid_to_anno[ranked_item]
                annotation.update(fault_type=0, repair_ops="repair_cls")
            elif fault_type == 2:
                # 定位错误：记录正确框，后续由 repair_anno_json 替换 bbox。
                correct_annotation = correct_annoid_to_anno[ranked_item]
                repair_info["loc"][ranked_item] = correct_annotation
                # 同步与位置相关的面积和分割信息，正确标注没有的字段则移除。
                for field in ("area", "segmentation"):
                    if field in correct_annotation:
                        annotation[field] = deepcopy(correct_annotation[field])
                    else:
                        annotation.pop(field, None)
                annotation.update(fault_type=0, repair_ops="repair_loc")
            elif fault_type == 3:
                # 冗余框：先记录，后续由 repair_anno_json 从输出标注中删除。
                repair_info["redun"][ranked_item] = annotation

    count_info = count_repair_info(repair_info, anno_fault_with_miss_json,
                                   anno_fault_json)
    pprint.pprint(count_info, indent=4, sort_dicts=False)
    anno_json = repair_anno_json(anno_json, repair_info)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "annotations.coco_repair.json"
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(anno_json, file, ensure_ascii=False)
    print(f"cost:{budget}, spent cost:{budget - cost}, remain cost:{cost}")
    print(f"repaired anno json saved at: {save_path}")

    coco_anno_json_path = save_path
    yolo_output_dir = SAVE_DIR/"yolo_format"
    os.makedirs(yolo_output_dir,exist_ok=True)
    coco2yolo(coco_anno_json_path,yolo_output_dir,"train")

    end_time = time.time()                   
    elapsed_time = end_time - start_time                                     
    hours = int(elapsed_time // 3600)                   
    minutes = int((elapsed_time % 3600) // 60)                     
    seconds = elapsed_time % 60                               
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02.0f}")
    now_timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"text: {now_timestr}")


if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    config = read_yaml("config.yaml")
    DATA_ROOT = Path(config['exp_data_dir'])
    METHOD = "datactive" # ours|datactive
    DATASET_NAME = "visdrone" # voc|kitti|visdrone
    MODEL_NAME = "yolov7"
    FAULT_RATIO = 0.1
    REPEAT_ID = 1
    # yolov7-voc: 1791094449
    # yolov7-kitti: 1792056154
    # yolov7-visdrone: 1791968956
    # rtdetr-voc: 1792398239
    # rtdetr-visdrone: 1792504350
    # rtdetr-kitti: 1792556872
    # frcnn-voc: 1792570209
    # frcnn-kitti: 1792642708
    EXPID = 1791968956
    if METHOD == "ours":
        # 排序结果路径
        RANK_PATH = DATA_ROOT / f"process_features/{DATASET_NAME}/{MODEL_NAME}/{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}_{EXPID}/merged_rank.json"
    elif METHOD == "datactive":
        RANK_PATH = DATA_ROOT / f"rank/datactive/{DATASET_NAME}/0.1/repeat_{REPEAT_ID}"/"converted_rank.joblib"
    # 故障（含有miss fault）标注文件
    ANNOTATIONS_WITHMISS_PATH = DATA_ROOT / f"fault_inject/{FAULT_RATIO}/{DATASET_NAME}/coco_format/annotations_with_miss.json"
    # 故障（不含有miss fault）标注文件
    ANNOTATIONS_FAULT_PATH = DATA_ROOT / f"fault_inject/{FAULT_RATIO}/{DATASET_NAME}/coco_format/annotations_no_miss.json"
    # 正确标注文件
    ANNOTATIONS_CORRECT_PATH = DATA_ROOT / f"datasets/{DATASET_NAME}-coco/train/_annotations.coco_correct.json"
    if METHOD == "datactive":
        SAVE_DIR = DATA_ROOT/f"repair/{METHOD}/{DATASET_NAME}/{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}"
    else:
        SAVE_DIR = DATA_ROOT/f"repair/{METHOD}/{DATASET_NAME}/{MODEL_NAME}/{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}_{EXPID}"
    os.makedirs(SAVE_DIR,exist_ok=True)
    main()
                                       
