import os
import random
import joblib
import pandas as pd
import json
from collections import defaultdict
from pycocotools.coco import COCO
from custom_module.small_utils import read_yaml
from custom_module.base_data_manager import get_annotations_with_miss_json_path

def convert_datactive_rank(datactive_rank:list, bg_catId:int) -> list:
    '''
    Convert the sequence from datactive (instances) to a unified (imgname or anno_id) sequence.
    '''
    converted_rank_list = []
    for instance in datactive_rank:
        gt_category_id = instance["gt_category_id"]                                                                        
        if gt_category_id == bg_catId:
            converted_rank_list.append(instance["image_name"])
        else:
            converted_rank_list.append(instance["anno_id"])
    return converted_rank_list

def aggregation(obj_list:list):
    '''
    textimg level
    '''
    image2loss = defaultdict(float)
    for obj in obj_list:
        img_name = obj["image_name"]
        loss = obj["loss"]
        image2loss[img_name] += loss
    
    sorted_img_name_list = sorted(image2loss, key=image2loss.get, reverse=True)
    return sorted_img_name_list

def calcu_afpd(ranked_results):
    fault_num = 0
    rank_sum = 0
    for i in range(len(ranked_results)):
        if ranked_results[i]['fault_type'] != FAULT_TYPE["no_fault"]:
            fault_num += 1
            rank_sum += i+1
    apfd = 1-(rank_sum-1)/(fault_num*len(ranked_results))
    apfd = round(apfd,3)
    return apfd


def main():
          
    with open(crop_infer_results_path, 'r') as f:
        crop_list = json.load(f)
    with open(others_infer_results_path, 'r') as f:
        others_list = json.load(f)
                       
    coco = COCO(annotation_path)
    imageId2boxes = defaultdict(list)
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    for instance in annotations:
        bbox = instance["bbox"]
        label  = instance["category_id"]
        imageId2boxes[instance["image_id"]].append([bbox,label])

    crop_list.extend(others_list)
    results = sorted(crop_list, key=lambda x: x['loss'], reverse=True)

                                      
    with open(annotation_with_miss_path,"r") as f:
        annotation_with_miss = json.load(f)
    images = annotation_with_miss["images"]
    image_id_to_image_name = {}
    for image in images:
        image_id_to_image_name[image["id"]] = image["file_name"]

    annos = annotation_with_miss["annotations"]
    miss_img_name_list = []
    for anno in annos:
        if anno["fault_type"] == FAULT_TYPE["missing_fault"]:
            miss_img_name_list.append(image_id_to_image_name[anno["image_id"]])
    
    for i in range(len(results)):
        if int(results[i]["gt_category_id"]) == bg_clss_id:
                          
            if results[i]["image_name"] in miss_img_name_list:
                                                                             
                results[i]["fault_type"] = FAULT_TYPE["missing_fault"]
            else:
                results[i]["fault_type"] = FAULT_TYPE["no_fault"]
    joblib.dump(results,rank_result_save_path)
    print(f"rankResult saved at:{rank_result_save_path}")
    afpd = calcu_afpd(results)
    afpd = round(afpd,3)
    converted_rank = convert_datactive_rank(results,bg_clss_id)
    converted_rank_save_path =os.path.join(rank_result_save_dir,"converted_rank.joblib")
    joblib.dump(converted_rank,converted_rank_save_path)
    print(f"converted rank saved at:{converted_rank_save_path}")
    return afpd


if __name__ == "__main__":
    config =read_yaml("config.yaml")
    FAULT_TYPE = {
            'no_fault': 0,
            'cls_fault': 1,
            'loc_fault': 2,
            'redundancy_fault': 3,
            'missing_fault': 4,
    }
    exp_data_root = config["exp_data_dir"]
    dataset_name = "voc"
    
    # crop_infer_results_path=f'{exp_data_root}/baselines/datactive/{dataset_name}/rank/infer/crop.json'
    # others_infer_results_path=f'{exp_data_root}/baselines/datactive/{dataset_name}/rank/infer/other_objects.json'
    if dataset_name == "voc":
        bg_clss_id = 20 # [0-19]
    elif dataset_name == "kitti":
        bg_clss_id = 8 # [0-7]
    elif dataset_name == "visdrone":
        bg_clss_id = 10 # [0-9]

    inject_ratio = 0.1
    annotation_path=f"{exp_data_root}/fault_inject/{str(inject_ratio)}/{dataset_name}/coco_format/annotations_no_miss.json"
    annotation_with_miss_path = f"{exp_data_root}/fault_inject/{str(inject_ratio)}/{dataset_name}/coco_format/annotations_with_miss.json"
    for repeat_id in [1,2,3,4,5,6,7,8,9,10]:
        crop_infer_results_path = os.path.join(exp_data_root,"datactive_infer_res",dataset_name,
                                               str(inject_ratio),f"repeat_{repeat_id}","crop.json")
        others_infer_results_path = os.path.join(exp_data_root,"datactive_infer_res",dataset_name,
                                                 str(inject_ratio),f"repeat_{repeat_id}","other_objects.json")
    
        rank_result_save_dir = os.path.join(exp_data_root,"rank","datactive", dataset_name, 
                                            str(inject_ratio),f"repeat_{repeat_id}")
        os.makedirs(rank_result_save_dir,exist_ok=True)
        rank_result_save_path = os.path.join(rank_result_save_dir,"rank.joblib")
        apfd = main()
        print(f"repeat:{repeat_id},APFD:{apfd}")