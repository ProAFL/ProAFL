
'''
负责将排序进行统一转换成coco中的annoid/imgname
'''
import os

import joblib
from helper.data_organization_tools import conver_ours_rank,conver_datactive_rank
from helper.base_data_manager import get_collected_gt_box_json_path,get_error_ann_file_path
from ours.small_utils import read_json
from pycocotools.coco import COCO
from ours.small_utils import read_yaml

config = read_yaml("config.yaml")
exp_data_root_dir = config["exp_data_dir"]
dataset_name_list = config["dataset_name_list"]
model_name_list = config["model_name_list"]
def convert_ours():
    for dataset_name in dataset_name_list:
        gt_json_path = get_collected_gt_box_json_path(dataset_name)
        g_boxes_json = read_json(gt_json_path)
        anno_error_json_path = get_error_ann_file_path(dataset_name)
        anno_error_json = read_json(anno_error_json_path)
        for model_name in model_name_list:
            print(f"{dataset_name}|{model_name}|OURS 转换中...")
            rank_path = os.path.join(exp_data_root_dir,"Results","ours",dataset_name,model_name,
                                                    "exp_01","rank","rank.joblib")
            rank_res = joblib.load(rank_path)
            converted_rank_list = conver_ours_rank(rank_res,g_boxes_json,anno_error_json)
            save_file = os.path.join(exp_data_root_dir,"Results","ours",dataset_name,model_name,
                                                    "exp_01","rank","converted_rank.joblib")
            joblib.dump(converted_rank_list,save_file)

def convert_otherbaselines():
    
    for dataset_name in dataset_name_list:
        gt_json_path = get_collected_gt_box_json_path(dataset_name)
        g_boxes_json = read_json(gt_json_path)
        anno_error_json_path = get_error_ann_file_path(dataset_name)
        anno_error_json = read_json(anno_error_json_path)
        for model_name in ["YOLOv7","FRCNN","rtdetr"]:
            for baseline_name in ["objectlab"]: # ["entropy","loss","deepgini","margin","objectlab"]:
                print(f"{dataset_name}|{model_name}|{baseline_name} 转换中...")
                rank_path = os.path.join(exp_data_root_dir,"Results",
                                        "other_baselines",baseline_name,dataset_name,model_name,
                                        "exp_01","rank","rank.joblib")
                rank_res = joblib.load(rank_path)
                converted_rank_list = conver_ours_rank(rank_res,g_boxes_json,anno_error_json)
                save_file = os.path.join(exp_data_root_dir,"Results",
                                            "other_baselines",baseline_name,dataset_name,model_name,
                                            "exp_01","rank","converted_rank.joblib")
                joblib.dump(converted_rank_list,save_file)

def convert_datactive():
    exp_id = "exp_02"
    for dataset_name in dataset_name_list:
        print(f"{dataset_name}|dataactive 转换中...")
        rank_path = f"{exp_data_root_dir}/Results/datactive/{dataset_name}/YOLOv7/{exp_id}/rank/rank.joblib"
        rank_res = joblib.load(rank_path)
        anno_coco_error_json_path = get_error_ann_file_path(dataset_name)
        coco = COCO(anno_coco_error_json_path)
        catIds = coco.getCatIds()
        bg_id = catIds[-1]+1
        converted_rank_list = conver_datactive_rank(rank_res,bg_id)
        save_file = f"{exp_data_root_dir}/Results/datactive/{dataset_name}/YOLOv7/{exp_id}/rank/converted_rank.joblib"
        joblib.dump(converted_rank_list,save_file)


if __name__ == "__main__":
    # convert_ours()
    convert_otherbaselines()
    # convert_datactive()
