'''
数据迁移整理
'''

import os
import shutil

def trans_ours():
    exp_root_dir = "/data/mml/data_debugging_data"
    for dataset_name in ["VOC2012","KITTI_8","VisDrone"]:
        for model_name in ["YOLOv7","FRCNN","rtdetr"]:
            print(f"{dataset_name}|{model_name}|OURS")
            source_rank_path = os.path.join(exp_root_dir,"Results","ours",dataset_name,model_name,"exp_01","rank","converted_rank.joblib")
            target_dir = f"/data/mml/datadebugging_AI/{dataset_name.lower()}/ours/{model_name.lower()}"
            os.makedirs(target_dir,exist_ok=True)
            destination_file = shutil.copy(source_rank_path, target_dir)
            print('文件已复制到:', destination_file)
            
def trans_datactive():
    exp_root_dir = "/data/mml/data_debugging_data"
    for dataset_name in ["VOC2012","KITTI_8","VisDrone"]:
        for model_name in ["YOLOv7"]:
            print(f"{dataset_name}|{model_name}|datactive")
            source_rank_path = os.path.join(exp_root_dir,"Results","datactive",dataset_name,model_name,"exp_02","rank","converted_rank.joblib")
            target_dir = f"/data/mml/datadebugging_AI/{dataset_name.lower()}/datactive"
            os.makedirs(target_dir,exist_ok=True)
            destination_file = shutil.copy(source_rank_path, target_dir)
            print('文件已复制到:', destination_file)

def trans_otherbaselines():
    exp_root_dir = "/data/mml/data_debugging_data"
    for dataset_name in ["VOC2012","KITTI_8","VisDrone"]:
        for model_name in ["YOLOv7","FRCNN","rtdetr"]:
            for baseline_name in ["entropy","loss","deepgini","margin"]:
                print(f"{dataset_name}|{model_name}|{baseline_name}")
                source_rank_path = os.path.join(exp_root_dir,"Results","other_baselines",f"{baseline_name}",dataset_name,model_name,"exp_01","rank","converted_rank.joblib")
                target_dir = f"/data/mml/datadebugging_AI/{dataset_name.lower()}/{baseline_name}/{model_name.lower()}"
                os.makedirs(target_dir,exist_ok=True)
                destination_file = shutil.copy(source_rank_path, target_dir)
                print('文件已复制到:', destination_file)

if __name__ == "__main__":
    trans_ours()
    trans_datactive()
    trans_otherbaselines()
