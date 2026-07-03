'''
For retraining after repair, split the train set into train and val.
'''
import os
from pathlib import Path
import shutil
from ours.small_utils import get_all_files,read_yaml
from tqdm import tqdm


def extract_ops(file_name_list,source_dir,target_dir):
    for file_name in tqdm(file_name_list):
        source_file_path = os.path.join(source_dir,file_name)
        target_file_path = os.path.join(target_dir,file_name)
        shutil.copyfile(source_file_path, target_file_path)

def extract_labels():
    def extract_labels_help(imgs_dir,source_labels_dir,target_labels_dir):
        img_path_list = get_all_files(imgs_dir)
        img_name_list = [Path(img_path).name for img_path in img_path_list]
        label_name_list = []
        for img_name in img_name_list:
            base_name,ext = os.path.splitext(img_name)
            label_name = base_name+".txt"
            label_name_list.append(label_name)
        extract_ops(label_name_list,source_labels_dir,target_labels_dir)

    print("Extracting train labels...")
    extract_labels_help(splitted_train_imgs_dir,source_labels_dir,splitted_train_labels_dir)
    print("Extracting val labels...")
    extract_labels_help(splitted_val_imgs_dir,source_labels_dir,splitted_val_labels_dir)


if __name__ == "__main__":
    config = read_yaml("config.yaml")
                       
    exp_root_dir = config["exp_data_dir"]
    dataset_name = "voc"                     

    splitted_train_imgs_dir = os.path.join(exp_root_dir, "retrain_dataset_split", dataset_name, 
                                         "images", "split", "train")
    splitted_val_imgs_dir = os.path.join(exp_root_dir, "retrain_dataset_split", dataset_name, 
                                       "images", "split", "val")
    
                           
    method_name = "ours"                                                         

    if method_name == "ours":
        source_labels_dir = os.path.join(exp_root_dir,"ours",dataset_name,"yolov7","repair","yolo_format","labels")
        splitted_train_labels_dir = os.path.join(exp_root_dir,"ours",dataset_name,"yolov7","retrain/splitted_labels/train")
                                    
        splitted_val_labels_dir = os.path.join(exp_root_dir,"ours",dataset_name,"yolov7","retrain/splitted_labels/val")
    elif method_name in config["baselines"]:
        if method_name == "datactive":
            source_labels_dir = os.path.join(exp_root_dir,"baselines","datactive","repair","yolo_format","labels")
            splitted_train_labels_dir = os.path.join(exp_root_dir,"baselines","datactive","retrain/splitted_labels/train")
            splitted_val_labels_dir = os.path.join(exp_root_dir,"baselines","datactive","retrain/splitted_labels/val")
        else:
            source_labels_dir = os.path.join(exp_root_dir,"baselines",method_name,dataset_name,"yolov7","repair","yolo_format","labels")
            splitted_train_labels_dir = os.path.join(exp_root_dir,"baselines",method_name,dataset_name,"yolov7","retrain/splitted_labels/train")
            splitted_val_labels_dir = os.path.join(exp_root_dir,"baselines",method_name,dataset_name,"yolov7","retrain/splitted_labels/val")

    else:
        raise Exception(f"{method_name} is set error")
    extract_labels()

