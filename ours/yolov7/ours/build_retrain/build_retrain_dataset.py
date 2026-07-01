'''
为了修复后的retrain, 我们将trainset split 为 train and val
'''
import os

from pathlib import Path
import random
import shutil
from ours.small_utils import get_all_files
from tqdm import tqdm


def extract_ops(file_name_list,source_dir,target_dir):
    for file_name in tqdm(file_name_list):
        source_file_path = os.path.join(source_dir,file_name)
        target_file_path = os.path.join(target_dir,file_name)
        shutil.copyfile(source_file_path, target_file_path)

def extract_imgs():
    random.seed(random_seed)
    img_path_list = get_all_files(source_imgs_dir)
    img_name_list = []
    for img_path in img_path_list:
        img_name = Path(img_path).name
        img_name_list.append(img_name)
    val_num = int(val_rate*len(img_name_list))

    val_img_name_list = random.sample(img_name_list,val_num)
    train_img_name_list = list(set(img_name_list) - set(val_img_name_list))
    print("抽取train imgs...")
    extract_ops(train_img_name_list,source_imgs_dir,target_train_imgs_dir)
    print("抽取val imgs...")
    extract_ops(val_img_name_list,source_imgs_dir,target_val_imgs_dir)

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

    print("抽取train labels...")
    extract_labels_help(target_train_imgs_dir,source_labels_dir,target_train_labels_dir)
    print("抽取val labels...")
    extract_labels_help(target_val_imgs_dir,source_labels_dir,target_val_labels_dir)


if __name__ == "__main__":
    # Only YOLOv7 
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    random_seed = 43
    # 切分出用于验证的比例
    val_rate = 0.2

    # origin trainset imgs
    source_imgs_dir = os.path.join(exp_root_dir, "retrain_dataset_split", dataset_name, "images", "origin")
    target_train_imgs_dir = os.path.join(exp_root_dir, "retrain_dataset_split", dataset_name, 
                                         "images", "split", "train")
    target_val_imgs_dir = os.path.join(exp_root_dir, "retrain_dataset_split", dataset_name, 
                                       "images", "split", "val")
    
    # 选择从总的训练集中选择的数据用于验证集 20%, 剩下80%数据用于训练集
    # 1.抽取img
    # extract_imgs()

    # 2.抽取label
    # origin train的labels dir(yolo style)
    method_name = "objectlab" # ours|datactive|entropy|loss|deepgini|margin|objectlab|clod !!!
    exp_id = "exp_01"
    if method_name in ["entropy","loss","deepgini","margin","objectlab","clod"]:
        source_labels_dir = f"{exp_root_dir}/Results/other_baselines/{method_name}/{dataset_name}/YOLOv7/{exp_id}/repair/yolo_format/labels"
        # 切出的train labels dir
        target_train_labels_dir = f"{exp_root_dir}/Results/other_baselines/{method_name}/{dataset_name}/YOLOv7/{exp_id}/retrain/label_split/splitted_labels/train"
        # 切出的val labels dir
        target_val_labels_dir = f"{exp_root_dir}/Results/other_baselines/{method_name}/{dataset_name}/YOLOv7/{exp_id}/retrain/label_split/splitted_labels/val"
    else:
        source_labels_dir = f"{exp_root_dir}/Results/{method_name}/{dataset_name}/YOLOv7/{exp_id}/repair/yolo_format/labels"
        # 切出的train labels dir
        target_train_labels_dir = f"{exp_root_dir}/Results/{method_name}/{dataset_name}/YOLOv7/{exp_id}/retrain/label_split/splitted_labels/train"
        # 切出的val labels dir
        target_val_labels_dir = f"{exp_root_dir}/Results/{method_name}/{dataset_name}/YOLOv7/{exp_id}/retrain/label_split/splitted_labels/val"
    extract_labels()




