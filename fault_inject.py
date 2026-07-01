
import os
import random
import json
from pycocotools.coco import COCO
import torch
from torchvision.ops import boxes as box_ops
import pandas as pd
import copy
from utils.common import read_yaml


def cal_IoU(X, Y):
    return box_ops.box_iou(torch.tensor([X]), torch.tensor([Y]))

def gen_missing_fault(object_id_list, anno_list):
    for object_id in object_id_list:
        for anno in anno_list:
            if anno["id"] == object_id:
                fault_info = {
                    "obj_id":object_id,
                    "img_id":anno["image_id"],
                    "img_name": coco.loadImgs(anno["image_id"])[0]["file_name"],
                    "fault_type":fault_type["missing_fault"]
                }
                fault_recorder.append(fault_info)
                anno["fault_type"] = fault_type["missing_fault"]
    return anno_list

def gen_class_fault(object_id_list,anno_list):
    for object_id in object_id_list:
        for anno in anno_list:
            if anno["id"] == object_id:
                # 准备篡改该obj
                original_cls = anno["category_id"]
                candi_cls_list = [cls for cls in catIds if cls != original_cls]
                error_cls = random.choice(candi_cls_list)
                anno["category_id"] = error_cls
                anno["fault_type"] = fault_type["cls_fault"]
                fault_info = {
                    "obj_id":object_id,
                    "img_id":anno["image_id"],
                    "img_name": coco.loadImgs(anno["image_id"])[0]["file_name"],
                    "fault_type":fault_type["cls_fault"]
                }
                fault_recorder.append(fault_info)
    return anno_list

def gen_loc_fault(object_id_list,anno_list):
    for object_id in object_id_list:
        for anno in anno_list:
            if anno["id"] == object_id:
                # 准备篡改该obj
                original_bbox = anno["bbox"] # xmin,ymin,w,h
                ori_x1 = original_bbox[0]
                ori_x2 = ori_x1+original_bbox[2]
                ori_y1 = original_bbox[1]
                ori_y2 = ori_y1+original_bbox[3]
                image_info = coco.loadImgs(anno["image_id"])[0]
                image_size = (image_info["width"],image_info["height"])


                # generate a random location while the IoU is in [0.1, 0.5]
                while True:
                    new_x1 = random.randint(int(max(0, ori_x1 - (ori_x2 - ori_x1) / 2)), int((ori_x1 + ori_x2) / 2))
                    new_y1 = random.randint(int(max(0, ori_y1 - (ori_y2 - ori_y1) / 2)), int((ori_y1 + ori_y2) / 2))
                    new_x2 = random.randint(int((ori_x1 + ori_x2) / 2), int(min(image_size[0], ori_x2 + (ori_x2 - ori_x1) / 2)))
                    new_y2 = random.randint(int((ori_y1 + ori_y2) / 2), int(min(image_size[1], ori_y2 + (ori_y2 - ori_y1) / 2)))

                    # garantee the width and height are not equal to 0
                    if new_x1 >= new_x2 or new_y1 >= new_y2:
                        continue

                    # calculate the IoU
                    IoU = cal_IoU([ori_x1, ori_y1, ori_x2, ori_y2], [new_x1, new_y1, new_x2, new_y2]).item()
                    if 0.1 <= IoU <= 0.5:
                        break
                anno["bbox"] = [new_x1, new_y1, new_x2-new_x1, new_y2-new_y1]
                anno["fault_type"] = fault_type["loc_fault"]
                fault_info = {
                    "obj_id":object_id,
                    "img_id":anno["image_id"],
                    "img_name": coco.loadImgs(anno["image_id"])[0]["file_name"],
                    "fault_type":fault_type["loc_fault"]
                }
                fault_recorder.append(fault_info)
    return anno_list

def gen_redundancy_fault(object_id_list, anno_list):
    new_id = anno_list[-1]["id"]+1
    new_obj_list = []
    for object_id in object_id_list:
        for anno in anno_list:
            if anno["id"] == object_id:
                image_info = coco.loadImgs(anno["image_id"])[0]
                image_size = (image_info["width"],image_info["height"])
                new_x1, new_y1, new_x2, new_y2 = None, None, None, None
                while True:
                    new_x1 = random.randint(0, image_size[0])
                    new_y1 = random.randint(0, image_size[1])
                    new_x2 = random.randint(new_x1, image_size[0])
                    new_y2 = random.randint(new_y1, image_size[1])

                    # garantee the width and height are not equal to 0
                    if new_x1 >= new_x2 or new_y1 >= new_y2:
                        continue
                    break
                new_cls = random.sample(catIds, 1)[0]
                new_bbox = [new_x1, new_y1, new_x2-new_x1, new_y2-new_y1]
                new_obj = {
                    "id":new_id,
                    "image_id":anno["image_id"],
                    "category_id":new_cls,
                    "bbox":new_bbox,
                    "fault_type":fault_type["redundancy_fault"]
                }
                new_obj_list.append(new_obj)
                fault_info = {
                    "obj_id":new_id,
                    "img_id":anno["image_id"],
                    "img_name": coco.loadImgs(anno["image_id"])[0]["file_name"],
                    "fault_type":fault_type["redundancy_fault"]
                }
                fault_recorder.append(fault_info)
                new_id += 1
    anno_list.extend(new_obj_list)
    return anno_list


def add_fault_type_attr(anno_list):
    for anno in anno_list:
        anno["fault_type"] = fault_type["no_fault"] # 无错
    return anno_list

def remove_miss_fault_anno(anno_list):
    new_anno_list = []
    for anno in anno_list:
        if anno["fault_type"] != fault_type["missing_fault"]:
            new_anno_list.append(anno)
    return new_anno_list
        

if __name__ == "__main__":
    # 设置随机数种子
    random.seed(42)
    # 设置实验数据保存目录
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    # 设置数据集名称
    dataset_name = config["dataset_name"]
    # 设置数据集correct anno path
    correct_anno_json_path = os.path.join(exp_data_root,"datasets", f"{dataset_name}-coco","train","_annotations.coco_correct.json")
    # 转换为COCO实例
    coco = COCO(correct_anno_json_path)
    # 设置每种fault的错误率
    fault_rate = 0.1 # 每种错误比例为10%
    # 保存为annotation json文件
    save_dir = os.path.join(exp_data_root,"fault_anno",f"{fault_rate}",dataset_name,"coco_format")
    os.makedirs(save_dir,exist_ok=True)

    # 数据集中anno的ids
    ann_ids = coco.getAnnIds()
    # 根据 annoIds 载入所有 annotation
    annotations = coco.loadAnns(ann_ids)
    # 获得anno json中目标的类别idx
    catIds = coco.getCatIds()
    # 错误类型编码映射
    fault_type = {
            'no_fault': 0,
            'cls_fault': 1,
            'loc_fault': 2,
            'redundancy_fault': 3,
            'missing_fault': 4,
    }
    # 错误记录者，非必须
    fault_recorder = []

    # 准备注入错误
    # 总共的objs数量
    total_object_num = len(ann_ids)
    sample_num = int(total_object_num*fault_rate)
    # missing fault id 采样
    candi_id_set = set(ann_ids)

    missing_fault_obj_id_list = random.sample(list(candi_id_set),sample_num) # 随机不重复抽取

    # cls fault id 采样
    candi_id_set = set(ann_ids) - set(missing_fault_obj_id_list)
    cls_fault_obj_id_list = random.sample(list(candi_id_set),sample_num)

    # loc fault id 采样
    candi_id_set = set(ann_ids) - set(missing_fault_obj_id_list) - set(cls_fault_obj_id_list)
    loc_fault_obj_id_list = random.sample(list(candi_id_set),sample_num)

    # redundancy fault id 采样
    candi_id_set = set(ann_ids) - set(missing_fault_obj_id_list) - set(cls_fault_obj_id_list) - set(loc_fault_obj_id_list)
    redundancy_fault_obj_id_list = random.sample(list(candi_id_set),sample_num)

    annotations = add_fault_type_attr(annotations)
    print("miss(4)错误注入...")
    annotations = gen_missing_fault(missing_fault_obj_id_list,annotations)
    print("cls(1)错误注入...")
    annotations = gen_class_fault(cls_fault_obj_id_list,annotations)
    print("loc(2)错误注入...")
    annotations = gen_loc_fault(loc_fault_obj_id_list,annotations)
    print("redundancy(3)错误注入...")
    annotations = gen_redundancy_fault(redundancy_fault_obj_id_list,annotations)
    
    print(f"数据集: {dataset_name} 注错完成")

    annotations_no_miss = remove_miss_fault_anno(annotations)

    # 数据集中图像ids
    img_ids = coco.getImgIds()
    # 基于图像ids加载imgs
    images = coco.loadImgs(img_ids)
    # 基于cat ids加载categories
    categories = coco.loadCats(catIds)

    _json = {
        "images":images,
        "categories":categories,
        "annotations":annotations
    }
    
    anno_save_path = os.path.join(save_dir,"annotations_with_miss.json")
    with open(anno_save_path, "w", encoding="utf-8") as f:
        json.dump(_json, f, ensure_ascii=False, indent=4)
    print(f"标注文件(带有mis)保存在:{anno_save_path}")

    _json = {
        "images":images,
        "categories":categories,
        "annotations":annotations_no_miss
    }
    # 保存为annotation json文件
    save_dir = os.path.join(exp_data_root,"error_anno",dataset_name,"coco_format")
    os.makedirs(save_dir,exist_ok=True)
    anno_save_path = os.path.join(save_dir,"annotations_no_miss.json")
    with open(anno_save_path, "w", encoding="utf-8") as f:
        json.dump(_json, f, ensure_ascii=False, indent=4)
    print(f"标注文件(不带有mis)保存在:{anno_save_path}")

    # 保存fault recorder to csv 文件
    df = pd.DataFrame(fault_recorder)
    record_save_path = os.path.join(save_dir,"fault_records.csv")
    df.to_csv(record_save_path, index=False, encoding="utf-8")

    print(f"错误记录文件保存在:{record_save_path}")




