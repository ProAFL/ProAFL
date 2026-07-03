'''
Docstring for collection
'''

import os
import time
import pprint
import json
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from datasets import CocoDetectionDataset
import torch,torchvision
from collections import defaultdict
from cutom_module.small_utils import read_yaml

# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()

def build_frcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_trainset():
    train_dataset = CocoDetectionDataset(
        image_dir=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train", 
        annotation_path=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json",
        transforms=get_transform()
    )
    return train_dataset

def get_model(num_classes):
    model = build_frcnn_model(num_classes)
    return model

def set_nms(model, model_name, conf_threshold=0.25,iou_threshold=0.65):
    if model_name == "SSD":
        model.score_thresh = conf_threshold
        model.nms_thresh = iou_threshold
    elif model_name == "FRCNN":
        model.roi_heads.score_thresh = conf_threshold
        model.roi_heads.nms_thresh = iou_threshold
        
    else:
        raise Exception("模型名称错误")
    return model

def model_load_weight(model,epoch):
    # 加载模型
    w_path = os.path.join(error_model_pth_dir,f"epoch_{epoch}.pth")
    state_dict = torch.load(w_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def collect_help(img_name,p_boxes,p_labels,confs, global_id):
    p_boxs = []
    for i in range(p_boxes.shape[0]):
        p_box = p_boxes[i].tolist()
        p_label = int(p_labels[i])
        conf = confs[i].item()
        p_box = {
            "predicted_box_id":global_id,
            "img_name":img_name,
            "predicted_cls":p_label,
            "conf":conf,
            "bbox":p_box
        }
        p_boxs.append(p_box)
        global_id += 1
    return p_boxs


def collect_one_epoch(model,dataset_loader,device):
    model.eval()
    global_id = 0
    collect_dict = {}
    for images, targets in dataset_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        predictions = model(images)
        for img,target,pred in zip(images,targets,predictions):
            img_name = target["image_path"].split("/")[-1]
            p_boxes = pred['boxes']
            p_labels = pred['labels']
            confs = pred['scores']
            if p_boxes.shape[0] > 0:
                # 模型对该图像有预测输出
                collected_p_boxs = collect_help(img_name,p_boxes,p_labels,confs, global_id)
                collect_dict[img_name] = {
                    "predicted_bboxs":collected_p_boxs
                } 
                global_id += len(collected_p_boxs)
    return collect_dict

def save_one_epoch(collect_dict,epoch):
    save_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_path = os.path.join(collect_save_dir,save_file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collect_dict,f,indent=4)
    return save_path


def collect_predicted_boxes():
    start_time = time.time()  # 记录开始时间
    # 构建数据集实例
    train_dataset = get_trainset()
    # 构建数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=16, 
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    # 得到数据集nc:
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    # 构建模型
    model = get_model(num_classes)
    if _args["custom_nms"] is True:
        model = set_nms(model,model_name=model_name)
    # 得到设备
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    # 开始收集
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        model = model_load_weight(model,epoch)
        collect_dict = collect_one_epoch(model,train_loader,device)
        save_path = save_one_epoch(collect_dict,epoch)
        print(f"收集的数据保存在: {save_path}")
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数
    print(f"耗时: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root_dir = config["exp_data_dir"]
    gpu_id = 0
    PID = os.getpid()
    print("PID:",PID)
    _args = {
        "dataset_name":"voc", # voc|kitti|visdrone
        "model_name":"frcnn",
        "num_epochs":50,
        "custom_nms":False
    }
    pprint.pprint(_args)
    
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]
    num_epochs = _args["num_epochs"]
    
    error_model_pth_dir = os.path.join(exp_data_root_dir,"models",dataset_name,
                                       model_name,"error")
    collect_save_dir = os.path.join(exp_data_root_dir,"collection_process_info",
                                    dataset_name,model_name,"collected_predict_boxes")
    os.makedirs(collect_save_dir,exist_ok=True)
    collect_predicted_boxes()