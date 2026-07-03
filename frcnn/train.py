
import pprint
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from torchvision.transforms import ToTensor,Normalize
from datasets import CocoDetectionDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights

import torch.distributed as dist
import torch,torchvision
from engine import train_one_epoch,train_one_epoch_for_distribution,evaluate
from small_utils import timestamp_to_hms
from torchvision import models, transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from frcnn.cutom_module.base_data_manager import (get_correct_ann_file_path,
                                                  get_error_ann_file_path,
                                                  get_imgs_dir)
import time
from cutom_module.small_utils import read_yaml
# Transform PIL image --> PyTorch tensor
def get_transform(train=False): 
    t = [ToTensor()]
    if train:
        if _args["ColorJitter"]:
            t.insert(0,transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
        if _args["Normal"]:
            t.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        if _args["Normal"]:
            t.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)

def get_train_and_val_dataset():
    # Load training dataset
    train_dataset = CocoDetectionDataset(
        image_dir=train_imgs_dir,
        annotation_path=train_annotation_path,
        transforms=get_transform(train=True)
    )

    # Load validation dataset
    val_dataset = CocoDetectionDataset(
        image_dir=val_imgs_dir,
        annotation_path=val_annotation_path,
        transforms=get_transform(train=False)
    )
    return train_dataset, val_dataset

def build_model(nc):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)
    return model

def save_epoch(model,epoch):
    save_file_name = f"epoch_{epoch}.pth"
    save_path = os.path.join(epoch_save_dir,save_file_name)
    torch.save(model.state_dict(), save_path)
    return save_path

def model_load_weight(model,model_weight_path):
    # 加载模型
    state_dict = torch.load(model_weight_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train(model_weight_path=None,distributed=False):
    start_time = time.time()  # 记录开始时间
    # 加载FRCNN模型（预训练）
    # Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN, , you change this 
    # weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    # weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT (up-to-date weights)
    # model =torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Number of classes in the dataset (including background)
    # +1 for bg class

    # 加载数据集
    train_dataset, val_dataset = get_train_and_val_dataset()
    # 构建模型
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    model = build_model(num_classes)
    # 加载权重
    if model_weight_path is not None:
        model_load_weight(model,model_weight_path)
    '''
    # 锁住Backbone params
    for param in model.backbone.parameters():
        param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    '''
    # 参数优化器
    # optimizer = torch.optim.Adam(params, lr=1e-4)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=init_lr,
                                momentum=0.9,weight_decay=0.0005)

    # lr 调度器：在 60%/80% epoch 处各衰减一次，适合 SSD 和 FRCNN
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.6), int(num_epochs * 0.8)],
        gamma=0.1
    )
    if distributed: # 开启单机多卡多进程训练
        model_train = model.train() # model_train将来会被DDP包裹,model_train与model是实时共享参数的
        ngpus_per_node  = torch.cuda.device_count() # 本机中gpu个数
        dist.init_process_group(backend="nccl") # 分布式后端使用nccl
        local_rank  = int(os.environ["LOCAL_RANK"]) # 当前进程在本机上的GPU编号
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank) # 指定当前进程使用哪块GPU
        if local_rank == 0: # 0号进程打印
            print("单机多卡多进程训练...")
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
        if ngpus_per_node > 1:
            # 多卡分布式训练 BatchNorma同步
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        model_train = model_train.cuda(local_rank) # 模型放到该该进程对应的gpu上
        model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        batch_size = _args["batch_size"] // ngpus_per_node # 你设定好的batchsize均分到多个gpu上
        # 如果 DataLoader 已经传入了 sampler，就不能再同时使用 shuffle=True。
        train_loader = DataLoader(train_dataset, shuffle = False, batch_size = batch_size, num_workers = 4, pin_memory=True,
                                    drop_last=True, sampler=train_sampler, collate_fn=lambda x: tuple(zip(*x)),
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        val_loader = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = 4, pin_memory=True, 
                                    drop_last=True, sampler=val_sampler, collate_fn=lambda x: tuple(zip(*x)),
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        loss_history = {
            "train_loss":[],
            "val_loss":[]
        }
        save_period = 10
        save_dir = epoch_save_dir
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch) # 让 DistributedSampler 每个 epoch 使用不同的随机打乱顺序。
            # train one epoch
            train_one_epoch_for_distribution(model_train,model,optimizer,train_loader,val_loader,
                                     epoch,num_epochs,loss_history,save_period,save_dir,local_rank=local_rank)
            lr_scheduler.step()
    else:
        # 非分布式训练
        # 设备
        device = torch.device(f"cuda:{gpu_id}")
        model.to(device)
        train_loader = DataLoader(train_dataset, batch_size=_args["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=_args["batch_size"], shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        # train loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)
            lr_scheduler.step()
            evaluate(model, val_loader, device=device)
            epoch_save_path = save_epoch(model,epoch)
            print(f"model pth is saved in: {epoch_save_path}")

    end_time = time.time()  # 记录结束时间
    cost_time = end_time - start_time  # 计算运行时间（秒）
    run_time = timestamp_to_hms(cost_time)
    print(f"运行时间: {run_time}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    PID = os.getpid()
    print("PID:",PID)
    seed = 42
    exp_data_root_dir = config["exp_data_dir"]
    gpu_id = 0
    print("gpu_id:",gpu_id)
    _args = {
        "dataset_name":"voc", # voc|kitti|visdrone
        "model_name":"frcnn",
        "trainset_status":"error", # clean|error|
    }
    
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]

    model_weight_path = None
    num_epochs = 50
    init_lr = 5e-3
    _args["model_weight_path"] = model_weight_path

    # 训练超参数
    _args["init_lr"] = init_lr
    _args["num_epochs"] = num_epochs
    _args["batch_size"] = 32
    _args["ColorJitter"] = True
    _args["Normal"] = False

    epoch_save_dir = os.path.join(exp_data_root_dir,"models",dataset_name,
                                  model_name, _args["trainset_status"])
    os.makedirs(epoch_save_dir,exist_ok=True)
    _args["epoch_save_dir"] = epoch_save_dir
    pprint.pprint(_args)

    train_imgs_dir = get_imgs_dir(dataset_name,"train",style="coco")
    val_imgs_dir = get_imgs_dir(dataset_name,"val",style="coco")

    if _args["trainset_status"] == "clean":
        train_annotation_path = get_correct_ann_file_path(dataset_name,"train")
    elif _args["trainset_status"] == "error":
        train_annotation_path = get_error_ann_file_path(dataset_name)

    val_annotation_path = get_correct_ann_file_path(dataset_name,"val")

    train(model_weight_path,distributed=True)
