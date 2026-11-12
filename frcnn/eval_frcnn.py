import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

from datasets import CocoDetectionDataset
from engine import evaluate, train_one_epoch
from small_utils import timestamp_to_hms


ROOT_DIR = Path("/data/mml/data_debugging_data/ProAFL_data")
DATASET_NAME = "voc"  # voc|kitti|visdrone
METHOD_NAME = "ours" # ours|datactive|objectlab|deepgini|entropy|margin|loss
GPU_ID = 0
FAULT_RATIO = 0.1
REPEAT_ID = 1
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42

TESTSET_ANNO_JSON_PATH = ROOT_DIR / "datasets" / f"{DATASET_NAME}-coco"/ "val" / "_annotations.coco.json" # 干净的测试集
TEST_IMGS_DIR = ROOT_DIR / "datasets" / f"{DATASET_NAME}-coco"/ "val"
# 不同故障率和重复实验分别保留一个最佳权重，避免互相覆盖。
WEIGHT_PATH = ROOT_DIR / "retrained_models" / METHOD_NAME / DATASET_NAME / "frcnn" / f"best_{FAULT_RATIO}_repeat_{REPEAT_ID}.pth"


def get_num_classes():
    if DATASET_NAME == "voc":
        return 20
    if DATASET_NAME == "kitti":
        return 8
    if DATASET_NAME == "visdrone":
        return 10

def get_transform(train=False):
    t = [ToTensor()]
    if train:
        t.insert(0, transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ))
    return transforms.Compose(t)

def get_test_dataset():
    test_dataset = CocoDetectionDataset(
        image_dir=str(TEST_IMGS_DIR),
        annotation_path=str(TESTSET_ANNO_JSON_PATH),
        transforms=get_transform(train=False)
    )
    if len(test_dataset) == 0:
        raise ValueError("Test datasets 中没有数据.")    
    return test_dataset


def build_model(nc):
    # 所有权重均从起点 checkpoint 加载，无需下载预训练权重。
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)
    return model


def model_load_weight(model, model_weight_path):
    state_dict = torch.load(model_weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def worker_init_fn(worker_id, seed):
    worker_seed = (seed + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def collate_fn(batch):
    '''
    batch = [
        (image1, target1),
        (image2, target2),
    ]

    整理后得到：

    images = (image1, image2)
    targets = (target1, target2)

    这样就保留了每张图片各自的标注，不要求它们具有相同数量的边界框。

    '''
    return tuple(zip(*batch))

def eval():
    start_time = time.time() # 启动时间
    # 设置SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(f"cuda:{GPU_ID}")

    test_dataset = get_test_dataset()
    num_classes = get_num_classes() + 1 # 0号被预留给了bgclass,所以对于frcnn模型来说模型分类数量就是实际业务分类数量+1
    # 构建模型与加载checkpoint
    model = model_load_weight(build_model(num_classes), WEIGHT_PATH)
    # 模型放到gpu上
    model.to(device)
    
    print(f"Eval checkpoint: {WEIGHT_PATH}")
    print(f"Device: {device}, batch size: {BATCH_SIZE}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True, # 将读取的数据放入锁页内存，通常有利于 CPU 到 GPU 的数据传输
        collate_fn=collate_fn, # 指定如何把多个样本整理成一个 batch。目标检测中，不同图片的框数量不同，需要专门的batch整理函数。
        worker_init_fn=partial(worker_init_fn, seed=SEED)
    )
    generator = torch.Generator().manual_seed(SEED) # torch随机数生成器用于打乱训练集的
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    
    coco_evaluator = evaluate(model, test_loader, device=device)
    map50 = float(coco_evaluator.coco_eval["bbox"].stats[1])
    print(f"mAP@0.5: {map50:.3f}")
    print(f"Elapsed time: {timestamp_to_hms(time.time() - start_time)}")


if __name__ == "__main__":
    print("PID:", os.getpid()) # 打印pid
    eval()
