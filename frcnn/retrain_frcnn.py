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
METHOD_NAME = "datactive" # ours|datactive|objectlab|deepgini|entropy|margin|loss
GPU_ID = 0
FAULT_RATIO = 0.1
REPEAT_ID = 1
EPOCH_ID = 49
WEIGHT_PATH = (
    ROOT_DIR / "models" / DATASET_NAME / "frcnn" / "fault_train"
    / str(FAULT_RATIO) / f"repeat_{REPEAT_ID}" / f"epoch_{EPOCH_ID}.pth"
)
EPOCHS = 25
INIT_LR = 5e-3
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42

if METHOD_NAME in ["ours", "objectlab", "deepgini", "entropy", "margin", "loss"]:
    REPAIR_DIR = (
        ROOT_DIR / "repair" / METHOD_NAME / DATASET_NAME / "frcnn"
        / f"{FAULT_RATIO}_repeat" / f"repeat_{REPEAT_ID}"
    )
elif METHOD_NAME == "datactive":
    # 按 prompt.txt 的要求，datactive 使用 repair/ours 下不含 frcnn 的目录。
    REPAIR_DIR = (
        ROOT_DIR / "repair" / METHOD_NAME / DATASET_NAME
        / f"{FAULT_RATIO}_repeat" / f"repeat_{REPEAT_ID}"
    )
else:
    raise ValueError(f"Unsupported method: {METHOD_NAME}")

TRAINSET_ANNO_JSON_PATH = REPAIR_DIR / "annotations_train.json"
VALSET_ANNO_JSON_PATH = REPAIR_DIR / "annotations_val.json"
TRAIN_IMGS_DIR = ROOT_DIR / "datasets" / f"{DATASET_NAME}-coco" / "train"
VAL_IMGS_DIR = TRAIN_IMGS_DIR # train和val的图像文件其实在一个目录里面
OUTPUT_DIR = ROOT_DIR / "retrained_models" / METHOD_NAME / DATASET_NAME / "frcnn"
# 不同故障率和重复实验分别保留一个最佳权重，避免互相覆盖。
BEST_WEIGHT_PATH = OUTPUT_DIR / f"best_{FAULT_RATIO}_repeat_{REPEAT_ID}.pth"


def get_transform(train=False):
    t = [ToTensor()]
    if train:
        t.insert(0, transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ))
    return transforms.Compose(t)


def get_train_and_val_dataset():
    train_dataset = CocoDetectionDataset(
        image_dir=str(TRAIN_IMGS_DIR),
        annotation_path=str(TRAINSET_ANNO_JSON_PATH),
        transforms=get_transform(train=True)
    )
    val_dataset = CocoDetectionDataset(
        image_dir=str(VAL_IMGS_DIR),
        annotation_path=str(VALSET_ANNO_JSON_PATH),
        transforms=get_transform(train=False)
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must both contain annotated images.")
    train_categories = train_dataset.coco.getCatIds()
    if set(train_categories) != set(val_dataset.coco.getCatIds()):
        raise ValueError("Training and validation category IDs must match.")
    # CocoDetectionDataset 使用 category_id + 1，checkpoint 也沿用该映射。
    if sorted(train_categories) != list(range(len(train_categories))):
        raise ValueError("Expected contiguous category IDs starting at 0.")
    return train_dataset, val_dataset


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

def train():
    start_time = time.time() # 启动时间
    # 设置SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(f"cuda:{GPU_ID}")

    train_dataset, val_dataset = get_train_and_val_dataset()
    num_classes = len(train_dataset.coco.getCatIds()) + 1 # 0号被预留给了bgclass,所以对于frcnn模型来说模型分类数量就是实际业务分类数量+1
    # 构建模型与加载checkpoint
    model = model_load_weight(build_model(num_classes), WEIGHT_PATH)
    # 模型放到gpu上
    model.to(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # `parents=True` ：递归创建路径中所有不存在的中间目录
    print(f"Retrain checkpoint: {WEIGHT_PATH}")
    print(f"Device: {device}, epochs: {EPOCHS}, batch size: {BATCH_SIZE}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True, # 将读取的数据放入锁页内存，通常有利于 CPU 到 GPU 的数据传输
        collate_fn=collate_fn, # 指定如何把多个样本整理成一个 batch。目标检测中，不同图片的框数量不同，需要专门的batch整理函数。
        worker_init_fn=partial(worker_init_fn, seed=SEED)
    )
    generator = torch.Generator().manual_seed(SEED) # torch随机数生成器用于打乱训练集的
    # 训练集加载器（随机打乱）
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    # 参数优化器
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=0.0005
    )
    # 学习率调节器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(EPOCHS * 0.6), int(EPOCHS * 0.8)], gamma=0.1
    )

    best_map50 = float("-inf")
    best_epoch = None
    for epoch in range(EPOCHS):
        print(f"\nRetrain epoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25) # 不开启混合精度(没有传入 scaler),print_freq=25表示每过25个batch就会打印。
        lr_scheduler.step()
        # evaluate 会把数据集 target 转成 COCO 格式，保留 category_id + 1 的映射。
        coco_evaluator = evaluate(model, val_loader, device=device)
        map50 = float(coco_evaluator.coco_eval["bbox"].stats[1])
        if not np.isfinite(map50) or map50 < 0:
            raise RuntimeError(f"Invalid validation mAP@0.5: {map50}")
        print(f"Validation mAP@0.5: {map50:.6f}")
        if map50 > best_map50:
            # 使用固定文件名覆盖，仅保留本次实验验证集上最好的权重。
            torch.save(model.state_dict(), BEST_WEIGHT_PATH)
            best_map50 = map50
            best_epoch = epoch + 1
            print(f"Best model saved to: {BEST_WEIGHT_PATH}")

    print(f"Best epoch: {best_epoch}, mAP@0.5: {best_map50:.6f}")
    print(f"Best model: {BEST_WEIGHT_PATH}")
    print(f"Elapsed time: {timestamp_to_hms(time.time() - start_time)}")


if __name__ == "__main__":
    print("PID:", os.getpid()) # 打印pid
    train()
