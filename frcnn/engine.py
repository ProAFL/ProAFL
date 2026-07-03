import math
import sys
import time
import os
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    warm_lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        warm_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # torch.amp.autocast
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if warm_lr_scheduler is not None:
            warm_lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def train_one_epoch_for_distribution(model,model_without_ddp,optimizer,train_dataloader,val_dataloader,
                                     epoch,Epoch,loss_history,save_period,save_dir,local_rank=0,scaler=None):
    # 训练集上的总损失
    train_loss = 0
    # 验证集上的损失
    val_loss = 0
    if local_rank == 0:
        print('Start Train')
        print(f"Epoch:{epoch}/{Epoch}")
    # model进入训练模式
    model.train()
    # 设置热身阶段的学习率衰减策略(学习率调度器)
    warm_lr_scheduler = None
    if epoch == 0:
        # epoch == 0时就是热身训练
        warmup_factor = 1.0 / 1000
        # 按照batch step进行lr调整
        warmup_iters = min(1000, len(train_dataloader) - 1)
        # 设置热身阶段的学习率衰减策略(学习率调度器)
        warm_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    batch_step = 0
    for images,targets in train_dataloader:
        images = list(image.cuda(local_rank) for image in images)
        targets = [{k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # 前向传播
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values()) # 分类损失和回归损失
        # 所有进程的loss进行规约
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # 获得规约后的loss value
        loss_value = losses_reduced.item()
        train_loss += loss_value
        batch_step += 1
        
        if not math.isfinite(loss_value):
            # 如果规约后的loss value是无限大了，训练终止
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        # 优化器清空上次计算的梯度
        optimizer.zero_grad()
        # 反向传播
        if scaler is not None:
            # 混合精度backward
            scaler.scale(losses).backward() # 计算参数梯度值
            scaler.step(optimizer) # 优化参数
            scaler.update() # 半精度更新
        else:
            # 计算参数梯度值
            losses.backward()
            # 优化参数
            optimizer.step()

        if warm_lr_scheduler is not None:
            # 按照batch step进行lr调整
            warm_lr_scheduler.step()
    train_loss = train_loss / batch_step
    # 一个epoch训练完毕，开始验证
    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    # 所有rank一起做验证 (val_dataloader 需使用 DistributedSampler 分片)
    eval_batch_step = 0
    for images, targets in val_dataloader:
        images = list(image.cuda(local_rank) for image in images)
        targets = [{k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # 前向传播 (model 仍在 train 模式才会返回 loss_dict)
        with torch.no_grad():
            loss_dict = model(images, targets)
        # 所有进程的loss进行规约
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # 获得规约后的loss value
        loss_value = losses_reduced.item()
        val_loss += loss_value
        eval_batch_step += 1
    val_loss = val_loss / eval_batch_step

    # 仅rank 0 负责记录历史与保存权重
    if local_rank == 0:
        print('Finish Validation')
        print('Train Loss: %.3f || Val Loss: %.3f ' % (train_loss, val_loss))
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        #   保存权值
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            # 定期间隔(save_period)和最后一个epoch都会保存模型权重
            torch.save(model_without_ddp.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))

        if len(loss_history["val_loss"]) <= 1 or val_loss <= min(loss_history["val_loss"]):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model_without_ddp.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(model_without_ddp.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
