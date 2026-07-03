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
                            
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
                                                          
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
          
    train_loss = 0
          
    val_loss = 0
    if local_rank == 0:
        print('Start Train')
        print(f"Epoch:{epoch}/{Epoch}")
               
    model.train()
                                                  
    warm_lr_scheduler = None
    if epoch == 0:
                        
        warmup_factor = 1.0 / 1000
                                 
        warmup_iters = min(1000, len(train_dataloader) - 1)
                                                      
        warm_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    batch_step = 0
    for images,targets in train_dataloader:
        images = list(image.cuda(local_rank) for image in images)
        targets = [{k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
              
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())       
                                          
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                                
        loss_value = losses_reduced.item()
        train_loss += loss_value
        batch_step += 1
        
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
    train_loss = train_loss / batch_step
                        
    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

                                                                
    eval_batch_step = 0
    for images, targets in val_dataloader:
        images = list(image.cuda(local_rank) for image in images)
        targets = [{k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                                                       
        with torch.no_grad():
            loss_dict = model(images, targets)
                                          
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                                
        loss_value = losses_reduced.item()
        val_loss += loss_value
        eval_batch_step += 1
    val_loss = val_loss / eval_batch_step

                     
    if local_rank == 0:
        print('Finish Validation')
        print('Train Loss: %.3f || Val Loss: %.3f ' % (train_loss, val_loss))
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
                
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                                            
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

                                         
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

                                            
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
