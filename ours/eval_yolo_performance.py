'''
用于评估YOLOv7模型的mAP
'''

import os
import yaml
from pathlib import Path
import argparse
import torch
from utils.datasets import create_dataloader
from utils.general import colorstr,non_max_suppression, non_max_suppression_with_probs,scale_coords,xyxy2xywh,xywh2xyxy,box_iou
from yolov7.models.yolo import Model
from utils.torch_utils import select_device,time_synchronized
from tqdm import tqdm
import numpy as np
from utils.metrics import ap_per_class,ConfusionMatrix
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from helper.base_data_manager import (get_correct_ann_file_path,get_error_ann_file_path,
                                    get_clean_train_model_weight_file_path,
                                    get_error_train_model_weight_file_path, 
                                    get_repair_train_model_weight_file_path)

def get_model(model,device):
    # 加载模型权重
    
    if weights_path.endswith("last.pt") or weights_path.endswith("best.pt"):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = state_dict['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    return model

def buquan_anno_file():
    
    if train_or_val == "train":
        in_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco","train","_annotations.coco_error.json")
        out_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco","train","_annotations.coco_error_buquan.json")
    elif train_or_val == "val":
        in_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco","val","_annotations.coco.json")
        out_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco","val","_annotations.coco_buquan.json")
    
    with open(in_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    for ann in coco.get("annotations", []):
        if "area" not in ann:
            x, y, w, h = ann["bbox"]  # COCO bbox: [x, y, w, h]
            ann["area"] = float(w * h)
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print("saved:", out_path)


def get_name_id_map(ANN_FILE):
    
    coco = COCO(ANN_FILE)

    # 1) file_name -> img_id（假设 file_name 唯一）
    name2id = {img_info["file_name"]: img_id for img_id, img_info in coco.imgs.items()}

    # 2) 反向：img_id -> file_name
    id2name = {img_id: img_info["file_name"] for img_id, img_info in coco.imgs.items()}

    return name2id,id2name


def xyxy2xywh(xyxy:list):
    x1,y1,x2,y2 = xyxy
    w = x2-x1
    h = y2-y1
    return [x1,y1,w,h]


def get_batch_res(imgs,outs,paths,shapes,name2id):
    batch_res = []
    for si, pred in enumerate(outs):
        # si: 这批图像的局部索引
        # 拷贝一份这张图像的预测
        predn = pred.clone()
        path = Path(paths[si])
        img_name = path.name
        img_id = name2id[img_name]
        scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
        for *xyxy, conf, cls in predn.tolist():
            xywh = xyxy2xywh(xyxy)
            batch_res.append({
                "image_id": int(img_id),
                "category_id": int(cls),
                "bbox":xywh,
                "score": float(conf),
            })
    return batch_res

def get_coco_results(dataloader,device,model,name2id):
    coco_res = []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, wid
        with torch.no_grad():
            # out:shape:(bs,anchors*grids,nc+5)
            outs, train_outs = model(imgs, augment=False)  # inference and training outputs
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            lb = []
            # [shape:(boxs_num,6),]
            outs = non_max_suppression_with_probs(outs, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
            for dets, probs in outs:
                print(dets.shape, probs.shape)
            batch_res = get_batch_res(imgs,outs,paths,shapes,name2id)
            coco_res.extend(batch_res)
    return coco_res


def add_area_iscrowd(coco:COCO):
    anns = coco.loadAnns(coco.getAnnIds())
    for ann in anns:
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = float(w * h)
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
    return coco



def eval_perform_coco_style():
    # 拿到数据yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    # 数据集中的类别数量
    nc = int(data['nc'])  # number of classes
    # 指定GPU设备

    device = select_device(f'{gpu_id}')
    # 获得模型
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    model = get_model(model,device)
    model.eval()
    name2id,id2name = get_name_id_map(ANN_FILE)
    # 获得数据集加载器
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    batch_size = 32
    imgsz = 640

    dataloader = create_dataloader(data[train_or_val], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{train_or_val}: '))[0]

    coco_results = get_coco_results(dataloader,device,model,name2id)
    cocoGt = COCO(ANN_FILE)
    cocoGt = add_area_iscrowd(cocoGt)
    cocoDt = cocoGt.loadRes(coco_results)
    coco_eval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

    '''
    save_dir = os.path.join(exp_data_root,"eval_model_performance",dataset_name,model_name)
    save_file_name = "coco_res.json"
    save_path = os.path.join(save_dir,save_file_name)
    with open(save_path,"w") as f:
        json.dump(coco_res,save_path)
    print(f"coco res is saved in: {save_path}")
    '''


def eval_performance_yolo_style():
    # 拿到数据yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # 数据集中的类别数量
    nc = int(data['nc'])  # number of classes
    # 指定GPU设备

    device = select_device('0')
    # 获得模型
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    epoch = 0
    model = get_model(epoch,model,device)
    model.eval()
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    # 获得数据集加载器
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    batch_size = 32
    imgsz = 640
    dataloader = create_dataloader(data["train"], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]

    # confusion_matrix = ConfusionMatrix(nc=nc)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0 # 统计总共的图像数量
    p, r, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0.

    jdict, stats, ap, ap_class = [], [], [], []

    # 批次遍历数据集
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, wid
        with torch.no_grad():
            # out:shape:(bs,anchors*grids,nc+5)
            t = time_synchronized()
            out, train_out = model(img, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            # [xyxy, conf, cls]
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # 在每张图像上做统计
        for si, pred in enumerate(out):
            seen += 1
            # si: 这批图像的局部索引
            # 拿到这张图像的所有标签: [[cls,xc,yc,width,height],...]
            labels = targets[targets[:, 0] == si, 1:]
            # 这张图像的标签数量
            nl = len(labels)
            path = Path(paths[si])
            # 这张图像的target class list
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # pred: shape: (n个预测框,xyxy+conf+cls)
            if len(pred) == 0:
                # 经过NMS后没有预测结果
                if nl:
                    # 本身具有label
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            # 拷贝一份这张图像的预测
            predn = pred.clone()
            # img[si].shape[1:]: 这张图像的chanel,width,height
            # shapes[si][0], shapes[si][1] 图像native原图size
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            '''
            # 这张图像的预测结果写入text
            for *xyxy, conf, cls in predn.tolist():
                line = (cls, *xyxy, conf)
                save_dir = os.path.join(exp_data_root,"eval_model_performance",dataset_name,model_name,"predicted_labels")
                save_file_name = path.stem + '.txt'
                save_file_path = os.path.join(save_dir,save_file_name)
                with open(save_file_path, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            '''
            img_name = path.name
            for *xyxy, conf, cls in predn.tolist():
                jdict.append({
                    "img_name":img_name,
                    "cls":int(cls),
                    "bbox_xyxy":xyxy,
                    "conf":round(conf, 5)
                })

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # 如果这张图像有预测输出
                detected = []  # target indices
                # 包含的所有cls
                tcls_tensor = labels[:, 0]
                # 转换target box的xywh => xyxy
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # 遍历unique cls
                for cls in torch.unique(tcls_tensor):
                    # 找gt_cls等于到当前cls的行索引list
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    # 找p_cls等于到当前cls的行索引list
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # Search for detections
                    if pi.shape[0]:
                        # 如果当前cls下有pi
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            # pi 中 的第j个位置的iou > 0.5
                            # i 中第j个位置存储了ti的位置
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                # pi[j]: 预测框p
                                # ious[j]: 这个预测框匹配到的最大iou
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # correct: 各个预测框在不同iouv下是否正确匹配了target
            # pred[:,4]: 各个预测框的坐标
            # pred[:,5]: 各个预测框的conf
            # tcls: 一个图像中所有目标的cls
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # Compute statistics
    # zip(*stats): *stats从stats列表中解包出多个tuple,zip(*stats)打包成所有图像的correct,..
    # *stats 把列表解包成多个 tuple 传给 zip。
    # zip(*stats) 会把“按图片存的 tuple 列表”转为“按字段分组”的迭代器：
    # 第 1 组：所有图片的 correct
    # 第 2 组：所有图片的 pred[:, 4]
    # 第 3 组：所有图片的 pred[:, 5]
    # 第 4 组：所有图片的 tcls
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any(): # .any() 表示里面至少有一个元素为 True / 非零。
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, names=names)
        # ap：shape=[len(ap_class),niou]
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # map就是map@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # 分类统计的评估结果
    if len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    # 打印每张图像的推理和NMS速度(ms)
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


def get_COCOANN_FILE(train_or_val:str):
    if train_or_val == "train":
        ANN_FILE = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",f"{train_or_val}","_annotations.coco_error.json")
    elif train_or_val == "val":
        ANN_FILE = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",f"{train_or_val}","_annotations.coco.json")
    else:
        raise Exception("get anno coco 错误")
    return ANN_FILE

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    model_state = "clean" # clean|error|repair_ours|repair_datactive
    train_or_val = "val"
    gpu_id = 1
    # ANN_FILE = get_correct_ann_file_path(dataset_name,train_or_val)
    # ANN_FILE = 
    cocoGt = COCO(ANN_FILE)

    weights_path = "/data/mml/data_debugging_data/models/visdrone/yolov7/repair_ours/alpha=1.5_val/weights/best.pt"
    '''
    if model_state == "clean":
        weights_path = get_clean_train_model_weight_file_path(dataset_name,model_name)
    elif model_state == "error":
        weights_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
    elif model_state == "repair_ours":
        weights_path = get_repair_train_model_weight_file_path(dataset_name,model_name,method_name="ours")
    elif model_state == "repair_datactive":
        weights_path = get_repair_train_model_weight_file_path(dataset_name,model_name,method_name="datactive")
    # buquan_anno_file()
    '''
    eval_perform_coco_style()

