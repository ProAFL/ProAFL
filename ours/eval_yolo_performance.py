'''
Evaluate YOLOv7 model mAP.
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
            x, y, w, h = ann["bbox"]                           
            ann["area"] = float(w * h)
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print("saved:", out_path)


def get_name_id_map(ANN_FILE):
    
    coco = COCO(ANN_FILE)

                                                          
    name2id = {img_info["file_name"]: img_id for img_id, img_info in coco.imgs.items()}

                                     
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
                                             
              
        predn = pred.clone()
        path = Path(paths[si])
        img_name = path.name
        img_id = name2id[img_name]
        scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])                     
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
        imgs /= 255.0                        
        targets = targets.to(device)
        nb, _, height, width = imgs.shape                                     
        with torch.no_grad():
                                               
            outs, train_outs = model(imgs, augment=False)                                  
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)             
                                                                       
            lb = []
                                   
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
                        
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
                               
    nc = int(data['nc'])                     
                    

    device = select_device(f'{gpu_id}')
               
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    model = get_model(model,device)
    model.eval()
    name2id,id2name = get_name_id_map(ANN_FILE)
                        
    gs = max(int(model.stride.max()), 32)                          
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
                 
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
                        
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

                               
    nc = int(data['nc'])                     
                    

    device = select_device('0')
               
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    epoch = 0
    model = get_model(epoch,model,device)
    model.eval()
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

                        
    gs = max(int(model.stride.max()), 32)                          
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
                 
    batch_size = 32
    imgsz = 640
    dataloader = create_dataloader(data["train"], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]

                                               
    iouv = torch.linspace(0.5, 0.95, 10).to(device)                               
    niou = iouv.numel()

    seen = 0            
    p, r, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0.

    jdict, stats, ap, ap_class = [], [], [], []

          
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0                        
        targets = targets.to(device)
        nb, _, height, width = img.shape                                     
        with torch.no_grad():
                                               
            t = time_synchronized()
            out, train_out = model(img, augment=False)                                  
            t0 += time_synchronized() - t
                     
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)             
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
                               
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

              
        for si, pred in enumerate(out):
            seen += 1
                                                 
                                                  
            labels = targets[targets[:, 0] == si, 1:]
                       
            nl = len(labels)
            path = Path(paths[si])
                                   
            tcls = labels[:, 0].tolist() if nl else []                
                                                
            if len(pred) == 0:
                             
                if nl:
                               
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

                         
                  
            predn = pred.clone()
                                                        
                                                             
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])                     

            '''
            # texttext
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

                                                 
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                            
                detected = []                  
                         
                tcls_tensor = labels[:, 0]
                                                
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])                       

                                
                for cls in torch.unique(tcls_tensor):
                                               
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)                  
                                              
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)                      
                                           
                    if pi.shape[0]:
                                       
                                                   
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)                      

                                           
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                                        
                                               
                            d = ti[i[j]]                   
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                              
                                                  
                                correct[pi[j]] = ious[j] > iouv                    
                                if len(detected) == nl:                                        
                                    break

                                                           
                                         
                             
                                 
                           
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                        
                                                                     
                                 
                                                      
                              
                                 
                                 
                           
    stats = [np.concatenate(x, 0) for x in zip(*stats)]            
    if len(stats) and stats[0].any():                           
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, names=names)
                                       
        ap50, ap = ap[:, 0], ap.mean(1)                       
                             
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)                               
    else:
        nt = torch.zeros(1)

                   
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4                
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

          
    if len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

                  
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)         
                     
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


def get_COCOANN_FILE(train_or_val:str):
    if train_or_val == "train":
        ANN_FILE = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",f"{train_or_val}","_annotations.coco_error.json")
    elif train_or_val == "val":
        ANN_FILE = os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",f"{train_or_val}","_annotations.coco.json")
    else:
        raise Exception("get anno coco text")
    return ANN_FILE

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone"                           
    model_name = "YOLOv7"
    model_state = "clean"                                           
    train_or_val = "val"
    gpu_id = 1
                                                                     
                 
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

