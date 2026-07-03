'''
other baselinestext
'''

import os
import time
import pprint
import json
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import ToTensor
from datasets import CocoDetectionDataset
import torch,torchvision
from collections import defaultdict
from customROI import RoIHeadsCustom
from cutom_module.small_utils import read_yaml
                                        
def get_transform():
    return ToTensor()

def build_frcnn_model(num_classes):
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                                                      
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


                               

                                                           
    rh = model.roi_heads

    model.roi_heads = RoIHeadsCustom(
        rh.box_roi_pool,
        rh.box_head,
        rh.box_predictor,

                            
        rh.proposal_matcher.high_threshold,
        rh.proposal_matcher.low_threshold,
        rh.fg_bg_sampler.batch_size_per_image,
        rh.fg_bg_sampler.positive_fraction,
        rh.box_coder.weights,

                             
        rh.score_thresh,
        rh.nms_thresh,
        rh.detections_per_img,

                                                    
        rh.mask_roi_pool,
        rh.mask_head,
        rh.mask_predictor,
        rh.keypoint_roi_pool,
        rh.keypoint_head,
        rh.keypoint_predictor,
    )
    
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

def set_nms(model, conf_threshold=0.25,iou_threshold=0.65):
    model.roi_heads.score_thresh = conf_threshold
    model.roi_heads.nms_thresh = iou_threshold
    return model

def model_load_weight(model,epoch):
                
    w_path = os.path.join(error_model_pth_dir,f"epoch_{epoch}.pth")
    state_dict = torch.load(w_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def collect_help(img_name,p_boxes,p_labels,confs,probs,global_id):
    p_boxs = []
    for i in range(p_boxes.shape[0]):
        p_box = p_boxes[i].tolist()
        p_label = int(p_labels[i])
        conf = confs[i].item()
        prob = probs[i].tolist()
        p_box = {
            "predicted_box_id":global_id,
            "img_name":img_name,
            "predicted_cls":p_label,
            "conf":conf,
            "bbox":p_box,
            "prob":prob
        }
        p_boxs.append(p_box)
        global_id += 1
    return p_boxs


def ssd_forward(model, images: list[torch.Tensor]):
    """
    SSD Manual forward inference, returning a predictions list in the same format as FRCNN.
    Each element is a dict containing:
        boxes  : Tensor[N, 4]          detection box coordinates (x1,y1,x2,y2)
        labels : Tensor[N]             predicted class id
        scores : Tensor[N]             confidence, the softmax value of the predicted class
        probs  : Tensor[N, num_classes] full-class softmax probability distribution for each detection box
    """
    from torchvision.ops import boxes as box_ops

    model.eval()
    with torch.no_grad():
                                                                                  
        images_transformed, _ = model.transform(images, None)

                                                                                   
        features = model.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

                                                                           
        head_outputs    = model.head(features)
        cls_logits      = head_outputs["cls_logits"]                                      
        bbox_regression = head_outputs["bbox_regression"]                       

                                                                              
        anchors = model.anchor_generator(images_transformed, features)
                                                           

                                                                                    
        pred_scores_all = F.softmax(cls_logits, dim=-1)                                   

        detections = []
        for boxes_reg, scores, ancs, image_shape in zip(
            bbox_regression, pred_scores_all, anchors, images_transformed.image_sizes
        ):
                                          
                                                                         
                                          

            num_classes = scores.shape[-1]
            device      = scores.device

                                                                                                
            decoded_boxes = model.box_coder.decode_single(boxes_reg, ancs)                    
            decoded_boxes = box_ops.clip_boxes_to_image(decoded_boxes, image_shape)

            img_boxes       = []
            img_scores_flat = []
            img_labels      = []
            img_anchor_idxs = []                                                                             

                                                                                              
            for label in range(1, num_classes):
                cls_score = scores[:, label]                                             
                keep_idxs = torch.where(cls_score > model.score_thresh)[0]
                if keep_idxs.numel() == 0:
                    continue

                cls_score   = cls_score[keep_idxs]
                cls_boxes   = decoded_boxes[keep_idxs]

                                                                               
                num_topk = min(model.topk_candidates, cls_score.size(0))
                cls_score, topk_idxs = cls_score.topk(num_topk)
                cls_boxes   = cls_boxes[topk_idxs]
                orig_idxs   = keep_idxs[topk_idxs]                          

                img_boxes.append(cls_boxes)
                img_scores_flat.append(cls_score)
                img_labels.append(torch.full((cls_score.shape[0],), label,
                                             dtype=torch.int64, device=device))
                img_anchor_idxs.append(orig_idxs)

                                                                                                           
            if len(img_boxes) == 0:
                detections.append({
                    "boxes":  torch.zeros((0, 4),        device=device),
                    "labels": torch.zeros((0,),          device=device, dtype=torch.int64),
                    "scores": torch.zeros((0,),          device=device),
                    "probs":  torch.zeros((0, num_classes - 1), device=device),
                })
                continue

            img_boxes        = torch.cat(img_boxes,        dim=0)          
            img_scores_flat  = torch.cat(img_scores_flat,  dim=0)       
            img_labels       = torch.cat(img_labels,       dim=0)       
            img_anchor_idxs  = torch.cat(img_anchor_idxs,  dim=0)       

                                                                           
            keep = box_ops.batched_nms(img_boxes, img_scores_flat, img_labels, model.nms_thresh)
            keep = keep[:model.detections_per_img]

                                                                                                     
            kept_anchor_idxs = img_anchor_idxs[keep]
                                                                                                             
                                                       
            probs = scores[kept_anchor_idxs][:, 1:]                                

            detections.append({
                "boxes":  img_boxes[keep],
                "labels": img_labels[keep],
                "scores": img_scores_flat[keep],
                "probs":  probs,
            })

    return detections


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
            probs = pred['probs']
            if p_boxes.shape[0] > 0:
                                                                
                collected_p_boxs = collect_help(img_name,p_boxes,p_labels,confs,probs,global_id)
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


def collect():
    start_time = time.time()                     
          
    train_dataset = get_trainset()
          
    train_loader = DataLoader(train_dataset, batch_size=16, 
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
             
    num_classes = len(train_dataset.coco.getCatIds()) + 1
                 
    model = get_model(num_classes)
    if _args["custom_nums"] is True:
        model = set_nms(model)
                
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
                      
    model = model_load_weight(model,_args["last_epoch"])
    collect_dict = collect_one_epoch(model,train_loader,device)
    save_path = save_one_epoch(collect_dict,_args["last_epoch"])
    print(f"Collected data saved at: {save_path}")

    end_time = time.time()                   
    elapsed_time = end_time - start_time                                     
    hours = int(elapsed_time // 3600)                   
    minutes = int((elapsed_time % 3600) // 60)                     
    seconds = elapsed_time % 60                               
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root_dir = config["exp_data_dir"]
    gpu_id = 0
    PID = os.getpid()
    print("PID:",PID)
    _args = {
        "dataset_name":"voc",                     
        "model_name":"frcnn",        
        "last_epoch":49,
        "custom_nums":False
    }
    pprint.pprint(_args)
    
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]
    error_model_pth_dir = os.path.join(exp_data_root_dir,"models",dataset_name,
                                       model_name,"error")
    collect_save_dir = os.path.join(exp_data_root_dir,"collection_process_info",
                                    dataset_name,model_name,"for_baselines",
                                    "collected_predict_boxes_withprobs")
    os.makedirs(collect_save_dir,exist_ok=True)
    collect()