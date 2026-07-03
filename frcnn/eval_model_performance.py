
'''
textFRCNNtextSSDtextmAP
'''
import os
import json
from datasets import CocoDetectionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import torch,torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from engine import evaluate
from frcnn.cutom_module.base_data_manager import (get_correct_ann_file_path,get_error_train_model_weight_file_path,
                               get_imgs_dir,exp_data_root_dir,get_clean_train_model_weight_file_path,
                               get_repair_train_model_weight_file_path,)
from customROI import RoIHeadsCustom
def get_transform():
    return ToTensor()

def build_ssd_model(num_classes):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.head.classification_head = SSDClassificationHead(
        [512, 1024, 512, 256, 256, 256],
        model.anchor_generator.num_anchors_per_location(), 
        num_classes
    )
    return model

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



def model_load_weight(model,model_weights_path):
                
    state_dict = torch.load(model_weights_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def get_coco_results(model, data_loader, device, score_thresh=0.5):
    results = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        outputs = model(images)

        for target, out in zip(targets, outputs):
            image_id = target["image_id"]
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

                          
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]     
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]     
            boxes_xywh[:, 0] = boxes[:, 0]                   
            boxes_xywh[:, 1] = boxes[:, 1]                   

            for box, score, label in zip(boxes_xywh, scores, labels):
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x) for x in box.tolist()],
                    "score": float(score),
                })
    return results


def set_nms(model, model_name, conf_threshold=0.25,iou_threshold=0.5):
    if model_name == "SSD":
        model.score_thresh = conf_threshold
        model.nms_thresh = iou_threshold
    elif model_name == "FRCNN":
        model.roi_heads.nms_thresh = iou_threshold
        model.roi_heads.score_thresh = conf_threshold
    else:
        raise Exception("Invalid model name")
    return model

def offset_category_id(cocoGt):
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    for cat in cats:
        cat["id"] += 1
    anns = cocoGt.loadAnns(cocoGt.getAnnIds())
    for ann in anns:
        ann["category_id"] = ann["category_id"] + 1
    return cocoGt
    


def eval_performance():
          
    dataset = CocoDetectionDataset(
        image_dir=get_imgs_dir(dataset_name,train_or_val,style="coco"),
        annotation_path=ANN_FILE,
        transforms=get_transform()
    )
    dataset_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

                
    num_classes = len(dataset.coco.getCatIds()) + 1
    if model_name == "SSD":
        model = build_ssd_model(num_classes)
    elif model_name == "FRCNN":
        model = build_frcnn_model(num_classes)
    else:
        raise Exception("Invalid model name")
    
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    model = model_load_weight(model,model_weights_path)
                                                                               
    
    model.eval()
    '''
    for images, targets in train_loader:
        all_labels = torch.cat([t["labels"] for t in targets])
        # print("num_classes in model:", model.roi_heads.box_predictor.cls_score.out_features)
        # print("max label in batch:", int(all_labels.max()))
        # print("min label in batch:", int(all_labels.min()))
        images = list(img.to(device) for img in images)
        outputs = model(images)
        print()
    '''

    evaluate(model, dataset_loader, device=device)

    '''
    # text
    coco_results = get_coco_results(model, dataset_loader, device, score_thresh=0.0)
    # textground truth data
    cocoGt = COCO(ANN_FILE)
    cocoGt = offset_category_id(cocoGt)
    cocoDt = cocoGt.loadRes(coco_results)
    coco_eval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
    '''

if __name__ == "__main__":
    dataset_name = "VOC2012"                           
    model_name = "SSD"            
    model_state = "clean"                                           
    gpu_id = 0
    train_or_val = "val"                         
    ANN_FILE = get_correct_ann_file_path(dataset_name,train_or_val)
    if model_state == "clean":
        model_weights_path = get_clean_train_model_weight_file_path(dataset_name,model_name)
    elif model_state == "error":
        model_weights_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
    elif model_state == "repair_ours":
        model_weights_path = get_repair_train_model_weight_file_path(dataset_name,model_name,method_name="ours")
    elif model_state == "repair_datactive":
        model_weights_path = get_repair_train_model_weight_file_path(dataset_name,model_name,method_name="datactive")
    eval_performance()