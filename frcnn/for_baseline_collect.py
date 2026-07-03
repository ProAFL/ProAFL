'''
other baselines收集预测信息脚本
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
# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()

def build_frcnn_model(num_classes):
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    # 替换成我们的 RoIHeads

    # 获取原 roi_heads 参数（很关键）
    rh = model.roi_heads

    model.roi_heads = RoIHeadsCustom(
        rh.box_roi_pool,
        rh.box_head,
        rh.box_predictor,

        # ---- Training ----
        rh.proposal_matcher.high_threshold,
        rh.proposal_matcher.low_threshold,
        rh.fg_bg_sampler.batch_size_per_image,
        rh.fg_bg_sampler.positive_fraction,
        rh.box_coder.weights,

        # ---- Inference ----
        rh.score_thresh,
        rh.nms_thresh,
        rh.detections_per_img,

        # ---- Mask / Keypoints（保持原样） ----
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
        image_dir=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train", # 全量train
        annotation_path=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json",
        transforms=get_transform()
    )
    return train_dataset

def get_model(num_classes):
    # 加载模型
    model = build_frcnn_model(num_classes)
    return model

def set_nms(model, conf_threshold=0.25,iou_threshold=0.65):
    model.roi_heads.score_thresh = conf_threshold
    model.roi_heads.nms_thresh = iou_threshold
    return model

def model_load_weight(model,epoch):
    # 加载模型
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
    SSD 手动前向推理，返回与 FRCNN 完全相同格式的 predictions 列表。
    每个元素是 dict，包含：
        boxes  : Tensor[N, 4]          检测框坐标 (x1,y1,x2,y2)
        labels : Tensor[N]             预测类别 id
        scores : Tensor[N]             置信度（预测类别的 softmax 值）
        probs  : Tensor[N, num_classes] 每个检测框的全类别 softmax 概率分布
    """
    from torchvision.ops import boxes as box_ops

    model.eval()
    with torch.no_grad():
        # ── 1. 图像预处理 ──────────────────────────────────────────────
        images_transformed, _ = model.transform(images, None)

        # ── 2. Backbone 特征提取 ───────────────────────────────────────
        features = model.backbone(images_transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # ── 3. SSD Head 输出 ───────────────────────────────────────────
        head_outputs    = model.head(features)
        cls_logits      = head_outputs["cls_logits"]       # [B, num_anchors, num_classes]
        bbox_regression = head_outputs["bbox_regression"]  # [B, num_anchors, 4]

        # ── 4. 生成 Anchor ─────────────────────────────────────────────
        anchors = model.anchor_generator(images_transformed, features)
        # anchors: list of Tensor[num_anchors, 4], len == B

        # ── 5. 全局 Softmax 概率 ───────────────────────────────────────
        pred_scores_all = F.softmax(cls_logits, dim=-1)    # [B, num_anchors, num_classes]

        detections = []
        for boxes_reg, scores, ancs, image_shape in zip(
            bbox_regression, pred_scores_all, anchors, images_transformed.image_sizes
        ):
            # boxes_reg : [num_anchors, 4]
            # scores    : [num_anchors, num_classes]  (softmax 已完成)
            # ancs      : [num_anchors, 4]

            num_classes = scores.shape[-1]
            device      = scores.device

            # ── 5a. 解码 anchor → 真实坐标，并裁剪到图像边界 ───────────
            decoded_boxes = model.box_coder.decode_single(boxes_reg, ancs)  # [num_anchors, 4]
            decoded_boxes = box_ops.clip_boxes_to_image(decoded_boxes, image_shape)

            img_boxes       = []
            img_scores_flat = []
            img_labels      = []
            img_anchor_idxs = []   # 记录每个候选框来自哪个 anchor，用于事后回查 probs

            # ── 5b. 逐类过滤（跳过背景类 0）─────────────────────────────
            for label in range(1, num_classes):
                cls_score = scores[:, label]                              # [num_anchors]
                keep_idxs = torch.where(cls_score > model.score_thresh)[0]
                if keep_idxs.numel() == 0:
                    continue

                cls_score   = cls_score[keep_idxs]
                cls_boxes   = decoded_boxes[keep_idxs]

                # top-k 候选（与 torchvision SSD 源码保持一致）
                num_topk = min(model.topk_candidates, cls_score.size(0))
                cls_score, topk_idxs = cls_score.topk(num_topk)
                cls_boxes   = cls_boxes[topk_idxs]
                orig_idxs   = keep_idxs[topk_idxs]   # 原始 anchor 索引

                img_boxes.append(cls_boxes)
                img_scores_flat.append(cls_score)
                img_labels.append(torch.full((cls_score.shape[0],), label,
                                             dtype=torch.int64, device=device))
                img_anchor_idxs.append(orig_idxs)

            # ── 5c. 无检测结果时返回空结构 ───────────────────────────────
            if len(img_boxes) == 0:
                detections.append({
                    "boxes":  torch.zeros((0, 4),        device=device),
                    "labels": torch.zeros((0,),          device=device, dtype=torch.int64),
                    "scores": torch.zeros((0,),          device=device),
                    "probs":  torch.zeros((0, num_classes - 1), device=device),
                })
                continue

            img_boxes        = torch.cat(img_boxes,        dim=0)  # [M, 4]
            img_scores_flat  = torch.cat(img_scores_flat,  dim=0)  # [M]
            img_labels       = torch.cat(img_labels,       dim=0)  # [M]
            img_anchor_idxs  = torch.cat(img_anchor_idxs,  dim=0)  # [M]

            # ── 5d. Batched NMS ──────────────────────────────────────────
            keep = box_ops.batched_nms(img_boxes, img_scores_flat, img_labels, model.nms_thresh)
            keep = keep[:model.detections_per_img]

            # ── 5e. 用 anchor 索引回查全类别 softmax 概率 ─────────────────
            kept_anchor_idxs = img_anchor_idxs[keep]
            # 去掉 index=0 的背景类，与 FRCNN 的 probs 维度语义保持一致：
            # probs[:, k] 对应真实类别 k+1
            probs = scores[kept_anchor_idxs][:, 1:]   # [num_kept, num_classes - 1]

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
        # predictions:
            # type:list, len(predictions) == batchsize
        # predictions[0]: 
        #   type:dict, dict_keys(['boxes', 'labels', 'scores', 'probs'])
        #   predictions[0]["boxes"].shape == (p_box_nums, 4), type:Tensor
        #   predictions[0]["labels"].shape == (p_box_nums), type:Tensor
        #   predictions[0]["scores"].shape == (p_box_nums), type:Tensor
        #   predictions[0]["probs"].shape == (p_box_nums, nc), type:Tensor
        # predictions = model(images)

        predictions = model(images)
        for img,target,pred in zip(images,targets,predictions):
            img_name = target["image_path"].split("/")[-1]
            p_boxes = pred['boxes']
            p_labels = pred['labels']
            confs = pred['scores']
            probs = pred['probs']
            if p_boxes.shape[0] > 0:
                # 模型对该图像有预测输出
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
    if _args["custom_nums"] is True:
        model = set_nms(model)
    # 得到设备
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    # 开始收集
    model = model_load_weight(model,_args["last_epoch"])
    collect_dict = collect_one_epoch(model,train_loader,device)
    save_path = save_one_epoch(collect_dict,_args["last_epoch"])
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
        "model_name":"frcnn", # frcnn
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