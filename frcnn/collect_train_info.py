"""逐轮采集 FRCNN 预测框，保留完整前景类别概率。"""

import os
import time
import pprint
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
import torch,torchvision
from cutom_module.small_utils import read_yaml
from customROI import RoIHeadsCustom


class CollectionDataset(Dataset):
    """采集只需图片与路径，避免每轮重复构建训练用标注张量。"""

    def __init__(self, image_dir, annotation_path, transforms=None):
        with open(annotation_path, encoding="utf-8") as file:
            annotations = json.load(file)
        visible_ids = {row["image_id"] for row in annotations["annotations"]}
        # 保持原数据集的图片顺序和筛选规则，预测框 ID 的生成顺序不变。
        self.image_paths = [
            os.path.join(image_dir, row["file_name"])
            for row in annotations["images"] if row["id"] in visible_ids
        ]
        self.num_classes = len({row["id"] for row in annotations["categories"]}) + 1
        self.transforms = transforms
        print(f"Total images: {len(annotations['images'])}, valid images with anns: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return image, {"image_path": path}


def collate_batch(batch):
    """顶层函数支持多进程加载，图片尺寸不同时不进行堆叠。"""
    return tuple(zip(*batch))


class CollectionRoIHeads(RoIHeadsCustom):
    """Expose foreground probabilities while retaining scalar detection scores."""

    def forward(self, features, proposals, image_shapes, targets=None):
        detections, losses = super().forward(features, proposals, image_shapes, targets)
        if not self.training:
            for detection in detections:
                # Some local torchvision versions already expose the matrix as
                # "probs" and reduce "scores" to one dimension in RoIHeads.forward.
                # Stock torchvision passes through our matrix as "scores" instead.
                probs = detection.get("probs", detection["scores"])
                detection["probs"] = probs
                detection["scores"] = probs.gather(
                    1, (detection["labels"] - 1).unsqueeze(1)
                ).squeeze(1)
        return detections, losses

                                        
def get_transform():
    return ToTensor()

def build_frcnn_model(num_classes):
    # Every epoch loads a complete checkpoint, so pretrained downloads are unnecessary.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        # 让更多位置候选进入分类和框回归阶段。
        # rpn_pre_nms_top_n_test=2000,
        # rpn_post_nms_top_n_test=2000,
        # VisDrone目标密集，提高阈值可减少相邻同类框被抑制。
        # box_nms_thresh=0.75,
        # 先保留当前置信度门槛，便于判断其他调整的效果。
        box_score_thresh=0.05,
        # 最终每图最多保存500个框。
        box_detections_per_img=500
    )
                                                      
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    rh = model.roi_heads
    model.roi_heads = CollectionRoIHeads(
        rh.box_roi_pool, rh.box_head, rh.box_predictor,
        rh.proposal_matcher.high_threshold, rh.proposal_matcher.low_threshold,
        rh.fg_bg_sampler.batch_size_per_image, rh.fg_bg_sampler.positive_fraction,
        rh.box_coder.weights, rh.score_thresh, rh.nms_thresh, rh.detections_per_img,
        rh.mask_roi_pool, rh.mask_head, rh.mask_predictor,
        rh.keypoint_roi_pool, rh.keypoint_head, rh.keypoint_predictor,
    )
    return model

def get_trainset():
    train_dataset = CollectionDataset(
        image_dir=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train", # 这其实就是全部的训练集了
        annotation_path=f"{exp_data_root_dir}/fault_inject/{fault_ratio}/{dataset_name}/coco_format/annotations_no_miss.json",
        transforms=get_transform()
    )
    return train_dataset

def get_model(num_classes):
    model = build_frcnn_model(num_classes)
    return model

def set_nms(model, model_name, conf_threshold=0.25,iou_threshold=0.65):
    if model_name == "SSD":
        model.score_thresh = conf_threshold
        model.nms_thresh = iou_threshold
    elif model_name.lower() == "frcnn":
        model.roi_heads.score_thresh = conf_threshold
        model.roi_heads.nms_thresh = iou_threshold
    else:
        raise Exception("Invalid model name")
    return model

def model_load_weight(model,epoch):
                
    w_path = os.path.join(error_model_pth_dir,f"epoch_{epoch}.pth")
    state_dict = torch.load(w_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model

def collect_help(img_name,p_boxes,p_labels,confs,probs,global_id):
    # 每组张量整批转到 CPU，避免每个框都触发 GPU 同步和小块数据传输。
    boxes = p_boxes.detach().cpu().tolist()
    labels = p_labels.detach().cpu().tolist()
    scores = confs.detach().cpu().tolist()
    probabilities = probs.detach().cpu().tolist()
    if not len(boxes) == len(labels) == len(scores) == len(probabilities):
        raise ValueError(f"预测框、类别、分数和概率数量不一致：{img_name}")
    p_boxs = []
    for prediction_id, (box, label, score, class_probs) in enumerate(
        zip(boxes, labels, scores, probabilities), start=global_id
    ):
        p_box = {
            "predicted_box_id":prediction_id,
            "img_name":img_name,
            # 训练时类别加了 1；输出时恢复从 0 开始的前景类别编号。
            "predicted_cls":int(label) - 1,
            "conf":round(score, 4),
            "bbox":box,
            "probs":[round(value, 4) for value in class_probs]
        }
        p_boxs.append(p_box)
    return p_boxs


@torch.inference_mode()
def collect_one_epoch(model,dataset_loader,device):
    model.eval()
    global_id = 0
    collect_dict = {}
    for images, targets in dataset_loader:
        images = list(image.to(device,non_blocking=True) for image in images)
        predictions = model(images)
        for target,pred in zip(targets,predictions):
            img_name = os.path.basename(target["image_path"])
            p_boxes = pred['boxes']
            p_labels = pred['labels']
            confs = pred['scores']
            probs = pred['probs']
            collected_p_boxs = collect_help(img_name,p_boxes,p_labels,confs,probs,global_id)
            collect_dict[img_name] = {"predicted_bboxs":collected_p_boxs}
            global_id += len(collected_p_boxs)
    return collect_dict

def save_one_epoch(collect_dict,epoch):
    save_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_path = os.path.join(collect_save_dir,save_file_name)
    with open(save_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
        json.dump(collect_dict, f, separators=(",", ":"))
    return save_path


def collect_predicted_boxes():
    start_time = time.time()                     
    train_dataset = get_trainset()
    device = torch.device(f"cuda:{gpu_id}")
    workers = _args["num_workers"]
    loader_options = {
        "batch_size": _args["batch_size"], "num_workers": workers,
        "shuffle": False, "collate_fn": collate_batch,
        "pin_memory": device.type == "cuda",
    }
    # num_workers=0 时不能配置 persistent_workers 或 prefetch_factor。
    if workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=_args["prefetch_factor"])
    train_loader = DataLoader(train_dataset, **loader_options)
    num_classes = train_dataset.num_classes
                 
    model = get_model(num_classes)
    if _args["custom_nms"] is True:
        model = set_nms(model,model_name=model_name)
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        epoch_start = time.perf_counter()
        model = model_load_weight(model,epoch)
        loaded_at = time.perf_counter()
        collect_dict = collect_one_epoch(model,train_loader,device)
        collected_at = time.perf_counter()
        save_path = save_one_epoch(collect_dict,epoch)
        saved_at = time.perf_counter()
        print(f"权重加载 {loaded_at - epoch_start:.1f}s，"
              f"图片加载/推理/转换 {collected_at - loaded_at:.1f}s，"
              f"JSON保存 {saved_at - collected_at:.1f}s，"
              f"采集吞吐 {len(collect_dict) / max(collected_at - loaded_at, 1e-9):.2f} 张/秒")
        print(f"Collected data saved at: {save_path}")
        # 下轮开始前释放上一轮结果，避免同时保留两轮的大字典。
        del collect_dict
    end_time = time.time()                   
    elapsed_time = end_time - start_time                                     
    hours = int(elapsed_time // 3600)                   
    minutes = int((elapsed_time % 3600) // 60)                     
    seconds = elapsed_time % 60                               
    print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02.0f}")

if __name__ == "__main__":
    PID = os.getpid()
    print("PID:",PID)
    config = read_yaml("config.yaml")
    exp_data_root_dir = config["exp_data_dir"]
    gpu_id = 1
    _args = {
        "dataset_name":"visdrone", # voc|kitti|visdrone
        "model_name":"frcnn",
        "num_epochs":50,
        "batch_size":64,
        "num_workers":16,  # CPU 并行读图数；设为 0 可切换单进程加载。
        "prefetch_factor":2,
        "custom_nms":False
    }
    pprint.pprint(_args)
    
    dataset_name = _args["dataset_name"]
    model_name = _args["model_name"]
    num_epochs = _args["num_epochs"]
    fault_ratio = 0.1
    repeat_id =1
    ts = int(time.time())
    error_model_pth_dir = f"{exp_data_root_dir}/models/{dataset_name}/frcnn/fault_train/{fault_ratio}/repeat_{repeat_id}"
    collect_save_dir = f"{exp_data_root_dir}/collection_process_info/{dataset_name}/frcnn/collected_predict_boxes/inject_{fault_ratio}_repeat/repeat_{repeat_id}_{ts}"
    os.makedirs(collect_save_dir,exist_ok=True)
    collect_predicted_boxes()
