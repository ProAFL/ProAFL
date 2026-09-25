'''
Collect gt_box and p_box information.
'''
import os
import argparse
import torch
from utils.datasets import create_dataloader
from models.yolo import Model
import yaml
import json
from utils.general import colorstr,non_max_suppression,scale_coords
from collections import defaultdict

from custom_module.base_data_manager import get_fault_train_model_weight_file_path,get_error_ann_file_path,get_nc_by_datasetname
from custom_module.small_utils import read_yaml

def collect_one_epoch(model, dataloader, epoch, device, save_dir,
                      conf_thres=0.25, iou_thres=0.65):
    predicted_box_dict = {}
    predicted_box_id = 0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255.0
        with torch.no_grad():
            out, _ = model(imgs, augment=False)
            out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True)
            for si, preds in enumerate(out):
                if len(preds) == 0:
                    continue
                img_name = os.path.basename(paths[si])
                predn = preds.clone()
                scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                predicted_bbox_list = []
                for *xyxy, conf, cls in predn.tolist():
                    predicted_bbox_list.append({
                        "predicted_box_id": predicted_box_id,
                        "img_name": img_name,
                        "predicted_cls": int(cls),
                        "conf": conf,
                        "bbox": xyxy,
                    })
                    predicted_box_id += 1
                predicted_box_dict[img_name] = {
                    "predicted_bboxs": predicted_bbox_list,
                    "height": shapes[si][0][0],
                    "weight": shapes[si][0][1],
                }

    os.makedirs(save_dir,exist_ok=True)
    save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(predicted_box_dict, f, indent=4)
    print(f"Data saved at:{save_json_path}")

def collect_predicted_box(model, device, batch_size, workers,
                          conf_thres=0.25, iou_thres=0.65):
                        
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)
    opt = argparse.Namespace(single_cls=False)
    dataloader, dataset = create_dataloader(data["train"], 640, batch_size, gs, opt,
                                             pad=0.5, rect=True, workers=workers,
                                             prefix=colorstr('train: '))
    print(f"Total image count:{len(dataset)}")

    for epoch in range(epochs):
        # 加载权重
        weights_path = get_fault_train_model_weight_file_path(dataset_name,model_name,inject_ratio,epoch)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        # eval mode
        model.eval()
        # 收集epoch_i
        collect_one_epoch(model, dataloader, epoch, device, collect_p_box_dir,
                          conf_thres, iou_thres)

'''
def collect_gt_box():
    with open(error_annotations_path, 'r') as f:
        error_annotations = json.load(f)
    images_list = error_annotations["images"]
    gt_box_dict  = defaultdict(list)
    box_id = 0
    no_anno_count = 0
    for image in images_list:
        img_id = image["id"]
                                
        annos_of_img = search_annotations_by_img_id(img_id,error_annotations)
        img_name = image["file_name"]
        imge_name_no_ext = img_name.split(".")[0]
                           
        txt_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-yolo","train","labels",f"{imge_name_no_ext}.txt")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            no_anno_count += 1                           
        assert len(lines) == len(annos_of_img), "Annotation mismatch"
        for l_id, line in enumerate(lines):
            box_line = line.split()
            cls = int(box_line[0])
            x_center = float(box_line[1])
            y_center = float(box_line[2])
            width = float(box_line[3])
            height = float(box_line[4])
            fault_type = annos_of_img[l_id]["fault_type"]
            box = {
                "box_id":box_id,
                "img_name":img_name,
                "cls":cls,
                "gt_bbox":[x_center,y_center,width,height],
                "fault_type":fault_type
            }
            box_id += 1
            gt_box_dict[img_name].append(box)
    save_dir = collect_gt_box_dir
    save_json_file_name = "gt_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_dict, f, indent=4)
    print(f"collect_gt_boxtext, Saved at:{save_json_path}")
'''

def search_annotations_by_img_id(img_id,annotations_no_miss):
    annos_of_img = []
    annotations = annotations_no_miss["annotations"]
               
    for anno in annotations:
        if anno["image_id"] == img_id:
            annos_of_img.append(anno)
              
    return annos_of_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='inference batch size')
    parser.add_argument('--workers', type=int, default=16, help='data loader workers')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = read_yaml("../config.yaml")
    exp_data_root = config["exp_data_dir"]
    dataset_name = "voc"
    model_name = "yolov7"
    inject_ratio = 0.01
    nc = get_nc_by_datasetname(dataset_name) # 数据集分类数
    # 基于yaml配置加载出model:Model,并放到device
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)

    pbox_confi_thres = 0.25 # 低于这个置信度的预测框会被丢弃。值越大保留的预测框越少。
    iou_thres = 0.65 # NMS(非极大抑制)的重叠门槛。同类预测框IoU超过0.65时，只保留最高的那个预测框，其他的丢弃。值越大保留的预测框越多
    epochs = 50

    collect_p_box_dir = os.path.join(exp_data_root,
                                     "collection_process_info",dataset_name,model_name,"collected_predict_boxes_test",f"inject_{inject_ratio}")
    # 收集整个训练轮次的预测框
    collect_predicted_box(model, device, args.batch_size, args.workers,
                          conf_thres=pbox_confi_thres, iou_thres=iou_thres)

    '''
    不需要重新收集这个gt_box信息
    error_annotations_path = get_error_ann_file_path(dataset_name)
    collect_gt_box_dir = os.path.join(exp_data_root,"collection_process_info",dataset_name)
    collect_gt_box()
    '''
