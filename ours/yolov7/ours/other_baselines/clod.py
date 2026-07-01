import os
import joblib
import numpy as np
from collections import defaultdict
from PIL import Image
from ours.base_data_manager import get_collected_gt_box_json_path
from ours.small_utils import read_json,xcycwh_to_x1y1x2y2,calu_iou,get_nc

def infer_clod_error_type(gts, preds):
    if len(gts) == 0 and len(preds) > 0:
        return "missing"

    if len(gts) > 0 and len(preds) == 0:
        return "spurious"

    gt_classes = set(g["cls"] for g in gts)
    pred_classes = set(p["cls"] for p in preds)

    if len(gt_classes & pred_classes) == 0:
        return "mislabeled"

    return "mislocated_or_correct"

def clod_aggregate(self_confidences, alpha=0.8):
    vals = sorted(self_confidences, reverse=True)
    S = vals[0]
    for v in vals[1:]:
        S = alpha * v + (1 - alpha) * S
    return S

def get_imgname2gboxs():
    imgname2gboxs = defaultdict(list)
    for img_name,gbox_list in g_json.items():
        img_path = os.path.join(train_img_dir,img_name)
        # 图像的width,height
        image = Image.open(img_path)
        width, height = image.size
        for gbox in gbox_list:
            x1y1x2y2 = xcycwh_to_x1y1x2y2(gbox["gt_bbox"],width,height)
            imgname2gboxs[img_name].append({
                "kind":"gbox",
                "id":gbox["box_id"],
                "cls":gbox["cls"],
                "x1y1x2y2":x1y1x2y2,
                "fault_type":gbox["fault_type"],
                "imgname":img_name
            })
    return imgname2gboxs

def get_imgname2pboxs():
    imgname2pboxs = defaultdict(list)
    for img_name in p_json.keys():
        for pbox in p_json[img_name]["predicted_bboxs"]:
            imgname2pboxs[img_name].append({
                "kind":"pbox",
                "id":pbox["predicted_box_id"],
                "cls":pbox["predicted_cls"],
                "x1y1x2y2":pbox["bbox"],
                "prob":pbox["prob"],
                "conf":pbox["conf"],
                "imgname":img_name
            })
            
    return imgname2pboxs

def clustering(gboxs,pboxs,iou_thr):
    all_boxes = []
    all_boxes.extend(gboxs)
    all_boxes.extend(pboxs)

    n = len(all_boxes)
    if n == 0:
        return []

    visited = [False] * n
    adjacency = [[] for _ in range(n)]

    # Build graph: connect boxes whose IoU >= threshold
    for i in range(n):
        for j in range(i + 1, n):
            iou = calu_iou(all_boxes[i]["x1y1x2y2"], all_boxes[j]["x1y1x2y2"])
            if iou >= iou_thr:
                adjacency[i].append(j)
                adjacency[j].append(i)

    clusters = []

    # Find connected components
    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component_indices = []

        while stack:
            u = stack.pop()
            component_indices.append(u)

            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        clusters.append([all_boxes[idx] for idx in component_indices])

    return clusters

def clod_score(imgname,gboxs,pboxs,iou_thr=0.5, alpha=0.8):
    res = {}
    res[imgname] = []
    # 1. 按 IoU >= 0.5 做 clustering
    clusters = clustering(gboxs,pboxs,iou_thr)
    for cluster in clusters:
        gts = [b for b in cluster if b["kind"] == "gbox"]
        preds = [b for b in cluster if b["kind"] == "pbox"]

        # 3. 构造 Y' 和 P'
        # 长度 num_classes + 1，最后一维是 background
        y = [0.0] * (num_classes + 1)
        p = [0.0] * (num_classes + 1)

        for gt in gts:
            y[gt["cls"]] = 1.0

        if len(gts) == 0:
            y[num_classes] = 1.0  # background
        else:
            y[num_classes] = 0.0

        for pred in preds:
            c = pred["cls"]
            p[c] = max(p[c], pred["conf"])

        if max(p[:num_classes]) == 0:
            p[num_classes] = 1.0
        else:
            p[num_classes] = 0.0

        # 4. multi-label self-confidence
        self_conf = []
        for ym, pm in zip(y, p):
            sm = ym * pm + (1 - ym) * (1 - pm)
            self_conf.append(sm)

        quality = clod_aggregate(self_conf, alpha=alpha) # quality 越高：标注越可信
        suspiciousness = 1.0 - quality # suspiciousness 越高：标注越不可信

        # 5. 判断类型
        err_type = infer_clod_error_type(gts, preds)

        # 6. 输出候选
        # candidates = []
        if len(gts) > 0:
            for gt in gts:
                res[gt["id"]] = {}
                res[gt["id"]] = suspiciousness
                # candidates.append({
                #     "image_name": imgname,
                #     "unit_type": "gbox",
                #     "box_id": gt["id"],
                #     "x1y1x2y2": gt["x1y1x2y2"],
                #     "quality": quality, 
                #     "suspiciousness": suspiciousness,
                #     "error_type": err_type,
                # })
        
        else:
            for pred in preds:
                res[imgname].append(suspiciousness)
                # candidates.append({
                #     "image_name": imgname,
                #     "unit_type": "pbox",
                #     "box_id": pred["id"],
                #     "x1y1x2y2": pred["box1y1x2y2x"],
                #     "pred_class": pred["cls"],
                #     "pred_score": pred["conf"],
                #     "quality": quality,
                #     "suspiciousness": suspiciousness,
                #     "error_type": "missing",
                # })
    if len(res[imgname]) > 0:
        res[imgname] = max(res[imgname]) 
    else:
        res[imgname] = 0.0
    return res



def main():
    imgname2gboxs = get_imgname2gboxs()
    imgname2pboxs = get_imgname2pboxs()
    print("gboxs and pboxs 重组完毕")
    res = {}
    for imgname in imgname2gboxs.keys():
        gboxs = imgname2gboxs[imgname]
        pboxs = imgname2pboxs[imgname]
        score_res = clod_score(imgname,gboxs,pboxs)
        for key,value in score_res.items():
            res[key] = value
    print("clod lab打分完毕")
    # ranking
    rank = sorted(res, key=res.get, reverse=True) # gid/imgname rank， 值越大越可疑越靠前

    # 保存rank
    save_dir = os.path.join(exp_root_dir,"Results","other_baselines","clod",
                            dataset_name,model_name,f"exp_{exp_id}","rank")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "rank_temp.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(rank,save_path)
    print(f"rank长度为:{len(rank)}")
    print(f'rank结果保存在:{save_path}')



if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    exp_id = "01"
    dataset_name = "VOC2012" # VOC2012|KITTI_8|VisDrone
    model_name = "FRCNN" # YOLOv7|FRCNN|rtdetr
    num_classes = get_nc(dataset_name)
    epoch = 99 if model_name == "rtdetr" else 49
    train_img_dir = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-yolo","origin","train","images")
    g_json_path = get_collected_gt_box_json_path(dataset_name)
    g_json = read_json(g_json_path)
    if model_name in ["YOLOv7","FRCNN"]:
        collect_p_box_dir = os.path.join(exp_root_dir,"collection_bbox_level",
                                        dataset_name,model_name,"other_baselines",
                                        "collected_predicted_box_withprobs")
    else:
        collect_p_box_dir = os.path.join(exp_root_dir,"collection_bbox_level",
                                        dataset_name,model_name,"other_baselines",
                                        "predicted_bbox_withprobs")
        
    p_json_path = os.path.join(collect_p_box_dir,
        f"epoch_{epoch}_predicted_bboxs.json"
    )
    p_json = read_json(p_json_path)
    main()