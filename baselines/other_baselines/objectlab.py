import os
import joblib
import numpy as np
from collections import defaultdict
from PIL import Image
from helper.base_data_manager import get_collected_gt_box_json_path
from ours.small_utils import read_json,xcycwh_to_x1y1x2y2,calu_iou

def softmin(q):
    q = np.asarray(q, dtype=float)
    logits = 1.0 - q
    exp_logits = np.exp(logits - logits.max())
    weights = exp_logits / exp_logits.sum()
    return float(np.sum(weights * q))

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
                "id":pbox["predicted_box_id"],
                "cls":pbox["predicted_cls"],
                "x1y1x2y2":pbox["bbox"],
                "prob":pbox["prob"],
                "conf":pbox["conf"],
                "imgname":img_name
            })
            
    return imgname2pboxs


def objectlab_score(imgname,gboxs,pboxs):
    

    res = {}
    iou_map = {}
    for gbox in gboxs:
        for pbox in pboxs:
            key = f"{gbox['id']}#{pbox['id']}"
            iou = calu_iou(gbox["x1y1x2y2"],pbox["x1y1x2y2"])
            iou_map[key] = iou

    for gbox in gboxs:
        res[gbox['id']] = {}
        # loc fault rule
        same_cls_preds = [
            pbox for pbox in pboxs
            if pbox["cls"]  == gbox['cls'] and iou_map[f"{gbox['id']}#{pbox['id']}"] > 0
        ]
        if len(same_cls_preds) == 0:
            badloc_score = 1.0 # 越靠近1表示标注越可靠，越靠近0表示标注越可疑
        else:
            badloc_score = max(iou_map[f"{gbox['id']}#{pbox['id']}"] for pbox in same_cls_preds)

        
        res[gbox['id']]["badloc_score"] = badloc_score

        # cls fault rule
        diff_cls_high_conf_preds = [
            pbox for pbox in pboxs
            if pbox["cls"] != gbox['cls'] and iou_map[f"{gbox['id']}#{pbox['id']}"] > 0.95
        ]
        if len(diff_cls_high_conf_preds) == 0:
            badcls_score = 1.0 # 越大越可信
        else:
            badcls_score = 1.0 - max(iou_map[f"{gbox['id']}#{pbox['id']}"] for pbox in diff_cls_high_conf_preds)
        res[gbox['id']]["badcls_score"] = badcls_score

    # miss fault rule
    miss_scores = [] # img level
    sim_min = 1e-6
    for pbox in pboxs:
        if pbox['conf'] <= 0.65:
            continue
        same_cls_nonoverlap_labels = [
            gbox for gbox in gboxs
            if gbox["cls"] == pbox["cls"] and iou_map[f"{gbox['id']}#{pbox['id']}"] == 0
        ]
        if len(same_cls_nonoverlap_labels) == 0:
            miss_score = sim_min * (1.0 - pbox['conf'])
        else:
            miss_score = max(calu_iou(gbox["x1y1x2y2"], pbox["x1y1x2y2"]) for gbox in same_cls_nonoverlap_labels)

        miss_scores.append(miss_score)
    miss_score_img = softmin(miss_scores) if miss_scores else 1.0
    res[imgname] = {"miss_score":miss_score_img}
    return res




def main():
    imgname2gboxs = get_imgname2gboxs()
    imgname2pboxs = get_imgname2pboxs()
    print("gboxs and pboxs 重组完毕")
    res = {}
    for imgname in imgname2gboxs.keys():
        gboxs = imgname2gboxs[imgname]
        pboxs = imgname2pboxs[imgname]
        score_res = objectlab_score(imgname,gboxs,pboxs)
        for key,value in score_res.items():
            res[key] = value
    print("object lab打分完毕")
    # ranking
    _dict = {}
    for key in res.keys():
        if type(key) == int:
            # gid
            gid = key
            score = min(res[key]["badloc_score"],res[key]["badcls_score"])
            _dict[gid] = score
        else:
            imgname = key
            score = res[key]["miss_score"]
            _dict[imgname] = score
    rank = sorted(_dict, key=_dict.get) # gid/imgname rank, 值越小越可疑越靠前

    # 保存rank
    save_dir = os.path.join(exp_root_dir,"Results","other_baselines","objectlab",
                            dataset_name,model_name,f"exp_{exp_id}","rank")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "rank.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(rank,save_path)
    print(f"rank长度为:{len(rank)}")
    print(f'rank结果保存在:{save_path}')



if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    exp_id = "01"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7" # YOLOv7|FRCNN|rtdetr
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