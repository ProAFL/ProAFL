'''
讨论下ours,在3个数据集上不同模型上的overlap情况
'''
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils.common import read_json,read_yaml

def get_miss_anno():
    annotation_withmiss_json = read_json(annotation_withmiss_json_path)
    images = annotation_withmiss_json["images"]
    imgId2imgname = {image["id"]:image["file_name"] for image in images}    
    annos = annotation_withmiss_json["annotations"]
    missannoid2info = {}
    for anno in annos:
        if int(anno["fault_type"]) != 4:
            continue
        imgname = imgId2imgname[anno["image_id"]]
        annoid = int(anno["id"])
        missannoid2info[annoid] = {
            "imgname":imgname,
            "anno":anno
        }
    return missannoid2info
        

def box_iou(box_a, box_b):
    """输入均为 xyxy；无交集或退化框返回 0。"""
    width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)

def main():
    miss_process_metric_json = read_json(miss_process_metric_json_path)
    missannoid2info = get_miss_anno()
    mingzhong_dict = {}
    tack_mingzhong_dict = {}
    for missannoid,info in missannoid2info.items():
        mingzhong_flag = False
        imgname = info["imgname"]
        tack_mingzhong_dict[imgname] = {}
        anno  = info["anno"]
        xmin,ymin,width,height = anno["bbox"]
        x1,y1,x2,y2 = xmin,ymin,xmin+width,ymin+height
        box_anno = [x1,y1,x2,y2]
        tracks = miss_process_metric_json["images"][imgname]["tracks"]
        last_epoches = 5
        last_epoch_tracks = []
        for track in tracks:
            bbox_list = track["bbox_list"][-last_epoches:]
            if all(x is None for x in bbox_list):
                # 这条track 后5个轮次没有bbox
                continue
            last_epoch_tracks.append(track)

        for track in last_epoch_tracks:
            track_id = track["track_id"]
            tack_mingzhong_dict[imgname][track_id] = False
            bbox_list = track["bbox_list"][-5:]
            for bbox in bbox_list:
                if bbox is None:
                    continue
                pbox = bbox
                iou = box_iou(box_anno,pbox)
                if iou >= iou_threshold:
                    mingzhong_flag = True # 如果当前track中存在一个bbox命中了这个missannoid则直接停止遍历track
                    break
            if mingzhong_flag is True:
                tack_mingzhong_dict[imgname][track_id] = True
                break # 如果当前track已经命中，则不需要遍历后面的track了。
        mingzhong_dict[missannoid] = mingzhong_flag
    recall_counter = 0
    for missannoid,isminghzhong in mingzhong_dict.items():
        if isminghzhong:
            recall_counter += 1
    recall = round(recall_counter / len(mingzhong_dict.keys()),4)

    precision_count = 0
    track_count =0 
    for imgname in tack_mingzhong_dict.keys():
        for trackid,ismingzhong in tack_mingzhong_dict[imgname].items():
            track_count += 1
            if ismingzhong:
                precision_count += 1

    precision = round((precision_count / track_count),4)
    print(f"iou_threshold:{iou_threshold},recall:{recall},precision:{precision}")


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "visdrone" # voc|kitti|visdrone
    model_name = "yolov7"
    fault_ratio = 0.1
    repeat_id = 1
    
    annotation_withmiss_json_path = root_dir/f"fault_inject/{fault_ratio}/{dataset_name}/coco_format/annotations_with_miss.json"
    miss_process_metric_json_path = root_dir/"dual_branch"/"process_metrics"/dataset_name/model_name/f"{fault_ratio}_repeat"/f"repeat_{repeat_id}"/"miss_process_metrics.json"

    for iou_threshold in [0.5,0.6,0.7,0.8,0.9]:
        main()