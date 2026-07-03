import os
import json
import yaml
from datetime import datetime

def calu_iou(gt_bbox,predicted_bbox):
    x1_min, y1_min, x1_max, y1_max = gt_bbox
    x2_min, y2_min, x2_max, y2_max = predicted_bbox

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def xcycwh_to_x1y1x2y2(bbox,W,H):
    xc = bbox[0]
    yc = bbox[1]
    w = bbox[2]
    h = bbox[3]

                            
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

                                          
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

                              
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]

def get_all_files(directory)->list[str]:
    files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files

def get_nc(dataset_name)->int:
    if dataset_name == "VOC2012":
        nc = 20
    elif dataset_name == "KITTI_8":
        nc = 8
    elif dataset_name == "KITTI":
        nc = 9
    elif dataset_name == "VisDrone":
        nc = 10
    else:
        raise Exception("Invalid dataset parameters")
    return nc

def read_json(json_path:str):
    _json = None
    with open(json_path, "r") as f:
        _json = json.load(f)
    return _json

def read_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return 

def save_json_file(data, file_path):
    """
    Save JSON data to a file
    
    Args:
        data (dict): JSON data to save
        file_path (str): Target file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_formatted_time():
    """Return the current time as a formatted string (YYYY-MM-DD_HH:MM:SS)"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")

def is_directory_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def add_path_value(d:dict, keys:list, value):
    '''
    Nested dictionary whose final value is [].
    '''
    cur = d
                                     
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur.setdefault(keys[-1], []).append(value)

def get_cost_time(cost_timetamp)->str:
    hours = int(cost_timetamp // 3600)                   
    minutes = int((cost_timetamp % 3600) // 60)                     
    seconds = cost_timetamp % 60                               
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"
