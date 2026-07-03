'''
查看排序后的前top rate(top40%)涉及到的imgname set
'''
import os
import joblib
from helper.base_data_manager import (exp_data_root_dir,get_collected_gt_box_json_path,get_error_ann_file_path)
from helper.data_organization_tools import (conver_ours_rank,conver_datactive_rank,
                                          get_imgid_to_imgname,get_annoId_to_anno,
                                          )
from ours.small_utils import read_json
from pycocotools.coco import COCO



def cut(rank_list:list,cut_rate:float=0.4)->list:
    print(f"rank的原始长度:{len(rank_list)}")
    cut_point = int(len(rank_list) * cut_rate)
    cutted_rank = rank_list[:cut_point]
    print(f"cut前{int(cut_rate*100)}%的长度:{len(cutted_rank)}")
    return cutted_rank

def get_suspicious_imgName_set(cutted_rank_list:list,anno_error_json:dict)->set:
    imgId_to_imgName = get_imgid_to_imgname(anno_error_json)
    annoId_to_anno = get_annoId_to_anno(anno_error_json)
    suspicious_imgName_set = set()
    for idd in cutted_rank_list:
        if type(idd) is str:
            img_name = idd
        else:
            annoid = idd
            anno = annoId_to_anno[annoid]
            img_name =  imgId_to_imgName[anno["image_id"]]
        suspicious_imgName_set.add(img_name)
    print(f"总的图像数量: {len(anno_error_json['images'])}")
    print(f"可疑的图像数量: {len(suspicious_imgName_set)}")
    return suspicious_imgName_set


def main_ours():
    g_boxes_json = read_json(g_boxes_json_path)
    anno_error_json =  read_json(anno_error_json_path)
    converted_rank_res = conver_ours_rank(rank_res,g_boxes_json,anno_error_json)
    # 切出排序的前40%idd(imgname or annoid)
    cut_off_rank = cut(converted_rank_res)
    # 得到top序中指向的imgset
    suspicious_imgName_set = get_suspicious_imgName_set(cut_off_rank,anno_error_json)
    

def main_datactive():
    coco = COCO(anno_error_json_path)
    bg_catId = coco.getCatIds()[-1]+1
    converted_rank_res = conver_datactive_rank(rank_res,bg_catId)
    anno_error_json =  read_json(anno_error_json_path)
    cut_off_rank = cut(converted_rank_res)
    suspicious_imgName_set = get_suspicious_imgName_set(cut_off_rank,anno_error_json)

if __name__ == "__main__":
    dataset_name = "VisDrone"
    
    # ours
    model_name = "YOLOv7"
    anno_error_json_path  = get_error_ann_file_path(dataset_name)
    rank_res = joblib.load(os.path.join(exp_data_root_dir,"final_res","ours",dataset_name,model_name,
                                        "rank_res","alpha=1.5","rank_topsis.joblib"))
    g_boxes_json_path = get_collected_gt_box_json_path(dataset_name)
    main_ours()

    '''
    # datactive
    rank_res = joblib.load("/data/mml/data_debugging_data/final_res/datactive/VisDrone/ranked_result/ranked_list.joblib")
    anno_error_json_path  = "/data/mml/data_debugging_data/datasets/VisDrone-coco/train/_annotations.coco_error.json"
    main_datactive()
    '''