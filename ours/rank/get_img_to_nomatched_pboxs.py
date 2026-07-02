'''
保存图像在整个训练周期中所有没匹配到gBox的pBoxs
'''
import os
import json
from ours.base_data_manager import exp_data_root_dir,get_all_img_name,get_annotations_with_miss_json_path
from ours.data_organization_tools import get_all_miss_error_img_name_set
from ours.small_utils import read_json,save_json_file,add_path_value


def get_epoch_to_matched_p_boxs(gt_match_dict):
    # 每个epoch中所有被匹配上的p_box
    epoch_to_match_info = {}
    # 遍历所有的g_box
    for g_box_id in gt_match_dict.keys():
        # 当前g_box的匹配信息
        match_info_list = gt_match_dict[g_box_id]
        for match_info in match_info_list:
            epoch = match_info["epoch"]
            p_box = match_info["p_box"]
            p_box_id = p_box["predicted_box_id"]
            if epoch in epoch_to_match_info:
                epoch_to_match_info[epoch][p_box_id] = p_box
            else:
                epoch_to_match_info[epoch] = {p_box_id:p_box}
    return epoch_to_match_info

def get_img_name_to_epoch_to_unmatched_p_boxs(epoch_to_matched_p_ids:dict,
                                              last_epoch: int=5, conf_threshold: float=0.6):
    '''
    得到图像在后面几个epoch中未得到匹配的高置信度p_box
    参数：
    ---
    epoch_to_matched_p_ids : dict
        每个epoch下的被匹配的p_ids
    last_epoch : int, default=5
    conf_threshold: float, default=0.6

    返回：
    ---
    img_name_to_epoch_to_no_match_p_boxs : dict
        数据结构示例：
        {
            img_name:{
                epoch:[p_box1,p_box2],
                ...
            },
            ...
        }
    '''
    img_name_to_no_match_p = {}
    # 只关心最后5个epoch的预测情况
    for epoch in range(epochs-last_epoch,epochs):
        # 加载当前epoch的预测结果
        predicted_epoch_json_path = os.path.join(predicted_bboxs_dir, f"epoch_{epoch}_predicted_bboxs.json")
        with open(predicted_epoch_json_path,mode="r") as f:
            predicted_epoch_dict = json.load(f)
        # 统计所有图像中没被gt_box匹配到的高置信度预测box
        for img_name in sorted(predicted_epoch_dict.keys()):
            # img_name在该epoch下的所有预测框
            p_box_list = predicted_epoch_dict[img_name]["predicted_bboxs"]
            # 遍历预测框
            for p_box in p_box_list:
                p_id = p_box["predicted_box_id"]
                # 没被匹配的和conf大于一定阈值的pid
                if p_id not in epoch_to_matched_p_ids[epoch] and p_box["conf"] > conf_threshold:
                    add_path_value(img_name_to_no_match_p,keys=[img_name,epoch],value=p_box)
    return img_name_to_no_match_p


def get_img_to_no_matched_pboxs(all_img_name_list, gt_match_json:dict)->dict:
    last_epoch_nums = 20 # default:5
    conf_threshold = 0.5 # default:0.6
    epoch_to_matched_p_ids = get_epoch_to_matched_p_boxs(gt_match_json)
    # 获得每张图像在后面几个epoch中没被g_box匹配的高置信度p_box
    img_name_to_epoch_to_no_match_p_boxs = get_img_name_to_epoch_to_unmatched_p_boxs(
        epoch_to_matched_p_ids,last_epoch_nums,conf_threshold)
    # 划分出带有miss fault的img set和不带有miss fault的img set
    with_miss_fault_img_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)
    # 展平epoch key
    '''
    数据结构: img_to_p_boxs = {
        img_name:{
            "with_miss_fault_flag":0/1 # 0表示该img不含有miss fault
            "No_matched_p_box_list":[pbox,..]
        }
    }
    '''
    img_to_p_boxs = {}

    for img_name in all_img_name_list:
        img_to_p_boxs[img_name] = {}
        if img_name in with_miss_fault_img_set:
            img_to_p_boxs[img_name]["with_miss_fault_flag"] = 1
        else:
            img_to_p_boxs[img_name]["with_miss_fault_flag"] = 0

        img_to_p_boxs[img_name]["No_matched_p_box_list"] = []
        if img_name in img_name_to_epoch_to_no_match_p_boxs.keys():
            for epoch in img_name_to_epoch_to_no_match_p_boxs[img_name].keys():
                for p_box in img_name_to_epoch_to_no_match_p_boxs[img_name][epoch]:
                    p_box["epoch"] = epoch
                    img_to_p_boxs[img_name]["No_matched_p_box_list"].append(p_box)
    return img_to_p_boxs

def main():
    all_img_name_list = get_all_img_name(imgs_dir)
    gt_match_json = read_json(match_json_path)
    img_to_p_boxs = get_img_to_no_matched_pboxs(all_img_name_list, gt_match_json)

    save_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                            dataset_name,model_name)
    save_file_name = "img_to_nomatched_pboxs_temp.json"
    save_path = os.path.join(save_dir,save_file_name)
    save_json_file(img_to_p_boxs,save_path)
    print(f"json保存在:{save_path}")
    


if __name__ == "__main__":

    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7" # YOLOv7|FRCNN|SSD|rtdetr
    epochs = 50
    if model_name == "rtdetr":
        epochs = 100
    # 一定要是全量的trainset的imgsdir
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split", dataset_name,
                             "images", "origin")
    match_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                dataset_name,model_name,"gp_box_match", "match_v2.json")
    
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    predicted_bboxs_dir = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                    dataset_name,model_name,"collected_predicted_box", "v2")
    
    main()