'''
中间实验数据管理
'''
import os
from custom_module.small_utils import read_yaml
# 实验数据存放目录
config = read_yaml["config.yaml"]
exp_data_root_dir = config["exp_data_dir"]
# 注错类型map
fault_type_map = {
    'no_fault': 0,
    'cls_fault': 1,
    'loc_fault': 2,
    'redundancy_fault': 3,
    'missing_fault': 4,
}

def get_correct_ann_file_path(dataset_name,train_or_val):
    '''
    得到正确的anno file
    '''
    ann_file_path = ""
    if train_or_val == "val":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,
                                     "_annotations.coco.json")
    else:
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,
                                     "_annotations.coco_correct.json")
        
    return ann_file_path

def get_error_ann_file_path(dataset_name):
    '''
    得到数据集（trainset）注入错的anno json path (coco-style)
    '''
    ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",
                                 "_annotations.coco_error.json")
    return ann_file_path

def get_repair_ann_file_path(dataset_name,
                             method_name,
                             model_name:None):
    ann_file_path = ""    
    if method_name == "ours":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",f"_annotations.coco_repair_ours_{model_name}.json")
    if method_name == "datactive":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",f"_annotations.coco_repair_datactive.json")
    return ann_file_path

def get_error_train_model_weight_file_path(dataset_name,model_name,epoch):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",f"{dataset_name.lower()}", model_name.lower(), "error", "weights", f"epoch_{epoch}.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",f"{dataset_name.lower()}", model_name.lower(), "error", f"epoch_{epoch}.pth")
    return model_weight_file_path

def get_repair_train_model_weight_file_path(dataset_name,model_name, method_name):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(), f"repair_{method_name}", "last.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(), f"repair_{method_name}", "epoch_49.pth")
    return model_weight_file_path

def get_clean_train_model_weight_file_path(dataset_name,model_name):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(),"clean","weights","last.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(),"clean", "epoch_49.pth")
    return model_weight_file_path

def get_imgs_dir(dataset_name,train_or_val,style):
    imgs_dir = ""
    if style == "coco":
        imgs_dir = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val)
    elif style == "yolo":
        imgs_dir = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-yolo",train_or_val,"images")
    return imgs_dir

def get_all_img_name(imgs_dir:str) -> list[str]:
    img_name_list = []
    for filename in sorted(os.listdir(imgs_dir)):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isfile(filepath):
            img_name_list.append(filename)
    return img_name_list



def get_ours_rank_res_path(dataset_name,model_name,istopsis:bool):
    if istopsis:
        return os.path.join(exp_data_root_dir,"final_res","ours",dataset_name,model_name,"rank_res","rank_topsis.joblib")
    return os.path.join(exp_data_root_dir,"final_res","ours",dataset_name,model_name,"rank_res","rank.joblib")

def get_datactive_rank_res_path(dataset_name):
    '''
    获得datactive的排序结果(instance list)
    '''
    return os.path.join(exp_data_root_dir, "final_res","datactive",dataset_name,"ranked_result","ranked_list.joblib")

def get_collected_gt_box_json_path(dataset_name):
    '''
    得到收集上来的数据集trainset的bboxs(不含miss falut,因为miss是无法收集到bbox的)
    '''
    return os.path.join(exp_data_root_dir,"collection_bbox_level",dataset_name,"gt_bboxs.json")

def get_annotations_with_miss_json_path(dataset_name):
    '''
    获得该数据集注错的anno json, 带有miss fault.
    '''
    return os.path.join(exp_data_root_dir,"error_anno",
                        dataset_name,"coco_format","annotations_with_miss.json")


def get_ours_gt_box_metric_path(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name, "collection_metric", "collection_metrics_v2.json")

def get_ours_match_path(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name, "gp_box_match", "match_v2.json")

def get_nc_by_datasetname(dataset_name) -> int:
    '''
    得到数据集的类别总数
    '''
    if dataset_name == "VOC2012":
        return 20
    elif dataset_name == "KITTI_8":
        return 8
    elif dataset_name == "VisDrone":
        return 10

if __name__ == "__main__":
    pass

