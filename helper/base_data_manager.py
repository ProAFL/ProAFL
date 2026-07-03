'''
Intermediate experiment data management
'''
import os
from ours.small_utils import read_yaml

config = read_yaml("config.yaml")
baselines = config["baselines"]
                           
exp_data_root_dir = config["exp_data_dir"]

                
fault_type_map = {
    'no_fault': 0,
    'cls_fault': 1,
    'loc_fault': 2,
    'redundancy_fault': 3,
    'missing_fault': 4,
}






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

def get_all_img_name(imgs_dir:str) -> list[str]:
    img_name_list = []
    for filename in sorted(os.listdir(imgs_dir)):
        filepath = os.path.join(imgs_dir, filename)
        if os.path.isfile(filepath):
            img_name_list.append(filename)
    return img_name_list


def get_annotations_with_miss_json_path(dataset_name):
    '''
    Get the faulty annotation JSON for this dataset, including miss faults.
    '''
    return os.path.join(exp_data_root_dir,"error",
                        dataset_name,"labels","coco_format","annotations_with_miss.json")

def get_error_ann_file_path(dataset_name):
    '''
    Get the injected-fault annotation JSON path for the train set (COCO style).
    '''
    ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",
                                 "_annotations.coco_error.json")
    return ann_file_path

def get_correct_ann_file_path(dataset_name,train_or_val):
    '''
    Get the clean annotation file.
    '''
    ann_file_path = ""
    if train_or_val == "val":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,
                                     "_annotations.coco.json")
    else:
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,
                                     "_annotations.coco_correct.json")
    return ann_file_path


def get_collected_gt_box_json_path(dataset_name):
    '''
    Get collected train-set bounding boxes, excluding miss faults because miss faults have no box to collect.
    gid format, using our own gid rather than the anno_id in the COCO annotation JSON.
    '''
    return os.path.join(exp_data_root_dir,"collection_process_info",dataset_name,"gt_bboxs.json")

def get_collected_predict_boxes_dir(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_process_info",
                                       dataset_name,model_name,"collected_predict_boxes")

def get_ours_gt_box_metric_path(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_process_info",dataset_name,model_name, "metrics.json")

def get_ours_match_path(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_process_info",dataset_name,model_name, "match.json")

def get_img_to_nomatched_pboxs_json_path(dataset_name,model_name):
    return os.path.join(exp_data_root_dir,"collection_process_info",dataset_name,model_name,"img_to_nomatched_pboxs.json")

def get_all_trainimgs_dir(dataset_name):
    return os.path.join(exp_data_root_dir,"datasets", f"{dataset_name}-yolo","train","images")

def get_rank_data_path(dataset_name,method_name,model_name=None):
    rank_data_path = None
    if method_name in baselines:
        if method_name == "datactive":
            rank_data_path = os.path.join(exp_data_root_dir,"baselines","datactive",dataset_name,"rank", "rank.joblib")
        else:
            assert model_name is not None, "model_name is None"
            rank_data_path = os.path.join(exp_data_root_dir,"baselines",method_name,dataset_name,model_name,"rank","rank.joblib")
    elif method_name == "ours":
        assert model_name is not None, "model_name is None"
        rank_data_path = os.path.join(exp_data_root_dir,"ours",dataset_name,model_name,"rank","rank.joblib")
    assert rank_data_path is not None, "Failed to get rank data path"
    return rank_data_path

def get_converted_rank_data_path(dataset_name,method_name,model_name=None):
    converted_rank_data_path = None
    if method_name in baselines:
        if method_name == "datactive":
            converted_rank_data_path = os.path.join(exp_data_root_dir,"baselines","datactive",dataset_name,"rank", "converted_rank.joblib")
        else:
            assert model_name is not None, "model_name is None"
            converted_rank_data_path = os.path.join(exp_data_root_dir,"baselines",method_name,dataset_name,model_name,"rank","converted_rank.joblib")
    elif method_name == "ours":
        assert model_name is not None, "model_name is None"
        converted_rank_data_path = os.path.join(exp_data_root_dir,"ours",method_name,dataset_name,model_name,"rank","converted_rank.joblib")
    assert converted_rank_data_path is not None, "Failed to get converted rank data path"
    return converted_rank_data_path

def get_nc_by_datasetname(dataset_name) -> int:
    '''
    Get the number of dataset classes.
    '''
    if dataset_name == "voc":
        return 20
    elif dataset_name == "kitti":
        return 8
    elif dataset_name == "visdrone":
        return 10

if __name__ == "__main__":
    pass

