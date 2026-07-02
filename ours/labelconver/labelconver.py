
'''
coco格式数据标注转yolo格式
'''
import os
from pathlib import Path
from labelformat.formats import (YOLOv7ObjectDetectionInput, COCOObjectDetectionOutput, 
                                 COCOObjectDetectionInput, PascalVOCObjectDetectionOutput, 
                                 KittiObjectDetectionInput, YOLOv7ObjectDetectionOutput,
                                 )
def coco2yolo(coco_anno_json_path:Path,yolo_output_dir:Path,tvt:str):
    '''
    tvt:"train"|"test"|"val"
    '''
    # coco -> yolov7
    coco_input = COCOObjectDetectionInput(input_file=coco_anno_json_path)
    yolo_output_path = yolo_output_dir.joinpath(Path("data.yaml"))
    yolo_output = YOLOv7ObjectDetectionOutput(
        output_file=yolo_output_path,
        output_split=tvt
    )
    yolo_output.save(label_input=coco_input)
    print("Conversion from COCO to YOLOv7 completed successfully!")

def coco2voc(coco_anno_json_path:Path,voc_output_dir:Path):
    '''
    coco -> voc
    tvt:"train"|"test"|"val"
    '''
    coco_input = COCOObjectDetectionInput(input_file=coco_anno_json_path)
    voc_output = PascalVOCObjectDetectionOutput(
        output_folder=voc_output_dir
    )
    voc_output.save(label_input=coco_input)
    print(f"Conversion from COCO to VOC completed successfully! XML 已保存到: {voc_output_dir}")


if __name__ == "__main__":

    '''
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    dataset_mode = "correct"
    tvt = "val"
    
    if tvt == "train":
        coco_anno_json_path = Path(
            os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",tvt,f"_annotations.coco_{dataset_mode}.json")
        )
    elif tvt == "val":
        coco_anno_json_path = Path(
            os.path.join(exp_data_root,"datasets",f"{dataset_name}-coco",tvt,f"_annotations.coco.json")
        )
    if dataset_mode ==  "correct":
        dataset_mode = "clean"
    if tvt == "val":
        tvt = "test"
    voc_output_dir = Path(
        os.path.join(exp_data_root,"Results",dataset_mode,dataset_name,"labels","voc_format",tvt)
    )
    coco2voc(coco_anno_json_path,voc_output_dir)
    '''

    # Only YOLOv7
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    method_name = "objectlab" # ours|datactive|entropy|loss|deepgini|margin|objectlab|clod
    exp_id = "01"
    
    if method_name in ["entropy","loss","deepgini","margin","objectlab","clod"]:
        coco_anno_json_path = Path(
            os.path.join(
                exp_data_root,"Results","other_baselines",method_name,dataset_name,
                "YOLOv7",f"exp_{exp_id}","repair","_annotations.coco_repair.json"
            )
        )
        yolo_output_dir = Path(
            os.path.join(
                exp_data_root,"Results","other_baselines",method_name,dataset_name,
                "YOLOv7",f"exp_{exp_id}","repair","yolo_format"
            )
        )
    else:
        coco_anno_json_path = Path(
            os.path.join(
                exp_data_root,"Results",method_name,dataset_name,
                "YOLOv7",f"exp_{exp_id}","repair","_annotations.coco_repair.json"
            )
        )
        yolo_output_dir = Path(
            os.path.join(
                exp_data_root,"Results",method_name,dataset_name,
                "YOLOv7",f"exp_{exp_id}","repair","yolo_format"
            )
        )
    coco2yolo(coco_anno_json_path,yolo_output_dir,tvt="train")




# Load KITTI labels
'''
split = "val"
kitti_input = KittiObjectDetectionInput(
    input_folder=Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_kitti_format/{split}/labels"),
    category_names="Car,Van,Truck,Pedestrian,Person_sitting,Cyclist,Tram,Misc,DontCare", # 9 个 categories
    images_rel_path=f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_kitti_format/{split}/images"
)
coco_output = COCOObjectDetectionOutput(output_file = 
            Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco.json")
            )
coco_output.save(label_input=kitti_input)
print(f"Conversion from KITTI to COCO completed successfully! split:{split}.")


# COCO Convert to YOLOv7 and save
split = "val"
coco_input_path = Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco_noDonCare.json")
yolo_output_path = Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_yolo_format/{split}/data.yaml")

coco_input = COCOObjectDetectionInput(input_file=coco_input_path)
yolo_output = YOLOv7ObjectDetectionOutput(
    output_file=yolo_output_path,
    output_split=split
)
yolo_output.save(label_input=coco_input)
print(f"Conversion from COCO to YOLOv7 completed successfully! split:{split}.")
'''


'''
exp_data_root = "/data/mml/data_debugging_data"
dataset_name = "KITTI" # VOC2012|VisDrone|KITTI
# Initialize input and output classes
split_flag = "train" # train|val
correct_or_error = "error"

# yolo -> coco
yolo_input = YOLOv7ObjectDetectionInput(input_file = Path(f"{exp_data_root}/datasets/{dataset_name}-yolo/data.yaml"), input_split=split_flag)
coco_output = COCOObjectDetectionOutput(output_file = Path(f"{exp_data_root}/datasets/{dataset_name}-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
coco_output.save(label_input=yolo_input)
print(f"Conversion from YOLOv7 to COCO completed successfully! split:{split_flag}. isError:{correct_or_error}")
'''

'''
# coco -> xml
coco_input = COCOObjectDetectionInput(input_file=Path(f"{exp_data_root}/datasets/{dataset_name}-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
pascal_output = PascalVOCObjectDetectionOutput(output_folder=Path(f"{exp_data_root}/datasets/{dataset_name}-xml/datasets_error/train"))
pascal_output.save(label_input=coco_input)
print(f"Conversion from COCO to XML completed successfully! split:{split_flag}. isError:{correct_or_error}.")
'''

