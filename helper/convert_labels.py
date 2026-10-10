
'''
text
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
                    
    coco_input = COCOObjectDetectionInput(input_file=coco_anno_json_path)
    yolo_output_path = yolo_output_dir.joinpath(Path("data.yaml"))
    yolo_output = YOLOv7ObjectDetectionOutput(
        output_file=yolo_output_path,
        output_split=tvt
    )
    yolo_output.save(label_input=coco_input)
    print(f"coco to yolo is saved in {yolo_output_dir}")

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
    print(f"Conversion from COCO to VOC completed successfully! XML text: {voc_output_dir}")


if __name__ == "__main__":

    inject_ratio = 0.1 # 0.01,0.05,0.1,0.15
    repeat_id = 1
    dataset_name = "visdrone" # voc|kitti|visdrone
    method_name = "ours"
    model_name = "yolov7"
    # coco_anno_json_path = Path(f"/data/mml/data_debugging_data/ProAFL_data/corrected_anno/{method_name}/{dataset_name}/{model_name}/{inject_ratio}_repeat/repeat_{repeat_id}/annotations_corrected.json")
    # yolo_output_dir = Path(f"/data/mml/data_debugging_data/ProAFL_data/corrected_anno/{method_name}/{dataset_name}/{model_name}/{inject_ratio}_repeat/repeat_{repeat_id}/yolo_format")
    coco_anno_json_path = Path("/data/mml/data_debugging_data/ProAFL_data/fault_inject/0.1/visdrone/coco_format/annotations_no_miss.json")
    yolo_output_dir = Path("/data/mml/data_debugging_data/ProAFL_data/fault_inject/0.1/visdrone/yolo_format")
    coco2yolo(coco_anno_json_path,yolo_output_dir,"train")
    