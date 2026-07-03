
'''
数据标注格式转换
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
    pass