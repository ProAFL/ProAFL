import os
import json
import time
from ultralytics import RTDETR
from rtdetr.cost_time_util import get_cost_time
from rtdetr.custom_module.small_utils import read_yaml

def collect_one_epoch(model:RTDETR,imgs_dir):
    results = model.predict(imgs_dir, stream=True, batch=256, device=f"cuda:{gpu_id}",verbose=False)
          
    predicted_box_dict = {}
    predicted_box_id = 0
    for r in results:
        img_path = r.path
        img_basename = img_path.split("/")[-1]
        predicted_box_dict[img_basename] = {}
        predicted_bboxs = []
        xyxy_list = r.boxes.xyxy.tolist()
        clsid_list = r.boxes.cls.tolist()
        conf_list = r.boxes.conf.tolist()
        
        for xyxy,clsid,conf in zip(xyxy_list,clsid_list,conf_list):
            bbox = {}
            bbox["predicted_cls"] = int(clsid)
            bbox["bbox"] = xyxy
            bbox["conf"] = conf
            bbox["img_name"] = img_basename
            bbox["predicted_box_id"] = predicted_box_id
            predicted_box_id += 1
            predicted_bboxs.append(bbox)
        predicted_box_dict[img_basename]["predicted_bboxs"] = predicted_bboxs
    return predicted_box_dict

def main():
          
    for epoch in range(Epochs):
        print(f"Epoch:{epoch}/{Epochs}...")
        e_start_timestamp = time.time()
              
        model = RTDETR(os.path.join(models_dir,f"epoch{epoch}.pt"))
        predicted_box_dict = collect_one_epoch(model,imgs_dir)
        print(f"Inference image count:{len(predicted_box_dict.keys())}")
              
        save_dir = collect_p_box_dir
        os.makedirs(save_dir,exist_ok=True)
        save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
        save_json_path = os.path.join(save_dir,save_json_file_name)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(predicted_box_dict, f, indent=4)
        print(f"Data saved at:{save_json_path}")
        e_end_timestamp = time.time()
        e_cost_timestamp = e_end_timestamp - e_start_timestamp
        e_cost_time = get_cost_time(e_cost_timestamp)
        print(f"textElapsed time:{e_cost_time}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    dataset_name = "voc"                     
    model_name = "rtdetr"
    gpu_id = 1
    Epochs = 100
    imgs_dir = f"{exp_data_root}/datasets/{dataset_name}-yolo/origin/train/images"
    collect_p_box_dir = os.path.join(exp_data_root,"collection_process_info",dataset_name,model_name,"collected_predict_boxes")
    models_dir = f"{exp_data_root}/models/{dataset_name}/rtdetr/error/weights"
    main()