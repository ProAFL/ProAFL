from ultralytics import RTDETR
from rtdetr.custom_module.small_utils import read_yaml

model = RTDETR("rtdetr-l.pt")
config = read_yaml("config.yaml")

dataset_name = "voc"                     
label_mode = "error"              

results = model.train(data=f"data/{dataset_name}.yaml",epochs=100,imgsz=640,batch=32,device=[0,1],
                      save_period = 1,
                       project=f'{config["exp_data_dir"]}/{dataset_name}/rtdetr',
                      name=label_mode)
