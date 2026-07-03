import os
import json
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from exp_reproduct.inference_disassemble_dataset import Inference_classificationDataSet

def build_dataset(mask_type):
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )
    dataset = Inference_classificationDataSet(
        img_root, 
        annotation_path,
        mask_type = mask_type,
        transforms=data_transform)
    return dataset

def build_model():
    # load model
    modelState = torch.load(trained_model_path, map_location="cpu")
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, class_num)
    model.load_state_dict(modelState["model"])
    return model

def infer():
    dataset = build_dataset(mask_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    model = build_model()
    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    # 所有解构出的instance_img的推理结果
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, targets = data
            outputs = model(images.to(device))
            # softmax outputs
            labels = targets['category_id'].to(device)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # print(label,labels)
            loss = loss_func(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1) # predictd其实是预测的label 
            # progress bar
            print("\rInference: {}/{}".format(i + 1, len(dataloader)), end="")
            # save softmax outputs image_name category_id boxes
            for j in range(len(predicted)):
                content_dic = {
                    "image_name": targets["image_name"][j],
                    "anno_id":targets["anno_id"][j].item(),
                    "full_scores": outputs[j].cpu().numpy().tolist(), # imgs[j]的prob_list
                    "pred_category_id":predicted[j].item(),
                    "gt_category_id": int(targets["category_id"][j]),
                    "bbox": targets["boxes"][j].numpy().tolist(),
                    "loss": loss, # imgs[j]的loss,其实就是一个instance的loss，因为由于解构了数据集所以一个instance就是一个img
                    "fault_type":targets["fault_type"].item()
                }
                results.append(content_dic)
    json_str = json.dumps(results, indent=4)
    with open(results_save_path,'w') as json_file:
        json_file.write(json_str)
    print(f"结果保存在:{results_save_path}")

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    img_root = f"{exp_data_root}/datasets/{dataset_name}-coco/train"
    annotation_path = f"{exp_data_root}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json"
    if dataset_name == "VOC2012":
        class_num = 21
    elif dataset_name == "VisDrone":
        class_num = 11
    elif dataset_name == "KITTI":
        class_num = 10
    elif dataset_name == "KITTI_8":
        class_num = 9
    gpu_id = 0
    mask_type = "other_objects" # crop (裁剪模式,更加耗时些) and other_objects (背景推理模式)
    trained_model_path = f"{exp_data_root}/final_res/datactive/{dataset_name}/saved_models/new/{mask_type}/epoch_12.pt"
    results_save_path = f"{exp_data_root}/final_res/datactive/{dataset_name}/infer_results/new/{mask_type}.json"
    infer()


'''
train_model(mask_type='crop', class_num=class_num, img_root=img_root,
            trainlabel_root=train_label_path,
            testlabel_root=test_label_path,
            model_save_path="./models/crop_model_epoch_{}.pt")
train_model(mask_type='other objects', class_num=class_num, img_root=img_root,
            trainlabel_root=train_label_path,
            testlabel_root=test_label_path,
            model_save_path="./models/mask_others_model_epoch_{}.pt")
inf_model(root_path=img_root, mask_type='crop',
              dirty_path=test_label_path,
              modelpath='./models/crop_model_epoch_13.pt',
              results_save_path='./crop_test_inf.json')
inf_model(root_path=img_root, mask_type='mask others',
              dirty_path=test_label_path,
              modelpath='./models/mask_others_model_epoch_13.pt',
              results_save_path='./mask_others_test_inf.json')
detective(crop_path='./crop_test_inf.json',
              mask_others_path='./mask_others_test_inf.json',
              dirty_path='./dataset/COCO/casestudy_test.json')
'''