import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from exp_reproduct.disassemble_dataset import DisassembledDataSet
from TruncatedLoss import TruncatedLoss
import time
from torchvision.models import ResNet50_Weights
from datetime import datetime

from custom_module.small_utils import read_yaml
from custom_module.base_data_manager import get_annotations_with_miss_json_path,get_all_trainimgs_dir

def build_dataset(mask_type,class_num):
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )
    disassembled_dataset = DisassembledDataSet(
        img_root_dir, 
        annotation_path,
        class_num = class_num,
        mask_type = mask_type,
        transforms=data_transform)
    return disassembled_dataset

def build_ResNet50(class_num):
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(2048, class_num)
    return model

def build_criterion(is_LNL,train_dataset):
    if is_LNL:
        criterion = TruncatedLoss(trainset_size=len(train_dataset))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion

def train_one_epoch(epoch,model,train_dataloader,
                    optimizer,criterion,
                    is_LNL,
                    model_save_dir,
                    device):
    print("epoch: %d, lr: %f" % (epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_sum = 0
    if (epoch + 1) >= 3 and (epoch + 1) % 3 == 0 and is_LNL:
        best_checkpoint = torch.load(os.path.join(model_save_dir,"best.pt"), map_location="cpu")
        model.load_state_dict(best_checkpoint["model"])
        model.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(train_dataloader):
                                                                                                       
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            criterion.update_weight(outputs, targets, indexes)
        last_checkpoint = torch.load(os.path.join(model_save_dir,f"epoch_{epoch-1}.pt"), map_location="cpu")
        model.load_state_dict(last_checkpoint['model'])
        model.train()

    for i, (inputs, labels, indexes) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
                 
        outputs = model(inputs)
        if is_LNL:
            loss = criterion(outputs, labels, indexes)
        else:
            loss = criterion(outputs, labels)
        loss_sum += loss.item()
                  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                               
                                                                                                                      
                                                                                           
    loss_avg = round(loss_sum / len(train_dataloader),4)
    return loss_avg


def val_one_epoch(model,val_dataloader,device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels, indexes in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                               
                                                                              

    print("\rAccuracy of the val images: {} %".format(100 * correct / total))

    acc = 100 * correct / total
    return acc

def train():
    start_time = time.time()
    train_disassembled_dataset = build_dataset(mask_type,class_num)
    train_dataloader = DataLoader(train_disassembled_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(train_disassembled_dataset, batch_size=32, shuffle=False, num_workers=4)

    
    model = build_ResNet50(class_num)
    model.to(device)
    is_LNL = True
    criterion = build_criterion(is_LNL=is_LNL,train_dataset=train_disassembled_dataset)
    criterion.to(device)
               
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 11], gamma=0.1)

    best_acc = 0.0
           
    for epoch in range(epoches):
        loss_avg = train_one_epoch(epoch,model,train_dataloader,
                    optimizer,criterion,
                    is_LNL,
                    model_save_dir,
                    device)
        lr_scheduler.step()
        print(" | Loss_avg: {:.4}".format(loss_avg))
        
        val_acc = val_one_epoch(model,val_dataloader,device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss_avg,
                "acc": val_acc
            }, os.path.join(model_save_dir,"best.pt"))

        print("Now best acc: {} %".format(best_acc))
                         
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_avg,
            "acc": val_acc
        }, os.path.join(model_save_dir,f"epoch_{epoch}.pt"))
        end_time = time.time()
        elapsed_time = end_time - start_time                                     
        hours = int(elapsed_time // 3600)                   
        minutes = int((elapsed_time % 3600) // 60)                     
        seconds = elapsed_time % 60                               
        print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02.0f}")
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"text: {now_str}")
        


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    dataset_name = "voc"                     
    img_root_dir = f"{exp_data_root}/datasets/{dataset_name}-coco/train"
    annotation_path = f"{exp_data_root}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json"
    mask_type = "crop"                       
    if dataset_name == "voc":
        class_num = 21         
    elif dataset_name == "visdrone":
        class_num = 11         
    elif dataset_name == "kitti":
        class_num = 9        
    else:
        raise Exception("text")
    epoches = 13
    device = torch.device("cuda:0")
    model_save_dir = f"{exp_data_root}/baselines/datactive/{dataset_name}/rank/models/{mask_type}"
    os.makedirs(model_save_dir,exist_ok=True)
    train()

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

        
                                                                                  
                                            
                                                    
                             
                           
                       
                                