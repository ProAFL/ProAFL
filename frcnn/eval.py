
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets import CocoDetectionDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import torch,torchvision
from engine import train_one_epoch, evaluate
from torchvision import models, transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import joblib

exp_data_root_dir = "/data/mml/data_debugging_data"
dataset_name = "KITTI" # VOC2012|VisDrone|KITTI
model_name = "FRCNN" # SSD|FRCNN
gpu_id = 0
num_epochs = 50
conf_threshold = 0.8
# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()
# Load training dataset
train_dataset = CocoDetectionDataset(
    image_dir=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train", 
    annotation_path=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json",
    transforms=get_transform()
)

# Load validation dataset
'''
val_dataset = CocoDetectionDataset(
    image_dir=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/val",
    annotation_path=f"{exp_data_root_dir}/datasets/{dataset_name}-coco/val/_annotations.coco.json",
    transforms=get_transform()
)
'''
 
# Load dataset with DataLoaders, you can change batch_size 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
# train_t_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def test():
    # class names
    label_list= ["","ball", "goalkeeper", "player", "referee",""]
    
    # Number of classes (include background)
    num_classes = 6   # this has to be 5 in normally, but because of some labeling issues in dataset this is 6.
    
    # Load the same model 
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    
    # Load trained Faster R-CNN model
    model.load_state_dict(torch.load("check_points/model_epoch_10.pth"))
    model.eval()
    
    # Load image with OpenCV and convert to RGB
    img_path = "football-players-detection/valid/2e57b9_1_6_png.rf.74724a3814311da25a648a48d778d589.jpg"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # detection data
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    """
    Higher threshold give you more accurate detections, 
    but number of predictions is reduced; there is a simple trade-off
    """
    conf_threshold = 0.8
    for i in range(len(boxes)):
        if scores[i] > conf_threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = label_list[labels[i]]
            score = scores[i].item()
            text = f"{label}: {score:.2f}"
            cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
            # Draw bbox and label
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    
    
    # Convert BGR --> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Show image with larger figure size
    plt.figure(figsize=(16, 12)) 
    plt.imshow(image_rgb)
    plt.axis('off')
    # plt.show()
    plt.savefig("test.png")

def build_ssd_model(num_classes):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.head.classification_head = SSDClassificationHead(
        [512, 1024, 512, 256, 256, 256],
        model.anchor_generator.num_anchors_per_location(), 
        num_classes
    )
    return model

def build_frcnn_model(num_classes):
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # 加载FRCNN模型（预训练）
    # +1 for bg class
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    if model_name == "SSD":
        model = build_ssd_model(num_classes)
    elif model_name == "FRCNN":
        model = build_frcnn_model(num_classes)
    else:
        raise Exception("模型名称错误")
    # Move the model to the GPU for faster training
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    # Loop through each epoch
    epoch_loss_value_list = []
    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        # 加载模型
        w_path = os.path.join(exp_data_root_dir,"models",f"{dataset_name}_error", model_name, f"epoch_{epoch}.pth")
        state_dict = torch.load(w_path,map_location="cpu")
        model.load_state_dict(state_dict)
        batch_loss_value_list = []
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            # torch.amp.autocast
            with torch.amp.autocast('cuda',enabled=False):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                batch_loss_value = losses.item()
                batch_loss_value_list.append(batch_loss_value)
        epoch_loss_value = round(sum(batch_loss_value_list) / len(batch_loss_value_list),4)
        print(f"Epoch:{epoch},loss:{epoch_loss_value}")
        epoch_loss_value_list.append(epoch_loss_value)
    save_dir = os.path.join(exp_data_root_dir,"check_train_effect",dataset_name,model_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_path = os.path.join(save_dir,"epoch_trian_loss_value_list.joblib")
    joblib.dump(epoch_loss_value_list,save_file_path)

if __name__ == "__main__":
    main()
    # test()