import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from UTILS import presets
from UTILS.mydataset import DetectionDataSet
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, \
    fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.ssd import SSDClassificationHead
from UTILS.engine import evaluate, train_one_epoch
from torch.utils.tensorboard import SummaryWriter

# parameters
model, optimizer, lr_scheduler, train_dataset, writer, epochs, modelsavepath = None, None, None, None, None, None, None
modeltype = 'ssd'  # 'ssd' or 'frcnn'
datatype = 'KITTI'  # 'VisDrone' or 'VOC' or 'COCO' or 'KITTI'
root_path, data_augmentation, layer_num = None, None, None
resume = None
batch_size = 4
lr = 0.002
traintype = 'dirty'
dirtypath = './data/fault_annotations/KITTItrain_mixedfault0.1.json'
# tensorboard
if modeltype == 'ssd':
    writer = SummaryWriter(log_dir='./' + modeltype + 'logs' + '/' + datatype + '_' + traintype,
                           comment="ssd_" + datatype)
elif modeltype == 'frcnn':
    writer = SummaryWriter(log_dir='./' + modeltype + 'logs' + '/' + datatype + '_' + traintype,
                           comment="frcnn_" + traintype + datatype)

if modeltype == 'ssd':
    epochs = 26
    data_augmentation = 'ssd'
    modelsavepath = "./autodl-tmp/models/ssd300" + traintype + "0.1_vgg16_" + datatype + "_epoch_{}.pth"

elif modeltype == 'frcnn':
    epochs = 26
    data_augmentation = 'hflip'
    modelsavepath = "./autodl-tmp/models/frcnn" + traintype + "0.1_resnet50_" + datatype + "_epoch_{}.pth"

if datatype == 'VOC':
    root_path = './autodl-tmp/dataset/VOCdevkit/VOC2012'
    layer_num = 21

elif datatype == 'VisDrone':
    root_path = './autodl-tmp/dataset'
    layer_num = 12

elif datatype == 'COCO':
    root_path = './autodl-tmp/dataset/COCO'
    layer_num = None # no need to set

elif datatype == 'KITTI':
    root_path = './autodl-tmp/dataset/KITTI'
    layer_num = 8

train_dataset = DetectionDataSet(root=root_path, runtype="train",
                                 transforms=presets.DetectionPresetTrain(data_augmentation=data_augmentation)
                                 , datatype=datatype, traintype='dirty',
                                 dirtypath=dirtypath)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=train_dataset.collate_fn)

val_dataset = DetectionDataSet(root=root_path, runtype="val" if datatype == 'VOC' else "test",
                               transforms=presets.DetectionPresetEval(),
                               datatype=datatype,
                               traintype='clean')

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=val_dataset.collate_fn)

if modeltype == 'ssd':
    # ssd model
    if datatype == 'COCO':# if coco dataset just load the pretrained model
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    else:
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

        model.head.classification_head = SSDClassificationHead([512, 1024, 512, 256, 256, 256],
                                                               model.anchor_generator.num_anchors_per_location(), layer_num)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 22], gamma=0.1)

elif modeltype == 'frcnn':
    if datatype =='COCO': # if coco dataset just load the pretrained model
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, layer_num)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 22], gamma=0.1)

device = torch.device('cuda')
model.to(device)
start_epoch = 0

if resume is not None:
    print('In resume training.')
    checkpoint = torch.load(resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"] + 1

for epoch in range(start_epoch, epochs):
    classloss, loss, boxloss, loss_classifier, loss_box_reg, loss_objectness = None, None, None, None, None, None
    if modeltype == 'ssd':
        metric_log, loss, classloss, boxloss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, 10,
                                                               modeltype)
    elif modeltype == 'frcnn':
        metric_log, loss, loss_classifier, loss_box_reg, loss_objectness = train_one_epoch(model, optimizer,
                                                                                           train_dataloader, device,
                                                                                           epoch, 10,
                                                                                           modeltype)
    lr_scheduler.step()
    with torch.no_grad():
        coco_evaluator = evaluate(model, val_dataloader, device=device)

    # checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "map": coco_evaluator.coco_eval["bbox"].stats[0]
    }, modelsavepath.format(epoch))

    # tensorboard
    writer.add_scalar("train/totalloss", loss, epoch)
    if modeltype == 'ssd':
        writer.add_scalar("train/classloss", classloss, epoch)
        writer.add_scalar("train/boxloss", boxloss, epoch)
    elif modeltype == 'frcnn':
        writer.add_scalar("train/loss_classifier", loss_classifier, epoch)
        writer.add_scalar("train/loss_box_reg", loss_box_reg, epoch)
        writer.add_scalar("train/loss_objectness", loss_objectness, epoch)

    # VOC map@0.5
    writer.add_scalar("val/map_0.5:0.95", coco_evaluator.coco_eval["bbox"].stats[1], epoch)

    # VOC map@  0.5:0.95
    writer.add_scalar("val/map", coco_evaluator.coco_eval["bbox"].stats[0], epoch)

writer.close()
# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
