import torch
import torchvision
# tensorboard
from torch.utils.tensorboard import SummaryWriter

from TruncatedLoss import TruncatedLoss
from UTILS.mydataset import classificationDataSet
from torchvision import transforms
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# parameters

mask_type = 'crop'

# 准备保存模型的路径使用的分类模型是ResNet50
model_save_path = "./autodl-tmp/models/crop_dirty_LNL_resnet50_kitti_epoch_{}_bs=32.pth"

# tensorboard路径
tensorboardpath = "./clslogs/kitti_LNL_crop_dirty_lr=0.001_bs=32"

# 损失函数选择
is_LNL = True
''
# 数据集类别数
class_num = 8  # 8 for KITTI, 11 for VisDrone

#############

# 构建分类数据集
# train dataset
train_dataset = classificationDataSet(root="./autodl-tmp/dataset/KITTI", transforms=data_transform,
                                      txt_name="", mask_type=mask_type, train_type='dirty',
                                      dirty_path='./data/fault_annotations/KITTItrain_mixedfault0.1.json',
                                      datatype='KITTI')

# train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# test dataset
test_dataset = classificationDataSet(root="./autodl-tmp/dataset/KITTI", transforms=data_transform,
                                     txt_name="", mask_type=mask_type, train_type='clean', datatype='KITTI',
                                     run_type='test')

# test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# gpu设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ResNet50
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, class_num)
model.to(device)

# loss function
# loss_func = torch.nn.CrossEntropyLoss()
if is_LNL:
    criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()
else:
    criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if is_LNL:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 11], gamma=0.1)
    epoches = 13
else:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 11], gamma=0.1)
    epoches = 13

# resume
# checkpoint = torch.load('./models/resnet50_voc_epoch_10.pth', map_location="cpu")
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# epoch = checkpoint["epoch"]
# loss = checkpoint["loss"]
# acc=checkpoint["acc"]
# print("checkpoint acc = ",acc)


# tensorboard
writer = SummaryWriter(log_dir=tensorboardpath, comment="resnet50_voc")
best_acc = 0.0
# train
for epoch in range(epoches):
    print("epoch: %d, lr: %f" % (epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_sum = 0

    if (epoch + 1) >= 3 and (epoch + 1) % 3 == 0 and is_LNL:
        # epoch = 2开始并且后续每3轮并且是LNL
        # 加载出当前最好model
        checkpoint = torch.load('./models/best.pth', map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        # 准备评估
        model.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(train_dataloader):
            print("\rrunning update_weight:{} / {}".format(batch_idx, len(train_dataloader)), end="")
            inputs, targets = inputs.to(device), targets.to(device)
            # 推理一遍训练集
            outputs = model(inputs)
            # 损失函数更新权重
            criterion.update_weight(outputs, targets, indexes)
        # 加载当前epoch模型
        now = torch.load(model_save_path.format(epoch), map_location="cpu")
        model.load_state_dict(now['model'])
        model.train()
    # 继续训练
    for i, (inputs, labels, indexes) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        outputs = model(inputs)

        if is_LNL:
            loss = criterion(outputs, labels, indexes)
        else:
            loss = criterion(outputs, labels)
        loss_sum += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training progress bar
        print("\rEpoch: {}/{} | Step: {}/{} | Loss: {:.4f}".format(epoch + 1, epoches, i + 1, len(train_dataloader),
                                                                   loss.item()), end="")
    # lr 更新步
    lr_scheduler.step()

    # tensorboard epoch loss
    writer.add_scalar('Train/Loss', loss_sum / len(train_dataloader), epoch)

    # loss average
    loss_avg = loss_sum / len(train_dataloader)
    print(" | Loss_avg: {:.4f}".format(loss_avg))

    # test
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels, indexes in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # test progress bar
            print("\rTest: {}/{}".format(total, len(test_dataloader)), end="")

    print("Accuracy of the test images: {} %".format(100 * correct / total))

    acc = 100 * correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_avg,
            "acc": 100 * correct / total
        }, './models/best.pth')

    print("Now best acc: {} %".format(best_acc))
    # tensorboard epoch acc
    writer.add_scalar('Test/Acc', 100 * correct / total, epoch)

    # save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss_avg,
        "acc": 100 * correct #/ total
    }, model_save_path.format(epoch + 1))

writer.close()
