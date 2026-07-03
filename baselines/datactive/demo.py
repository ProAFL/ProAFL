import json
import os
import random

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
# tensorboard
from torch.utils.tensorboard import SummaryWriter

from TruncatedLoss import TruncatedLoss
from UTILS.demo_dataset import classificationDataSet, inference_classificationDataSet
from torchvision import transforms
from torch.utils.data import DataLoader


def train_model(mask_type='crop', class_num=9, img_root="./autodl-tmp/dataset/COCO",
                trainlabel_root='./autodl-tmp/dataset/COCO/casestudy_train.json',
                testlabel_root='./autodl-tmp/dataset/COCO/casestudy_test.json',
                model_save_path="./autodl-tmp/models/crop_model_epoch_{}.pth"):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # parameters

    tensorboardpath = "./casestudy/" + mask_type

    is_LNL = True
    ''
    #############

    # train dataset
    train_dataset = classificationDataSet(root=img_root, transforms=data_transform,
                                          txt_name="", mask_type=mask_type,
                                          dirty_path=trainlabel_root,
                                          datatype='COCO')

    # train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # test dataset
    test_dataset = classificationDataSet(root=img_root, transforms=data_transform,
                                         txt_name="", mask_type=mask_type,
                                         dirty_path=testlabel_root,
                                         datatype='COCO')

    # test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

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

            checkpoint = torch.load('./models/best.pth', map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.eval()
            for batch_idx, (inputs, targets, indexes) in enumerate(train_dataloader):
                print("\rrunning update_weight:{} / {}".format(batch_idx, len(train_dataloader)), end="")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                criterion.update_weight(outputs, targets, indexes)
            now = torch.load(model_save_path.format(epoch), map_location="cpu")
            model.load_state_dict(now['model'])
            model.train()

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
            "acc": 100 * correct  # / total
        }, model_save_path.format(epoch + 1))

    writer.close()


def inf_model(root_path='./autodl-tmp/dataset/COCO', mask_type='mask others',
              dirty_path='./autodl-tmp/dataset/COCO/casestudy_test.json',
              modelpath='./autodl-tmp/models/crop_model_epoch_13.pth',
              results_save_path='./autodl-tmp/mask_others_test_inf.json'):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = inference_classificationDataSet(root=root_path,
                                              transforms=data_transform,
                                              mask_type=mask_type,
                                              dirty_path=dirty_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    loss_func = torch.nn.CrossEntropyLoss()
    model_path = modelpath
    # load model
    modelState = torch.load(model_path, map_location="cpu")
    model = torchvision.models.resnet50()

    model.fc = torch.nn.Linear(2048, class_num)

    model.load_state_dict(modelState["model"])
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
            # print(loss)
            _, predicted = torch.max(outputs.data, 1)

            # progress bar
            print("\rInference: {}/{}".format(i + 1, len(dataloader)), end="")

            # save softmax outputs image_name category_id boxes
            for j in range(len(predicted)):
                content_dic = {
                    "image_name": targets["image_name"][j],
                    "full_scores": outputs[j].cpu().numpy().tolist(),
                    "detectiongt_category_id": int(targets["category_id"][j]),
                    "bbox": targets["boxes"][j].numpy().tolist(),
                    "fault_type": targets["fault_type"][j].item(),
                    "loss": loss
                }
                results.append(content_dic)
    json_str = json.dumps(results, indent=4)
    with open(results_save_path,
              'w') as json_file:
        json_file.write(json_str)


def detective(crop_path='./casestudydata/crop_test_inf.json',
              mask_others_path='./casestudydata/mask_others_test_inf.json',
              dirty_path='./dataset/COCO/casestudy_test.json'):
    with open(crop_path, 'r') as f:
        crop_list = json.load(f)
    with open(mask_others_path, 'r') as f:
        mask_others_list = json.load(f)
    with open(dirty_path, 'r') as f:
        dirty_list = json.load(f)

    imagename2boxes = {}
    for instance in dirty_list:
        if instance["labels"] != -1:
            if instance["image_name"] not in imagename2boxes:
                imagename2boxes[instance["image_name"]] = []
            imagename2boxes[instance["image_name"]].append([instance["boxes"], instance["labels"]])

    labelmap = {1: 'person', 2: 'car', 3: 'chair', 4: 'book', 5: 'bottle',
                6: 'cup', 7: 'dining table', 8: 'traffic light'}
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(len(crop_list)):
        scores = crop_list[i]['full_scores']
        label = crop_list[i]['detectiongt_category_id']

        loss = loss_func(torch.tensor([scores]), torch.tensor([label]))
        crop_list[i]['loss'] = loss.item()
    crop_list.extend(mask_others_list)
    results = sorted(crop_list, key=lambda x: x['loss'], reverse=True)

    # random.seed(2023)
    # random.shuffle(results) # 随机500

    falut_imagename2boxes = {}

    for i in range(500):
        imagename = results[i]['image_name']
        if imagename not in falut_imagename2boxes:
            falut_imagename2boxes[imagename] = []
        falut_imagename2boxes[imagename].append([results[i]['bbox'], results[i]['detectiongt_category_id']])

    for i, imagename in enumerate(falut_imagename2boxes):
        img_path = os.path.join('./dataset/COCO/val2017', imagename)
        img = Image.open(img_path).convert("RGB")
        ft_img = falut_imagename2boxes[imagename]
        have_missing = False
        vis_list = []
        for item in ft_img:
            fault_box = item[0]
            fault_label = item[1]
            if fault_label == 0:
                have_missing = True
                continue
            plt.gca().add_patch(
                plt.Rectangle((fault_box[0], fault_box[1]), fault_box[2] - fault_box[0], fault_box[3] - fault_box[1],
                              fill=False,
                              edgecolor='red',
                              linewidth=1))
            plt.gca().text(fault_box[0], fault_box[1] - 2, labelmap[fault_label],
                           fontsize=6, color='red')
            vis_list.append([int(fault_box[0]), int(fault_box[1]), int(fault_box[2]), int(fault_box[3])])

        for item in imagename2boxes[imagename]:
            box = item[0]
            label = item[1]
            box = [int(x) for x in box]
            if box in vis_list:
                continue
            plt.gca().add_patch(
                plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='green',
                              linewidth=1))  # xmin, ymin, w, h\
            plt.gca().text(box[0], box[1] - 2, labelmap[label],
                           fontsize=6, color='green')

        # save plt as image without x y axis
        plt.axis('off')
        plt.imshow(img)
        if have_missing:
            plt.savefig('./casestudydata/images2/missing_{}.png'.format(i), bbox_inches='tight', pad_inches=0, dpi=400)
        else:
            plt.savefig('./casestudydata/images2/{}.png'.format(i), bbox_inches='tight', pad_inches=0, dpi=400)

        plt.close()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./dataset/COCO', help='input dataset')
parser.add_argument('--trainlabel', default='./dataset/COCO/casestudy_train.json',
                    help='input trainlabel root path')
parser.add_argument('--testlabel', default='./dataset/COCO/casestudy_test.json', help='input testlabel path')
parser.add_argument('--classnum', default=9, help='input class num')
args = parser.parse_args()
img_root = args.dataset
train_label_path = args.trainlabel
test_label_path = args.testlabel
class_num = int(args.classnum)

train_model(mask_type='crop', class_num=class_num, img_root=img_root,
            trainlabel_root=train_label_path,
            testlabel_root=test_label_path,
            model_save_path="./models/crop_model_epoch_{}.pth")
train_model(mask_type='other objects', class_num=class_num, img_root=img_root,
            trainlabel_root=train_label_path,
            testlabel_root=test_label_path,
            model_save_path="./models/mask_others_model_epoch_{}.pth")
inf_model(root_path=img_root, mask_type='crop',
              dirty_path=test_label_path,
              modelpath='./models/crop_model_epoch_13.pth',
              results_save_path='./crop_test_inf.json')
inf_model(root_path=img_root, mask_type='mask others',
              dirty_path=test_label_path,
              modelpath='./models/mask_others_model_epoch_13.pth',
              results_save_path='./mask_others_test_inf.json')
detective(crop_path='./crop_test_inf.json',
              mask_others_path='./mask_others_test_inf.json',
              dirty_path='./dataset/COCO/casestudy_test.json')