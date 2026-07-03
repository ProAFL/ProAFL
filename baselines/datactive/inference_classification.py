import json
import time

import torch
import torchvision.models
from torch.utils.data import DataLoader

from UTILS.mydataset import inference_VOCGt_classificationDataSet, inference_VOCinf_classificationDataSet, \
    inference_VOCgtfault_classificationDataSet
from torchvision import transforms

from UTILS.parameters import parameters


def inference_VOCclassification(dataloader, inference_type, modelpath='', datatype='VOC'):
    loss_func = torch.nn.CrossEntropyLoss()
    model_path = modelpath

    # load model
    modelState = torch.load(model_path, map_location="cpu")
    model = torchvision.models.resnet50()
    if datatype == 'VOC':
        model.fc = torch.nn.Linear(2048, 21)
    elif datatype == 'VisDrone':
        model.fc = torch.nn.Linear(2048, 12)
    elif datatype == 'COCO':
        model.fc = torch.nn.Linear(2048, 91)
    elif datatype == 'KITTI':
        model.fc = torch.nn.Linear(2048, 8)
    model.load_state_dict(modelState["model"])
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = []
    start_time = time.time()
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
                if inference_type == "gt":
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectiongt_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                    }
                    results.append(content_dic)
                elif inference_type == "inf":
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectioninf_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                    }
                    results.append(content_dic)

                # inference_type == 'class fault' or 'location fault' or 'redundancy fault' or 'missing fault'
                else:
                    content_dic = {
                        "image_name": targets["image_name"][j],
                        "full_scores": outputs[j].cpu().numpy().tolist(),
                        "detectiongt_category_id": int(targets["category_id"][j]),
                        "bbox": targets["boxes"][j].numpy().tolist(),
                        "fault_type": targets["fault_type"][j].item(),
                        "loss": loss
                    }
                    results.append(content_dic)
    end_time = time.time()
    print("\nInference time: {:.4f}s".format(end_time - start_time))
    return results


params = parameters()
if __name__ == '__main__':
    # params
    inference_type = 'mixed fault'
    datatype = 'KITTI'
    modeltype = 'frcnn'
    mask_type = 'crop'
    runtype = 'train'
    faultratio = params.fault_ratio
    results_save_path = './data/classification_results/crop_dirty_LNL_classification_bs=32_' + datatype + runtype + 'mixedfault' + str(
        faultratio) + '_inferences.json'
    dirty_path = './data/fault_annotations/' + datatype + runtype + '_mixedfault0.1.json'
    modelpath = './autodl-tmp/models/crop_dirty_LNL_resnet50_kitti_epoch_13_bs=32.pth'

    #############################
    if datatype == 'VOC':
        root_path = './autodl-tmp/dataset/VOCdevkit/VOC2012'
    elif datatype == 'VisDrone':
        root_path = './autodl-tmp/dataset/VisDrone2019-DET-' + runtype
    elif datatype == 'COCO':
        root_path = './autodl-tmp/dataset/COCO'
    elif datatype == 'KITTI':
        root_path = './autodl-tmp/dataset/KITTI'
    print(root_path)
    # if mask_type == 'mask others':
    #     results_save_path = './data/classification_results/mask_others_classification_VOCgtmixedfault'+str(faultratio)+'_inferences.json'
    # elif mask_type == 'mask all':
    #     results_save_path = './data/classification_results/mask_all_classification_VOCgtmixedfault'+str(faultratio)+'_inferences.json'
    # elif mask_type == 'crop':
    #     results_save_path = './data/classification_results/crop_classification_VOCgtmixedfault'+str(faultratio)+'_inferences.json'
    #####################

    detection_results = {
        "ssd": "./data/detection_results/ssd_VOCval_inferences.json",
        "frcnn": "./data/detection_results/frcnn_VOCval_inferences.json",
    }
    clssification_results = {
        "ssd": './data/classification_results/classification_VOCssdinf' + str(params.m_t) + '_inferences.json',
        "frcnn": './data/classification_results/classification_VOCfrcnninf' + str(params.m_t) + '_inferences.json',
    }

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataloader = None

    if inference_type == 'gt':
        dataset = inference_VOCGt_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                        transforms=data_transform,
                                                        txt_name="val.txt")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'inf':
        dataset = inference_VOCinf_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                         inferences_root=detection_results[modeltype],
                                                         transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 

    elif inference_type == 'class fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="class fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'location fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="location fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'redundancy fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="redundancy fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    elif inference_type == 'missing fault':
        dataset = inference_VOCgtfault_classificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012",
                                                             fault_type="missing fault",
                                                             transforms=data_transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    elif inference_type == 'mixed fault':
        dataset = inference_VOCgtfault_classificationDataSet(root=root_path,
                                                             fault_type="mixed fault",
                                                             transforms=data_transform,
                                                             mask_type=mask_type,
                                                             datatype=datatype,
                                                             dirty_path=dirty_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if inference_type == 'gt':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgt_inferences.json', 'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'inf':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open(clssification_results[modeltype], 'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'class fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtclassfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'location fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtlocationfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'redundancy fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtredundancyfault_inferences.json',
                  'w') as json_file:
            json_file.write(json_str)

    elif inference_type == 'missing fault':
        results = inference_VOCclassification(dataloader, inference_type)
        json_str = json.dumps(results, indent=4)
        with open('./data/classification_results/classification_VOCgtmissingfault_inferences.json', 'w') as json_file:
            json_file.write(json_str)
    elif inference_type == 'mixed fault':
        results = inference_VOCclassification(dataloader, inference_type, modelpath=modelpath, datatype=datatype)
        json_str = json.dumps(results, indent=4)

        with open(results_save_path,
                  'w') as json_file:
            json_file.write(json_str)

    else:
        print("inference_type error")
