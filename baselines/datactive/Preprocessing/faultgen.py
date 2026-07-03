import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from UTILS import presets
from UTILS.mydataset import inferenceVOCDetectionDataSet
from UTILS.parameters import parameters
from torchvision.ops import boxes as box_ops

params = parameters()
falut_type = params.fault_type


def cal_IoU(X, Y):
    return box_ops.box_iou(torch.tensor([X]), torch.tensor([Y]))


def getinstances(dataset='VOC', datatype='val'):
    val_dataset = None
    val_dataloader = None
    result = []
    if dataset == 'VOC':
        val_dataset = inferenceVOCDetectionDataSet(voc_root="../dataset/VOCdevkit/VOC2012",
                                                   transforms=presets.DetectionPresetEval(),
                                                   txt_name=datatype + '.txt')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                     collate_fn=val_dataset.collate_fn)

        print('images num:', len(val_dataset))

        for images, targets in tqdm(val_dataloader, desc='getinstances'):
            # get image size
            images = list(image for image in images)
            image_size = [images[0].shape[2], images[0].shape[1]]  # it is converted to [width, height]

            # split into instances
            for i in range(len(targets[0]['labels'])):
                instance = {
                    'image_name': targets[0]['image_name'],
                    'image_size': image_size,
                    'boxes': targets[0]['boxes'][i].numpy().tolist(),
                    'labels': targets[0]['labels'][i].item(),
                    'image_id': targets[0]['image_id'][0].item(),
                    'area': targets[0]['area'][i].item(),
                    'iscrowd': targets[0]['iscrowd'][i].item(),
                    'fault_type': falut_type['no fault'],
                }
                result.append(instance)
        print('instances num:', len(result))

    if dataset == 'VisDrone' or dataset == 'KITTI':

        annotation_path, image_path = None, None
        if dataset == 'VisDrone':
            annotation_path = '../dataset/VisDrone2019-DET-' + datatype + '/annotations.json'
            image_path = '../dataset/VisDrone2019-DET-' + datatype + '/images/'
        if dataset == 'KITTI':
            annotation_path = '../dataset/KITTI/' + datatype + '_annotations.json'
            image_path = '../dataset/KITTI/training/image_2/'

        with open(annotation_path, 'r') as f:
            dataset = json.load(f)
        print('image num',len(dataset))
        for id, image in enumerate(dataset):
            # progress bar
            print('\r', 'getinstances: ', id, '/', len(dataset), end='')
            img = Image.open(image_path + image['image_name'])
            # [width, height]
            image_size = [img.size[0], img.size[1]]

            for i in range(len(image['categories'])):
                instance = {
                    'image_name': image['image_name'],
                    'image_size': image_size,
                    'boxes': image['boxes'][i],
                    'labels': image['categories'][i],
                    'image_id': id,
                    'area': (image['boxes'][i][2] - image['boxes'][i][0]) * (
                            image['boxes'][i][3] - image['boxes'][i][1]),
                    'iscrowd': 0,
                    'fault_type': falut_type['no fault'],
                }
                result.append(instance)
        print('instances num:', len(result))

    if dataset == 'COCO':
        with open('../dataset/COCO/instances_val2017_' + datatype + '.json', 'r') as f:
            result = json.load(f)

    return result, len(result)


def genmixedfault(dataset='VOC', datatype='val'):
    random.seed(2023)
    np.random.seed(2023)
    
    instances, num = getinstances(dataset, datatype)
    
    
    class_falut_index, location_falut_index, redundancy_falut_index, missing_falut_index = None, None, None, None
    if dataset == 'VOC' or dataset == 'COCO' or dataset == 'KITTI':
        class_falut_index = random.sample([i for i in range(num)], int(num * params.fault_ratio))
        # get location_falut_index except class_falut_index
        location_falut_index = random.sample([i for i in range(num) if i not in class_falut_index],
                                             int(num * params.fault_ratio))
        # get redundancy_falut_index except class_falut_index and location_falut_index
        redundancy_falut_index = random.sample(
            [i for i in range(num) if i not in class_falut_index and i not in location_falut_index],
            int(num * params.fault_ratio))
        # get missing_falut_index except class_falut_index and location_falut_index and redundancy_falut_index
        missing_falut_index = random.sample([i for i in range(num) if
                                             i not in class_falut_index and i not in location_falut_index and i not in redundancy_falut_index],
                                            int(num * params.fault_ratio))

    if dataset == 'VisDrone':  # improve the fault generation efficiency
        idx_set = {i for i in range(num)}
        class_falut_index = random.sample(list(idx_set), int(num * params.fault_ratio))
        print('class_falut_index finished with idx_set len {}, and class_falut_index len {}'.format(len(idx_set),
                                                                                                    len(class_falut_index)))
        idx_set = idx_set - set(class_falut_index)
        # get location_falut_index except class_falut_index
        location_falut_index = random.sample(list(idx_set),
                                             int(num * params.fault_ratio))

        print('location_falut_index finished with idx_set len {}, and location_falut_index len {}'.format(len(idx_set),
                                                                                                          len(location_falut_index)))
        idx_set = idx_set - set(location_falut_index)
        # get redundancy_falut_index except class_falut_index and location_falut_index
        redundancy_falut_index = random.sample(list(idx_set), int(num * params.fault_ratio))
        print('redundancy_falut_index finished with idx_set len {}, and redundancy_falut_index len {}'.format(
            len(idx_set),
            len(redundancy_falut_index)))
        idx_set = idx_set - set(redundancy_falut_index)
        # get missing_falut_index except class_falut_index and location_falut_index and redundancy_falut_index
        missing_falut_index = random.sample(list(idx_set), int(num * params.fault_ratio))

        print('missing_falut_index finished with idx_set len {}, and missing_falut_index len {}'.format(len(idx_set),
                                                                                                        len(missing_falut_index)))

    class_num, class_list = None, None
    if dataset == 'VOC':
        class_num = 21
    elif dataset == 'VisDrone':
        class_num = 12
    elif dataset == 'KITTI':
        class_num = 8

    elif dataset == 'COCO':
        class_list = params.coco_class

    for index in tqdm(class_falut_index, desc='genclassfault'):
        # random choose a class except background and the original class
        if dataset == 'COCO':
            instances[index]['labels'] = random.sample(class_list, 1)[0]
        else:
            instances[index]['labels'] = \
                random.sample([i for i in range(1, class_num) if i != instances[index]['labels']], 1)[0]
        instances[index]['fault_type'] = falut_type['class fault']

    for index in tqdm(location_falut_index, desc='genlocationfault'):
        # generate a random location with IoU in [0.1, 0.5] with the original location

        # get the original location
        ori_x1, ori_y1, ori_x2, ori_y2 = instances[index]['boxes']

        # get the image size
        image_size = instances[index]['image_size']

        # generate a random location while the IoU is in [0.1, 0.5]
        while True:
            new_x1 = random.randint(int(max(0, ori_x1 - (ori_x2 - ori_x1) / 2)), int((ori_x1 + ori_x2) / 2))
            new_y1 = random.randint(int(max(0, ori_y1 - (ori_y2 - ori_y1) / 2)), int((ori_y1 + ori_y2) / 2))
            new_x2 = random.randint(int((ori_x1 + ori_x2) / 2), int(min(image_size[0], ori_x2 + (ori_x2 - ori_x1) / 2)))
            new_y2 = random.randint(int((ori_y1 + ori_y2) / 2), int(min(image_size[1], ori_y2 + (ori_y2 - ori_y1) / 2)))

            # garantee the width and height are not equal to 0
            if new_x1 >= new_x2 or new_y1 >= new_y2:
                continue

            # calculate the IoU
            IoU = cal_IoU([ori_x1, ori_y1, ori_x2, ori_y2], [new_x1, new_y1, new_x2, new_y2]).item()
            if 0.1 <= IoU <= 0.5:
                break

        instances[index]['boxes'] = [new_x1, new_y1, new_x2, new_y2]
        instances[index]['fault_type'] = falut_type['location fault']

    for index in tqdm(missing_falut_index, desc='genmissingfault'):
        # delete the instance
        instances[index]['fault_type'] = falut_type['missing fault']

    for index in tqdm(redundancy_falut_index, desc='genredundancyfault'):
        # generate a new random instance and insert it into the instances list
        # get the image size
        image_size = instances[index]['image_size']

        # generate a random location
        new_x1, new_y1, new_x2, new_y2 = None, None, None, None
        while True:
            new_x1 = random.randint(0, image_size[0])
            new_y1 = random.randint(0, image_size[1])
            new_x2 = random.randint(new_x1, image_size[0])
            new_y2 = random.randint(new_y1, image_size[1])

            # garantee the width and height are not equal to 0
            if new_x1 >= new_x2 or new_y1 >= new_y2:
                continue
            break

        # generate a random class
        if dataset == 'COCO':
            new_class = random.sample(class_list, 1)[0]
        else:
            new_class = random.randint(1, class_num - 1)

        # generate a new instance
        new_instance = {
            'image_name': instances[index]['image_name'],
            'image_size': image_size,
            'boxes': [new_x1, new_y1, new_x2, new_y2],
            'labels': new_class,
            'image_id': instances[index]['image_id'],
            'area': (new_x2 - new_x1) * (new_y2 - new_y1),
            'iscrowd': 0,
            'fault_type': falut_type['redundancy fault'],
        }

        # insert the new instance into the instances list
        instances.insert(index, new_instance)

    print('fault instances num:',
          len(class_falut_index) + len(location_falut_index) + len(missing_falut_index) + len(redundancy_falut_index))

    print('fault ratio:', (len(class_falut_index) + len(location_falut_index) + len(missing_falut_index) + len(
        redundancy_falut_index)) / num)

    # json_str = json.dumps(instances, indent=4)
    # with open('../data/fault_annotations/' + dataset + datatype + '_mixedfault' + str(params.fault_ratio) + '.json',
    #           'w') as json_file:
    #     json_file.write(json_str)


if __name__ == '__main__':
    # genclassfault(dataset='VOC')
    # genlocationfault(dataset='VOC')
    # genredundancyfalut(dataset='VOC')
    # genmissingfault(dataset='VOC')

    genmixedfault(dataset='KITTI', datatype='test')
