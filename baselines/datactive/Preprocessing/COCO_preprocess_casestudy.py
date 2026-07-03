import json

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
# read COCO instances_val2017.json and transform it to
# {
#         "image_name": "xxx.jpg",
#         "image_size": [
#             1400,
#             788
#         ],
#         "boxes": [
#             x1,y1,x2,y2
#         ],
#         "labels": 1,
#         "image_id": 469,
#         "area": 24,
#         "iscrowd": 0,
# },
from UTILS.parameters import parameters

params = parameters()
falut_type = params.fault_type


def xywh2xyxy(boxes):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[0] + boxes[2]
    y2 = boxes[1] + boxes[3]
    return [x1, y1, x2, y2]


path = '../dataset/COCO/instances_val2017.json'

with open(path, 'r') as f:
    data = json.load(f)

labelmap = {1: 1, 3: 2, 62: 3, 84: 4, 44: 5, 47: 6, 67: 7, 10: 8}

instances = []
for image in tqdm(data['images']):
    image_name = image['file_name']
    image_size = [image['width'], image['height']]
    image_id = image['id']

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            instance = {
                'image_name': image_name,
                'image_size': image_size,
                'boxes': xywh2xyxy(annotation['bbox']),
                'labels': annotation['category_id'],
                'image_id': image_id,
                'area': annotation['area'],
                'iscrowd': annotation['iscrowd'],
                'fault_type': falut_type['no fault']
            }
            instances.append(instance)

#############case study###############

for instance in instances:
    if instance['labels'] not in [1, 3, 62, 84, 44, 47, 67, 10]:
        instance['labels'] = -1

for instance in instances:
    if instance['labels'] != -1:
        instance['labels']=labelmap[int(instance['labels'])]
#############


COCO_label_name = {'1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus',
                   '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant',
                   '13': 'stop sign',
                   '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse',
                   '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe',
                   '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee',
                   '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat',
                   '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket',
                   '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon',
                   '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli',
                   '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair',
                   '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet',
                   '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone',
                   '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book',
                   '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddy bear', '89': 'hair drier',
                   '90': 'toothbrush'}

# 统计每个类别的数量 并作柱状图
num = {}
for instance in instances:
    if instance['labels'] not in num:
        num[instance['labels']] = 1
    else:
        num[instance['labels']] += 1

plt.bar(num.keys(), num.values())
plt.show()
print('number of categories: ', len(num))
# num.keys() to list and sort
print('categories: ', sorted(list(num.keys())))

# get top 8 num.values() corresponding num.keys()
print(num)
print('top 8 categories: ', sorted(num.items(), key=lambda x: x[1], reverse=True)[:8])

# top 8 categories:  [(1, 11004), (3, 1932), (62, 1791), (84, 1161), (44, 1025), (47, 899), (67, 697), (10, 637)]

# transform instances to {imagename:[]} format dict

instances_dict = {}

for instance in instances:
    image_name = instance['image_name']
    if image_name not in instances_dict:
        instances_dict[image_name] = []
    instances_dict[image_name].append(instance)

# random split instances_dict to 2 equal parts and assert the number of categories in each part is 80

import random

random.seed(2023)

instances_dict_keys = list(instances_dict.keys())
random.shuffle(instances_dict_keys)

print('number of images: ', len(instances_dict_keys))

instances_dict_keys1 = instances_dict_keys[:len(instances_dict_keys) // 2]
instances_dict_keys2 = instances_dict_keys[len(instances_dict_keys) // 2:]

print('number of instances in instances_dict_keys1: ', len(instances_dict_keys1))
print('number of instances in instances_dict_keys2: ', len(instances_dict_keys2))

instances_dict1 = {}
instances_dict2 = {}

for key in instances_dict_keys1:
    instances_dict1[key] = instances_dict[key]

for key in instances_dict_keys2:
    instances_dict2[key] = instances_dict[key]

# assert the number of categories in each part is 80

num1 = {}
num2 = {}

print('number of instances_dict1: ', len(instances_dict1))
print('number of instances_dict2: ', len(instances_dict2))

for key in instances_dict1:
    for instance in instances_dict1[key]:
        if instance['labels'] not in num1:
            num1[instance['labels']] = 1
        else:
            num1[instance['labels']] += 1

for key in instances_dict2:
    for instance in instances_dict2[key]:
        if instance['labels'] not in num2:
            num2[instance['labels']] = 1
        else:
            num2[instance['labels']] += 1

print('number of categories in instances_dict1: ', len(num1))
print('number of categories in instances_dict2: ', len(num2))

# assert len(num1) == 80 and len(num2) == 80, 'number of categories is not 80'

# transform instances_dict to instances_list

instances_list1 = []
instances_list2 = []

for key in instances_dict1:
    for instance in instances_dict1[key]:
        instances_list1.append(instance)

for key in instances_dict2:
    for instance in instances_dict2[key]:
        instances_list2.append(instance)

# save instances_list to json file

print('number of instances in instances_list1: ', len(instances_list1))
print('number of instances in instances_list2: ', len(instances_list2))


with open('../dataset/COCO/casestudy_train.json', 'w') as f:
    json.dump(instances_list1, f, indent=4)

with open('../dataset/COCO/casestudy_test.json', 'w') as f:
    json.dump(instances_list2, f, indent=4)


# # plt some images with instances
#
#
#
# for key in instances_dict:
#     img = Image.open('../dataset/COCO/val2017/' + key)
#     for instance in instances_dict[key]:
#
#         boxes = instance['boxes']
#         plt.gca().add_patch(
#             plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
#                           linewidth=3))
#
#         class_name = str(instance['labels'])
#         plt.gca().text(boxes[0], boxes[1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5),
#                         fontsize=14, color='white')
#
#     plt.imshow(img)
#     plt.show()
