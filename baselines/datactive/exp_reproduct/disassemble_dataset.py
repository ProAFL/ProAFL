import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from collections import defaultdict
'''
数据集解构
'''
class DisassembledDataSet(Dataset):
    def __init__(self, 
                img_root_dir, 
                annotation_path,
                class_num,
                mask_type,
                transforms=None):  # run_type 仅用来指定是测试集还是训练集
        assert mask_type in ['crop', 'other_objects', 'all_backgrounds'], "mask_type must be in ['other objects', 'all backgrounds', 'crop']"
        
        self.img_root_dir = img_root_dir
        self.mask_type = mask_type
        self.transforms = transforms
        self.coco = COCO(annotation_path)
        self.catIds = self.coco.getCatIds()
        self.background_id = self.catIds[-1]+1
        # 获取所有 annotation 的 ID
        ann_ids = self.coco.getAnnIds()
        # 根据 ID 载入所有 annotation
        annotations = self.coco.loadAnns(ann_ids)
        for instance in annotations:
            xmin, ymin, width, height = instance["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            instance["bbox"] = [int(xmin),int(ymin),int(xmax),int(ymax)]
            # 保证box是个有效的整数矩形
            if instance["bbox"][0] == instance["bbox"][2]:
                instance["bbox"][2] += 1
            if instance["bbox"][1] == instance["bbox"][3]:
                instance["bbox"][3] += 1
        '''
        将label!=-1的实例收集到self.instances_list中
        构建一个imgid2boxes的字典：{imgid:[box1,box2]}
        '''
        # 用于存储实例的list
        self.instances_list = []
        # 一个图像映射到其boxes
        self.imageid2boxes = defaultdict(list)
        # 遍历每个实例
        for instance in annotations:
            self.instances_list.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])

        len_before = len(self.instances_list)

        # 判断一下数据集结构类型
        if self.mask_type == 'all_backgrounds' or self.mask_type == 'other_objects':
            '''
            从self.instances_list选择一些实例将其改为背景实例，并添加到self.instances_list中
            '''
            print('INFO: all_backgrounds or other objects')
            # 随机选择一些obj实例作为背景
            self.background_instances_list = random.sample(self.instances_list,
                                                           int(len(self.instances_list) / class_num))
            # 遍历这些作为背景的obj实例
            for instance in self.background_instances_list:
                # 拷贝一个新的obj实例出来
                new_instance = {key: value for key, value in instance.items()}
                # 把obj实例的label改为 0（背景类别）
                new_instance["category_id"] = self.background_id # bg instance
                # 把这个背景obj加入到self.instances_list
                self.instances_list.append(new_instance)

            assert len(self.instances_list) == len_before + len(self.background_instances_list)
        else:
            # 不需要背景实例
            self.background_instances_list = []
            assert len(self.instances_list) == len_before

        print("INFO: {} instances loaded. including {} instances and {} background instances".format(
            len(self.instances_list), len_before, len(self.background_instances_list)))

    def gaussian_blur(self, img, box):
        '''
        将img中的box中的obj进行模糊
        '''
        # 裁剪
        img_box = img.crop(box)
        # 模糊
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
        # 粘回
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):
        # 拿到一个实例
        instance = self.instances_list[idx]
        # 拿到该实例所属image path
        image_id = instance["image_id"]
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_root_dir, image_info['file_name'])

        img, label = None, None
        # 判断一下解构类型
        # 如果是模糊其他obj
        if self.mask_type == 'other_objects':
            # 拿到instance 所在图像
            img = Image.open(img_path).convert("RGB")
            cur_instance_bbox = instance["bbox"]
            label = instance["category_id"]
            # 用于存在该实例内部的实例的box
            in_boxes_list = []
            img_need = None

            '''
            除了instance其他都模糊掉：
            注意instance 内部 obj的模糊
            '''
            if label != self.background_id: # 非背景obj,裁出
                img_need = img.crop(cur_instance_bbox) # 裁剪出一个obj
            
            '''模糊掉不是内部box的其他实例'''
            # 遍历该图像的所有bbox
            for bbox in self.imageid2boxes[instance["image_id"]]:
                if bbox == cur_instance_bbox and label != self.background_id:
                    # 如果遍历到了自己（instance,obj,target）且自己不是背景，跳过该box，准备直接处理下个box
                    continue
                if bbox[0] > cur_instance_bbox[0] and bbox[1] > cur_instance_bbox[1] and bbox[2] < cur_instance_bbox[2] and bbox[3] < cur_instance_bbox[3]:
                    # 当前bbox是当前instance的子bbox,把这个box坐标存入in_boxes_list
                    in_boxes_list.append(bbox)
                else:
                    # 如果bbox不是内部，把它模糊掉
                    img = self.gaussian_blur(img, bbox) # img已经是将instance裁出了

            if label != self.background_id:
                # 该instance 不是 bg instance, 把它贴回到图像中
                img.paste(img_need, cur_instance_bbox)

            '''模糊掉内部其他实例'''
            for bbox in in_boxes_list: # 把它内部box模糊掉
                img = self.gaussian_blur(img, bbox)

        # 如果结构方式是裁出，直接裁出并且resize就可以了。
        elif self.mask_type == 'crop':
            img = Image.open(img_path).convert("RGB")
            cur_instance_bbox = instance['bbox']
            label = instance["category_id"]
            # Crop out the boxes part of the image
            img = img.crop(cur_instance_bbox)

        img = img.resize((224, 224))
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(label)
        return img, label, idx # idx 其实是实例idx
    
    def __len__(self):
            return len(self.instances_list)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    

# if idx == 3952:
# print(instance)
# for box in self.imageid2boxes[instance["image_name"]]:
#     if box == instance["boxes"]:
#         continue
#     # box = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
#     plt.gca().add_patch(
#         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
#                       linewidth=3))  # xmin, ymin, w, h
# plt.gca().add_patch(
#     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='green',
#                   linewidth=3))
# #
# plt.imshow(img)
# plt.show()





    