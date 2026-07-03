import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageFilter


    
def xminyminwh_to_xyxy(xmin,ymin,width,height):
        xmax = xmin + width
        ymax = ymin + height
        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax += 1
        return [xmin,ymin,xmax,ymax]

class Inference_classificationDataSet(Dataset):
    def __init__(self, 
                img_root,
                annotation_path,
                mask_type,
                transforms=None):
        '''
        :param img_root: trainset imgs root dir
        :param annotation_path: coco style annotations.json
        :param mask_type: 
            other_objects|all_background: 把该图像中, 除了该obj的other obj给mask
                该模式下只推理bg instance
            crop: 把该图像中的该obj给裁剪出来
        :param transforms: toTenser,normalize
        '''
        self.img_root = img_root
        self.mask_type = mask_type # crop | other_objects | all_background
        self.coco = COCO(annotation_path) # 基于annotation json 得到 coco 对象
        self.catIds = self.coco.getCatIds() # 所有的分类id, 从0开始
        self.background_id = self.catIds[-1] + 1 # 设置bg catgory id
        ann_ids = self.coco.getAnnIds() # 获得所有的anno_ids
        annotations = self.coco.loadAnns(ann_ids) # 基于 anno_ids 得到 annos
        # 存放不同推理模式下的instance
        self.instances = []
        # bbox -> xmin,ymin,xmax,ymax
        for instance in annotations:
            xmin, ymin, width, height = instance["bbox"]
            instance["bbox"] = xminyminwh_to_xyxy(xmin, ymin, width, height)
        # 图像id => 其boxes
        self.imageid2boxes = defaultdict(list)
        
        # 遍历每个实例
        for instance in annotations:
            self.instances.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])
            
        print(f"INFO: {len(self.instances)} instances nomissing-loaded.")

        if self.mask_type == 'other_objects' or self.mask_type == 'all_background':
            # 背景推理模式
            bg_instances = [] # 用于存放对应图像name 的 bg instance
            bg_image_names = [] # 用于存放图像name
            for instance in self.instances:
                # 拿到该实例所属image path
                image_id = instance["image_id"]
                image_info = self.coco.loadImgs(image_id)[0] 
                image_name = image_info['file_name']
                item = {}
                item["image_id"] = image_id
                item["image_name"] = image_name
                item["image_size"] = [image_info["width"],image_info["height"]]
                item["anno_id"] = instance["id"]
                item["bbox"] = instance["bbox"] # xyxy
                item["label"] = self.background_id # 所有instance的label先都初始化为bg cls id
                item["area"] = self.caclu_area(instance["bbox"]) # xyxy -> area
                item["iscrowd"] = 0 # 都是不拥挤obj
                item["fault_type"] = instance["fault_type"]

                if item["image_name"] not in bg_image_names:
                    bg_instances.append(item)
                    bg_image_names.append(item["image_name"])

            self.instances = bg_instances #只需推理背景部分即可
            print(f"INFO: {len(self.instances)} instances only bkg-loaded.")
        elif self.mask_type == "crop": 
            # 裁剪推理模式
            temp_instances = []
            for instance in self.instances:
                # 拿到该实例所属image path
                image_id = instance["image_id"]
                image_info = self.coco.loadImgs(image_id)[0] 
                image_name = image_info['file_name']
                item = {}
                item["image_id"] = image_id
                item["image_name"] = image_name
                item["image_size"] = [image_info["width"],image_info["height"]]
                item["anno_id"] = instance["id"]
                item["bbox"] = instance["bbox"]
                item["label"] = instance["category_id"]
                item["image_id"] = instance["image_id"]
                item["area"] = self.caclu_area(instance["bbox"])
                item["fault_type"] = instance["fault_type"]
                item["iscrowd"] = 0
                temp_instances.append(item)
            self.instances = temp_instances

        self.transforms = transforms

    def caclu_area(self,bbox:list):
        xmin,ymin,xmax,ymax = bbox
        area = int((xmax-xmin) * (ymax - ymin))
        return area
    
    def gaussian_blur(self, img, box):
        '''把img box 处给模糊掉'''
        if box[2] - box[0] <= 0:
            box[2] = box[0] + 1
        if box[3] - box[1] <= 0:
            box[3] = box[1] + 1
        img_box = img.crop(box) # 把这个box挖出来
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20)) # 把这个box给模糊
        img.paste(img_box, box) # 粘回去
        return img
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        img_path = os.path.join(self.img_root, instance['image_name'])
        img = Image.open(img_path).convert("RGB")
        cur_instance_bbox = instance["bbox"]
        # 当前instance box中的box
        in_boxes_list = []
        label = instance["label"]
        img_need = None
        if label != self.background_id:
            # 正常instance,非背景推理模式
            img_need = img.crop(cur_instance_bbox)

        if self.mask_type == 'other_objects':
            # 背景模式推理
            for bbox in self.imageid2boxes[instance["image_id"]]:
                bbox = [int(i) for i in bbox]
                if bbox == cur_instance_bbox and label != self.background_id:
                    continue
                # Gaussian blur the box area of the image
                # if box belong to the part of boxes

                if bbox[0] > cur_instance_bbox[0] and bbox[1] > cur_instance_bbox[1] and bbox[2] < cur_instance_bbox[2] and bbox[3] < cur_instance_bbox[3]:
                    in_boxes_list.append(bbox)
                else:
                    # mask other obj
                    img = self.gaussian_blur(img, bbox)
            if label != self.background_id:
                img.paste(img_need, cur_instance_bbox)

            # 内部bbox也mask掉
            for bbox in in_boxes_list:
                img = self.gaussian_blur(img, bbox)
        elif self.mask_type == 'crop':
            img = img.crop(cur_instance_bbox)

        target = {}
        target["image_name"] = instance["image_name"]
        target["anno_id"] = instance["anno_id"]
        target["category_id"] = torch.tensor(instance["label"])
        target["boxes"] = torch.tensor(cur_instance_bbox)
        target["fault_type"] = instance["fault_type"]
        img = img.resize((224, 224))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.instances)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))