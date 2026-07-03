import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from lxml import etree

import torch.utils.data as data
from matplotlib import pyplot as plt
from UTILS.parameters import parameters

params = parameters()
fault_type_dict = parameters().fault_type


class classificationDataSet(data.Dataset):
    def __init__(self, root, transforms=None, txt_name: str = "train.txt", mask_type='other objects', dirty_path='',
                 datatype: str = 'VOC', run_type='test'):                                                                       
        assert mask_type in ['other objects', 'all backgrounds',
                             'crop'], "mask_type must be in ['other objects', 'all backgrounds', 'crop']"
        self.dirtypath = dirty_path
        self.root = root
        self.mask_type = mask_type
        self.transforms = transforms
        self.datatype = datatype

        self.img_root = None
        class_num = None
        if datatype == 'VOC':
            class_num = 20
            self.img_root = os.path.join(self.root, "JPEGImages")
        elif datatype == 'VisDrone':
            class_num = 11
            self.img_root = os.path.join(self.root, "images")
        elif datatype == 'COCO':
            class_num = 8
            self.img_root = os.path.join(self.root, "val2017")
        elif datatype == 'KITTI':
            class_num = 8
            self.img_root = os.path.join(self.root, "training/image_2")

        with open(self.dirtypath, 'r') as f:
            self.dirtylist = json.load(f)
        if datatype == 'COCO' or datatype == 'KITTI':
                               
            for instance in self.dirtylist:
                instance["boxes"] = [int(i) for i in instance["boxes"]]
                if instance["boxes"][0] == instance["boxes"][2]:
                    instance["boxes"][2] += 1
                if instance["boxes"][1] == instance["boxes"][3]:
                    instance["boxes"][3] += 1

        self.instances_list = []
        self.imageid2boxes = {}
        for target in self.dirtylist:
            if int(target['labels']) != -1:
                self.instances_list.append(target)
            if target["image_name"] not in self.imageid2boxes:
                self.imageid2boxes[target["image_name"]] = []
            self.imageid2boxes[target["image_name"]].append(target["boxes"])
        len_before = len(self.instances_list)

        if self.mask_type == 'all backgrounds' or self.mask_type == 'other objects':
            print('INFO: all backgrounds or other objects')
                                          
            self.background_instances_list = random.sample(self.instances_list,
                                                           int(len(self.instances_list) / class_num))

            for instance in self.background_instances_list:
                new_instance = {key: value for key, value in instance.items()}
                new_instance["labels"] = 0              
                self.instances_list.append(new_instance)

            assert len(self.instances_list) == len_before + len(self.background_instances_list)
        else:
            self.background_instances_list = []
            assert len(self.instances_list) == len(self.instances_list)

        print("INFO: {} instances loaded. including {} instances and {} background instances".format(
            len(self.instances_list), len_before, len(self.background_instances_list)))

    def gaussian_blur(self, img, box):

        img_box = img.crop(box)
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):

        instance = self.instances_list[idx]

        img, label = None, None
        if self.mask_type == 'other objects':
                               
                img_path = os.path.join(self.img_root, instance["image_name"])
                img = Image.open(img_path).convert("RGB")
                boxes = [int(float(instance["boxes"][0])), int(float(instance["boxes"][1])),
                         int(float(instance["boxes"][2])), int(float(instance["boxes"][3]))]
                label = instance["labels"]
                in_boxes_list = []
                img_need = None

                if label != 0:
                    img_need = img.crop(boxes)                 

                for box in self.imageid2boxes[instance["image_name"]]:
                                                                           
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

                    if box == boxes and label != 0:
                                                                                                          
                        continue
                    if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                                   
                        in_boxes_list.append(box)
                    else:
                                          
                        img = self.gaussian_blur(img, box)

                if label != 0:
                                                                                         
                    img.paste(img_need, boxes)

                for box in in_boxes_list:                       
                    img = self.gaussian_blur(img, box)


        elif self.mask_type == 'crop':
            img_path = os.path.join(self.img_root, instance["image_name"])
            img = Image.open(img_path).convert("RGB")
            boxes = instance['boxes']
            label = instance["labels"]


                                                  
            img = img.crop(boxes)
             


                         
                         
                                                                
                                          
                          
                                                                                              
                                  
                                                                                                                
                                                                 
                              
                                                                                                                          
                                         
           
                         
                    
         
        img = img.resize((224, 224))
                         
                    

                                 
         
                      
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(label)

        return img, label, idx                                

    def __len__(self):
        return len(self.instances_list)

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inference_classificationDataSet(data.Dataset):
    def __init__(self, root,  transforms=None, mask_type='mask others',
                 dirty_path='../data/fault_annotations/VOCval_mixedfault0.1.json'):
        self.root = root
        path = "val2017"
        self.img_root = os.path.join(self.root, path)
        self.mask_type = mask_type
                                  
        fault_gt = json.load(open(dirty_path, "r"))
        self.fault_gt_instances = []
                                                         
        print(f"INFO: {len(fault_gt)} instances pre-loaded.")

        self.imagename2boxes = {}
        params = parameters()
        fault_type_dict = params.fault_type

        for instance in fault_gt:
            if instance["labels"] != -1:               
                self.fault_gt_instances.append(instance)
            if instance["image_name"] not in self.imagename2boxes:
                self.imagename2boxes[instance["image_name"]] = []
            self.imagename2boxes[instance["image_name"]].append(instance["boxes"])


        print(f"INFO: {len(self.fault_gt_instances)} instances nomissing-loaded.")

        if self.mask_type == 'mask others' or self.mask_type == 'mask all':
            bkg_fault_gt_instances = []                           
            bkg_image_names = []                                                                            
            for instance in self.fault_gt_instances:
                item = {}
                item["image_name"] = instance["image_name"]                                           
                item["image_size"] = instance["image_size"]

                item["boxes"] = instance["boxes"]
                item["labels"] = 0
                item["image_id"] = instance["image_id"]
                item["area"] = instance["area"]
                item["iscrowd"] = instance["iscrowd"]
                item["fault_type"] = instance["fault_type"]

                if item["image_name"] not in bkg_image_names:
                    bkg_fault_gt_instances.append(item)
                    bkg_image_names.append(item["image_name"])

                                                              
                                                                 
                                                                    
                                                           
            self.fault_gt_instances = bkg_fault_gt_instances                                

            print(f"INFO: {len(self.fault_gt_instances)} instances only bkg-loaded.")

        self.transforms = transforms

    def gaussian_blur(self, img, box):
        '''
        Blur the object in the box
        '''
        if box[2] - box[0] <= 0:
            box[2] = box[0] + 1
        if box[3] - box[1] <= 0:
            box[3] = box[1] + 1
        img_box = img.crop(box)
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):
                                   
        instance = self.fault_gt_instances[idx]
                                 
        img_path = os.path.join(self.img_root, instance["image_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = instance["boxes"]
                                     
        boxes = [int(i) for i in boxes]

        in_boxes_list = []
                                          
        label = instance["labels"]
        img_need = None
        if label != 0:
            img_need = img.crop(boxes)
        if self.mask_type == 'mask others':
            for box in self.imagename2boxes[instance["image_name"]]:
                box = [int(i) for i in box]
                if box == boxes and label != 0:
                    continue
                                                         

                                                    
                if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                    in_boxes_list.append(box)
                else:
                    img = self.gaussian_blur(img, box)
            if label != 0:
                img.paste(img_need, boxes)

            for box in in_boxes_list:
                img = self.gaussian_blur(img, box)


        elif self.mask_type == 'crop':
                          
                                                                      
                                              
                              
                                      
                                                                                                                    
                                                                     
                                  
                                                                                                           
                                                  
                                             
                             
                        
            img = img.crop(boxes)
                             
                        



        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["labels"])
        target["boxes"] = torch.tensor(boxes)
        target["fault_type"] = instance["fault_type"]

                                     
        img = img.resize((224, 224))

                           
                         
                    

                                                
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.fault_gt_instances)

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))