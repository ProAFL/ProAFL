import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from collections import defaultdict
'''
text
'''
class DisassembledDataSet(Dataset):
    def __init__(self, 
                img_root_dir, 
                annotation_path,
                class_num,
                mask_type,
                transforms=None):                                                                       
        assert mask_type in ['crop', 'other_objects', 'all_backgrounds'], "mask_type must be in ['other objects', 'all backgrounds', 'crop']"
        
        self.img_root_dir = img_root_dir
        self.mask_type = mask_type
        self.transforms = transforms
        self.coco = COCO(annotation_path)
        self.catIds = self.coco.getCatIds()
        self.background_id = self.catIds[-1]+1
                                 
        ann_ids = self.coco.getAnnIds()
                                 
        annotations = self.coco.loadAnns(ann_ids)
        for instance in annotations:
            xmin, ymin, width, height = instance["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            instance["bbox"] = [int(xmin),int(ymin),int(xmax),int(ymax)]
                         
            if instance["bbox"][0] == instance["bbox"][2]:
                instance["bbox"][2] += 1
            if instance["bbox"][1] == instance["bbox"][3]:
                instance["bbox"][3] += 1
        '''
        textlabel!=-1textself.instances_listtext
        textimgid2boxestext:{imgid:[box1,box2]}
        '''
                  
        self.instances_list = []
                   
        self.imageid2boxes = defaultdict(list)
              
        for instance in annotations:
            self.instances_list.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])

        len_before = len(self.instances_list)

              
        if self.mask_type == 'all_backgrounds' or self.mask_type == 'other_objects':
            '''
            textself.instances_listtext,textself.instances_listtext
            '''
            print('INFO: all_backgrounds or other objects')
                         
            self.background_instances_list = random.sample(self.instances_list,
                                                           int(len(self.instances_list) / class_num))
                         
            for instance in self.background_instances_list:
                             
                new_instance = {key: value for key, value in instance.items()}
                                              
                new_instance["category_id"] = self.background_id              
                                                
                self.instances_list.append(new_instance)

            assert len(self.instances_list) == len_before + len(self.background_instances_list)
        else:
                  
            self.background_instances_list = []
            assert len(self.instances_list) == len_before

        print("INFO: {} instances loaded. including {} instances and {} background instances".format(
            len(self.instances_list), len_before, len(self.background_instances_list)))

    def gaussian_blur(self, img, box):
        '''
        textimgtextboxtextobjtext
        '''
              
        img_box = img.crop(box)
              
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
              
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):
              
        instance = self.instances_list[idx]
                        
        image_id = instance["image_id"]
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_root_dir, image_info['file_name'])

        img, label = None, None
              
                 
        if self.mask_type == 'other_objects':
                               
            img = Image.open(img_path).convert("RGB")
            cur_instance_bbox = instance["bbox"]
            label = instance["category_id"]
                     
            in_boxes_list = []
            img_need = None

            '''
            textinstancetext:
            textinstance text objtext
            '''
            if label != self.background_id:               
                img_need = img.crop(cur_instance_bbox)                 
            
            '''textinner boxtext'''
                      
            for bbox in self.imageid2boxes[instance["image_id"]]:
                if bbox == cur_instance_bbox and label != self.background_id:
                                                                                                                     
                    continue
                if bbox[0] > cur_instance_bbox[0] and bbox[1] > cur_instance_bbox[1] and bbox[2] < cur_instance_bbox[2] and bbox[3] < cur_instance_bbox[3]:
                                                                           
                    in_boxes_list.append(bbox)
                else:
                                       
                    img = self.gaussian_blur(img, bbox)                      

            if label != self.background_id:
                                                                                     
                img.paste(img_need, cur_instance_bbox)

            '''text'''
            for bbox in in_boxes_list:                       
                img = self.gaussian_blur(img, bbox)

                              
        elif self.mask_type == 'crop':
            img = Image.open(img_path).convert("RGB")
            cur_instance_bbox = instance['bbox']
            label = instance["category_id"]
                                                  
            img = img.crop(cur_instance_bbox)

        img = img.resize((224, 224))
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(label)
        return img, label, idx              
    
    def __len__(self):
            return len(self.instances_list)

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    

                 
                 
                                                        
                                  
                  
                                                                                      
                          
                                                                                                        
                                                         
                      
                                                                                                                  
                                 
   
                 
            





