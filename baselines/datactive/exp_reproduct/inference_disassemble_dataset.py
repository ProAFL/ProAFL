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
            other_objects|all_background: text, textobjtextother objtextmask
                textbg instance
            crop: textobjtextcliptext
        :param transforms: toTenser,normalize
        '''
        self.img_root = img_root
        self.mask_type = mask_type                                        
        self.coco = COCO(annotation_path)                                     
        self.catIds = self.coco.getCatIds()                    
        self.background_id = self.catIds[-1] + 1                    
        ann_ids = self.coco.getAnnIds()               
        annotations = self.coco.loadAnns(ann_ids)                           
                      
        self.instances = []
                                     
        for instance in annotations:
            xmin, ymin, width, height = instance["bbox"]
            instance["bbox"] = xminyminwh_to_xyxy(xmin, ymin, width, height)
                             
        self.imageid2boxes = defaultdict(list)
        
              
        for instance in annotations:
            self.instances.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])
            
        print(f"INFO: {len(self.instances)} instances nomissing-loaded.")

        if self.mask_type == 'other_objects' or self.mask_type == 'all_background':
                  
            bg_instances = []                            
            bg_image_names = []           
            for instance in self.instances:
                                
                image_id = instance["image_id"]
                image_info = self.coco.loadImgs(image_id)[0] 
                image_name = image_info['file_name']
                item = {}
                item["image_id"] = image_id
                item["image_name"] = image_name
                item["image_size"] = [image_info["width"],image_info["height"]]
                item["anno_id"] = instance["id"]
                item["bbox"] = instance["bbox"]       
                item["label"] = self.background_id                                     
                item["area"] = self.caclu_area(instance["bbox"])               
                item["iscrowd"] = 0          
                item["fault_type"] = instance["fault_type"]

                if item["image_name"] not in bg_image_names:
                    bg_instances.append(item)
                    bg_image_names.append(item["image_name"])

            self.instances = bg_instances                                
            print(f"INFO: {len(self.instances)} instances only bkg-loaded.")
        elif self.mask_type == "crop": 
                      
            temp_instances = []
            for instance in self.instances:
                                
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
        '''textimg box text'''
        if box[2] - box[0] <= 0:
            box[2] = box[0] + 1
        if box[3] - box[1] <= 0:
            box[3] = box[1] + 1
        img_box = img.crop(box)              
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))              
        img.paste(img_box, box)       
        return img
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        img_path = os.path.join(self.img_root, instance['image_name'])
        img = Image.open(img_path).convert("RGB")
        cur_instance_bbox = instance["bbox"]
                                 
        in_boxes_list = []
        label = instance["label"]
        img_need = None
        if label != self.background_id:
                               
            img_need = img.crop(cur_instance_bbox)

        if self.mask_type == 'other_objects':
                  
            for bbox in self.imageid2boxes[instance["image_id"]]:
                bbox = [int(i) for i in bbox]
                if bbox == cur_instance_bbox and label != self.background_id:
                    continue
                                                         
                                                    

                if bbox[0] > cur_instance_bbox[0] and bbox[1] > cur_instance_bbox[1] and bbox[2] < cur_instance_bbox[2] and bbox[3] < cur_instance_bbox[3]:
                    in_boxes_list.append(bbox)
                else:
                                    
                    img = self.gaussian_blur(img, bbox)
            if label != self.background_id:
                img.paste(img_need, cur_instance_bbox)

                                  
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

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))