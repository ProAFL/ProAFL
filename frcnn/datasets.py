import os
import torch
from torch.utils.data import Dataset

from PIL import Image
from pycocotools.coco import COCO

                                                                   
class CocoDetectionDataset(Dataset):
                                                                         
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        all_image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

              
        self.image_ids = []
        for img_id in all_image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) == 0:
                           
                continue
            self.image_ids.append(img_id)
        print(f"Total images: {len(all_image_ids)}, valid images with anns: {len(self.image_ids)}")
 
                                    
    def __len__(self):
        return len(self.image_ids)
 
                                                
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
 
                                             
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
 
                                                            
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            if xmax <= xmin:
                xmax += 1
                print(f"img:{image_path}, boxtext")
            if ymax <= ymin:
                ymax += 1
                print(f"img:{image_path}, boxtext")
            boxes.append([xmin, ymin, xmax, ymax])
                                                           
            labels.append(int(obj['category_id'])+1)
            obj["area"] = width * height
 
                                                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)
 
                                                     
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "image_path": image_path,
            "area": area,
            "iscrowd": iscrowd
        }
 
                                             
        if self.transforms:
            image = self.transforms(image)
 
        return image, target