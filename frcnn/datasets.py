import os
import torch
from torch.utils.data import Dataset

from PIL import Image
from pycocotools.coco import COCO

# Custom PyTorch Dataset to load COCO-format annotations and images
class CocoDetectionDataset(Dataset):
    # Init function: loads annotation file and prepares list of image IDs
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        all_image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        # 过滤掉没有任何标注的图片
        self.image_ids = []
        for img_id in all_image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) == 0:
                # 这张图没有任何目标，跳过
                continue
            self.image_ids.append(img_id)
        print(f"Total images: {len(all_image_ids)}, valid images with anns: {len(self.image_ids)}")
 
    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)
 
    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
 
        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
 
        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            if xmax <= xmin:
                xmax += 1
                print(f"img:{image_path}, box宽度有问题")
            if ymax <= ymin:
                ymax += 1
                print(f"img:{image_path}, box高度有问题")
            boxes.append([xmin, ymin, xmax, ymax])
            # anno file 中 category_id 我是从 0 开始的所以要从 1 开始
            labels.append(int(obj['category_id'])+1)
            obj["area"] = width * height
 
        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)
 
        # Package everything into a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "image_path": image_path,
            "area": area,
            "iscrowd": iscrowd
        }
 
        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)
 
        return image, target