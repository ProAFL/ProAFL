import os
import json
import random
import hashlib
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                transforms=None,
                cache_root=None):
        assert mask_type in ['crop', 'other_objects', 'all_backgrounds'], "mask_type must be in ['other objects', 'all backgrounds', 'crop']"
        
        self.img_root_dir = img_root_dir
        self.mask_type = mask_type
        self.transforms = transforms
        self.cache_dir = None
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
                         
            sampler = random.Random(42) if cache_root is not None else random
            self.background_instances_list = sampler.sample(self.instances_list,
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
        if cache_root is not None:
            self.prepare_png_cache(annotation_path, cache_root)

    def _cache_path(self, idx):
        return os.path.join(self.cache_dir, f"{idx // 1000:05d}", f"{idx:08d}.png")

    def _cache_fingerprint(self, annotation_path):
        digest = hashlib.sha256()
        digest.update(b"DisassembledDataSet-PNG-v1")
        digest.update(json.dumps([os.path.abspath(self.img_root_dir), self.mask_type,
                                  len(self.instances_list)], separators=(',', ':')).encode())
        with open(annotation_path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        for image_id in sorted({item['image_id'] for item in self.instances_list}):
            name = self.coco.imgs[image_id]['file_name']
            stat = os.stat(os.path.join(self.img_root_dir, name))
            digest.update(json.dumps([image_id, name, stat.st_size, stat.st_mtime_ns],
                                     separators=(',', ':')).encode())
        return digest.hexdigest()[:20]

    def _build_image_cache(self, image_id, indexes):
        missing = [idx for idx in indexes if not os.path.isfile(self._cache_path(idx))]
        if not missing:
            return
        image_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.img_root_dir, image_info['file_name'])
        with Image.open(img_path) as source:
            original = source.convert('RGB')
        for idx in missing:
            instance = self.instances_list[idx]
            img = self._render_instance(instance, original)
            path = self._cache_path(idx)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = path + '.tmp'
            img.save(temporary, format='PNG')
            os.replace(temporary, path)

    def prepare_png_cache(self, annotation_path, cache_root, workers=4):
        if self.mask_type not in ('crop', 'other_objects'):
            raise ValueError(f'PNG cache does not support mask_type={self.mask_type}')
        fingerprint = self._cache_fingerprint(annotation_path)
        self.cache_dir = os.path.join(cache_root, self.mask_type, fingerprint)
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, '.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = os.path.join(self.cache_dir, 'complete.json')
            if os.path.isfile(marker):
                with open(marker) as source:
                    if json.load(source).get('count') == len(self.instances_list):
                        print(f'INFO: reusing PNG cache: {self.cache_dir}')
                        return
            groups = defaultdict(list)
            for idx, instance in enumerate(self.instances_list):
                groups[instance['image_id']].append(idx)
            print(f'INFO: building PNG cache for {len(self.instances_list)} samples: {self.cache_dir}')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._build_image_cache, image_id, indexes)
                           for image_id, indexes in groups.items()]
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if completed % 500 == 0:
                        print(f'INFO: cached {completed}/{len(groups)} source images', flush=True)
            temporary = marker + '.tmp'
            with open(temporary, 'w') as output:
                json.dump({'count': len(self.instances_list), 'fingerprint': fingerprint}, output)
            os.replace(temporary, marker)
            print('INFO: PNG cache complete', flush=True)

    def gaussian_blur(self, img, box):
        '''
        textimgtextboxtextobjtext
        '''
              
        img_box = img.crop(box)
              
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
              
        img.paste(img_box, box)
        return img

    def _render_instance(self, instance, original):
        """Produce the same 224x224 image used by the uncached dataset."""
        if self.mask_type == 'other_objects':
            img = original.copy()
            cur_instance_bbox = instance["bbox"]
            label = instance["category_id"]
            in_boxes_list = []
            img_need = None
            if label != self.background_id:
                img_need = img.crop(cur_instance_bbox)
            for bbox in self.imageid2boxes[instance["image_id"]]:
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
            img = original.crop(instance['bbox'])
        else:
            raise ValueError(f'Unsupported mask_type: {self.mask_type}')
        return img.resize((224, 224))

    def __getitem__(self, idx):
        instance = self.instances_list[idx]
        if self.cache_dir is not None:
            with Image.open(self._cache_path(idx)) as source:
                img = source.convert('RGB')
        else:
            image_info = self.coco.loadImgs(instance['image_id'])[0]
            img_path = os.path.join(self.img_root_dir, image_info['file_name'])
            with Image.open(img_path) as source:
                original = source.convert('RGB')
            img = self._render_instance(instance, original)
        if self.transforms is not None:
            img = self.transforms(img)
        label = torch.tensor(instance['category_id'])
        return img, label, idx
    
    def __len__(self):
            return len(self.instances_list)

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    

                 
                 
                                                        
                                  
                  
                                                                                      
                          
                                                                                                        
                                                         
                      
                                                                                                                  
                                 
   
                 
            





