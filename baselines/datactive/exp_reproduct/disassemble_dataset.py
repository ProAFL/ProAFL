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
        self.coco = COCO(annotation_path) # no miss fault
        self.catIds = self.coco.getCatIds()
        self.background_id = self.catIds[-1]+1 # 背景类id，就是最后个类id+1，确保与所有其他类id不同，比如对于10个类，背景类id就是11
                                 
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
                  
        self.instances_list = [] # 每个anno其实就是instance
                   
        self.imageid2boxes = defaultdict(list)
              
        for instance in annotations:
            self.instances_list.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])

        len_before = len(self.instances_list)

              
        if self.mask_type == 'all_backgrounds' or self.mask_type == 'other_objects':
            print(f'MASK TYPE: {self.mask_type}')
            sampler = random.Random(42) if cache_root is not None else random
            # 无放回采样，采样一部分的instance作为背景instance，即修改category_id为背景类id
            self.background_instances_list = sampler.sample(self.instances_list,int(len(self.instances_list) / class_num))
            # 修改category_id为背景类id，将背景instance转换为背景类
            for instance in self.background_instances_list:     
                new_instance = {key: value for key, value in instance.items()}
                new_instance["category_id"] = self.background_id                                                      
                self.instances_list.append(new_instance)
            # 这些背景instance是append进来，所以需要检查是否等于原始instance数加上背景instance数
            assert len(self.instances_list) == len_before + len(self.background_instances_list)
        else:
            print(f'MASK TYPE: {self.mask_type}') # crop
            self.background_instances_list = [] # 不需要背景instance
            assert len(self.instances_list) == len_before
        print("INFO: {} instances loaded. including {} instances and {} background instances".format(
            len(self.instances_list), len_before, len(self.background_instances_list)))
        if cache_root is not None: # 如果设置了缓存目录，就准备缓存png图片
            self.prepare_png_cache(annotation_path, cache_root)

    def _cache_path(self, idx):
        # 例如idx=123456，返回路径为：/cache_dir/00001/0000123456.png
        # 这里每 1000 个样本分到一个子目录，避免所有文件堆在同一个目录里。
        return os.path.join(self.cache_dir, f"{idx // 1000:05d}", f"{idx:08d}.png")

    def _cache_fingerprint(self, annotation_path):
        '''
        计算缓存指纹，用于判断是否需要重新缓存png图片。
        '''
        digest = hashlib.sha256() # hash计算器
        #  DisassembledDataSet-PNG-v1 是人为定义的版本标识。如果以后修改图片处理逻辑，可以把 v1 改为 v2，让新逻辑使用新的缓存，避免继续读取旧结果。这个版本号不会自动更新。 
        digest.update(b"DisassembledDataSet-PNG-v1") #   b""表示字节串，update() 的意思是“把这部分数据加入计算”，多次调用会累积输入，不会覆盖前面的数据。 
        # 因此，即使图片内容一样，换了图片目录，也会得到不同的指纹。
        digest.update(json.dumps([os.path.abspath(self.img_root_dir), self.mask_type,
                                  len(self.instances_list)], separators=(',', ':')).encode()) # .encode()把字符串转换为字节串
        with open(annotation_path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''): # 每次读取1MB数据
                digest.update(chunk) # 加入数据到hash计算器， 分块读取是为了避免把整个文件一次性放进内存。
        #   所以，标注文件内容一旦改变，指纹就通常会改变。哪怕只调整 JSON 文件中的空格，也会影响指纹，因为这里读取的是原始文件字节。
        for image_id in sorted({item['image_id'] for item in self.instances_list}): # sorted(...)：排序，保证每次加入哈希的信息顺序一致。
            name = self.coco.imgs[image_id]['file_name']
            stat = os.stat(os.path.join(self.img_root_dir, name)) # os.stat统计文件大小单位是字节，stat.st_mtime_ns是文件修改时间，单位是纳秒级
            digest.update(json.dumps([image_id, name, stat.st_size, stat.st_mtime_ns], 
                                     separators=(',', ':')).encode())
        return digest.hexdigest()[:20] # hexdigest() 把哈希结果转换为十六进制字符串，完整 SHA-256 结果有 64 个字符。这里只取前 20 个字符作为缓存编号。

    def _build_image_cache(self, image_id, indexes):
        '''
        为一个原图生成多个样本的png图片。
        image_id: 原图在 COCO 标注中的 ID。
        indexes:  这张原图对应的样本在 self.instances_list 中的下标列表，比如 [0, 1, 3]。
        '''
        # 收集这个image id下所有的instance缺失缓存的
        missing = [idx for idx in indexes if not os.path.isfile(self._cache_path(idx))]
        if not missing:
            # 这个image id下所有instance没有缺失直接return了，啥也不做
            return
        image_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.img_root_dir, image_info['file_name'])
        with Image.open(img_path) as source:
            original = source.convert('RGB')
        for idx in missing:
            instance = self.instances_list[idx]
            img = self._render_instance(instance, original) # 这个缓存缺失实例和原图，根据mask_type渲染出instance图片
            path = self._cache_path(idx)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = path + '.tmp'
            img.save(temporary, format='PNG')
            os.replace(temporary, path) #  先完整写入临时文件，再通过 os.replace() 原子替换为正式文件，可以避免程序中断时留下只写了一半的正式 PNG。

    def prepare_png_cache(self, annotation_path, cache_root, workers=4):
        if self.mask_type not in ('crop', 'other_objects'):
            raise ValueError(f'PNG cache does not support mask_type={self.mask_type}')
        fingerprint = self._cache_fingerprint(annotation_path)
        # 创建缓存目录：{cache_root}/{mask_type}/{fingerprint}
        self.cache_dir = os.path.join(cache_root, self.mask_type, fingerprint)
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, '.lock'), 'w') as lock: # • 这两行是给缓存目录加一个进程间的排他锁，避免多个进程同时生成同一份缓存。
            fcntl.flock(lock, fcntl.LOCK_EX) #   这行向操作系统申请该文件的排他锁，LOCK_EX 就是 exclusive lock。
            # completed.json 内容最终类似
            # { "count":len(self.instances_list),"fingerprint": "a12b34c56d78e90f1234"}
            marker = os.path.join(self.cache_dir, 'complete.json')
            if os.path.isfile(marker):
                #   意思是：如果缓存目录中已经有 complete.json，而且里面记录的样本数量与当前数据集一致，就认为缓存已经完成，可以复用。
                with open(marker) as source:
                    if json.load(source).get('count') == len(self.instances_list):
                        print(f'INFO: reusing PNG cache: {self.cache_dir}')
                        return
            #   把实例按所属原图分组
            #   为什么按原图分组？ 因为同一张原图可能包含多个目标。如果每个实例都单独读取原图，会重复读取和解码；分组后，一张原图只需打开一次，就能生成它对应的多个样本。
            groups = defaultdict(list)
            for idx, instance in enumerate(self.instances_list):
                groups[instance['image_id']].append(idx)
            print(f'INFO: building PNG cache for {len(self.instances_list)} samples: {self.cache_dir}')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # 每个image_id都会有个Future对象，用于等待线程池完成任务
                futures = [pool.submit(self._build_image_cache, image_id, indexes)
                           for image_id, indexes in groups.items()] # 线程池安排线程执行（_build_image_cache），每个线程处理一个原图的多个样本
                #   as_completed(futures) 会按照任务实际完成的顺序返回 Future，不一定按照提交顺序。
                #  enumerate(..., 1) 则从 1 开始计数，记录已经完成了多少个任务。
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result() #   非常关键：如果工作线程里的任务报错，这里会把异常重新抛出来，让整个缓存准备过程失败，避免后面把不完整的缓存标记为完成。这段代码没有使用任务返回值，只需要确认任务执行成功。
                    if completed % 500 == 0:
                        #   每完成 500 个任务，就打印一次进度。注意进度的单位是原图数量，因为每个任务对应一张原图，而不是一个instance。 
                        print(f'INFO: cached {completed}/{len(groups)} source images', flush=True)
            #  全部成功后，写入完成标记 
            temporary = marker + '.tmp'
            with open(temporary, 'w') as output:
                json.dump({'count': len(self.instances_list), 'fingerprint': fingerprint}, output)
            os.replace(temporary, marker)
            print('INFO: PNG cache complete', flush=True)

    def gaussian_blur(self, img, box):
        img_box = img.crop(box)
              
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
              
        img.paste(img_box, box)
        return img

    def _render_instance(self, instance, original):
        """Produce the same 224x224 image used by the uncached dataset."""
        if self.mask_type == 'other_objects':
            img = original.copy() # copy a new image to avoid modifying the original
            cur_instance_bbox = instance["bbox"]
            label = instance["category_id"]
            in_boxes_list = []
            img_need = None
            if label != self.background_id:
                # 不是背景instance，才需要裁剪出来
                img_need = img.crop(cur_instance_bbox)
            for bbox in self.imageid2boxes[instance["image_id"]]:
                # 遍历出这个image id下的所有框框
                if bbox == cur_instance_bbox and label != self.background_id:
                    # 如果是本instance的框框且不是背景instance，就不要模糊处理
                    continue
                if bbox[0] > cur_instance_bbox[0] and bbox[1] > cur_instance_bbox[1] and bbox[2] < cur_instance_bbox[2] and bbox[3] < cur_instance_bbox[3]:
                    # 遇到本框中框
                    in_boxes_list.append(bbox)
                else:
                    # 遇到其他框框，就模糊处理一下
                    img = self.gaussian_blur(img, bbox)
            if label != self.background_id:
                img.paste(img_need, cur_instance_bbox)
            for bbox in in_boxes_list:
                img = self.gaussian_blur(img, bbox)
        elif self.mask_type == 'crop':
            # 把这个图的instance裁剪出来，作为缓存png图片
            img = original.crop(instance['bbox'])
        else:
            raise ValueError(f'Unsupported mask_type: {self.mask_type}')
        return img.resize((224, 224))

    def __getitem__(self, idx):
        instance = self.instances_list[idx]
        if self.cache_dir is not None:
            # 有缓存直接读取缓存
            with Image.open(self._cache_path(idx)) as source:
                img = source.convert('RGB')
        else:
            # 没有缓存，从原始图片渲染一个instance
            image_info = self.coco.loadImgs(instance['image_id'])[0]
            img_path = os.path.join(self.img_root_dir, image_info['file_name'])
            with Image.open(img_path) as source:
                original = source.convert('RGB')
            img = self._render_instance(instance, original)
        if self.transforms is not None:
            # 对图片进行transform
            img = self.transforms(img)
        label = torch.tensor(instance['category_id'])
        return img, label, idx
    
    def __len__(self):
            return len(self.instances_list)

                                
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    

                 
                 
                                                        
                                  
                  
                                                                                      
                          
                                                                                                        
                                                         
                      
                                                                                                                  
                                 
   
                 
            





