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
    def __init__(self, root, transforms=None, txt_name: str = "train.txt", mask_type='other objects',
                 train_type='clean', dirty_path='', datatype: str = 'VOC', run_type='test'):  # run_type 仅用来指定是测试集还是训练集
        assert mask_type in ['other objects', 'all backgrounds',
                             'crop'], "mask_type must be in ['other objects', 'all backgrounds', 'crop']"
        assert train_type in ['clean', 'dirty'], "train_type must be in ['clean', 'dirty']"
        self.dirtypath = dirty_path
        self.train_type = train_type
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
            class_num = 80  # 仅用来添加bkg，所以不用区分extra class
            self.img_root = os.path.join(self.root, "val2017")
        elif datatype == 'KITTI':
            class_num = 8
            self.img_root = os.path.join(self.root, "training/image_2")

        if train_type == 'clean' and datatype == 'VOC':
            self.annotations_root = os.path.join(self.root, "Annotations")
            text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
            assert os.path.exists(text_path), "file not found"
            with open(text_path, "r") as f:
                xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                            for line in f.readlines() if len(line.strip()) > 0]
            self.xml_list = []
            # check file
            for xml_path in xml_list:
                assert os.path.exists(xml_path), "xml file not found"
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                xml = etree.fromstring(xml_str)
                data = self.parse_xml_to_dict(xml)["annotation"]
                if "object" not in data:
                    print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                    continue
                self.xml_list.append(xml_path)
            # read class_indict
            json_file = os.path.join(self.root, 'pascal_voc_classes.json')
            assert os.path.exists(json_file), 'json file not found'
            with open(json_file, 'r') as fp:
                self.class_dict = json.load(fp)

            self.imageid2boxes = {}
            # split into instances
            self.instances_list = []
            for xml_path in self.xml_list:
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                xml = etree.fromstring(xml_str)
                data = self.parse_xml_to_dict(xml)["annotation"]
                for obj in data["object"]:
                    instance = {}
                    instance["image_id"] = data["filename"]
                    instance["category_id"] = self.class_dict[obj["name"]]
                    instance["bbox"] = obj["bndbox"]
                    self.instances_list.append(instance)
                    if instance["image_id"] not in self.imageid2boxes:
                        self.imageid2boxes[instance["image_id"]] = []
                    self.imageid2boxes[instance["image_id"]].append(instance["bbox"])
            len_before = len(self.instances_list)
            if self.mask_type == 'all backgrounds' or self.mask_type == 'other objects':
                print('INFO: all backgrounds or other objects')
                # random select 1000 instances
                # need revised
                self.background_instances_list = random.sample(self.instances_list,
                                                               int(len(self.instances_list) / class_num))

                for instance in self.background_instances_list:
                    new_instance = {key: value for key, value in instance.items()}
                    new_instance["category_id"] = 0
                    self.instances_list.append(new_instance)
                assert len(self.instances_list) == len_before + len(self.background_instances_list)
            else:
                self.background_instances_list = []
                assert len(self.instances_list) == len(self.instances_list)

            print("INFO: {} instances loaded. including {} instances and {} background instances".format(
                len(self.instances_list), len_before, len(self.background_instances_list)))

        elif train_type == 'clean' and (datatype == 'VisDrone' or datatype == 'KITTI'):
            if datatype == 'VisDrone':
                self.annotations_root = os.path.join(self.root, 'annotations.json')
            elif datatype == 'KITTI':
                self.annotations_root = os.path.join(self.root, run_type+'_annotations.json')
            with open(self.annotations_root, 'r') as f:
                self.annotations = json.load(f)

            self.imageid2boxes = {}
            self.instances_list = []

            for id, image in enumerate(self.annotations):

                img = Image.open(os.path.join(self.img_root,image['image_name']))

                image_size = [img.size[0], img.size[1]]

                for i in range(len(image['categories'])):
                    boxes = image['boxes'][i]
                    instance = {
                        "image_name": image['image_name'],
                        "image_size": image_size,
                        "boxes": boxes,
                        "labels": image['categories'][i],
                        "image_id": id,
                        "area": (boxes[2] - boxes[0]) * (boxes[3] - boxes[1]),
                        "iscrowd": 0,
                        "fault_type": 0
                    }
                    self.instances_list.append(instance)
                    if instance["image_name"] not in self.imageid2boxes:
                        self.imageid2boxes[instance["image_name"]] = []
                    self.imageid2boxes[instance["image_name"]].append(instance["boxes"])

            len_before = len(self.instances_list)
            if self.mask_type == 'all backgrounds' or self.mask_type == 'other objects':
                print('INFO: all backgrounds or other objects')
                # random select 1000 instances
                # need revised
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

            if datatype == 'VisDrone':
                random.seed(2023)

                # random shuffle
                random.shuffle(self.instances_list)

                # sample 1/7 for
                if self.mask_type == 'crop':
                    self.instances_list = self.instances_list[:int(len(self.instances_list) / 10)]
                elif self.mask_type == 'other objects':
                    self.instances_list = self.instances_list[:int(len(self.instances_list) / 50)]

            for instance in self.instances_list:
                instance["boxes"] = [int(i) for i in instance["boxes"]]
                if instance["boxes"][0] == instance["boxes"][2]:
                    instance["boxes"][2] += 1
                if instance["boxes"][1] == instance["boxes"][3]:
                    instance["boxes"][3] += 1

            print("INFO: {} instances sampled".format(len(self.instances_list)))

        elif train_type == 'clean' and datatype == 'COCO':
            self.instances_root = os.path.join(self.root, 'instances_val2017_' + run_type + '.json')
            with open(self.instances_root, 'r') as f:
                self.instances_list = json.load(f)

            # make boxes be int
            for instance in self.instances_list:
                instance["boxes"] = [int(i) for i in instance["boxes"]]
                if instance["boxes"][0] == instance["boxes"][2]:
                    instance["boxes"][2] += 1
                if instance["boxes"][1] == instance["boxes"][3]:
                    instance["boxes"][3] += 1

            self.imageid2boxes = {}
            for instance in self.instances_list:
                if instance["image_name"] not in self.imageid2boxes:
                    self.imageid2boxes[instance["image_name"]] = []
                self.imageid2boxes[instance["image_name"]].append(instance["boxes"])
            len_before = len(self.instances_list)
            if self.mask_type == 'all backgrounds' or self.mask_type == 'other objects':
                print('INFO: all backgrounds or other objects')
                # random select 1000 instances
                # need revised
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




        elif train_type == 'dirty':  # 这里处理方式是一致的
            with open(self.dirtypath, 'r') as f:
                self.dirtylist = json.load(f)
            if datatype == 'COCO' or datatype == 'KITTI':
                # make boxes be int
                for instance in self.dirtylist:
                    instance["boxes"] = [int(i) for i in instance["boxes"]]
                    if instance["boxes"][0] == instance["boxes"][2]:
                        instance["boxes"][2] += 1
                    if instance["boxes"][1] == instance["boxes"][3]:
                        instance["boxes"][3] += 1

            self.instances_list = []
            self.imageid2boxes = {}
            for target in self.dirtylist:
                if target['fault_type'] != fault_type_dict['missing fault']:
                    self.instances_list.append(target)
                    if target["image_name"] not in self.imageid2boxes:
                        self.imageid2boxes[target["image_name"]] = []
                    self.imageid2boxes[target["image_name"]].append(target["boxes"])
            len_before = len(self.instances_list)
            if self.mask_type == 'all backgrounds' or self.mask_type == 'other objects':
                print('INFO: all backgrounds or other objects')
                # random select 1000 instances
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

            if datatype == 'VisDrone':
                random.seed(2023)

                # random shuffle
                random.shuffle(self.instances_list)

                # sample 1/7 for train
                if self.mask_type == 'crop':
                    self.instances_list = self.instances_list[:int(len(self.instances_list) / 10)]
                elif self.mask_type == 'other objects':
                    self.instances_list = self.instances_list[:int(len(self.instances_list) / 50)]

                print("INFO: {} instances sampled".format(len(self.instances_list)))

    def gaussian_blur(self, img, box):

        img_box = img.crop(box)
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):

        instance = self.instances_list[idx]

        img, label = None, None
        if self.mask_type == 'other objects':
            if self.train_type == 'clean':
                if self.datatype == 'VOC':
                    img_path = os.path.join(self.img_root, instance["image_id"])
                    img = Image.open(img_path).convert("RGB")
                    boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                             int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]
                    label = instance["category_id"]
                    img_name = instance["image_id"]
                elif (self.datatype == 'VisDrone' or self.datatype =='KITTI') or self.datatype == 'COCO':
                    img_path = os.path.join(self.img_root, instance["image_name"])
                    img = Image.open(img_path).convert("RGB")
                    boxes = instance["boxes"]
                    label = instance["labels"]
                    img_name = instance["image_name"]

                in_boxes_list = []
                # get img boxes except current box

                img_need = None
                if label != 0:
                    img_need = img.crop([int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])])

                for box in self.imageid2boxes[img_name]:
                    if self.datatype == 'VOC':
                        box = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]

                    if box == boxes and label != 0:
                        continue
                    # Gaussian blur the box area of the image

                    # if box belong to the part of boxes
                    if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                        in_boxes_list.append(box)
                    else:
                        img = self.gaussian_blur(img, [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                if label != 0:
                    img.paste(img_need, [int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])])

                for box in in_boxes_list:
                    img = self.gaussian_blur(img, [int(box[0]), int(box[1]), int(box[2]), int(box[3])])

            elif self.train_type == 'dirty':

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

        elif self.mask_type == 'all backgrounds' and self.train_type == 'clean':
            img_path = os.path.join(self.img_root, instance["image_id"])
            img = Image.open(img_path).convert("RGB")
            boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                     int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]
            in_boxes_list = []
            # get img boxes except current box
            label = instance["category_id"]
            img_need = None
            if label != 0:
                img_need = img.crop(boxes)
            for box in self.imageid2boxes[instance["image_id"]]:
                box = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
                if box == boxes and label != 0:
                    continue
                # Gaussian blur the box area of the image

                # if box belong to the part of boxes
                if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                    in_boxes_list.append(box)
            img = self.gaussian_blur(img, [0, 0, img.size[0], img.size[1]])
            if label != 0:
                img.paste(img_need, boxes)
            for box in in_boxes_list:
                img = self.gaussian_blur(img, box)

        elif self.mask_type == 'crop':
            if self.train_type == 'dirty':
                img_path = os.path.join(self.img_root, instance["image_name"])
                img = Image.open(img_path).convert("RGB")
                boxes = instance['boxes']
                label = instance["labels"]
            elif self.train_type == 'clean':
                if self.datatype == 'VOC':
                    img_path = os.path.join(self.img_root, instance["image_id"])
                    img = Image.open(img_path).convert("RGB")
                    boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                             int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]
                    label = instance["category_id"]
                elif self.datatype == 'VisDrone' or self.datatype == 'COCO' or self.datatype == 'KITTI':
                    img_path = os.path.join(self.img_root, instance["image_name"])
                    img = Image.open(img_path).convert("RGB")
                    boxes = [int(x) for x in instance["boxes"]]
                    label = instance["labels"]

            # Crop out the boxes part of the image
            img = img.crop(boxes)
            #


        # if idx == 3952:
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
        # print(instance)
        # plt.imshow(img)
        # plt.show()
        #
        img = img.resize((224, 224))
        # plt.imshow(img)
        # plt.show()

        # convert everything into
        #
        # torch.Tensor
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

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inference_VOCGt_classificationDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

        # split into instances
        self.instances_list = []
        for xml_path in self.xml_list:
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            for obj in data["object"]:
                instance = {}
                instance["image_name"] = data["filename"]
                instance["category_id"] = self.class_dict[obj["name"]]
                instance["bbox"] = obj["bndbox"]
                self.instances_list.append(instance)
        print(f"INFO: {len(self.instances_list)} instances loaded.")

    def __getitem__(self, idx):
        instance = self.instances_list[idx]
        img_path = os.path.join(self.img_root, instance["image_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = [int(instance["bbox"]["xmin"]), int(instance["bbox"]["ymin"]),
                 int(instance["bbox"]["xmax"]), int(instance["bbox"]["ymax"])]

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["category_id"])
        target["boxes"] = torch.tensor(boxes)

        # # # draw img with bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
        #                   linewidth=3)) # xmin, ymin, w, h
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        # Crop out the boxes part of the image
        img = img.crop(boxes)

        # # plt cropped image
        # plt.imshow(img)
        # plt.show()

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

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

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


from UTILS.parameters import parameters


class inference_VOCinf_classificationDataSet(data.Dataset):
    def __init__(self, voc_root, inferences_root="../data/detection_results/ssd_VOCval_inferences.json",
                 transforms=None):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        full_inference_results = json.load(open(inferences_root, "r"))

        self.inference_results = []
        # get inference results score > m_t
        params = parameters()
        for inference_result in full_inference_results:
            if inference_result["score"] > params.m_t:
                self.inference_results.append(inference_result)
        print(f"INFO: {len(self.inference_results)} instances loaded.")

        self.transforms = transforms

    def __getitem__(self, idx):
        instance = self.inference_results[idx]
        img_path = os.path.join(self.img_root, instance["image_name"])
        img = Image.open(img_path).convert("RGB")
        boxes = instance["bbox"]

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["category_id"])
        target["boxes"] = torch.tensor(boxes)

        # # draw img with bounding box
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='red',
        #                   linewidth=3)) # xmin, ymin, w, h
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        # Crop out the boxes part of the image
        img = img.crop(boxes)

        # plt cropped image
        # plt.imshow(img)
        # plt.show()

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.inference_results)

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

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inference_VOCgtfault_classificationDataSet(data.Dataset):
    def __init__(self, root, fault_type='mixed fault', transforms=None, mask_type='mask others', datatype='VOC',
                 dirty_path='../data/fault_annotations/VOCval_mixedfault0.1.json'):
        self.root = root
        self.datatype = datatype
        path = None
        if datatype == 'VOC':
            path = "JPEGImages"
        elif datatype == 'VisDrone':
            path = "images"
        elif datatype == 'COCO':
            path = "val2017"
        elif datatype == 'KITTI':
            path = "training/image_2"
        self.img_root = os.path.join(self.root, path)
        self.mask_type = mask_type
        # falut_type without space
        fault_gt = json.load(open(dirty_path, "r"))
        self.fault_gt_instances = []
        # if fault_type == 'missing fault' then remove it
        print(f"INFO: fault type is {fault_type}")
        print(f"INFO: {len(fault_gt)} instances pre-loaded.")

        self.imagename2boxes = {}
        params = parameters()
        fault_type_dict = params.fault_type

        if fault_type == 'missing fault' or fault_type == 'mixed fault':
            for instance in fault_gt:
                if instance["fault_type"] != fault_type_dict['missing fault']:
                    self.fault_gt_instances.append(instance)
                    if instance["image_name"] not in self.imagename2boxes:
                        self.imagename2boxes[instance["image_name"]] = []
                    self.imagename2boxes[instance["image_name"]].append(instance["boxes"])
        else:
            for instance in fault_gt:
                self.fault_gt_instances.append(instance)

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

            # self.fault_gt_instances = bkg_fault_gt_instances
            # merge fault_gt_instances and bkg_fault_gt_instances
            # self.fault_gt_instances.extend(bkg_fault_gt_instances)
            self.fault_gt_instances = bkg_fault_gt_instances #只需推理背景部分即可

            print(f"INFO: {len(self.fault_gt_instances)} instances only bkg-loaded.")

        self.transforms = transforms

    def gaussian_blur(self, img, box):
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
        # get img boxes except current box
        label = instance["labels"]
        img_need = None
        if label != 0:
            img_need = img.crop(boxes)
        if self.mask_type == 'mask others':
            for box in self.imagename2boxes[instance["image_name"]]:
                box = [int(i) for i in box]
                if box == boxes and label != 0:
                    continue
                # Gaussian blur the box area of the image

                # if box belong to the part of boxes
                if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                    in_boxes_list.append(box)
                else:
                    img = self.gaussian_blur(img, box)
            if label != 0:
                img.paste(img_need, boxes)

            for box in in_boxes_list:
                img = self.gaussian_blur(img, box)

        elif self.mask_type == 'mask all':
            for box in self.imagename2boxes[instance["image_name"]]:
                box = [int(i) for i in box]
                if box == boxes and label != 0:
                    continue
                # Gaussian blur the box area of the image

                # if box belong to the part of boxes
                if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                    in_boxes_list.append(box)
            img = self.gaussian_blur(img, [0, 0, img.size[0], img.size[1]])
            if label != 0:
                img.paste(img_need, boxes)
            for box in in_boxes_list:
                img = self.gaussian_blur(img, box)

        elif self.mask_type == 'crop':
            img = img.crop(boxes)

        # print(label)
        # for box in self.imagename2boxes[instance["image_name"]]:
        #     if box == instance["boxes"]:
        #         continue
        #     plt.gca().add_patch(
        #         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
        #                       linewidth=3))  # xmin, ymin, w, h
        # plt.gca().add_patch(
        #     plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], fill=False, edgecolor='green',
        #                   linewidth=3))
        # plt.imshow(img)
        # plt.show()

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["labels"])
        target["boxes"] = torch.tensor(boxes)
        target["fault_type"] = instance["fault_type"]

        # Resize the image to 224x224
        img = img.resize((224, 224))

        # plt resized image
        # plt.imshow(img)
        # plt.show()

        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.fault_gt_instances)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class DetectionDataSet(data.Dataset):
    def __init__(self, root, transforms=None, runtype: str = 'train', datatype: str = 'VOC', traintype='clean',
                 dirtypath=''):
        self.datatype = datatype # COCO/VOC
        self.root = root
        self.traintype = traintype # clean dirty
        self.dirtypath = dirtypath
        if self.datatype == 'VOC' and self.traintype == 'clean':
            self.img_root = os.path.join(self.root, "JPEGImages")
            self.annotations_root = os.path.join(self.root, "Annotations")
            self.transforms = transforms
            text_path = os.path.join(self.root, "ImageSets/Main", runtype + '.txt')
            assert os.path.exists(text_path), "file not found"
            with open(text_path, "r") as f:
                xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                            for line in f.readlines() if len(line.strip()) > 0]
            self.xml_list = []
            # check file
            for xml_path in xml_list:
                assert os.path.exists(xml_path), "xml file not found"
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                xml = etree.fromstring(xml_str)
                data = self.parse_xml_to_dict(xml)["annotation"]
                if "object" not in data:
                    print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                    continue
                self.xml_list.append(xml_path)
            # read class_indict
            json_file = os.path.join(self.root, 'pascal_voc_classes.json')
            assert os.path.exists(json_file), 'json file not found'
            with open(json_file, 'r') as fp:
                self.class_dict = json.load(fp)

        elif self.traintype == 'dirty':
            with open(self.dirtypath, 'r') as f:
                self.dirtylist = json.load(f)
            # transform dirtylist to {imagename:[]} format dict
            self.imagename2targets = {}
            for target in self.dirtylist:
                if target['image_name'] not in self.imagename2targets:
                    self.imagename2targets[target['image_name']] = []
                if target['fault_type'] != fault_type_dict['missing fault']:  # missing is not a target
                    self.imagename2targets[target['image_name']].append(target)
            print('imagename2targets len:', len(self.imagename2targets))
            self.keys_list = list(self.imagename2targets.keys())

            self.new_keys_list = []
            # delete the image which has no target
            for key in self.keys_list:
                if len(self.imagename2targets[key]) != 0:
                    self.new_keys_list.append(key)
            self.keys_list = self.new_keys_list

            if datatype == 'VOC':
                self.img_root = os.path.join(self.root, "JPEGImages")
                # read class_indict
                json_file = os.path.join(self.root, 'pascal_voc_classes.json')
                assert os.path.exists(json_file), 'json file not found'
                with open(json_file, 'r') as fp:
                    self.class_dict = json.load(fp)

            elif datatype == 'VisDrone':
                self.img_root = os.path.join(self.root, 'VisDrone2019-DET-' + runtype + '/images')

            elif datatype == 'COCO':
                self.img_root = os.path.join(self.root, 'val2017')

            elif datatype == 'KITTI':
                self.img_root = os.path.join(self.root, 'training/image_2')

            self.transforms = transforms



        elif datatype == 'VisDrone' and self.traintype == 'clean':
            self.img_root = os.path.join(self.root, 'VisDrone2019-DET-' + runtype + '/images')
            self.annotations_root = os.path.join(self.root, 'VisDrone2019-DET-' + runtype + '/annotations.json')
            self.transforms = transforms
            self.row_targets = json.load(open(self.annotations_root, "r"))

        elif datatype == 'KITTI' and self.traintype == 'clean':
            self.img_root = os.path.join(self.root, 'training/image_2')
            self.annotations_root = os.path.join(self.root, runtype + '_annotations.json')
            self.transforms = transforms
            self.row_targets = json.load(open(self.annotations_root, "r"))

        elif datatype == 'COCO' and self.traintype == 'clean':
            self.img_root = os.path.join(self.root, 'val2017')
            self.transforms = transforms
            self.annotations_root = os.path.join(self.root, 'instances_val2017_' + runtype + '.json')
            self.clean_list = json.load(open(self.annotations_root, "r"))

            self.imagename2targets = {}

            for target in self.clean_list:
                if target['image_name'] not in self.imagename2targets:
                    self.imagename2targets[target['image_name']] = []
                self.imagename2targets[target['image_name']].append(target)
            print('imagename2targets len:', len(self.imagename2targets))
            self.keys_list = list(self.imagename2targets.keys())

    def __getitem__(self, idx):
        boxes, labels, img, iscrowd, img_name = None, None, None, None, None
        if self.datatype == 'VOC' and self.traintype == 'clean':

            xml_path = self.xml_list[idx]
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            img_path = os.path.join(self.img_root, data["filename"])
            img = Image.open(img_path).convert("RGB")
            img_name = data["filename"]
            boxes = []
            labels = []
            iscrowd = []
            for obj in data["object"]:
                boxes.append([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]),
                              int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
                labels.append(self.class_dict[obj["name"]])

                # check if the boxes are valid
                if boxes[-1][2] <= boxes[-1][0] or boxes[-1][3] <= boxes[-1][1]:
                    print(f"INFO: invalid box in {xml_path}, skip this annotation file.")
                    continue

                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
        elif self.traintype == 'dirty':
            img_name = self.keys_list[idx]
            img_path = os.path.join(self.img_root, img_name)
            img = Image.open(img_path).convert("RGB")
            targets = self.imagename2targets[img_name]

            boxes = [x['boxes'] for x in targets]
            labels = [x['labels'] for x in targets]
            iscrowd = [x['iscrowd'] for x in targets]
            fault_type = [x['fault_type'] for x in targets]

            # print(targets)
            # # # draw img with bounding box and labels
            # for i in range(len(boxes)):
            #     plt.gca().add_patch(
            #         plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], fill=False, edgecolor='red',
            #                       linewidth=3))
            #     # reverse class_dict to get class name
            #     # class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(labels[i])]
            #     class_name = str(labels[i])
            #     plt.gca().text(boxes[i][0], boxes[i][1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5),
            #                     fontsize=14, color='white')
            # # plt original image
            # plt.imshow(img)
            # plt.show()

        elif (self.datatype == 'VisDrone' or self.datatype == 'KITTI') and self.traintype == 'clean':

            data = self.row_targets[idx]
            img_name = data["image_name"]
            img_path = os.path.join(self.img_root, data["image_name"])
            img = Image.open(img_path).convert("RGB")
            boxes = data["boxes"]
            labels = data['categories']
            iscrowd = [0] * len(labels)

        elif self.datatype == 'COCO' and self.traintype == 'clean':
            img_name = self.keys_list[idx]
            img_path = os.path.join(self.img_root, img_name)
            img = Image.open(img_path).convert("RGB")
            targets = self.imagename2targets[img_name]

            boxes = [x['boxes'] for x in targets]
            labels = [x['labels'] for x in targets]
            iscrowd = [x['iscrowd'] for x in targets]
            fault_type = [x['fault_type'] for x in targets]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["image_name"] = img_name
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])

        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # print(target)
        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if self.datatype == 'VOC' and self.traintype == 'clean':
            return len(self.xml_list)
        elif self.traintype == 'dirty':
            return len(self.keys_list)
        elif (self.datatype == 'VisDrone' or self.datatype == 'KITTI') and self.traintype == 'clean':
            return len(self.row_targets)
        elif self.datatype == 'COCO':
            return len(self.keys_list)

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

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class inferenceVOCDetectionDataSet(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train.txt"):
        self.root = voc_root
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.transforms = transforms
        text_path = os.path.join(self.root, "ImageSets/Main", txt_name)
        assert os.path.exists(text_path), "file not found"
        with open(text_path, "r") as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        # check file
        for xml_path in xml_list:
            assert os.path.exists(xml_path), "xml file not found"
            with open(xml_path, "r") as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            self.xml_list.append(xml_path)
        # read class_indict
        json_file = os.path.join(self.root, 'pascal_voc_classes.json')
        assert os.path.exists(json_file), 'json file not found'
        with open(json_file, 'r') as fp:
            self.class_dict = json.load(fp)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path, "r") as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            boxes.append([int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"]),
                          int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])])
            labels.append(self.class_dict[obj["name"]])

            # check if the boxes are valid
            if boxes[-1][2] <= boxes[-1][0] or boxes[-1][3] <= boxes[-1][1]:
                print(f"INFO: invalid box in {xml_path}, skip this annotation file.")
                continue

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # # # draw img with bounding box and labels
        # for i in range(len(boxes)):
        #     plt.gca().add_patch(
        #         plt.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1], fill=False, edgecolor='red',
        #                       linewidth=3))
        #     # reverse class_dict to get class name
        #     class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(labels[i])]
        #     plt.gca().text(boxes[i][0], boxes[i][1], '{:s}'.format(class_name), bbox=dict(facecolor='blue', alpha=0.5),
        #                     fontsize=14, color='white')
        # # plt original image
        # plt.imshow(img)
        # plt.show()

        target = {}
        target["image_name"] = data["filename"]
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        # print(target)
        # convert everything into a torch.Tensor
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.xml_list)

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

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# test
if __name__ == "__main__":
    # dataset = DetectionDataSet(root="../dataset/COCO", runtype='train', datatype='COCO', traintype='clean',
    #                            dirtypath='../data/fault_annotations/COCOtrain_mixedfault0.1.json')
    # test inference_VOCinf_classificationDataSet
    # dataset = inference_VOCgtfault_classificationDataSet(voc_root="../dataset/VOCdevkit/VOC2012",
    #                                                      mask_type="crop")

    # "../dataset/VisDrone2019-DET-train/"
    dataset = classificationDataSet(root="../dataset/KITTI", mask_type="other objects",
                                    train_type='clean',
                                    dirty_path='../data/fault_annotations/KITTItrain_mixedfault0.1.json',
                                    datatype='KITTI')

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    # test dataloader for 10 images
    params = parameters()
    fault_type_dict = params.fault_type
    for i, (img, target, idx) in enumerate(dataloader):
        print(idx)
        # # print(target)
        # if i == 100:
        #     break

        # if target[0]["fault type"] == fault_type_dict["class fault"]:
        #     print(target)

    #     # test VOCDetectionDataSet
    # dataset = inferenceVOCDetectionDataSet(voc_root="../dataset/VOCdevkit/VOC2012/", txt_name="train.txt")
    # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    # # test dataloader for 10 images
    # for i, (img, target) in enumerate(dataloader):
    #     print(target)
    #     if i == 10:
    #         break

    # from torchvision import transforms
    # from torch.utils.data import DataLoader
    #
    # data_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    #
    # dataset = VOCclassificationDataSet(voc_root="./dataset/VOCdevkit/VOC2012", transforms=data_transforms, txt_name="train.txt")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    #
    #
    # # test dataloader for 10 images
    # for i, (img, label) in enumerate(dataloader):
    #     if i == 10:
    #         break
    #     print(f"img shape: {img[0].shape}, label: {label[0]}")
