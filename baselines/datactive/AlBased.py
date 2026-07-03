import json
import time

import numpy as np
import torch

from UTILS.metric import Metric

metric = Metric()
from UTILS.parameters import parameters

params = parameters()
fault_type_dict = parameters().fault_type


class ALBased:
    def __init__(self,
                 config={"dataset": "VOC", "model": "frcnn", "fault_ratio": 0.1, "is_dirty": True, "set": "val",
                         "al_type": "entrtopy"},
                 missing_dict= None,excel=None):
        self.excel = excel
        self.config = config
        self.al_type = self.config['al_type']
        self.gt_path = './data/fault_annotations/' + self.config["dataset"] + self.config['set'] + '_mixedfault' + str(
            self.config["fault_ratio"]) + '.json'
        det_name_ = lambda d, m, f, x: m + 'dirty' + str(
            f) + '_' + d + self.config['set'] + '_inferences.json' if x else m + 'clean_' + d + '_inferences.json'
        self.det_path = './data/detection_results/' + det_name_(self.config["dataset"], self.config["model"],
                                                                self.config["fault_ratio"], self.config["is_dirty"])

        print(self.gt_path, self.det_path)
        with open(self.gt_path, 'r') as f:
            self.gt = json.load(f)
        with open(self.det_path, 'r') as f:
            self.det = json.load(f)
        self.det = [i for i in self.det if i["score"] > params.m_t]

    def run(self):

        start_time = time.time()

        result = []
        if self.al_type == 'entropy':
            for instance in self.det:
                instance['entropy'] = self.compute_entropy(instance['full_scores'])

        elif self.al_type == 'margin':
            for instance in self.det:
                instance['margin'] = self.compute_margin(instance['full_scores'])

        elif self.al_type == 'gini':
            for instance in self.det:
                instance['gini'] = self.compute_gini(instance['full_scores'])

        # transform dec to {imagename:[]} format dict
        self.dec_dict = {}
        for i in range(len(self.det)):
            if self.det[i]["image_name"] in self.dec_dict:
                self.dec_dict[self.det[i]["image_name"]].append(self.det[i])
            else:
                self.dec_dict[self.det[i]["image_name"]] = [self.det[i]]

        for i in range(len(self.gt)):
            print('\r', 'progress: ', i, '/', len(self.gt), end='')
            if self.gt[i]["image_name"] in self.dec_dict:
                dec_boxes = [self.dec_dict[self.gt[i]["image_name"]][j]['bbox'] for j in
                             range(len(self.dec_dict[self.gt[i]["image_name"]]))]

                IoUs = metric.cal_IoU([self.gt[i]["boxes"]], dec_boxes)

                max_iou_index = torch.argmax(IoUs, dim=1).item()

                score = self.dec_dict[self.gt[i]["image_name"]][max_iou_index][self.al_type]

                result.append({"score": score, "fault_type": self.gt[i]["fault_type"],
                               'detectiongt_category_id': 0 if self.gt[i]["fault_type"] == fault_type_dict[
                                   'missing fault'] else -1, 'image_name': self.gt[i]["image_name"]})

        result.sort(key=lambda x: x["score"], reverse=True)

        end_time = time.time()

        print(self.al_type + " al time: ", end_time - start_time)
        print(metric.APFD(result))

        EXAM_F, EXAM_F_rel, Top_1, Top_3 = metric.EXAM_F(result)
        col_offset = None
        if self.config['al_type']=='entropy':
            col_offset = 4
        elif self.config['al_type']=='margin':
            col_offset = 5
        elif self.config['al_type']=='gini':
            col_offset = 6
        self.excel.run([EXAM_F_rel, EXAM_F, Top_1, Top_3], [0, 12, 24, 36], col_offset)
        print('albased EXAM_F: ', EXAM_F)
        print('albased EXAM_F_rel: ', EXAM_F_rel)
        print('albased Top_1: ', Top_1)
        print('albased Top_3: ', Top_3)

        return result

    def compute_entropy(self, prob_list):
        entropy = sum([-p * np.log(p) for p in prob_list])
        return entropy

    def compute_margin(self, prob_list):
        prob_list = sorted(prob_list)

        def one_two_(x):
            if x[0] < 0:
                return 0
            return (1 - (x[0] - x[1])) ** 2

        # get first two max prob in prob_list(unsorted)
        first_two = sorted(prob_list, reverse=True)[:2]

        margin = one_two_(first_two)

        return margin

    def compute_gini(self, prob_list):
        gini = 1 - sum([p ** 2 for p in prob_list])
        return gini
