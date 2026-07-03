# smooth L1 Loss computation
import json
import math
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random
from UTILS.parameters import parameters
from UTILS.metric import Metric

from UTILS.FocalLoss import FocalLoss

fault_type_dict = parameters().fault_type
params = parameters()
metric = Metric()

# convert fault_type_dict to number2fault
number2fault = {}
for key in fault_type_dict.keys():
    number2fault[fault_type_dict[key]] = key


class LossBased:
    def __init__(self,
                 config={"dataset": "VOC", "model": "frcnn", "fault_ratio": 0.1, "is_dirty": True, "set": "train",
                         "loss_type": "ce"},
                 missing_dict=None,excel=None):
        print('LossBased\n')
        self.excel=excel
        self.config = config
        self.missing_dict = missing_dict
        self.gt_path = './data/fault_annotations/' + self.config["dataset"] + self.config['set'] + '_mixedfault' + str(
            self.config["fault_ratio"]) + '.json'
        det_name_ = lambda d, m, f, x: m + 'dirty' + str(
            f) + '_' + d + self.config['set'] + '_inferences.json' if x else m + 'clean_' + d + '_inferences.json'
        self.det_path = './data/detection_results/' + det_name_(self.config["dataset"], self.config["model"],
                                                                self.config["fault_ratio"], self.config["is_dirty"])

        self.losstype = self.config['loss_type']
        print(self.gt_path, self.det_path)
        with open(self.gt_path, 'r') as f:
            self.gt = json.load(f)
        with open(self.det_path, 'r') as f:
            self.det = json.load(f)
        self.det = [i for i in self.det if i["score"] > params.m_t]

        fault_num = {
            'no fault': 0,
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
            'findable missing fault': 0,
        }
        with open(self.gt_path, 'r') as f:
            gt = json.load(f)
        for i in gt:
            fault_num[number2fault[i["fault_type"]]] += 1

        self.fault_num = fault_num

        # transform dec to {imagename:[]} format dict
        self.dec_dict = {}
        for i in range(len(self.det)):
            if self.det[i]["image_name"] in self.dec_dict:
                self.dec_dict[self.det[i]["image_name"]].append(self.det[i])
            else:
                self.dec_dict[self.det[i]["image_name"]] = [self.det[i]]
        self.imagename2boxes = {}
        for instance in self.gt:
            if instance["fault_type"] != fault_type_dict['missing fault']:
                if instance["image_name"] not in self.imagename2boxes:
                    self.imagename2boxes[instance["image_name"]] = []
                self.imagename2boxes[instance["image_name"]].append(instance["boxes"])

    def run(self):
        start_time = time.time()
        results = []
        for i in range(len(self.gt)):
            # print progress bar
            print('\r', 'progress: ', i, '/', len(self.gt), end='')

            if self.gt[i]['fault_type'] != fault_type_dict['missing fault'] and self.gt[i][
                "image_name"] in self.dec_dict:
                # loss = min loss of all decs
                min_loss = 100000

                # decious=[self.dec_dict[self.gt[i]["image_name"]][j]['bbox'] for j in range(len(self.dec_dict[self.gt[i]["image_name"]]))]
                #
                # IoUs = metric.cal_IoU([self.gt[i]["boxes"]], decious)
                #
                # max_iou_index = torch.argmax(IoUs, dim=1).item()
                #
                # min_loss = self.compute_loss(self.dec_dict[self.gt[i]["image_name"]][max_iou_index]['full_scores'],
                #                                 self.gt[i]['labels'],
                #                                 self.dec_dict[self.gt[i]["image_name"]][max_iou_index]['bbox'],
                #                                 self.gt[i]['boxes'])

                for j in range(len(self.dec_dict[self.gt[i]["image_name"]])):

                    loss = self.compute_loss(self.dec_dict[self.gt[i]["image_name"]][j]["full_scores"],
                                             self.gt[i]["labels"],
                                             self.dec_dict[self.gt[i]["image_name"]][j]["bbox"],
                                             self.gt[i]["boxes"], self.losstype)
                    if loss < min_loss:
                        min_loss = loss

                results.append({"loss": min_loss, "fault_type": self.gt[i]["fault_type"], 'detectiongt_category_id': -1,
                                'image_name': self.gt[i]["image_name"]})

        # add len(self.dec_dict) to results

        # random shuffle self.gt_image_names
        self.gt_image_names = [i for i in self.imagename2boxes.keys()]
        random.shuffle(self.gt_image_names)

        for name in self.gt_image_names:
            if name in self.missing_dict:
                results.append({"loss": 0, "fault_type": fault_type_dict['missing fault'], 'detectiongt_category_id': 0,
                                'image_name': name})
            else:
                results.append({"loss": 0, "fault_type": fault_type_dict['no fault'], 'detectiongt_category_id': -1,
                                'image_name': name})

        # sort results by loss from large to small
        results.sort(key=lambda x: x["loss"], reverse=True)
        end_time = time.time()
        print(self.losstype + " loss time: ", end_time - start_time)
        # X = [i for i in range(len(results))]
        # Y = [0 for i in range(len(results))]
        # # print(results)
        # fault_t = []
        #
        # flag_list = []
        print(metric.APFD(results))
        EXAM_F, EXAM_F_rel, Top_1, Top_3 = metric.EXAM_F(results)

        col_offset = None
        if self.config['loss_type'] == 'ce':
            col_offset = 2
        elif self.config['loss_type'] == 'focal':
            col_offset = 3
        self.excel.run([EXAM_F_rel, EXAM_F, Top_1, Top_3], [0, 12, 24, 36], col_offset)
        print('lossbased EXAM_F: ', EXAM_F)
        print('lossbased EXAM_F_rel: ', EXAM_F_rel)
        print('lossbased Top_1: ', Top_1)
        print('lossbased Top_3: ', Top_3)
        # for i in range(len(results)):
        #
        #     if results[i]['fault_type'] != fault_type_dict['no fault']:
        #         Y[i] = Y[i - 1] + 1
        #         fault_t.append(results[i]['fault_type'])
        #     else:
        #         Y[i] = Y[i - 1]
        #         fault_t.append(fault_type_dict["no fault"])
        #
        # # different color for different fault type
        # color_dict = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w'}
        #
        # # delete no fault
        # for i in range(len(fault_t)):
        #     if fault_t[i] == fault_type_dict["no fault"]:
        #         fault_t[i] = -1
        # XX = [X[i] for i in range(len(X)) if fault_t[i] != -1]
        # YY = [Y[i] for i in range(len(Y)) if fault_t[i] != -1]
        # fault_t = [i for i in fault_t if i != -1]
        #
        # plt.plot(X, Y, color='b')
        # # plt.scatter(XX, YY, c=[color_dict[i] for i in fault_t])
        # # plt.plot([0, len(results)], [0, len(fault_t)], color='r')
        # plt.show()
        # # color legend
        # for i in range(len(fault_type_dict)):
        #     plt.scatter([], [], c=color_dict[i], label=number2fault[i])

        return results
        # plt.legend()
        #
        # plt.show()

        # print('\n')
        # metric.RAUC(results, self.fault_num, rauc_num=params.rauc_num)
        #
        # print(len(results))
        # # get top params.vocfrcnn_FaultSet_length results
        # if self.detective_cofig == 'vocfrcnn':
        #     results = results[:params.vocfrcnn_FaultSet_length]
        #     fault_ratio, fault_inclusiveness = metric.RateAndInclusiveness(results, self.fault_num)
        #     print('fault_ratio: ', fault_ratio)
        #     print('fault_inclusiveness: ', fault_inclusiveness)

    def compute_loss(self, full_scores, label, box_pre, box_gt, loss_type):

        if loss_type == 'ce':
            return F.cross_entropy(torch.tensor(full_scores), torch.tensor(label)).item() \
                + F.smooth_l1_loss(torch.tensor(box_pre), torch.tensor(box_gt)).item()

        if loss_type == 'focal':
            return FocalLoss(gamma=2)(torch.tensor(full_scores), torch.tensor(label)).item() \
                + F.smooth_l1_loss(torch.tensor(box_pre), torch.tensor(box_gt)).item()


if __name__ == "__main__":
    loss_based = LossBased()
    loss_based.run()
