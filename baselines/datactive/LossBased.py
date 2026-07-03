                            
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
                                
            print('\r', 'progress: ', i, '/', len(self.gt), end='')

            if self.gt[i]['fault_type'] != fault_type_dict['missing fault'] and self.gt[i][
                "image_name"] in self.dec_dict:
                                             
                min_loss = 100000

                                                                                                                                           
                 
                                                                       
                 
                                                                  
                 
                                                                                                                     
                                                                       
                                                                                                                 
                                                                      

                for j in range(len(self.dec_dict[self.gt[i]["image_name"]])):

                    loss = self.compute_loss(self.dec_dict[self.gt[i]["image_name"]][j]["full_scores"],
                                             self.gt[i]["labels"],
                                             self.dec_dict[self.gt[i]["image_name"]][j]["bbox"],
                                             self.gt[i]["boxes"], self.losstype)
                    if loss < min_loss:
                        min_loss = loss

                results.append({"loss": min_loss, "fault_type": self.gt[i]["fault_type"], 'detectiongt_category_id': -1,
                                'image_name': self.gt[i]["image_name"]})

                                           

                                            
        self.gt_image_names = [i for i in self.imagename2boxes.keys()]
        random.shuffle(self.gt_image_names)

        for name in self.gt_image_names:
            if name in self.missing_dict:
                results.append({"loss": 0, "fault_type": fault_type_dict['missing fault'], 'detectiongt_category_id': 0,
                                'image_name': name})
            else:
                results.append({"loss": 0, "fault_type": fault_type_dict['no fault'], 'detectiongt_category_id': -1,
                                'image_name': name})

                                                  
        results.sort(key=lambda x: x["loss"], reverse=True)
        end_time = time.time()
        print(self.losstype + " loss time: ", end_time - start_time)
                                              
                                              
                          
                      
         
                        
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
                                       
         
                                                                         
                                     
                                                          
                   
                                 
                                                             
         
                                                    
                                                                                       
         
                           
                                       
                                                           
                                 
                                                                
                                                                
                                                   
         
                                   
                                                                   
                                                                     
                    
                        
                                               
                                                                         

        return results
                      
         
                    

                     
                                                                        
         
                             
                                                           
                                                
                                                                 
                                                                                                     
                                                 
                                                                 

    def compute_loss(self, full_scores, label, box_pre, box_gt, loss_type):

        if loss_type == 'ce':
            return F.cross_entropy(torch.tensor(full_scores), torch.tensor(label)).item()\
                + F.smooth_l1_loss(torch.tensor(box_pre), torch.tensor(box_gt)).item()

        if loss_type == 'focal':
            return FocalLoss(gamma=2)(torch.tensor(full_scores), torch.tensor(label)).item()\
                + F.smooth_l1_loss(torch.tensor(box_pre), torch.tensor(box_gt)).item()


if __name__ == "__main__":
    loss_based = LossBased()
    loss_based.run()
