import os

import torch


class parameters:
    def __init__(self):
        self.fault_type = {
            'no fault': 0,
            'class fault': 1,
            'location fault': 2,
            'redundancy fault': 3,
            'missing fault': 4,
        }
        self.fault_ratio = 0.1  # 0.05
        self.m_t = 0.5
        self.t_b = 0.1
        self.t_f = 0.5
        self.t_p = 0.5
        self.missing_threshold = 0.5  # if IoU > missing_threshold, then it is a missing fault
        self.rauc_num = 500
        self.adaptive_epoch = 17
        self.adaptive_batchsize = 200

        self.vocfrcnn_FaultSet_length = 9059

        self.coco_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
                           28, 31, 32, 33,
                           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           58, 59, 60, 61,
                           62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,
                           90]


# test
if __name__ == "__main__":
    para = parameters()
    print(para.m_t)
