import datetime
import json
import os
import random
import time
from scipy import stats
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from AlBased import ALBased
from LossBased import LossBased
from UTILS.parameters import parameters
from torchvision.ops import boxes as box_ops

from UTILS.metric import Metric
from b_cleanlab import CleanLab

params = parameters()
fault_type_dict = parameters().fault_type

metric = Metric()
# convert fault_type_dict to number2fault
number2fault = {}
for key in fault_type_dict.keys():
    number2fault[fault_type_dict[key]] = key

from UTILS.write_excel import WRITE_EXCEL

excel = WRITE_EXCEL("idea2_detective.xlsx")

DATA_NAME = "KITTI"
SET_TYPE = "test"
MODEL_NAME = "frcnn"
LOSS_TYPE = ""
AL_TYPE = ""
METHOD = "cleanlab"


class FaultDetective:
    def __init__(
        self,
        config={
            "dataset": DATA_NAME,
            "fault_ratio": 0.1,
            "is_dirty": True,
            "is_LNL": True,
            "set": SET_TYPE,
        },
        loss_based_config={
            "dataset": DATA_NAME,
            "model": MODEL_NAME,
            "fault_ratio": 0.1,
            "is_dirty": True,
            "set": SET_TYPE,
            "loss_type": LOSS_TYPE,
        },
        al_based_config={
            "dataset": DATA_NAME,
            "model": MODEL_NAME,
            "fault_ratio": 0.1,
            "is_dirty": True,
            "set": SET_TYPE,
            "al_type": AL_TYPE,
        },
        cleanlab_based_config={
            "dataset": DATA_NAME,
            "model": MODEL_NAME,
            "fault_ratio": 0.1,
            "is_dirty": True,
            "set": SET_TYPE,
        },
    ):
        self.config = config
        self.dataset = config["dataset"]
        self.setname = config["set"]
        self.model_name = loss_based_config["model"]
        self.loss_type = loss_based_config["loss_type"]
        self.al_type = al_based_config["al_type"]
        print("FaultDetective init\n")
        namer_ = (
            lambda x: "./data/classification_results/"
            + x
            + ("_dirty" if config["is_dirty"] else "_clean")
            + ("_LNL" if config["is_LNL"] else "")
            + "_classification_bs="
            + ("64_" if (config["dataset"] == "VisDrone" and x == "crop") else "32_")
            + config["dataset"]
            + config["set"]
            + "mixedfault"
            + str(config["fault_ratio"])
            + "_inferences.json"
        )
        self.mask_others_path = namer_("mask_others")
        self.crop_path = namer_("crop")
        # self.mask_all_path = namer_('mask_all')
        self.gt_path = "./data/fault_annotations/" + config["dataset"] + config["set"] + "_mixedfault" + str(config["fault_ratio"]) + ".json"
        self.loss_based_config = loss_based_config
        self.al_based_config = al_based_config
        self.cleanlab_based_config = cleanlab_based_config
        print(self.mask_others_path, self.crop_path, self.gt_path)
        self.max_fault_num = {
            "no fault": 0,
            "class fault": 0,
            "location fault": 0,
            "redundancy fault": 0,
            "missing fault": 0,
            "findable missing fault": 0,
        }

        with open(self.mask_others_path, "r") as f:
            self.mask_others_list = json.load(f)

        with open(self.crop_path, "r") as f:
            self.crop_list = json.load(f)

        # with open(self.mask_all_path, 'r') as f:
        #     self.mask_all_list = json.load(f)

        with open(self.gt_path, "r") as f:
            fullgt_list = json.load(f)  # with missing fault

        for i in range(len(fullgt_list)):
            if fullgt_list[i]["fault_type"] == fault_type_dict["class fault"]:
                self.max_fault_num["class fault"] += 1
            elif fullgt_list[i]["fault_type"] == fault_type_dict["location fault"]:
                self.max_fault_num["location fault"] += 1
            elif fullgt_list[i]["fault_type"] == fault_type_dict["redundancy fault"]:
                self.max_fault_num["redundancy fault"] += 1
            elif fullgt_list[i]["fault_type"] == fault_type_dict["missing fault"]:
                self.max_fault_num["missing fault"] += 1
            elif fullgt_list[i]["fault_type"] == fault_type_dict["no fault"]:
                self.max_fault_num["no fault"] += 1

        loss_func = torch.nn.CrossEntropyLoss()
        for i in range(len(self.crop_list)):
            scores = self.crop_list[i]["full_scores"]
            label = self.crop_list[i]["detectiongt_category_id"]

            loss = loss_func(torch.tensor([scores]), torch.tensor([label]))
            self.crop_list[i]["loss"] = loss.item()

        # mutiple loss
        if config["dataset"] == "VOC" or (config["dataset"] == "VisDrone" and config["set"] == "test"):
            for i in range(len(self.mask_others_list)):
                #
                # assert self.mask_others_list[i]['image_name'] == self.mask_all_list[i]['image_name']
                if self.mask_others_list[i]["detectiongt_category_id"] != 0:
                    assert self.mask_others_list[i]["image_name"] == self.crop_list[i]["image_name"]
                    self.mask_others_list[i]["loss"] = self.crop_list[i]["loss"]
        else:
            self.crop_list.extend(self.mask_others_list)
            self.mask_others_list = self.crop_list

        # process fullgt_list
        gt_list = []  # without missing fault
        self.missing_list = []  # only missing fault
        for i in range(len(fullgt_list)):
            if fullgt_list[i]["fault_type"] != fault_type_dict["missing fault"]:
                gt_list.append(fullgt_list[i])
            else:
                self.missing_list.append(fullgt_list[i])

        self.missing_dict = {}
        for i in range(len(self.missing_list)):
            if self.missing_list[i]["image_name"] in self.missing_dict:
                self.missing_dict[self.missing_list[i]["image_name"]].append(self.missing_list[i])
            else:
                self.missing_dict[self.missing_list[i]["image_name"]] = [self.missing_list[i]]

    def load_json(self, json_file):
        with open(json_file, "r") as fp:
            data = json.load(fp)
        return data

    def get_retrain_data(self, method="ours"):
        res = None
        loss_name = "loss"
        if method == "ours":
            res = self.mask_others_list
        elif method == "lossbased":
            loss_based = LossBased(self.loss_based_config, self.missing_dict, excel=excel)
            res = loss_based.run(early_return=True)
            res = [i for i in res if i["fault_type"] != fault_type_dict["missing fault"]]
            self.mask_others_list = None
        elif method == "albased":
            al_based = ALBased(self.al_based_config, excel=excel)
            res = al_based.run(early_return=True)
            self.mask_others_list = None
            loss_name = "score"
        elif method == "cleanlab":
            cleanlab = CleanLab(self.cleanlab_based_config, excel=excel)
            res = cleanlab.run(early_return=True)
            # make all score in res to negative
            for i in range(len(res)):
                res[i]["score"] = -res[i]["score"]
            self.mask_others_list = None
            loss_name = "score"

        dirty_data = self.load_json(
            "./data/fault_annotations/" + self.config["dataset"] + self.config["set"] + "_mixedfault" + str(self.config["fault_ratio"]) + ".json"
        )
        # argsort mask_others_list by loss in descending order
        print(len(res))
        tmp = [i.copy() for i in res]
        results = sorted(tmp, key=lambda x: x[loss_name], reverse=True)
        cnt = 0
        for i in range(int(len(results) * 0.4)):
            if results[i]["fault_type"] != fault_type_dict["no fault"]:
                cnt += 1
        print("0.4 results:" + str(int(len(results) * 0.4)))
        clean_data = self.load_json("./data/fault_annotations/" + self.config["dataset"] + self.config["set"] + "_mixedfault" + str(0.0) + ".json")
        count = 0
        incert_index = 0
        len_orgclean = len(clean_data)
        len_orgmasklist = len(res)
        # 对齐
        while incert_index < len(dirty_data):
            if dirty_data[incert_index]["fault_type"] == fault_type_dict["redundancy fault"]:
                # incert dirty_data[incert_index] to clean_data after the index of incert_index
                clean_data.insert(incert_index, dirty_data[incert_index])
            if dirty_data[incert_index]["fault_type"] == fault_type_dict["missing fault"] and method != "albased" and method != "cleanlab":
                res.insert(
                    incert_index,
                    {
                        "image_name": dirty_data[incert_index]["image_name"],
                        "full_scores": [],
                        "detectiongt_category_id": 4,
                        "bbox": [],
                        "fault_type": 0,
                        loss_name: -100,
                    },
                )
            assert dirty_data[incert_index]["image_name"] == clean_data[incert_index]["image_name"]
            assert dirty_data[incert_index]["image_name"] == res[incert_index]["image_name"]
            incert_index += 1

        assert len(dirty_data) == len(clean_data)
        # 插入了miss fault的 res 的 排序
        sorted_index = np.argsort([i[loss_name] for i in res])[::-1]
        # 代价之前的排序
        filter_index = sorted_index[: int(len_orgmasklist * 0.4)]
        # 代价
        print("0.4 lenorg" + str(int(len_orgmasklist * 0.4)))
        print(len(filter_index))

        del_list = []
        pp_cnt = 0
        is_mis_list = []
        for index in filter_index:
            if res[index]["fault_type"] != fault_type_dict["no fault"]:
                pp_cnt += 1
            if res[index]["detectiongt_category_id"] == 0:
                is_mis_list.append(res[index]["image_name"])

            else:
                assert dirty_data[index]["image_name"] == res[index]["image_name"]
                assert dirty_data[index]["image_name"] == clean_data[index]["image_name"]
                if dirty_data[index]["fault_type"] == fault_type_dict["redundancy fault"]:
                    count += 1
                    del_list.append(index)
                else:
                    dirty_data[index] = clean_data[index]
                    if dirty_data[index]["fault_type"] != fault_type_dict["no fault"]:
                        count += 1
        for i in range(len(dirty_data)):
            if dirty_data[i]["image_name"] in is_mis_list and dirty_data[i]["fault_type"] == fault_type_dict["missing fault"]:
                count += 1
                dirty_data[i]["fault_type"] = fault_type_dict["no fault"]
        print("fliter:", count)
        for i in reversed(sorted(del_list)):
            # delete index i in dirty_data
            del dirty_data[i]
        print(pp_cnt, cnt)
        # assert pp_cnt == cnt
        json_str = json.dumps(dirty_data, indent=4)
        filename = None
        if method == "cleanlab":
            filename = "cleanlab"
        else:
            filename = self.loss_based_config["loss_type"] if method == "lossbased" else self.al_based_config["al_type"]
        file_path = "./data/fault_annotations/" + self.config["dataset"] + self.config["set"] + filename + "_mixedfault_retrain.json"
        with open(file_path, "w") as json_file:
            json_file.write(json_str)

        print("get_retrain_data done")

    def run(self):
        start_time = time.time()
        # sort self.c_list by self.c_list[i]['loss']
        tmp = [i.copy() for i in self.mask_others_list]
        results = sorted(tmp, key=lambda x: x["loss"], reverse=True)

        end_time = time.time()
        print("ours time: ", end_time - start_time)

        X = [i for i in range(len(results))]
        Y = [0 for i in range(len(results))]
        # print(results)
        fault_t = []

        flag_list = []

        Stacked_line_chart = {
            "cls": [0 for i in range(len(results))],
            "cls+loc": [0 for i in range(len(results))],
            "cls+loc+red": [0 for i in range(len(results))],
            "cls+loc+red+mis": [0 for i in range(len(results))],
        }

        for i in range(len(results)):
            fault_ = "no"
            if int(results[i]["detectiongt_category_id"]) == 0 and results[i]["image_name"] in self.missing_dict:
                # 该实例是背景
                results[i]["fault_type"] = fault_type_dict["missing fault"]
                Y[i] = Y[i - 1] + 1
                fault_ = "mis"
                for key in Stacked_line_chart.keys():
                    if "mis" in key:
                        Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1
                fault_t.append(fault_type_dict["missing fault"])
                # box = results[i]["bbox"]
                # image_name = results[i]['image_name']
                # missing_boxes = [j["boxes"] for j in self.missing_dict[image_name]]
                # missing_max_IoU = torch.max(metric.cal_IoU([box], missing_boxes))
                #
                # if missing_max_IoU > params.missing_threshold:
                #     results[i]['fault_type'] = fault_type_dict['missing fault']
                #     Y[i] = Y[i - 1] + 1
                #     fault_t.append(fault_type_dict['missing fault'])
                # else:
                #     Y[i] = Y[i - 1]
                #     fault_t.append(fault_type_dict["no fault"])
            elif results[i]["fault_type"] != fault_type_dict["no fault"] and int(results[i]["detectiongt_category_id"]) != 0:
                if results[i]["fault_type"] == fault_type_dict["class fault"]:
                    fault_ = "cls"
                    for key in Stacked_line_chart.keys():
                        if "cls" in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1
                elif results[i]["fault_type"] == fault_type_dict["location fault"]:
                    fault_ = "loc"
                    for key in Stacked_line_chart.keys():
                        if "loc" in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1
                elif results[i]["fault_type"] == fault_type_dict["redundancy fault"]:
                    fault_ = "red"
                    for key in Stacked_line_chart.keys():
                        if "red" in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1

                Y[i] = Y[i - 1] + 1
                fault_t.append(results[i]["fault_type"])
            else:
                Y[i] = Y[i - 1]
                fault_t.append(fault_type_dict["no fault"])

            for key in Stacked_line_chart.keys():
                if fault_ not in key:
                    Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1]
        APFD = metric.APFD(results)
        print("ours: ", APFD)
        EXAM_F, EXAM_F_rel, Top_1, Top_3 = metric.EXAM_F(results)
        excel.run([EXAM_F_rel, EXAM_F, Top_1, Top_3], [0, 12, 24, 36], 1)
        # excel.run([APFD],[0],1)
        print("ours EXAM_F: ", EXAM_F)
        print("ours EXAM_F_rel: ", EXAM_F_rel)
        print("ours Top_1: ", Top_1)
        print("ours Top_3: ", Top_3)

        # color_dict = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w'}

        # delete no fault
        for i in range(len(fault_t)):
            if fault_t[i] == fault_type_dict["no fault"]:
                fault_t[i] = -1
        # XX = [X[i] for i in range(len(X)) if fault_t[i] != -1]
        # YY = [Y[i] for i in range(len(Y)) if fault_t[i] != -1]
        # fault_t = [i for i in fault_t if i != -1]
        #
        # plt.plot(X, Y, color='g')
        # # plt.scatter(XX, YY, c=[color_dict[i] for i in fault_t])
        # plt.plot([0, len(results)], [0, len(fault_t)], color='r')
        #
        # # color legend
        # for i in range(len(fault_type_dict)):
        #     plt.scatter([], [], c=color_dict[i], label=number2fault[i])

        # loss_based = LossBased(self.loss_based_config, self.missing_dict, excel=excel)
        # results_lossbased = loss_based.run()
        # #
        al_based = ALBased(self.al_based_config, excel=excel)
        results_albased = al_based.run()

        Clean_Lab = CleanLab(self.cleanlab_based_config, excel=excel)
        results_cleanlab = Clean_Lab.run()

        # plt.legend()
        max_fault = (
            self.max_fault_num["class fault"]
            + self.max_fault_num["location fault"]
            + self.max_fault_num["redundancy fault"]
            + self.max_fault_num["missing fault"]
        )
        print("max_fault: ", max_fault)

        # delete missing fault in results_lossbased
        # results_lossbased = [
        #     i
        #     for i in results_lossbased
        #     if i["fault_type"] != fault_type_dict["missing fault"]
        # ]

        # plt1 = metric.plt_stack_line_chart(
        #     results_lossbased,
        #     missing_dict=self.missing_dict,
        #     x_max=len(results) * 1.1,
        #     y_max=len([i for i in fault_t if i != -1]) * 1.1,
        #     max_fault=max_fault,
        # )
        # plt1.savefig(
        #     "./pictures/lossbased/"
        #     + self.dataset
        #     + "_"
        #     + self.setname
        #     + "_"
        #     + self.loss_type
        #     + "_"
        #     + self.model_name
        #     + ".pdf",
        #     bbox_inches="tight",
        # )
        # plt1.close()
        # plt2 = metric.plt_stack_line_chart(
        #     results_albased,
        #     missing_dict=self.missing_dict,
        #     x_max=len(results) * 1.1,
        #     y_max=len([i for i in fault_t if i != -1]) * 1.1,
        #     max_fault=max_fault,
        # )
        # plt2.savefig(
        #     "./pictures/albased/"
        #     + self.dataset
        #     + "_"
        #     + self.setname
        #     + "_"
        #     + self.al_type
        #     + "_"
        #     + self.model_name
        #     + ".pdf",
        #     bbox_inches="tight",
        # )
        # plt2.close()
        # plt3 = metric.plt_stack_line_chart(
        #     results,
        #     missing_dict=self.missing_dict,
        #     x_max=len(results) * 1.1,
        #     y_max=max_fault * 1.1,
        #     max_fault=max_fault,
        # )
        # plt3.savefig(
        #     "./pictures/ours/" + self.dataset + "_" + self.setname + ".pdf",
        #     bbox_inches="tight",
        # )
        # plt3.close()

        plt4 = metric.plt_stack_line_chart(
            results_cleanlab,
            missing_dict=self.missing_dict,
            x_max=len(results) * 1.1,
            y_max=len([i for i in fault_t if i != -1]) * 1.1,
            max_fault=max_fault,
        )
        plt4.savefig(
            "./pictures/cleanlab/" + self.dataset + "_" + self.setname + "_" + self.model_name + ".pdf",
            bbox_inches="tight",
        )
        plt4.close()


if __name__ == "__main__":
    detective = FaultDetective()
    # detective.run()
    detective.get_retrain_data(METHOD)
