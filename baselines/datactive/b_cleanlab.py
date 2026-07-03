import json

import numpy as np
from cleanlab.object_detection.rank import get_label_quality_scores
from UTILS.metric import Metric

from UTILS.mydataset import DetectionDataSet
from UTILS import presets
from UTILS.parameters import parameters

metric = Metric()
params = parameters()
fault_type_dict = parameters().fault_type


class CleanLab:
    def __init__(self, config=None, excel=None):
        self.excel = excel
        if config is None:
            config = {
                "dataset": "VOC",
                "model": "frcnn",
                "fault_ratio": 0.1,
                "is_dirty": True,
                "set": "val",
            }
        root_path = "./data/fault_annotations/" + config["dataset"] + config["set"] + "_mixedfault0.1.json"
        self.gt = self.load_json(root_path)
        data = self.load_json(root_path)
        # transform dirtylist to {imagename:[]} format dict
        imagename2targets = {}
        for target in data:
            if target["image_name"] not in imagename2targets:
                imagename2targets[target["image_name"]] = []
            if target["fault_type"] != fault_type_dict["missing fault"]:  # missing is not a target
                imagename2targets[target["image_name"]].append(target)
        print("imagename2targets len:", len(imagename2targets))
        names_list = list(imagename2targets.keys())
        Labels = []
        label_length = 0
        for image_name in names_list:
            bboxes = []
            labels = []
            for target in imagename2targets[image_name]:
                bboxes.append(target["boxes"])
                labels.append(target["labels"])
                label_length = max(label_length, target["labels"])
            Labels.append(
                {
                    "bboxes": np.array(bboxes, dtype=np.float32),
                    "labels": np.array(labels),
                    "image_name": image_name,
                }
            )
        label_length += 1
        print("label_length:", label_length)

        pre_path = "./data/detection_results/" + config["model"] + "dirty0.1_" + config["dataset"] + config["set"] + "_inferences.json"
        pre_data = self.load_json(pre_path)
        # transform dirtylist to {imagename:[]} format dict
        pre_imagename2targets = {}
        for target in pre_data:
            if target["image_name"] not in pre_imagename2targets:
                pre_imagename2targets[target["image_name"]] = []
            pre_imagename2targets[target["image_name"]].append(target)

        # pre_names_list = list(pre_imagename2targets.keys())
        # assert len(pre_names_list) == len(names_list)
        predictions = []
        for image_name in names_list:
            pred = [[] for _ in range(label_length)]
            if image_name in pre_imagename2targets:
                for target in pre_imagename2targets[image_name]:
                    pred[target["category_id"]].append([*target["bbox"], target["score"]])
            pred_final = []
            for p in pred:
                # convert to numpy array
                # convert to : array([], shape=(0, 5), dtype=float32)
                tmp = np.zeros((len(p), 5))
                for i in range(len(p)):
                    tmp[i] = p[i]
                p = np.array(tmp, dtype=np.float32)
                pred_final.append(p)
            predictions.append(np.array(pred_final, dtype=object))

        self.names_list = names_list
        self.Labels = Labels
        self.predictions = predictions

    def load_json(self, json_file):
        with open(json_file, "r") as fp:
            data = json.load(fp)
        return data

    def run(self, early_return=False):
        scores = get_label_quality_scores(self.Labels, self.predictions)
        results = []
        name2score = {self.names_list[i]: scores[i] for i in range(len(self.names_list))}
        for i in range(len(self.gt)):
            results.append(
                {
                    "score": name2score[self.gt[i]["image_name"]],
                    "fault_type": self.gt[i]["fault_type"],
                    "detectiongt_category_id": (0 if self.gt[i]["fault_type"] == fault_type_dict["missing fault"] else -1),
                    "image_name": self.gt[i]["image_name"],
                }
            )
        if early_return:
            return results
        results.sort(key=lambda x: x["score"])

        print(metric.APFD(results))
        EXAM_F, EXAM_F_rel, Top_1, Top_3 = metric.EXAM_F(results)
        col_offset = 7
        self.excel.run([EXAM_F_rel, EXAM_F, Top_1, Top_3], [0, 12, 24, 36], col_offset)
        print("cleanlab EXAM_F: ", EXAM_F)
        print("cleanlab EXAM_F_rel: ", EXAM_F_rel)
        print("cleanlab Top_1: ", Top_1)
        print("cleanlab Top_3: ", Top_3)

        return results
