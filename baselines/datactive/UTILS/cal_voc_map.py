import json

import torch

from UTILS.coco_eval import CocoEvaluator
from UTILS.coco_utils import get_coco_api_from_dataset
from UTILS.engine import _get_iou_types


def cal_voc_map(model, results, val_dataset):
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # instances results to image results
    image_results = {}
    for result in results:
        image_id = result["image_id"]
        if image_id not in image_results:
            image_results[image_id] = {'boxes': torch.Tensor([]), 'labels': torch.Tensor([]),
                                       'scores': torch.Tensor([])}
        image_results[image_id]['boxes'] = torch.cat((image_results[image_id]['boxes'], torch.Tensor([result['bbox']])), dim=0)
        image_results[image_id]['labels'] = torch.cat((image_results[image_id]['labels'], torch.Tensor([result['category_id']])), dim=0)
        image_results[image_id]['scores'] = torch.cat((image_results[image_id]['scores'], torch.Tensor([result['score']])), dim=0)

    coco_evaluator.update(image_results)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator





