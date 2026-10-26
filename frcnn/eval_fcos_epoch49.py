"""用项目中的 VisDrone COCO 格式验证集评估 FCOS 第 49 轮权重。"""

import copy
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __package__:
    from . import fcos_pipeline as pipeline
else:
    import fcos_pipeline as pipeline


DATA_ROOT = Path('/data/mml/data_debugging_data/ProAFL_data')
DEVICE = 'cuda:0'
EPOCH = 49
CHECKPOINT_DIR = DATA_ROOT / 'models/visdrone/fcos/fault_train/0.1/repeat_1'
IMAGE_DIR = DATA_ROOT / 'datasets/visdrone-coco/val'
ANNOTATIONS = IMAGE_DIR / '_annotations.coco.json'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'experiments' / (
    'fcos_eval_epoch49_' + datetime.now().strftime('%Y%m%d_%H%M%S'))


def evaluate(coco_gt, predictions, categories):
    """沿用 COCOeval 的 IoU、召回采样点和 maxDets=[1,10,100] 定义。"""
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = dict(images=coco_gt.dataset['images'],
                               categories=coco_gt.dataset['categories'], annotations=[])
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval['precision']
    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))[0])
    by_class = {}
    names = {x['id']: x['name'] for x in categories}
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[iou_index, :, index, 0, -1]
        valid = values[values >= 0]
        by_class[names[category]] = float(valid.mean()) if len(valid) else None
    return dict(mAP50=float(evaluator.stats[1]), mAP50_95=float(evaluator.stats[0]),
                AP_small=float(evaluator.stats[3]), AR100=float(evaluator.stats[8]),
                per_class_AP50=by_class, prediction_count=len(predictions),
                coco_max_dets=evaluator.params.maxDets)


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('验证需要可用的 CUDA GPU')
    torch.cuda.set_device(DEVICE)
    torch.set_num_threads(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    metadata = pipeline.load_json(CHECKPOINT_DIR / 'run_config.json')
    options = SimpleNamespace(device=DEVICE, min_size=800, max_size=1333,
                              score_threshold=.05, nms_threshold=.65,
                              max_detections=300, topk_candidates=1000,
                              batch_size=16, workers=4, seed=42)
    dataset = pipeline.DetectionDataset(IMAGE_DIR, ANNOTATIONS, training=False)
    if dataset.categories != metadata['categories']:
        raise ValueError('验证类别编号或名称与训练记录不一致')
    model = pipeline.build_model(dataset.num_classes, options,
                                 pretrained_layout=metadata['pretrained_layout']).eval()
    checkpoint = CHECKPOINT_DIR / f'epoch_{EPOCH}.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    data_loader = pipeline.loader(dataset, options, training=False)
    results = {'evaluation_005_cap300': [], 'collection_02_cap100': []}
    started = time.monotonic()
    completed = 0
    print(f'评估 {len(dataset)} 张验证图片，权重：{checkpoint}', flush=True)
    for images, targets in data_loader:
        outputs = model([image.to(DEVICE, non_blocking=True) for image in images])
        for target, output in zip(targets, outputs):
            image_id = int(target['image_id'])
            boxes, scores, labels = [output[key].cpu() for key in ('boxes', 'scores', 'labels')]
            boxes[:, 2:] -= boxes[:, :2]
            for name, threshold, cap in [('evaluation_005_cap300', .05, 300),
                                          ('collection_02_cap100', .2, 100)]:
                # 候选按分数降序；过滤低分框后取前 100，等价于同样 NMS 下的采集设置。
                indices = torch.where(scores > threshold)[0][:cap]
                for index in indices.tolist():
                    results[name].append(dict(image_id=image_id, category_id=int(labels[index]),
                                              bbox=boxes[index].tolist(), score=float(scores[index])))
        completed += len(images)
        if completed % 256 == 0 or completed == len(dataset):
            print(f'推理进度：{completed}/{len(dataset)}，耗时 {time.monotonic() - started:.1f}s', flush=True)
    inference_seconds = time.monotonic() - started
    # 原转换文件没有 area/iscrowd，只在内存中补齐 COCOeval 所需字段。
    # 标注 ID 从 1 重新编号，避免 COCOeval 把匹配到 ID=0 当作未匹配。
    gt = copy.deepcopy(pipeline.load_json(ANNOTATIONS))
    for index, anno in enumerate(gt['annotations'], 1):
        anno['id'] = index
        anno.setdefault('area', anno['bbox'][2] * anno['bbox'][3])
        anno.setdefault('iscrowd', 0)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    report = dict(checkpoint=str(checkpoint), annotations=str(ANNOTATIONS), images=len(dataset),
                  annotations_count=len(gt['annotations']), torch=torch.__version__,
                  device=torch.cuda.get_device_name(DEVICE), inference_seconds=inference_seconds,
                  protocol='COCO bbox evaluation on this project validation split; class-aware AP; category IDs 0..9',
                  input_size=[800, 1333], nms_threshold=.65, topk_candidates=1000,
                  results={})
    for name, predictions in results.items():
        with (OUTPUT_DIR / f'{name}_predictions.json').open('w') as file:
            json.dump(predictions, file, separators=(',', ':'), allow_nan=False)
        print(f'计算指标：{name}', flush=True)
        report['results'][name] = evaluate(coco_gt, predictions, dataset.categories)
        print(json.dumps(report['results'][name]), flush=True)
    pipeline.save_json(OUTPUT_DIR / 'report.json', report)
    print(f'报告：{OUTPUT_DIR / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
