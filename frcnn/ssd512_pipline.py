"""训练 SSD512–VGG16，并逐轮采集供 ProAFL 排序使用的预测框。

直接修改下方 Python 常量，然后运行：python frcnn/ssd512_pipline.py。
MODE="train" 表示训练，MODE="collect" 表示采集；仅保存普通 JSON。
训练标签加 1，为背景保留类别 0；采集时恢复原始类别编号。
probs 保存 softmax 后的全部前景类别概率，不对去掉背景后的概率重新归一化。
网络包含 7 个检测特征层和 24,564 个默认框，输入固定为 512×512。
默认从 SSD300 COCO 权重做部分迁移，并非原生 SSD512 COCO 预训练权重。
"""

import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16
from torchvision.models.detection.ssd import SSD, SSDHead, SSDFeatureExtractorVGG, SSD300_VGG16_Weights
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torchvision.transforms.functional import to_tensor

# 同时支持直接运行文件，以及 python -m frcnn.ssd512_pipline。
if __package__:
    from . import transforms as detection_transforms
else:
    import transforms as detection_transforms


# ======================== 在这里修改实验配置 ========================
MODE = "train"                         # "train"：训练；"collect"：逐轮采集
DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
DATASET_NAME = "visdrone"               # visdrone / voc / kitti
FAULT_RATIO = 0.1
REPEAT_ID = 1
DEVICE = "cuda:0"                      # 仅使用 GPU；按实际 GPU 编号修改
BATCH_SIZE = 16
NUM_WORKERS = 8
CPU_THREADS = 4
SEED = 42

# SSD512 的结构配置；不能只把 SSD300 的输入尺寸改成 512。
INPUT_SIZE = 512
FEATURE_CHANNELS = [512, 1024, 512, 256, 256, 256, 256]
ANCHOR_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
ANCHOR_SCALES = [0.04, 0.10, 0.26, 0.42, 0.58, 0.74, 0.90, 1.06]
ANCHOR_STEPS = [8, 16, 32, 64, 128, 256, 512]

# 推理及采集设置；TOPK_CANDIDATES 是每个前景类别的 NMS 前候选上限。
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 200                    # 每张图片最终最多保存的预测框数量
TOPK_CANDIDATES = 400

# 训练使用普通 FP32，不启用 AMP；每个 epoch 结束后打印汇总。
TRAIN_EPOCHS = 50
LEARNING_RATE = 0.002
PRETRAINED_WEIGHTS = "coco_transfer"     # SSD300 COCO 部分迁移；"none" 表示随机初始化
ENABLE_SSD_AUGMENTATION = True           # 颜色扰动、扩图、随机 IoU 裁剪、水平翻转
RESUME_PATH = None                      # 续训时填写 training_state.pth 的路径

# 采集设置；仅输出普通 JSON，不压缩。
COLLECT_EPOCHS = None                    # None：全部已有权重；或填写 [0, 10, 25, 49]
EXPERIMENT_ID = None                     # None：自动使用当前时间戳
COLLECTION_OUTPUT_DIR = None

# 路径覆盖；None 表示根据数据集、错误比例和 repeat 编号自动生成。
ANNOTATIONS_PATH = None
IMAGE_DIR = None
CHECKPOINT_DIR = None

# 小规模功能检查；正式实验保持 0。
MAX_IMAGES = 0
MAX_TRAIN_STEPS = 0
# ==================================================================


def load_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_path, training=False, limit=0, augmentation=True):
        data = load_json(annotation_path)
        self.categories = sorted(data['categories'], key=lambda x: x['id'])
        self.num_classes = len(self.categories)
        if [x['id'] for x in self.categories] != list(range(self.num_classes)):
            raise ValueError('This ProAFL adapter requires contiguous category IDs 0..C-1')
        self.image_dir = Path(image_dir)
        self.annos = defaultdict(list)
        known_images = {x['id'] for x in data['images']}
        seen_annos = set()
        for anno in data['annotations']:
            x, y, width, height = anno['bbox']
            if anno['image_id'] not in known_images or anno['id'] in seen_annos:
                raise ValueError(f'Unknown image or duplicate annotation: {anno["id"]}')
            if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError(f'Invalid annotation box: {anno["id"]}')
            if not 0 <= anno['category_id'] < self.num_classes:
                raise ValueError(f'Invalid annotation category: {anno["id"]}')
            seen_annos.add(anno['id'])
            self.annos[anno['image_id']].append(anno)
        # 训练时与原 FRCNN 保持一致，只使用有标注的图片。
        # 采集时也处理无标注图片，以保留图片分支需要的预测证据。
        self.images = [x for x in data['images'] if not training or self.annos[x['id']]]
        names = [x['file_name'] for x in self.images]
        if len(names) != len(set(names)) or len(known_images) != len(data['images']):
            raise ValueError('Image IDs and file names must be unique')
        if limit:
            self.images = self.images[:limit]
        if not self.images:
            raise ValueError('The selected dataset has no images')
        # SSD 的几何增强同步修改框坐标和标签，避免图片与标注错位。
        self.augmentation = detection_transforms.Compose([
            detection_transforms.RandomPhotometricDistort(),
            detection_transforms.RandomZoomOut(fill=[123., 117., 104.]),
            detection_transforms.RandomIoUCrop(),
            detection_transforms.RandomHorizontalFlip(p=.5),
        ]) if training and augmentation else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.image_dir / info['file_name']) as source:
            image = source.convert('RGB')
        if image.size != (info['width'], info['height']):
            raise ValueError(f'Image dimensions disagree with annotations: {info["file_name"]}')
        annos = self.annos[info['id']]
        boxes = torch.tensor([a['bbox'] for a in annos], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        target = {
            'boxes': boxes,
            'labels': torch.tensor([a['category_id'] + 1 for a in annos], dtype=torch.int64),
            'image_id': torch.tensor(info['id']),
            'image_name': info['file_name'],
        }
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        # 像素归一化及固定 512×512 缩放在 SSD 模型内部完成。
        return to_tensor(image), target


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(_):
    random.seed(torch.initial_seed() % 2**32)


def loader(dataset, options, training):
    """训练时打乱图片，采集时保持顺序；图片尺寸不同，因此不直接堆叠。"""
    return DataLoader(dataset, batch_size=options.batch_size, shuffle=training,
                      num_workers=options.workers, pin_memory=options.device.startswith('cuda'),
                      persistent_workers=options.workers > 0, collate_fn=collate,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(options.seed))


def postprocess_with_probs(self, head_outputs, image_anchors, image_shapes):
    """保留原版 SSD 的筛选和 NMS，只增加每个保留框的完整前景概率。"""
    probabilities = head_outputs['cls_logits'].softmax(-1)
    detections = []
    for regression, scores, anchors, shape in zip(
            head_outputs['bbox_regression'], probabilities, image_anchors, image_shapes):
        boxes = box_ops.clip_boxes_to_image(self.box_coder.decode_single(regression, anchors), shape)
        selected_boxes, selected_scores, selected_labels, selected_probs = [], [], [], []
        # 背景占第 0 列，前景类别从第 1 列开始。
        for label in range(1, scores.shape[1]):
            indices = torch.where(scores[:, label] > self.score_thresh)[0]
            values, order = scores[indices, label].topk(min(len(indices), self.topk_candidates))
            indices = indices[order]
            selected_boxes.append(boxes[indices])
            selected_scores.append(values)
            selected_labels.append(torch.full_like(indices, label))
            selected_probs.append(scores[indices, 1:])
        boxes, scores = torch.cat(selected_boxes), torch.cat(selected_scores)
        labels, probs = torch.cat(selected_labels), torch.cat(selected_probs)
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[:self.detections_per_img]
        detections.append({'boxes': boxes[keep], 'scores': scores[keep],
                           'labels': labels[keep], 'probs': probs[keep]})
    return detections


def build_model(num_classes, options, pretrained=False):
    """组装真正的七层 SSD512，并按需迁移 SSD300 COCO 中形状兼容的权重。"""
    backbone = SSDFeatureExtractorVGG(vgg16(weights=None).features, highres=True)
    # Torchvision 高分辨率分支只增加了末层；还需要修正中间层的步长和 padding。
    # 对应特征尺寸应为 64、32、16、8、4、2、1，而不是 64、32、16、8、6、4、1。
    for level in (3, 4):
        backbone.extra[level][2].stride = (2, 2)
        backbone.extra[level][2].padding = (1, 1)
    backbone.extra[5][2].padding = (1, 1)
    backbone.out_channels = list(FEATURE_CHANNELS)
    anchors = DefaultBoxGenerator(
        ANCHOR_RATIOS, scales=ANCHOR_SCALES, steps=ANCHOR_STEPS)
    head = SSDHead(FEATURE_CHANNELS, anchors.num_anchors_per_location(), num_classes + 1)
    model = SSD(
        backbone, anchors, size=(INPUT_SIZE, INPUT_SIZE), num_classes=num_classes + 1,
        head=head, image_mean=[0.48235, 0.45882, 0.40784],
        image_std=[1. / 255., 1. / 255., 1. / 255.],
        score_thresh=options.score_threshold, nms_thresh=options.nms_threshold,
        detections_per_img=options.max_detections, topk_candidates=options.topk_candidates,
    )
    model.initialization = {'method': 'none', 'transferred_tensors': 0}
    if pretrained:
        # SSD300 的 stride/padding 不保存在权重里，兼容的特征层可直接迁移。
        # 类别头不迁移；新增特征层、anchor 数不同的回归层保持随机初始化。
        weights = SSD300_VGG16_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        target = model.state_dict()
        compatible = {key: value for key, value in weights.items()
                      if key in target and value.shape == target[key].shape
                      and not key.startswith('head.classification_head.')}
        model.load_state_dict(compatible, strict=False)
        model.initialization = dict(method='ssd300_coco_partial_transfer',
                                    transferred_tensors=len(compatible),
                                    randomly_initialized_tensors=len(target) - len(compatible))
        del weights
    model.postprocess_detections = MethodType(postprocess_with_probs, model)
    return model.to(options.device)


def architecture():
    """明确记录结构，避免更改 anchors 后仍加载同名旧权重。"""
    return dict(input_size=INPUT_SIZE, feature_channels=FEATURE_CHANNELS,
                anchor_ratios=ANCHOR_RATIOS, anchor_scales=ANCHOR_SCALES,
                anchor_steps=ANCHOR_STEPS)


def paths(options):
    root = options.data_root
    annotation = options.annotations or root / f'fault_inject/{options.fault_ratio}/{options.dataset}/coco_format/annotations_no_miss.json'
    images = options.image_dir or root / f'datasets/{options.dataset}-coco/train'
    checkpoints = options.checkpoint_dir or root / f'models/{options.dataset}/ssd512/fault_train/{options.fault_ratio}/repeat_{options.repeat_id}'
    return images, annotation, checkpoints


def settings(options):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}


def train(options):
    """每轮保存模型权重，同时保存优化器和调度器状态以便续训。"""
    image_dir, annotation, directory = paths(options)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = DetectionDataset(image_dir, annotation, training=True, limit=options.max_images,
                               augmentation=options.augmentation)
    data_loader = loader(dataset, options, True)
    state = torch.load(options.resume, map_location='cpu', weights_only=True) if options.resume else None
    if state is not None and state['categories'] != dataset.categories:
        raise ValueError('Resume checkpoint category mapping differs from the current dataset')
    model = build_model(dataset.num_classes, options,
                        pretrained=options.weights == 'coco_transfer' and state is None)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=.9, weight_decay=.0005)
    milestones = sorted({int(options.epochs * fraction) for fraction in (.6, .8)
                         if 0 < int(options.epochs * fraction) < options.epochs})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=.1)
    start_epoch = 0
    if state is not None:
        state_initialization = state.get('initialization', {'method': 'unknown_legacy'})
        if state.get('architecture') != architecture():
            raise ValueError('续训状态的 SSD512 结构或 anchor 配置不匹配')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        del state
    if start_epoch >= options.epochs:
        raise ValueError('No epochs remain; TRAIN_EPOCHS 表示总轮数，而不是额外训练轮数')
    metadata = dict(model='ssd512_vgg16', categories=dataset.categories,
                    label_convention='training_category_plus_one_output_zero_based',
                    probs='softmax(class_logits)[:, 1:] without renormalization',
                    architecture=architecture(),
                    initialization=state_initialization if options.resume else model.initialization,
                    torch=torch.__version__, torchvision=torchvision.__version__, settings=settings(options))
    # 续训配置另存，保留首次训练的类别映射和模型结构说明。
    save_json(directory / ('resume_config.json' if options.resume else 'run_config.json'), metadata)
    # 重新训练时直接覆盖同名权重、配置，并清空旧日志；续训时保留日志。
    if not options.resume:
        (directory / 'train_log.jsonl').write_text('', encoding='utf-8')
    print(f'Training {len(dataset)} images, {dataset.num_classes} classes; checkpoints: {directory}', flush=True)
    for epoch in range(start_epoch, options.epochs):
        model.train()
        sums, count = defaultdict(float), 0
        started = time.monotonic()
        for step, (images, targets) in enumerate(data_loader):
            images = [x.to(options.device, non_blocking=True) for x in images]
            targets = [{k: v.to(options.device) for k, v in t.items() if k in ('boxes', 'labels')} for t in targets]
            # 只在第 0 轮预热学习率，后续轮次由调度器控制。
            if epoch == 0:
                factor = .001 + .999 * min(1., (step + 1) / min(1000, len(data_loader)))
                for group in optimizer.param_groups:
                    group['lr'] = options.lr * factor
            losses = model(images, targets)
            total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite losses: {losses}')
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            for name, loss in losses.items():
                sums[name] += float(loss.detach())
            count += 1
            if options.max_steps and count >= options.max_steps:
                break
        # 即使功能检查提前结束本轮，也要恢复正常学习率。
        if epoch == 0:
            for group in optimizer.param_groups:
                group['lr'] = options.lr
        scheduler.step()
        torch.save(model.state_dict(), directory / f'epoch_{epoch}.pth')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, categories=dataset.categories,
                        architecture=architecture(), initialization=metadata['initialization']),
                   directory / 'training_state.pth')
        report = dict(epoch=epoch, steps=count, losses={k: v / count for k, v in sums.items()},
                      seconds=round(time.monotonic() - started, 2), lr=optimizer.param_groups[0]['lr'])
        with (directory / 'train_log.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(report) + '\n')
        print(json.dumps(report), flush=True)


def prediction_rows(name, pred, start):
    boxes = pred['boxes'].cpu().tolist()
    labels = (pred['labels'] - 1).cpu().tolist()
    scores = pred['scores'].cpu().tolist()
    probabilities = pred['probs'].cpu().tolist()
    return [dict(predicted_box_id=start + i, img_name=name, predicted_cls=int(label),
                 conf=round(score, 4), bbox=box, probs=[round(x, 4) for x in probs])
            for i, (box, label, score, probs) in enumerate(zip(boxes, labels, scores, probabilities))]


@torch.inference_mode()
def collect(options):
    """逐轮加载训练权重，按图片流式保存普通 JSON，避免累积整轮预测。"""
    image_dir, annotation, checkpoint_dir = paths(options)
    config = load_json(checkpoint_dir / 'run_config.json')
    dataset = DetectionDataset(image_dir, annotation, limit=options.max_images)
    if config['categories'] != dataset.categories or config['model'] != 'ssd512_vgg16':
        raise ValueError('Checkpoint categories or model differ from this collection dataset')
    epochs = sorted(int(m.group(1)) for p in checkpoint_dir.glob('epoch_*.pth')
                    if (m := re.fullmatch(r'epoch_(\d+)\.pth', p.name))) if options.epochs == ['all'] else sorted(set(map(int, options.epochs)))
    if not epochs:
        raise FileNotFoundError(f'No epoch checkpoints in {checkpoint_dir}')
    for epoch in epochs:
        if not (checkpoint_dir / f'epoch_{epoch}.pth').is_file():
            raise FileNotFoundError(checkpoint_dir / f'epoch_{epoch}.pth')
    expid = options.experiment_id or str(int(time.time()))
    output = options.output_dir or options.data_root / (
        f'collection_process_info/{options.dataset}/ssd512/collected_predict_boxes/'
        f'inject_{options.fault_ratio}_repeat/repeat_{options.repeat_id}_{expid}')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'collection_config.json', dict(settings=settings(options), categories=dataset.categories,
              experiment_id=expid, checkpoint_dir=str(checkpoint_dir), epochs=epochs,
              probs=config['probs'], images=len(dataset)))
    data_loader = loader(dataset, options, False)
    if config.get('architecture') != architecture():
        raise ValueError('采集配置的 SSD512 结构或 anchors 与训练记录不一致')
    model = build_model(dataset.num_classes, options).eval()
    print(f'Collecting {len(dataset)} images; output: {output}', flush=True)
    for epoch in epochs:
        state = torch.load(checkpoint_dir / f'epoch_{epoch}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        path = output / f'epoch_{epoch}_predicted_bboxs.json'
        temporary = path.with_name(path.name + '.tmp')
        total, first, started = 0, True, time.monotonic()
        with temporary.open('w', encoding='utf-8') as file:
            file.write('{')
            for images, targets in data_loader:
                predictions = model([x.to(options.device, non_blocking=True) for x in images])
                for target, pred in zip(targets, predictions):
                    name = target['image_name']
                    rows = prediction_rows(name, pred, total)
                    total += len(rows)
                    if not first:
                        file.write(',')
                    first = False
                    file.write(json.dumps(name) + ':')
                    json.dump({'predicted_bboxs': rows}, file, separators=(',', ':'), allow_nan=False)
            file.write('}')
        # 本轮完整写入后再改为正式文件名，避免下游误读未写完的结果。
        temporary.replace(path)
        print(f'epoch {epoch}: {total} boxes, {path.stat().st_size / 1024**2:.2f} MiB, '
              f'{time.monotonic() - started:.1f}s -> {path}', flush=True)


def make_config():
    """将顶部常量组织为内部配置，并在加载数据和模型前检查参数。"""
    as_path = lambda value: Path(value) if value is not None else None
    options = SimpleNamespace(
        mode=MODE, data_root=Path(DATA_ROOT), dataset=DATASET_NAME,
        fault_ratio=FAULT_RATIO, repeat_id=REPEAT_ID,
        annotations=as_path(ANNOTATIONS_PATH), image_dir=as_path(IMAGE_DIR),
        checkpoint_dir=as_path(CHECKPOINT_DIR), device=DEVICE,
        batch_size=BATCH_SIZE, workers=NUM_WORKERS, cpu_threads=CPU_THREADS,
        seed=SEED, augmentation=ENABLE_SSD_AUGMENTATION,
        score_threshold=SCORE_THRESHOLD, nms_threshold=NMS_THRESHOLD,
        max_detections=MAX_DETECTIONS, topk_candidates=TOPK_CANDIDATES,
        max_images=MAX_IMAGES,
        epochs=TRAIN_EPOCHS if MODE == "train" else (
            ['all'] if COLLECT_EPOCHS is None else list(COLLECT_EPOCHS)),
        lr=LEARNING_RATE, weights=PRETRAINED_WEIGHTS, resume=as_path(RESUME_PATH),
        max_steps=MAX_TRAIN_STEPS, experiment_id=EXPERIMENT_ID,
        output_dir=as_path(COLLECTION_OUTPUT_DIR),
    )
    if MODE not in ("train", "collect"):
        raise ValueError('MODE 只能设置为 "train" 或 "collect"')
    if DATASET_NAME not in ("visdrone", "voc", "kitti"):
        raise ValueError('DATASET_NAME 只能设置为 visdrone、voc 或 kitti')
    if PRETRAINED_WEIGHTS not in ("coco_transfer", "none"):
        raise ValueError('PRETRAINED_WEIGHTS 只能设置为 "coco_transfer" 或 "none"')
    if min(BATCH_SIZE, CPU_THREADS, MAX_DETECTIONS,
           TOPK_CANDIDATES) < 1 or min(NUM_WORKERS, MAX_IMAGES) < 0:
        raise ValueError('数量和尺寸必须为正；NUM_WORKERS、MAX_IMAGES 可以为 0')
    if INPUT_SIZE != 512:
        raise ValueError('此脚本是 SSD512，INPUT_SIZE 必须为 512')
    if not 0 <= SCORE_THRESHOLD <= 1 or not 0 <= NMS_THRESHOLD <= 1:
        raise ValueError('分数和 NMS 阈值必须位于 [0, 1]')
    if torch.device(DEVICE).type != 'cuda':
        raise ValueError('本脚本仅使用 GPU，请将 DEVICE 设置为 "cuda:0" 等 CUDA 设备')
    if not torch.cuda.is_available():
        raise RuntimeError('当前环境无法使用 CUDA，请检查 GPU、驱动和 PyTorch CUDA 环境')
    if MODE == 'train' and (type(TRAIN_EPOCHS) is not int or TRAIN_EPOCHS < 1
                            or LEARNING_RATE <= 0 or MAX_TRAIN_STEPS < 0):
        raise ValueError('训练轮数、学习率或步数限制不合法')
    if MODE == 'collect' and COLLECT_EPOCHS is not None:
        if not COLLECT_EPOCHS or any(type(x) is not int or x < 0 for x in COLLECT_EPOCHS):
            raise ValueError('COLLECT_EPOCHS 应为 None 或非空的非负整数列表')
    return options


def main():
    options = make_config()
    torch.cuda.set_device(options.device)
    torch.set_num_threads(options.cpu_threads)
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    train(options) if options.mode == 'train' else collect(options)


if __name__ == '__main__':
    main()
