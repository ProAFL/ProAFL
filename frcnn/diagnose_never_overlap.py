"""Trace VisDrone never-overlapped annotations through Faster R-CNN inference.

Run from any directory (use the data_debugging Python environment)::

    python frcnn/diagnose_never_overlap.py --device cuda:1 --max-images 20
    python frcnn/diagnose_never_overlap.py --device cuda:1 --epochs all

The default epochs are 0, 10, 25, 49; --epochs all inspects all 50 checkpoints.
All fault types are included by default; --fault-types 0 restricts to clean GT.
Original checkpoints, predictions and metrics are read-only. Outputs are CSV
and JSON, without saving the very large intermediate candidate-box arrays.
"""

import argparse
import csv
import json
import gzip
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import MethodType

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import boxes as box_ops
from torchvision.transforms.functional import to_tensor


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data/mml/data_debugging_data/ProAFL_data')
STAGES = (
    'rpn_decoded', 'rpn_topk', 'rpn_score_size', 'rpn_nms', 'rpn_final',
    'roi_decoded', 'roi_score', 'roi_size', 'roi_nms', 'final',
)
REASONS = (
    ('final', 'reproduced_overlap'),
    ('roi_nms', 'roi_output_limit'),
    ('roi_size', 'roi_nms'),
    ('roi_score', 'roi_small_box_filter'),
    ('roi_decoded', 'roi_score_threshold'),
    ('rpn_final', 'roi_box_regression'),
    ('rpn_nms', 'rpn_output_limit'),
    ('rpn_score_size', 'rpn_nms'),
    ('rpn_topk', 'rpn_clip_score_size_filter'),
    ('rpn_decoded', 'rpn_topk'),
)


def read_json(path):
    path = json_path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def json_path(path):
    path = Path(path)
    compressed = path.with_suffix('.json.gz')
    return compressed if not path.exists() and path.suffix == '.json' and compressed.exists() else path


def iter_json_items(path, chunk_size=1024 * 1024):
    """Stream a top-level JSON object, keeping only one image/anno in memory."""
    decoder = json.JSONDecoder()
    path = json_path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        buffer, position, eof = '', 0, False

        def refill():
            nonlocal buffer, position, eof
            block = f.read(chunk_size)
            buffer, position = buffer[position:] + block, 0
            eof = not block

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ''
                refill()

        def expect(character):
            nonlocal position
            if peek() != character:
                raise ValueError(f'Expected {character!r} in {path}')
            position += 1

        def decode():
            nonlocal position
            peek()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    # A number may parse successfully before its remaining
                    # digits/exponent have arrived in the next read chunk.
                    if (end == len(buffer) and not eof) or (
                            end < len(buffer) and buffer[end] not in ' \t\r\n,:}'):
                        if eof:
                            raise ValueError(f'Invalid JSON token in {path}')
                        refill()
                        continue
                    position = end
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        expect('{')
        if peek() != '}':
            while True:
                key = decode()
                if not isinstance(key, str):
                    raise ValueError(f'Non-string JSON object key in {path}')
                expect(':')
                value = decode()
                yield key, value
                if peek() == '}':
                    break
                expect(',')
        expect('}')
        if peek():
            raise ValueError(f'Trailing content in {path}')


def reason(stats):
    """Locate the last stage with coverage, allowing regression to gain coverage."""
    return next((label for stage, label in REASONS if stats[stage] > 0),
                'no_decoded_rpn_overlap')


def max_iou(targets, candidates, chunk=4096):
    """Bound memory even for the raw RPN's hundreds of thousands of boxes."""
    result = targets.new_zeros(len(targets))
    for boxes in candidates.split(chunk):
        if len(boxes):
            result = torch.maximum(result, box_ops.box_iou(targets, boxes).max(1).values)
    return result


class Trace:
    def __init__(self, targets):
        self.targets = targets
        self.values = [dict() for _ in targets]
        self.counts = [dict() for _ in targets]
        self.max_foreground_score = [None for _ in targets]

    def record(self, image, stage, boxes):
        self.values[image][stage] = max_iou(self.targets[image], boxes).cpu().tolist()
        self.counts[image][stage] = len(boxes)

    def record_scores(self, image, boxes, scores):
        result = scores.new_zeros(len(self.targets[image]))
        for start in range(0, len(boxes), 4096):
            candidates = boxes[start:start + 4096]
            overlap = box_ops.box_iou(self.targets[image], candidates) > 0
            if len(candidates):
                values = scores[start:start + 4096].unsqueeze(0).expand(len(result), -1)
                result = torch.maximum(result, values.masked_fill(~overlap, 0).max(1).values)
        self.max_foreground_score[image] = result.cpu().tolist()


def install_rpn_trace(model):
    """Keep the original RPN filter outputs; inspect its exact intermediate steps."""
    original = model.rpn.filter_proposals

    def traced(rpn, proposals, objectness, image_shapes, num_anchors_per_level):
        final_boxes, final_scores = original(
            proposals, objectness, image_shapes, num_anchors_per_level)
        trace = model._diagnostic_trace
        logits = objectness.detach().reshape(len(proposals), -1)
        levels = torch.cat([
            torch.full((count,), level, dtype=torch.int64, device=proposals.device)
            for level, count in enumerate(num_anchors_per_level)
        ])
        top = rpn._get_top_n_idx(logits, num_anchors_per_level)
        for i, shape in enumerate(image_shapes):
            trace.record(i, 'rpn_decoded', proposals[i])
            boxes = proposals[i][top[i]]
            scores = logits[i][top[i]].sigmoid()
            labels = levels[top[i]]
            trace.record(i, 'rpn_topk', boxes)
            boxes = box_ops.clip_boxes_to_image(boxes, shape)
            keep = box_ops.remove_small_boxes(boxes, rpn.min_size)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = torch.where(scores >= rpn.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            trace.record(i, 'rpn_score_size', boxes)
            keep = box_ops.batched_nms(boxes, scores, labels, rpn.nms_thresh)
            trace.record(i, 'rpn_nms', boxes[keep])
            trace.record(i, 'rpn_final', final_boxes[i])
        return final_boxes, final_scores

    model.rpn.filter_proposals = MethodType(traced, model.rpn)


@torch.inference_mode()
def diagnose_batch(model, images, gt_boxes):
    original_shapes = [tuple(image.shape[-2:]) for image in images]
    transformed, _ = model.transform(images, None)
    targets = [resize_boxes(boxes, old, new) for boxes, old, new in
               zip(gt_boxes, original_shapes, transformed.image_sizes)]
    trace = Trace(targets)
    model._diagnostic_trace = trace
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = {'0': features}
    proposals, _ = model.rpn(transformed, features, None)
    head = model.roi_heads
    pooled = head.box_roi_pool(features, proposals, transformed.image_sizes)
    logits, regressions = head.box_predictor(head.box_head(pooled))
    counts = [len(x) for x in proposals]
    decoded = head.box_coder.decode(regressions, proposals).split(counts, 0)
    probabilities = logits.softmax(-1).split(counts, 0)
    output = []
    for i, (boxes, scores, shape) in enumerate(zip(decoded, probabilities, transformed.image_sizes)):
        boxes = box_ops.clip_boxes_to_image(boxes, shape)[:, 1:].reshape(-1, 4)
        labels = torch.arange(1, scores.shape[1], device=scores.device).expand(len(scores), -1).reshape(-1)
        scores = scores[:, 1:].reshape(-1)
        trace.record(i, 'roi_decoded', boxes)
        trace.record_scores(i, boxes, scores)
        keep = torch.where(scores > head.score_thresh)[0]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        trace.record(i, 'roi_score', boxes)
        keep = box_ops.remove_small_boxes(boxes, 1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        trace.record(i, 'roi_size', boxes)
        keep = box_ops.batched_nms(boxes, scores, labels, head.nms_thresh)
        trace.record(i, 'roi_nms', boxes[keep])
        keep = keep[:head.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        trace.record(i, 'final', boxes)
        output.append({'boxes': boxes, 'scores': scores, 'labels': labels})
    output = model.transform.postprocess(output, transformed.image_sizes, original_shapes)
    del model._diagnostic_trace
    return trace, output


def build_model(classes, args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        box_score_thresh=args.score_threshold, box_nms_thresh=args.nms_threshold,
        box_detections_per_img=args.max_detections,
        rpn_pre_nms_top_n_test=args.rpn_pre_nms,
        rpn_post_nms_top_n_test=args.rpn_post_nms,
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, classes)
    return model.eval().to(args.device)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--experiment', default='1792933936')
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--epochs', nargs='+', default=['0', '10', '25', '49'], help='Epoch IDs, or all (0..49)')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=0, help='0 = all affected images; otherwise deterministic sample')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fault-types', nargs='+', type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--nms-threshold', type=float, default=0.75)
    parser.add_argument('--max-detections', type=int, default=500)
    parser.add_argument('--rpn-pre-nms', type=int, default=1000)
    parser.add_argument('--rpn-post-nms', type=int, default=1000)
    parser.add_argument('--verify-saved', action='store_true', help='Read each selected epoch JSON and compare target coverage with re-inference')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        args.epochs = list(range(50)) if args.epochs == ['all'] else sorted(set(map(int, args.epochs)))
    except ValueError:
        parser.error('--epochs requires integer IDs or all')
    if not args.epochs or any(e < 0 or e >= 50 for e in args.epochs):
        parser.error('Epochs must be in 0..49')
    if min(args.batch_size, args.cpu_threads, args.max_detections, args.rpn_pre_nms, args.rpn_post_nms) < 1 or args.max_images < 0:
        parser.error('Counts must be positive; --max-images may be zero')
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        parser.error('Thresholds must be in [0,1]')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; select --device cpu or run in a GPU environment')
    return args


def main():
    args = arguments()
    torch.set_num_threads(args.cpu_threads)
    root = args.data_root
    metrics_path = args.metrics or root / f'dual_branch/process_metrics/visdrone/frcnn/0.1_repeat/repeat_1_{args.experiment}/process_metrics.json'
    checkpoint_dir = args.checkpoint_dir or root / 'models/visdrone/frcnn/fault_train/0.1/repeat_1'
    prediction_dir = root / f'collection_process_info/visdrone/frcnn/collected_predict_boxes/inject_0.1_repeat/repeat_1_{args.experiment}'
    annotation_path = root / 'fault_inject/0.1/visdrone/coco_format/annotations_no_miss.json'
    print(f'Loading never-overlap IDs from {metrics_path}', flush=True)
    lengths, never = Counter(), set()
    for key, value in iter_json_items(metrics_path):
        history = value['has_overlap_list']
        if not history:
            raise ValueError('Empty overlap histories cannot establish never-overlapped status')
        lengths[len(history)] += 1
        if not any(history):
            never.add(int(key))
    annotations = read_json(annotation_path)
    image_info = {x['id']: x for x in annotations['images']}
    selected = [x for x in annotations['annotations'] if x['id'] in never and x['fault_type'] in args.fault_types]
    if not never.issubset({x['id'] for x in annotations['annotations']}):
        raise ValueError('Metric IDs are absent from annotations')
    grouped = defaultdict(list)
    for anno in selected:
        grouped[anno['image_id']].append(anno)
    image_ids = sorted(grouped)
    if args.max_images and len(image_ids) > args.max_images:
        import random
        image_ids = sorted(random.Random(args.seed).sample(image_ids, args.max_images))
    selected = [anno for i in image_ids for anno in grouped[i]]
    if not selected:
        raise ValueError('No annotations match the selection')
    for epoch in args.epochs:
        if not (checkpoint_dir / f'epoch_{epoch}.pth').is_file():
            raise FileNotFoundError(checkpoint_dir / f'epoch_{epoch}.pth')
    output_dir = args.output_dir or REPO / 'experiments/frcnn_never_overlap' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f'Diagnosing {len(selected)} annotations in {len(image_ids)} images; epochs={args.epochs}; output={output_dir}', flush=True)
    model = build_model(len(annotations['categories']) + 1, args)
    install_rpn_trace(model)
    aggregate = {a['id']: dict.fromkeys(STAGES, 0.0) for a in selected}
    best_scores = dict.fromkeys(aggregate, 0.0)
    epoch_reports = []
    columns = ['epoch', 'anno_id', 'image_name', 'fault_type', 'category_id', 'width', 'height',
               'resized_width', 'resized_height', 'reason', 'max_overlapping_foreground_score',
               'saved_max_iou', 'saved_coverage_disagrees'] + [f'{s}_max_iou' for s in STAGES]
    with (output_dir / 'per_epoch.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for epoch in args.epochs:
            started = time.monotonic()
            state = torch.load(checkpoint_dir / f'epoch_{epoch}.pth', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            saved = None
            if args.verify_saved:
                selected_names = {image_info[i]['file_name'] for i in image_ids}
                saved = {name: value for name, value in iter_json_items(
                    prediction_dir / f'epoch_{epoch}_predicted_bboxs.json') if name in selected_names}
            reasons, coverage, candidate_counts = Counter(), Counter(), Counter()
            mismatches, missing_saved_images = 0, 0
            for offset in range(0, len(image_ids), args.batch_size):
                ids = image_ids[offset:offset + args.batch_size]
                images, targets, sizes = [], [], []
                for image_id in ids:
                    info = image_info[image_id]
                    path = root / 'datasets/visdrone-coco/train' / info['file_name']
                    with Image.open(path) as source:
                        image = to_tensor(source.convert('RGB')).to(args.device)
                    h, w = image.shape[-2:]
                    if (w, h) != (info['width'], info['height']):
                        raise ValueError(f'Image/annotation dimensions disagree: {path}')
                    sizes.append((h, w))
                    images.append(image)
                    xywh = torch.tensor([a['bbox'] for a in grouped[image_id]], device=args.device, dtype=torch.float32)
                    targets.append(torch.cat((xywh[:, :2], xywh[:, :2] + xywh[:, 2:]), 1))
                trace, predictions = diagnose_batch(model, images, targets)
                for i, image_id in enumerate(ids):
                    name = image_info[image_id]['file_name']
                    saved_ious = None
                    if saved is not None:
                        missing_saved_images += name not in saved
                        boxes = torch.tensor([p['bbox'] for p in saved.get(name, {}).get('predicted_bboxs', [])], device=args.device).reshape(-1, 4)
                        saved_ious = max_iou(targets[i], boxes).cpu().tolist()
                    for stage in STAGES:
                        candidate_counts[stage] += trace.counts[i][stage]
                    # Transformed GT size includes the exact resize rounding.
                    resized = trace.targets[i][:, 2:] - trace.targets[i][:, :2]
                    for j, anno in enumerate(grouped[image_id]):
                        stats = {stage: trace.values[i][stage][j] for stage in STAGES}
                        cause = reason(stats)
                        reasons[cause] += 1
                        for stage, value in stats.items():
                            aggregate[anno['id']][stage] = max(aggregate[anno['id']][stage], value)
                            coverage[stage] += value > 0
                        confidence = trace.max_foreground_score[i][j]
                        best_scores[anno['id']] = max(best_scores[anno['id']], confidence)
                        disagrees = saved_ious is not None and ((saved_ious[j] > 0) != (stats['final'] > 0))
                        mismatches += bool(disagrees)
                        row = dict(epoch=epoch, anno_id=anno['id'], image_name=name,
                                   fault_type=anno['fault_type'], category_id=anno['category_id'],
                                   width=anno['bbox'][2], height=anno['bbox'][3],
                                   resized_width=float(resized[j, 0]), resized_height=float(resized[j, 1]),
                                   reason=cause, max_overlapping_foreground_score=confidence,
                                   saved_max_iou='' if saved_ious is None else saved_ious[j],
                                   saved_coverage_disagrees='' if saved_ious is None else disagrees)
                        row.update({f'{k}_max_iou': v for k, v in stats.items()})
                        writer.writerow(row)
                if offset == 0 or (offset // args.batch_size + 1) % 50 == 0:
                    print(f'epoch {epoch}: {min(offset + args.batch_size, len(image_ids))}/{len(image_ids)} images', flush=True)
                del images, targets, trace, predictions
            del saved
            report = dict(epoch=epoch, seconds=round(time.monotonic() - started, 2),
                          reasons=dict(reasons), covered_annotations=dict(coverage),
                          candidate_boxes=dict(candidate_counts),
                          saved_coverage_mismatches=mismatches if args.verify_saved else None,
                          missing_saved_images=missing_saved_images if args.verify_saved else None)
            epoch_reports.append(report)
            f.flush()
            print(json.dumps(report, ensure_ascii=False), flush=True)
    all_reasons, by_fault, by_size = Counter(), defaultdict(Counter), defaultdict(Counter)
    columns = ['anno_id', 'image_name', 'fault_type', 'category_id', 'bbox', 'reason', 'max_overlapping_foreground_score'] + [f'{s}_max_iou' for s in STAGES]
    with (output_dir / 'per_anno.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for anno in selected:
            stats = aggregate[anno['id']]
            cause = reason(stats)
            all_reasons[cause] += 1
            by_fault[str(anno['fault_type'])][cause] += 1
            area = anno['bbox'][2] * anno['bbox'][3]
            size = '<16^2' if area < 256 else '16^2..32^2' if area < 1024 else '32^2..96^2' if area < 9216 else '>=96^2'
            by_size[size][cause] += 1
            row = dict(anno_id=anno['id'], image_name=image_info[anno['image_id']]['file_name'],
                       fault_type=anno['fault_type'], category_id=anno['category_id'], bbox=json.dumps(anno['bbox']),
                       reason=cause, max_overlapping_foreground_score=best_scores[anno['id']])
            row.update({f'{k}_max_iou': v for k, v in stats.items()})
            writer.writerow(row)
    report = dict(
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        resolved_paths=dict(metrics=str(metrics_path), annotations=str(annotation_path), checkpoints=str(checkpoint_dir)),
        versions=dict(torch=torch.__version__, torchvision=torchvision.__version__),
        source_history_lengths=dict(lengths), source_never_overlapped=len(never),
        selected_annotations=len(selected), selected_images=len(image_ids),
        selected_fault_types=dict(Counter(str(a['fault_type']) for a in selected)),
        reason_counts=dict(all_reasons), by_fault_type=dict(by_fault), by_area=dict(by_size),
        ever_covered={stage: sum(stats[stage] > 0 for stats in aggregate.values()) for stage in STAGES},
        ever_iou_ge_05={stage: sum(stats[stage] >= .5 for stats in aggregate.values()) for stage in STAGES},
        epochs=epoch_reports,
        interpretation=[
            'All coverage is class-agnostic IoU > 0, matching process_metrics.',
            'Per-anno maxima aggregate ONLY the selected checkpoints and images.',
            'Reasons locate the last stage with overlap, not a proven training cause. Box regression can gain or lose overlap.',
            'roi_score_threshold means no overlapping decoded foreground candidate exceeded the score threshold.',
            'rpn_topk means decoded RPN overlap existed but per-FPN-level objectness top-k lost it.',
            'reproduced_overlap for a historical never-overlap target requires checking saved predictions/settings/batch size.',
            'Batch padding can affect predictions; collection used batch_size=64. Use --verify-saved to check reproducibility.',
            'A single restored pixel of overlap does not establish correct object detection; compare IoU >= 0.5 too.',
        ],
    )
    (output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished. Reason counts: {dict(all_reasons)}\nReport: {output_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
