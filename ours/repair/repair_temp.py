


"""按 ours 排名和 repair.py 的严格预算修复 VOC COCO 标签。

默认运行：python ours/repair/repair_temp.py
调整预算：python ours/repair/repair_temp.py --cut-off-rate 0.2
fault_type=4 的框仅在被预算选中时补回，未修复的缺失框不写入输出。
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from numbers import Integral

import joblib
import os





def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def repair_annotations(rank, annotations_with_miss, correct_annotations, cut_off_rate=0.5):
    """返回修复标注和统计；输入保持不变，预算规则与 strict_cost 一致。"""
    if not 0 <= cut_off_rate <= 1:
        raise ValueError('cut_off_rate 必须在 [0, 1] 内')
    result = deepcopy(annotations_with_miss)
    image_names = {image['id']: image['file_name'] for image in result['images']}
    clean_images = {image['id']: image['file_name'] for image in correct_annotations['images']}
    if image_names != clean_images or result['categories'] != correct_annotations['categories']:
        raise ValueError('正确标注与错误标注的图片或类别映射不一致')
    clean = {ann['id']: ann for ann in correct_annotations['annotations']}
    visible = {}
    missing = defaultdict(list)
    for ann in result['annotations']:
        if ann['fault_type'] not in (0, 1, 2, 3, 4):
            raise ValueError(f"未知 fault_type: {ann}")
        if ann['fault_type'] == 4:
            missing[image_names[ann['image_id']]].append(ann['id'])
        else:
            visible[ann['id']] = ann
        if ann['fault_type'] in (1, 2, 4):
            reference = clean[ann['id']]
            if reference['image_id'] != ann['image_id']:
                raise ValueError(f"标注 {ann['id']} 的图片不一致")

    # 此版本 rank 的整数已经是 annotation ID，无需旧版 gid 转换。
    seen = set()
    for item in rank:
        if item in seen:
            raise ValueError(f'排名含重复条目: {item}')
        seen.add(item)
        if isinstance(item, str):
            if item not in image_names.values():
                raise ValueError(f'排名图片不存在: {item}')
        elif not isinstance(item, Integral) or item not in visible:
            raise ValueError(f'排名 annotation ID 不存在: {item}')

    budget = int((len(result['images']) + len(visible)) * cut_off_rate)
    remaining = budget
    repaired = Counter({'cls': 0, 'loc': 0, 'redun': 0, 'miss': 0})
    removed = set()
    restored = []
    inspected = 0
    for item in rank:
        if remaining <= 0:
            break
        inspected += 1
        if isinstance(item, str):
            # 固定按 ID 顺序补框，预算在图片中途耗尽时结果仍可复现。
            missed_ids = sorted(missing[item])
            if not missed_ids:
                remaining -= 1
            for ann_id in missed_ids:
                if remaining <= 0:
                    break
                ann = deepcopy(clean[ann_id])
                ann.update(fault_type=0, repair_ops='repair_miss')
                restored.append(ann)
                repaired['miss'] += 1
                remaining -= 1
        else:
            remaining -= 1
            ann = visible[item]
            fault = ann['fault_type']
            if fault == 1:
                ann['category_id'] = clean[item]['category_id']
                ann.update(fault_type=0, repair_ops='repair_cls')
                repaired['cls'] += 1
            elif fault == 2:
                ann['bbox'] = deepcopy(clean[item]['bbox'])
                for field in ('area', 'segmentation'):
                    if field in clean[item]:
                        ann[field] = deepcopy(clean[item][field])
                    else:
                        ann.pop(field, None)
                ann.update(fault_type=0, repair_ops='repair_loc')
                repaired['loc'] += 1
            elif fault == 3:
                removed.add(item)
                repaired['redun'] += 1

    result['annotations'] = [ann for ann_id, ann in visible.items() if ann_id not in removed] + restored
    all_faults = sum(ann['fault_type'] != 0 for ann in annotations_with_miss['annotations'])
    stats = {
        'cut_off_rate': cut_off_rate, 'budget': budget,
        'spent': budget - remaining, 'remaining': remaining,
        'rank_entries_inspected': inspected, 'repaired': dict(repaired),
        'total_repaired': sum(repaired.values()), 'all_faults': all_faults,
        'repair_rate': sum(repaired.values()) / all_faults if all_faults else 0,
        'unrepaired_missing': sum(map(len, missing.values())) - repaired['miss'],
        'output_annotations': len(result['annotations']),
    }
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_name = "voc"
    model_name = "yolov7"
    inject_ratio = 0.1
    repeat_id = 10
    method_name = "ours"
    DATA_ROOT ='/data/mml/data_debugging_data/ProAFL_data'
    rank_path = os.path.join(DATA_ROOT,"rank",method_name,dataset_name,model_name,f"{inject_ratio}_repeat",f"repeat_{repeat_id}","rank.joblib")
    # rank_path = os.path.join(DATA_ROOT,"rank",method_name,dataset_name,f"{inject_ratio}_repeat",f"repeat_{repeat_id}","converted_rank.joblib")
    annotations_path = os.path.join(DATA_ROOT,"fault_inject",str(inject_ratio),dataset_name,"coco_format","annotations_with_miss.json")
    correct_path = os.path.join(DATA_ROOT,"clean",dataset_name,"labels", "coco_format","_annotations.coco_correct.json")
    output_path = os.path.join(DATA_ROOT,"corrected_anno",method_name,dataset_name,model_name,f"{inject_ratio}_repeat",f"repeat_{repeat_id}","annotations_corrected.json")
    # output_path = os.path.join(DATA_ROOT,"corrected_anno",method_name,dataset_name,f"{inject_ratio}_repeat",f"repeat_{repeat_id}","annotations_corrected.json")
    parser.add_argument('--rank-path', type=str, default=rank_path)
    parser.add_argument('--annotations-path', type=str, default=annotations_path)
    parser.add_argument('--correct-path', type=str, default=correct_path)
    parser.add_argument('--output-path', type=str, default=output_path)
    parser.add_argument('--cut-off-rate', type=float, default=0.5)
    args = parser.parse_args()
    if os.path.realpath(args.output_path) in {os.path.realpath(args.annotations_path), os.path.realpath(args.correct_path), os.path.realpath(args.rank_path)}:
        parser.error('输出路径不能覆盖输入文件')
    result, stats = repair_annotations(joblib.load(args.rank_path), read_json(args.annotations_path), read_json(args.correct_path), args.cut_off_rate)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'Saved: {args.output_path}')


if __name__ == '__main__':
    main()
