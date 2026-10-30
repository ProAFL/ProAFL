"""统一流水线：可选原六维或对称八维，直接复用各模块底层函数。"""

import gc
from pathlib import Path

from ours.dual_branch import (
    process_metrics, process_metrics_img,
    process_feature, process_feature_img, merge_rank,
)
from ours.dual_branch.common import read_json, save_json
from ours.dual_branch.config import ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS, LATE_EPOCHS


# ==================== 运行配置：直接修改以下常量 ====================
FEATURE_DIMENSIONS = 8  # 6：原版类别 3 维 + 几何 3 维；8：扩展类别 4 维 + 几何 4 维
GEOMETRY_WEIGHT = 1.0  # 两分支等优先级；可填 [0,1]，减小后优先检查类别异常
PREDICTIONS_DIR_OVERRIDE = None  # None：使用 config.py；也可填完整预测目录字符串或 Path
REUSE_METRICS = True  # 首次切换预测目录必须重新收集；同一来源采集完成后可改 True 复用
# 图片系数仍在 config.py 的 IMAGE_PERCENTILE_SCALE 设置，目前 VisDrone/YOLOv7 为 0.5。


def main(predictions_dir=None, geometry_weight=None, reuse_metrics=None, feature_dimensions=None):
    # 脚本和无参数 Python 调用都读取上面的常量；显式函数参数用于对照调用。
    predictions_dir = PREDICTIONS_DIR_OVERRIDE if predictions_dir is None else predictions_dir
    geometry_weight = GEOMETRY_WEIGHT if geometry_weight is None else geometry_weight
    reuse_metrics = REUSE_METRICS if reuse_metrics is None else reuse_metrics
    feature_dimensions = FEATURE_DIMENSIONS if feature_dimensions is None else feature_dimensions
    process_feature.get_branch_feature_names(feature_dimensions)
    process_feature.validate_geometry_weight(geometry_weight)
    predictions_dir = PREDICTIONS_DIR if predictions_dir is None else Path(predictions_dir)
    if reuse_metrics:
        for path in (process_metrics.OUTPUT_PATH, process_metrics_img.OUTPUT_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"无法复用过程度量，文件不存在：{path}")
        print("\n[1/5, 2/5] 复用已有标注和图片过程度量", flush=True)
    else:
        # 保存后释放大对象，避免两种轨迹同时驻留内存。
        print("\n[1/5] 收集标注过程度量", flush=True)
        metrics = process_metrics.collect_process_metrics(ANNOTATIONS_PATH, predictions_dir, EPOCHS)
        save_json(process_metrics.OUTPUT_PATH, metrics)
        del metrics
        gc.collect()

        print("\n[2/5] 收集图片过程度量", flush=True)
        metrics = process_metrics_img.collect_metrics(ANNOTATIONS_PATH, predictions_dir, EPOCHS)
        save_json(process_metrics_img.OUTPUT_PATH, metrics)
        del metrics
        gc.collect()

    # 3. 根据常量选择原六维或扩展八维；保存函数返回合并需要的数据。
    print(f"\n[3/5] 标注 {feature_dimensions} 维特征与双分支排序", flush=True)
    features, ranking, branches, no_overlap = process_feature.build_results(
        ANNOTATIONS_PATH, process_metrics.OUTPUT_PATH, predictions_dir=predictions_dir,
        geometry_weight=geometry_weight, feature_dimensions=feature_dimensions,
    )
    anno_source = process_feature.save_results(features, ranking, branches, no_overlap,
                                              geometry_weight=geometry_weight,
                                              feature_dimensions=feature_dimensions)
    del features, ranking, branches, no_overlap
    gc.collect()

    # 4. 从图片轨迹提取六项特征，使用原图片 TOPSIS 规则排序。
    print("\n[4/5] 图片特征提取与排序", flush=True)
    metrics = read_json(process_metrics_img.OUTPUT_PATH)
    features, epochs = process_feature_img.build_image_features(metrics, LATE_EPOCHS)
    del metrics
    ranking, used_names = process_feature_img.build_ranking(features)
    image_source = process_feature_img.save_results(features, epochs, ranking, used_names)
    del features, ranking, used_names
    gc.collect()

    # 5. 两个排名直接通过 Python 对象传给合并函数，无需再读取排名 JSON。
    # 异常自然向上传递，任何一步失败都会停止；最终 JSON 结构保持原格式。
    print("\n[5/5] 合并标注与图片排名", flush=True)
    result = merge_rank.build_merged_rank(anno_source, image_source)
    save_json(merge_rank.OUTPUT_PATH, result)
    print(f"流程完成，共 {result['counts']['total_items']} 个条目：{merge_rank.OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
