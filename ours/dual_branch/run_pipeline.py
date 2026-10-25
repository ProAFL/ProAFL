"""统一流程入口：在一个 main() 内复用五个模块的底层函数，不调用它们的 main()。"""

import gc

from ours.dual_branch import (
    process_metrics, process_metrics_img,
    process_feature, process_feature_img, merge_rank,
)
from ours.dual_branch.common import read_json, save_json
from ours.dual_branch.config import ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS, LATE_EPOCHS


def main():
    # 1. 收集标注轨迹。保存中间结果后释放大对象，避免两种轨迹同时驻留内存。
    print("\n[1/5] 收集标注过程度量", flush=True)
    metrics = process_metrics.collect_process_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    save_json(process_metrics.OUTPUT_PATH, metrics)
    del metrics
    gc.collect()

    # 2. 收集图片中的漏标候选轨迹，沿用原来的筛选和位置关联规则。
    print("\n[2/5] 收集图片过程度量", flush=True)
    metrics = process_metrics_img.collect_metrics(ANNOTATIONS_PATH, PREDICTIONS_DIR, EPOCHS)
    save_json(process_metrics_img.OUTPUT_PATH, metrics)
    del metrics
    gc.collect()

    # 3. 提取完整六特征并做双分支排序；保存函数同时返回合并需要的数据。
    print("\n[3/5] 标注六特征双分支排序", flush=True)
    features, ranking, branches, no_overlap = process_feature.build_results(
        ANNOTATIONS_PATH, process_metrics.OUTPUT_PATH,
    )
    anno_source = process_feature.save_results(features, ranking, branches, no_overlap)
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
