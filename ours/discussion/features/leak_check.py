'''
排查 box_features 的 AUC 是否被"未匹配 gid 默认极值"放大
'''
import os
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
import topsispy as tp

from ours.data_organization_tools import (get_all_gids, get_g_id_to_metric,
                                          get_all_errored_g_box_id_set,
                                          get_all_correct_g_box_id_set)
from ours.base_data_manager import exp_data_root_dir
from ours.discussion.features.box_features import build_gid_feature, split_gid_clean_error
from ours.small_utils import read_json
from ours.base_data_manager import get_collected_gt_box_json_path


def per_fault_type_unmatched(gt_json, matched_gid_set):
    counter_total = Counter()
    counter_unmatched = Counter()
    for img_name, g_boxs in gt_json.items():
        for gb in g_boxs:
            ft = gb["fault_type"]
            counter_total[ft] += 1
            if gb["box_id"] not in matched_gid_set:
                counter_unmatched[ft] += 1
    name = {0: "no_fault", 1: "cls_fault", 2: "loc_fault",
            3: "redundancy_fault", 4: "missing_fault"}
    print(f"{'fault_type':<20}{'total':>8}{'unmatched':>12}{'ratio':>10}")
    for ft in sorted(counter_total):
        t = counter_total[ft]
        u = counter_unmatched[ft]
        print(f"{name.get(ft,ft):<20}{t:>8}{u:>12}{u/t:>10.3f}")


def auc_with_and_without_unmatched(g_id_to_features, feature_name_to_sign,
                                    correct_gid_set, error_gid_set, matched_gid_set):
    print(f"\n{'feature':<22}{'AUC(all)':>12}{'AUC(matched only)':>20}"
          f"{'#err_match':>12}{'#cor_match':>12}")
    for fn, sign in feature_name_to_sign.items():
        y_all, s_all = [], []
        y_m, s_m = [], []
        n_e_m = n_c_m = 0
        for gid in correct_gid_set:
            v = sign * float(g_id_to_features[gid][fn])
            y_all.append(0); s_all.append(v)
            if gid in matched_gid_set:
                y_m.append(0); s_m.append(v); n_c_m += 1
        for gid in error_gid_set:
            v = sign * float(g_id_to_features[gid][fn])
            y_all.append(1); s_all.append(v)
            if gid in matched_gid_set:
                y_m.append(1); s_m.append(v); n_e_m += 1
        auc_all = roc_auc_score(y_all, s_all)
        auc_m = roc_auc_score(y_m, s_m) if (n_e_m > 0 and n_c_m > 0) else float('nan')
        print(f"{fn:<22}{auc_all:>12.4f}{auc_m:>20.4f}{n_e_m:>12}{n_c_m:>12}")

    # 综合 TOPSIS score：所有 feature 一起跑 TOPSIS，看综合 score 的 AUC
    g_id_list = sorted(g_id_to_features.keys())
    gid_to_idx = {gid: i for i, gid in enumerate(g_id_list)}
    feature_names = list(feature_name_to_sign.keys())
    sign_list = [feature_name_to_sign[fn] for fn in feature_names]
    data = np.array([[float(g_id_to_features[gid][fn]) for fn in feature_names] for gid in g_id_list])
    weights = np.ones(len(feature_names)) / len(feature_names)
    _, score_array = tp.topsis(data, weights, sign_list)
    score_array = np.asarray(score_array)

    y_all, s_all, y_m, s_m = [], [], [], []
    n_e_m = n_c_m = 0
    for gid in correct_gid_set:
        v = float(score_array[gid_to_idx[gid]])
        y_all.append(0); s_all.append(v)
        if gid in matched_gid_set:
            y_m.append(0); s_m.append(v); n_c_m += 1
    for gid in error_gid_set:
        v = float(score_array[gid_to_idx[gid]])
        y_all.append(1); s_all.append(v)
        if gid in matched_gid_set:
            y_m.append(1); s_m.append(v); n_e_m += 1
    auc_all = roc_auc_score(y_all, s_all)
    auc_m = roc_auc_score(y_m, s_m) if (n_e_m > 0 and n_c_m > 0) else float('nan')
    print(f"{'TOPSIS_score':<22}{auc_all:>12.4f}{auc_m:>20.4f}{n_e_m:>12}{n_c_m:>12}")


def run(dataset_name, model_name="YOLOv7"):
    print("="*80)
    print(f"DATASET = {dataset_name}, MODEL = {model_name}")
    print("="*80)
    gt_json = read_json(get_collected_gt_box_json_path(dataset_name))
    metric_path = os.path.join(exp_data_root_dir, "collection_bbox_level",
                               dataset_name, model_name,
                               "collection_metric", "collection_metrics_v2.json")
    gid_to_metric = get_g_id_to_metric(metric_path)
    matched_gid_set = set(gid_to_metric.keys())

    all_gids = get_all_gids(gt_json)
    correct_gid_set, error_gid_set = split_gid_clean_error(gt_json)

    print(f"\n#all gids = {len(all_gids)}, #matched = {len(matched_gid_set)}, "
          f"unmatched ratio = {1 - len(matched_gid_set)/len(all_gids):.3f}")
    print(f"#correct = {len(correct_gid_set)}, #error = {len(error_gid_set)}")

    # 1. unmatched 分布
    print("\n[1] unmatched 比例按 fault_type 拆分:")
    per_fault_type_unmatched(gt_json, matched_gid_set)

    # 2. error / correct 中 unmatched 占比
    err_unm = sum(1 for g in error_gid_set if g not in matched_gid_set)
    cor_unm = sum(1 for g in correct_gid_set if g not in matched_gid_set)
    print(f"\n[2] error 中 unmatched: {err_unm}/{len(error_gid_set)} = "
          f"{err_unm/max(1,len(error_gid_set)):.3f}")
    print(f"    correct 中 unmatched: {cor_unm}/{len(correct_gid_set)} = "
          f"{cor_unm/max(1,len(correct_gid_set)):.3f}")

    # 3. 仅用"是否未匹配"这一个二值特征算 AUC
    y, s = [], []
    for gid in correct_gid_set:
        y.append(0); s.append(0 if gid in matched_gid_set else 1)
    for gid in error_gid_set:
        y.append(1); s.append(0 if gid in matched_gid_set else 1)
    print(f"\n[3] 仅用 'unmatched=1' 二值特征的 AUC = {roc_auc_score(y, s):.4f}")

    # 4. 各 feature 的 AUC：全集 vs 只用 matched gid
    g_id_to_features, feature_name_to_sign = build_gid_feature(all_gids, gid_to_metric, K=0.2)
    print("\n[4] 各 feature AUC: 全集 vs 仅 matched gid (剔除默认极值)")
    auc_with_and_without_unmatched(g_id_to_features, feature_name_to_sign,
                                    correct_gid_set, error_gid_set, matched_gid_set)


if __name__ == "__main__":
    for ds in ["VOC2012", "KITTI_8", "VisDrone"]:
        run(ds, "YOLOv7")
