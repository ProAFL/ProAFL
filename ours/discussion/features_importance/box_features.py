'''
对排序使用的features进行讨论
'''
import os
import csv
import numpy as np
from helper.data_organization_tools import (get_all_gids,get_g_id_to_metric,
                                          get_all_errored_g_box_id_set,get_all_correct_g_box_id_set)
from helper.base_data_manager import get_collected_gt_box_json_path,exp_data_root_dir
from ours.small_utils import read_json
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import topsispy as tp
import pandas as pd


FEATURE_NAME_TO_SIGN = {
    "early_conf_mean": -1,
    "early_iou_mean": -1,
    "lastly_conf_mean": -1,
    "lastly_iou_mean": -1,
    "conf_mean": -1,
    "iou_mean": -1,
    "D_conf": 1,
    "D_iou": 1,
}


def split_gid_clean_error(gt_json):
    error_gid_set = get_all_errored_g_box_id_set(gt_json)
    correct_gid_set = get_all_correct_g_box_id_set(gt_json)
    return correct_gid_set,error_gid_set

def build_gid_feature(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
    g_id_to_features = {}
    for g_id in g_box_id_to_metric.keys():
        conf_list = g_box_id_to_metric[g_id]["conf_list"]
        iou_list = g_box_id_to_metric[g_id]["iou_list"]
        epochs = len(conf_list)
        W_e = int(K*epochs)
        W_l = int(K*epochs)
        # 早期置信度均值，越小越可疑
        early_conf_mean = np.mean(conf_list[0:W_e])
        # 后期置信度均值，越小越可疑
        lastly_conf_mean = np.mean(conf_list[-W_l:])
        # 早期iou均值，越小越可疑
        early_iou_mean = np.mean(iou_list[0:W_e])
        # 后期iou均值，越小越可疑
        lastly_iou_mean = np.mean(iou_list[-W_l:])

        # 全局均值，越小越可疑
        conf_mean = np.mean(conf_list)
        iou_mean = np.mean(iou_list)

        conf_threshold = 0.5*lastly_conf_mean
        iou_threshold = 0.5*lastly_iou_mean

        min_e_conf = epochs
        min_e_iou = epochs
        for e in range(epochs):
            if conf_list[e] > conf_threshold:
                min_e_conf = e
                break
        for e in range(epochs):
            if iou_list[e] > iou_threshold:
                min_e_iou = e
                break
        # 起量延迟（显式刻画“涨得晚”）
        # 越大越可疑
        D_conf = min_e_conf / epochs
        D_iou = min_e_iou / epochs

        g_id_to_features[g_id] = {
            "early_conf_mean":early_conf_mean, # 早期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "early_iou_mean":early_iou_mean, # 早期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_conf_mean":lastly_conf_mean, # 后期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_iou_mean":lastly_iou_mean, # 后期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "conf_mean":conf_mean, # 全期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "iou_mean":iou_mean, # 全期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "D_conf":D_conf, # 起量延迟 conf，越大越可疑 -> topsis分数越高 -> 1
            "D_iou":D_iou, # 起量延迟 iou，越大越可疑 -> topsis分数越高 -> 1
        }
    feature_name_to_sign = FEATURE_NAME_TO_SIGN.copy()

    print(f"all gbox数量:{len(all_gids)}")
    print(f"matched gbox数量:{len(g_id_to_features)}")
    
    for g_id in all_gids:
        if g_id not in g_id_to_features:
            # 没有匹配上的gid都是最可疑的
            g_id_to_features[g_id] = {
                "early_conf_mean":0,
                "early_iou_mean":0,
                "lastly_conf_mean":0,
                "lastly_iou_mean":0,
                "conf_mean":0,
                "iou_mean":0,
                "D_conf":1,
                "D_iou":1, 
            }
    return (g_id_to_features,feature_name_to_sign)

def hypothesis_testing(list_1:list[float],list_2:list[float],alternative:str="two-sided"):
    def mannwhitneyu_effect_size(u_stat, n1, n2):
        """
        计算Mann-Whitney U检验的效应量r（正确版本）
        参数：
            u_stat: mannwhitneyu返回的U统计量
            n1: 第一组数据的样本量
            n2: 第二组数据的样本量
        返回：
            效应量r（绝对值），越大表示差异越明显
        """
        # 步骤1：计算U统计量的均值（零假设下的期望U值）
        mean_u = (n1 * n2) / 2
        # 步骤2：计算U统计量的标准差
        std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        # 步骤3：将U值转换为Z分数（标准化）
        z = (u_stat - mean_u) / std_u
        # 步骤4：计算效应量r（Cohen's r）
        r = abs(z) / np.sqrt(n1 + n2)
        return r
    
    u_stat, u_p = stats.mannwhitneyu(list_1, list_2, alternative=alternative)
    
    print(f"Mann-Whitney U检验：U值={u_stat:.3f}, p值={u_p:.3f}")
    # 计算效应量r
    r = mannwhitneyu_effect_size(u_stat, len(list_1), len(list_2))
    print(f"效应量r：{r:.3f}")

def visualization(correct_list,error_list,save_file_name:str):
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 箱线图
    ax1.boxplot([correct_list, error_list], labels=['correct', 'error'])
    ax1.set_title('Box plot: Data distribution comparison')
    ax1.set_ylabel('Numerical value')
    # 直方图+核密度估计（KDE）
    # 1. 可视化：箱线图（看分布位置、离散程度）+ 直方图（看分布形态）

    sns.histplot(correct_list, kde=True, ax=ax2, label='correct', alpha=0.5)
    sns.histplot(error_list, kde=True, ax=ax2, label='error', alpha=0.5)
    ax2.set_title('Histogram +KDE: Shape of distribution')
    ax2.legend()
    plt.savefig(f"/data/mml/data_debugging_data/temp/{save_file_name}.png")

def plot_roc_auc(g_id_to_features, feature_name_to_sign, correct_gid_set, error_gid_set, save_file_name:str="roc_auc"):
    """
    把所有 feature 各自的 ROC 以及 topsis 综合 score 的 ROC 画在同一张图上。
    label: error=1, correct=0
    feature score: 把 feature 转成"越大越可疑"。sign==1 直接用；sign==-1 取相反数。
    topsis score: 所有 feature 一起传入 topsis 得到的综合分数（越大越可疑）。
    """
    plt.figure(figsize=(8, 7))
    name_to_auc = {}

    # 各个 feature 的 ROC
    for feature_name, sign in feature_name_to_sign.items():
        y_true = []
        y_score = []
        for gid in correct_gid_set:
            y_true.append(0)
            y_score.append(sign * float(g_id_to_features[gid][feature_name]))
        for gid in error_gid_set:
            y_true.append(1)
            y_score.append(sign * float(g_id_to_features[gid][feature_name]))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        name_to_auc[feature_name] = roc_auc
        plt.plot(fpr, tpr, lw=1.5, label=f"{feature_name} (AUC={roc_auc:.3f})")

    # topsis 综合 score 的 ROC
    g_id_list = sorted(g_id_to_features.keys())
    gid_to_idx = {gid: i for i, gid in enumerate(g_id_list)}
    feature_names = list(feature_name_to_sign.keys())
    sign_list = [feature_name_to_sign[fn] for fn in feature_names]
    data = np.array([[float(g_id_to_features[gid][fn]) for fn in feature_names] for gid in g_id_list])
    weights = np.ones(len(feature_names)) / len(feature_names)
    _, score_array = tp.topsis(data, weights, sign_list)
    score_array = np.asarray(score_array)

    eval_gids = [gid for gid in g_id_list if gid in correct_gid_set or gid in error_gid_set]
    y_true_t = np.array([1 if gid in error_gid_set else 0 for gid in eval_gids])
    y_score_t = np.array([score_array[gid_to_idx[gid]] for gid in eval_gids])
    fpr_t, tpr_t, _ = roc_curve(y_true_t, y_score_t)
    roc_auc_t = auc(fpr_t, tpr_t)
    name_to_auc["TOPSIS_score"] = roc_auc_t
    plt.plot(fpr_t, tpr_t, lw=2.5, color='black', label=f"TOPSIS_score (AUC={roc_auc_t:.3f})")

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='random (AUC=0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves of Box Features & TOPSIS Score ({save_file_name})')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/data/mml/data_debugging_data/temp/{save_file_name}.png", dpi=150)
    plt.close()

    print("="*60)
    print("AUC 排名:")
    for fn, a in sorted(name_to_auc.items(), key=lambda x: -x[1]):
        print(f"  {fn:20s}  AUC={a:.4f}")
    return name_to_auc


def _safe_corr(func, x, y):
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    stat, p_value = func(x, y)
    return float(stat), float(p_value)

def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))

def _discretize_by_quantile(values, n_bins=10):
    """
    用分位数离散化连续特征，便于计算信息熵/互信息。
    重复取值很多时，实际 bin 数可能小于 n_bins。
    """
    values = np.asarray(values, dtype=float)
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        return np.zeros(len(values), dtype=int)
    quantiles = np.linspace(0, 1, min(n_bins, len(unique_values)) + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) == 0:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges, right=False)

def _mutual_information_discrete(feature_bins, labels):
    feature_bins = np.asarray(feature_bins, dtype=int)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    feature_values = np.unique(feature_bins)
    label_values = np.unique(labels)
    joint_counts = np.zeros((len(feature_values), len(label_values)), dtype=float)
    f_to_i = {v: i for i, v in enumerate(feature_values)}
    y_to_i = {v: i for i, v in enumerate(label_values)}
    for f, y in zip(feature_bins, labels):
        joint_counts[f_to_i[f], y_to_i[y]] += 1

    pxy = joint_counts / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))

    h_feature = _entropy_from_counts(joint_counts.sum(axis=1))
    h_label = _entropy_from_counts(joint_counts.sum(axis=0))
    mi_over_h_label = mi / h_label if h_label > 0 else float("nan")
    mi_over_h_feature = mi / h_feature if h_feature > 0 else float("nan")
    return float(mi), float(h_feature), float(h_label), float(mi_over_h_label), float(mi_over_h_feature)

def correlation_importance_analysis(df, feature_names, subset_name):
    labels = _series_to_bool(df["is_error"]).astype(int)
    rows = []
    for feature_name in feature_names:
        raw_values = df[feature_name].astype(float).to_numpy()
        sign = int(df[f"{feature_name}_sign"].iloc[0])
        suspicious_values = sign * raw_values
        pearson_r, pearson_p = _safe_corr(stats.pearsonr, suspicious_values, labels)
        spearman_r, spearman_p = _safe_corr(stats.spearmanr, suspicious_values, labels)
        point_biserial_r, point_biserial_p = _safe_corr(stats.pointbiserialr, raw_values, labels)
        rows.append({
            "subset": subset_name,
            "feature": feature_name,
            "n": len(labels),
            "n_error": int(labels.sum()),
            "n_correct": int(len(labels) - labels.sum()),
            "pearson_r_suspicious": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r_suspicious": spearman_r,
            "spearman_p": spearman_p,
            "point_biserial_r_raw": point_biserial_r,
            "point_biserial_p": point_biserial_p,
            "abs_spearman_r": abs(spearman_r) if not np.isnan(spearman_r) else float("nan"),
        })
    rows.sort(key=lambda row: -row["abs_spearman_r"] if not np.isnan(row["abs_spearman_r"]) else float("-inf"))
    return rows

def mutual_information_importance_analysis(df, feature_names, subset_name, n_bins=10):
    labels = _series_to_bool(df["is_error"]).astype(int)
    rows = []
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return rows

    suspicious_matrix = np.column_stack([
        int(df[f"{feature_name}_sign"].iloc[0]) * df[feature_name].astype(float).to_numpy()
        for feature_name in feature_names
    ])
    n_neighbors = max(1, min(3, len(labels) - 1))
    mi_continuous = mutual_info_classif(
        suspicious_matrix, labels, discrete_features=False,
        n_neighbors=n_neighbors, random_state=0
    )
    for idx, feature_name in enumerate(feature_names):
        suspicious_values = suspicious_matrix[:, idx]
        feature_bins = _discretize_by_quantile(suspicious_values, n_bins=n_bins)
        mi_bits, h_feature, h_label, mi_over_h_label, mi_over_h_feature = (
            _mutual_information_discrete(feature_bins, labels)
        )
        rows.append({
            "subset": subset_name,
            "feature": feature_name,
            "n": len(labels),
            "n_error": int(labels.sum()),
            "n_correct": int(len(labels) - labels.sum()),
            "bins": int(len(np.unique(feature_bins))),
            "mi_bits_quantile": mi_bits,
            "h_feature_bits": h_feature,
            "h_label_bits": h_label,
            "mi_over_h_label": mi_over_h_label,
            "mi_over_h_feature": mi_over_h_feature,
            "mi_knn_nats": float(mi_continuous[idx]),
        })
    rows.sort(key=lambda row: -row["mi_over_h_label"] if not np.isnan(row["mi_over_h_label"]) else float("-inf"))
    return rows

def write_rows_to_csv(rows, output_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def plot_importance_bar(rows, metric_name, title, output_path):
    if not rows:
        return
    feature_names = [row["feature"] for row in rows]
    values = [row[metric_name] for row in rows]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=values, y=feature_names, orient="h")
    plt.xlabel(metric_name)
    plt.ylabel("feature")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_feature_correlation_heatmap(df, feature_names, subset_name, output_path):
    matrix = np.column_stack([
        int(df[f"{feature_name}_sign"].iloc[0]) * df[feature_name].astype(float).to_numpy()
        for feature_name in feature_names
    ])
    if matrix.shape[0] < 2:
        return
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title(f"Feature Correlation Heatmap ({subset_name})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def print_importance_rows(title, rows, metric_name):
    print("="*60)
    print(title)
    for row in rows:
        value = row[metric_name]
        if np.isnan(value):
            value_text = "nan"
        else:
            value_text = f"{value:.4f}"
        print(f"  {row['feature']:20s}  {metric_name}={value_text}")

def _series_to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool).to_numpy()
    true_values = {"true", "1", "yes", "y", "t"}
    return series.astype(str).str.strip().str.lower().isin(true_values).to_numpy()

def get_feature_names_from_df(df:pd.DataFrame):
    required_columns = {"is_matched", "is_error"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV缺少必要列: {missing_columns}")

    feature_names = [fn for fn in FEATURE_NAME_TO_SIGN.keys() if fn in df.columns]
    if len(feature_names) != len(FEATURE_NAME_TO_SIGN):
        missing_features = sorted(set(FEATURE_NAME_TO_SIGN.keys()) - set(feature_names))
        raise ValueError(f"CSV缺少过程特征列: {missing_features}")

    missing_sign_columns = [
        f"{feature_name}_sign"
        for feature_name in feature_names
        if f"{feature_name}_sign" not in df.columns
    ]
    if missing_sign_columns:
        raise ValueError(f"CSV缺少feature sign列: {missing_sign_columns}")
    return feature_names

def run_feature_importance_analysis(df:pd.DataFrame):
    feature_names = get_feature_names_from_df(df)
    result_dir = RESULT_DIR
    output_prefix = f"box_feature_importance_{dataset_name}_{model_name}"
    # all gid set / matched gid set
    subsets = {
        "all": df,
        "matched_only": df[_series_to_bool(df["is_matched"])],
    }
    all_corr_rows = [] # 相关性
    all_mi_rows = [] # 互信息
    for subset_name, subset_df in subsets.items():
        labels = _series_to_bool(subset_df["is_error"]).astype(int)
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            print(f"[skip] {subset_name}: 样本为空或只包含单一类别，无法计算重要性。")
            continue
        corr_rows = correlation_importance_analysis(subset_df, feature_names, subset_name)
        mi_rows = mutual_information_importance_analysis(subset_df, feature_names, subset_name)
        all_corr_rows.extend(corr_rows)
        all_mi_rows.extend(mi_rows)

        print_importance_rows(
            f"相关性重要性排名 ({subset_name}, 按 |Spearman r|)",
            corr_rows,
            "abs_spearman_r",
        )
        print_importance_rows(
            f"信息熵/互信息重要性排名 ({subset_name}, 按 MI/H(label))",
            mi_rows,
            "mi_over_h_label",
        )

        plot_importance_bar(
            corr_rows,
            "abs_spearman_r",
            f"Correlation Importance ({subset_name})",
            f"{result_dir}/{output_prefix}_{subset_name}_correlation_importance.png",
        )
        plot_importance_bar(
            mi_rows,
            "mi_over_h_label",
            f"Mutual Information Importance ({subset_name})",
            f"{result_dir}/{output_prefix}_{subset_name}_mi_importance.png",
        )
        plot_feature_correlation_heatmap(
            subset_df,
            feature_names,
            subset_name,
            f"{result_dir}/{output_prefix}_{subset_name}_feature_corr_heatmap.png",
        )

    write_rows_to_csv(all_corr_rows, f"{result_dir}/{output_prefix}_correlation_importance.csv")
    write_rows_to_csv(all_mi_rows, f"{result_dir}/{output_prefix}_mutual_information_importance.csv")
    return all_corr_rows, all_mi_rows

def get_csv_table_for_feature(save_file_name=None):
    """
    导出 gid 级 feature 表。
    每行是一个 gid，列包括 8 个过程特征、是否 matched、以及 correct/error。
    """
    # 加载gt box的json数据
    gt_json = read_json(gt_json_path)
    # 得到所有的gids
    all_gids = get_all_gids(gt_json)
    # 得到被matched gid的conf/iou list
    gid_to_metric = get_g_id_to_metric(g_box_metrics_json_path)
    # 得到每个gid的features和每个feature的sign
    g_id_to_features,feature_name_to_sign = build_gid_feature(all_gids,gid_to_metric,K=0.2)
    # correct/error gid set
    correct_gid_set,error_gid_set = split_gid_clean_error(gt_json)
    # 单独拎出来 matched gid set
    matched_gid_set = set(gid_to_metric.keys())

    # csv 保存路径
    if save_file_name is None:
        save_file_name = f"box_feature_table_{dataset_name}_{model_name}.csv"
    output_path = os.path.join(RESULT_DIR, save_file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feature_names = list(feature_name_to_sign.keys())
    sign_fieldnames = [f"{feature_name}_sign" for feature_name in feature_names]
    fieldnames = ["gid", "is_matched", "is_error"] + feature_names + sign_fieldnames
    rows = []
    for gid in sorted(all_gids):
        row = {
            "gid": gid,
            "is_matched": gid in matched_gid_set,
            "is_error": gid in error_gid_set
        }
        for feature_name in feature_names:
            row[feature_name] = float(g_id_to_features[gid][feature_name])
            row[f"{feature_name}_sign"] = int(feature_name_to_sign[feature_name])
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"gid feature csv saved: {output_path}")
    return output_path

def main():
    mode = 0 # 0:全程贯通,1:基于csv进行特征重要性分析
    if mode == 0: 
        csv_path = get_csv_table_for_feature()
    if mode == 0 or mode == 1:
        if mode == 1:
            csv_path = os.path.join(RESULT_DIR,f"box_feature_table_{dataset_name}_{model_name}.csv")
        df = pd.read_csv(csv_path)
        run_feature_importance_analysis(df)


if __name__ == "__main__":
    RESULT_DIR = "/data/mml/data_debugging_data/discussion/"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    g_box_metrics_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                           dataset_name,model_name,"collection_metric",
                                           "collection_metrics_v2.json")
    if dataset_name == "VisDrone":
        g_box_metrics_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                           dataset_name,model_name,"collection_metric",
                                           "collection_metrics_v3.json")
    main()
