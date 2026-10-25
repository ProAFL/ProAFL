"""JSON 读写、等权 TOPSIS 和同分排名的公共函数。"""

import json

import numpy as np
from scipy.stats import rankdata


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def topsis_scores(matrix):
    """全部特征为正向、等权；分数越大越可疑，与实验脚本公式一致。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not matrix.shape[1]:
        raise ValueError("TOPSIS 需要非空的二维特征矩阵")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("特征不能包含 NaN 或无穷值")
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-15)
    # 等权的共同系数不会影响最终距离比值，因此无需再乘 1 / 特征数。
    positive = np.linalg.norm(normalized - normalized.max(axis=0), axis=1)
    negative = np.linalg.norm(normalized - normalized.min(axis=0), axis=1)
    return negative / np.maximum(positive + negative, 1e-15)


def suspicious_percentiles(scores, evidence):
    """仅在有证据的标注内计算升序平均名次百分位，越大越可疑。"""
    percentiles = np.zeros(len(scores), dtype=np.float64)
    count = int(np.count_nonzero(evidence))
    if count:
        percentiles[evidence] = (rankdata(scores[evidence], method="average") - 0.5) / count
    return percentiles


def rank_rows(rows, id_key):
    """分数降序排列；同分共用最小名次，ID 只用于固定展示顺序。"""
    rows = sorted(rows, key=lambda row: (-row["topsis_score"], row[id_key]))
    previous_score = None
    tied_rank = 0
    for position, row in enumerate(rows, start=1):
        if previous_score is None or row["topsis_score"] != previous_score:
            tied_rank = position
            previous_score = row["topsis_score"]
        row["rank"] = tied_rank
    return rows
