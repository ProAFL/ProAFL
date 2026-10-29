"""按组内百分位合并标注与图片，保持原 merged_rank.json 的结构和字段。"""

import math

from ours.dual_branch.common import read_json, save_json
from ours.dual_branch.config import FEATURES_DIR, IMAGE_PERCENTILE_SCALE


RANK_DIR = FEATURES_DIR
ANNO_RANK_PATH = RANK_DIR / "topsis_ranking.json"
IMAGE_RANK_PATH = RANK_DIR / "miss_image_ranking.json"
OUTPUT_PATH = RANK_DIR / "merged_rank.json"

def split_ranking(source, item_type, id_key, no_evidence_key):
    """将有证据和无证据条目分开，输出字段沿用原 merge_rank。"""
    rows = source["ranking"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{item_type} 排名不能为空")
    evidence, no_evidence, seen = [], [], set()
    for row in rows:
        item_id = row[id_key]
        if item_id in seen:
            raise ValueError(f"重复条目：{item_type}, {item_id}")
        seen.add(item_id)
        score, source_rank = row["topsis_score"], row["rank"]
        if type(source_rank) is not int or source_rank < 1:
            raise ValueError(f"条目名次不合法：{item_id}")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"条目分数不合法：{item_id}")
        if type(row[no_evidence_key]) is not bool:
            raise ValueError(f"条目证据标记不合法：{item_id}")
        no_support = row[no_evidence_key] if item_type == "anno" else not row[no_evidence_key]
        entry = {"item_type": item_type, **row}
        entry["source_rank"] = entry.pop("rank")
        # 标注此字段保存双分支融合分数，保留原字段名以兼容后续读取。
        entry["source_topsis_score"] = entry.pop("topsis_score")
        (no_evidence if no_support else evidence).append(entry)
    evidence.sort(key=lambda entry: (-entry["source_topsis_score"], entry[id_key]))
    no_evidence.sort(key=lambda entry: entry[id_key])
    return evidence, no_evidence


def add_within_group_percentiles(evidence):
    """同分使用平均名次百分位；这里越小越靠前，与分支异常百分位方向相反。"""
    total, start = len(evidence), 0
    while start < total:
        end = start + 1
        score = evidence[start]["source_topsis_score"]
        while end < total and evidence[end]["source_topsis_score"] == score:
            end += 1
        percentile = (start + end) / (2 * total)
        for entry in evidence[start:end]:
            entry["group_rank"] = start + 1
            entry["group_percentile"] = percentile
        start = end


def merge_evidence(annos, images):
    if (isinstance(IMAGE_PERCENTILE_SCALE, bool)
            or not math.isfinite(IMAGE_PERCENTILE_SCALE) or IMAGE_PERCENTILE_SCALE <= 0):
        raise ValueError("图片百分位缩放系数必须为正数")
    add_within_group_percentiles(annos)
    add_within_group_percentiles(images)
    for entry in annos:
        entry["merge_percentile"] = entry["group_percentile"]
    for entry in images:
        entry["merge_percentile"] = entry["group_percentile"] * IMAGE_PERCENTILE_SCALE
    merged = sorted(annos + images, key=lambda entry: (
        entry["merge_percentile"], 0 if entry["item_type"] == "anno" else 1,
        entry["anno_id"] if entry["item_type"] == "anno" else entry["image_name"],
    ))
    previous_percentile, tied_rank = None, 0
    for position, entry in enumerate(merged, start=1):
        if previous_percentile is None or entry["merge_percentile"] != previous_percentile:
            tied_rank = position
            previous_percentile = entry["merge_percentile"]
        entry["merged_rank"] = tied_rank
    return merged


def build_merged_rank(anno_source, image_source):
    annos, no_overlap_annos = split_ranking(anno_source, "anno", "anno_id", "never_overlapped")
    images, no_candidate_images = split_ranking(image_source, "image", "image_name", "has_candidate")
    if "no_candidate_images" in image_source:
        reported = image_source["no_candidate_images"]
        if (not isinstance(reported, list) or len(reported) != len(set(reported))
                or set(reported) != {entry["image_name"] for entry in no_candidate_images}):
            raise ValueError("无候选图片清单与图片排名不一致")
    merged = merge_evidence(annos, images)
    # 保留原来的三段结构：无重叠标注、混合排序、无候选图片。
    all_ranking = no_overlap_annos + merged + no_candidate_images
    for position, entry in enumerate(all_ranking, start=1):
        entry["total_rank"] = position
    return {
        "merge_rule": {
            "description": "Sort evidence-backed items by merge_percentile, ascending",
            "percentile_formula": "(first_tied_position + last_tied_position - 1) / (2 * group_size)",
            "image_percentile_scale": IMAGE_PERCENTILE_SCALE,
            "merge_percentile_formula": "anno: group_percentile; image: group_percentile * image_percentile_scale",
            "group_rank_direction": "1 is most suspicious within its own group",
            "percentile_direction": "smaller is more suspicious",
            "cross_group_ties": "share merged_rank; anno is displayed first",
            "total_rank_order": "never_overlapped_annos, merged_ranking, no_candidate_images",
            "note": "Percentiles express relative group position, not fault probabilities.",
        },
        "counts": {
            "evidence_annos": len(annos), "evidence_images": len(images),
            "merged_evidence_items": len(merged), "never_overlapped_annos": len(no_overlap_annos),
            "no_candidate_images": len(no_candidate_images), "total_items": len(all_ranking),
        },
        "merged_ranking": merged,
        "never_overlapped_annos": no_overlap_annos,
        "no_candidate_images": no_candidate_images,
        "all_ranking": all_ranking,
    }


def main():
    result = build_merged_rank(read_json(ANNO_RANK_PATH), read_json(IMAGE_RANK_PATH))
    save_json(OUTPUT_PATH, result)
    print(f"已合并 {result['counts']['total_items']} 个条目：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
