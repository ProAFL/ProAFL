"""Merge the two evidence-backed rankings by their within-group percentiles.

Annotation and image TOPSIS scores are never compared directly. The
never-overlapped annotations and images without candidate tracks are kept in
separate, unranked groups because their scores have no within-group resolution.

Edit the paths below, then run ``python -m ours.merge_rank`` from the repo root.
"""

import json
import math
from pathlib import Path


DATA_ROOT = Path("/data/mml/data_debugging_data/ProAFL_data")
RANK_DIR = DATA_ROOT / (
    "process_features/voc/yolov7/0.1_repeat/repeat_1_1791094449"
)
ANNO_RANK_PATH = RANK_DIR / "topsis_ranking.json"
IMAGE_RANK_PATH = RANK_DIR / "miss_image_ranking.json"
OUTPUT_PATH = RANK_DIR / "merged_rank.json"


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def split_ranking(source, item_type, id_key, no_evidence_key):
    """Separate scored items from items with no distinguishing process evidence."""
    rows = source["ranking"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{item_type}: ranking must be a nonempty list")

    evidence = []
    no_evidence = []
    seen_ids = set()
    for row in rows:
        item_id = row[id_key]
        if item_id in seen_ids:
            raise ValueError(f"{item_type}: duplicate {id_key} {item_id}")
        seen_ids.add(item_id)

        source_rank = row["rank"]
        score = row["topsis_score"]
        if type(source_rank) is not int or source_rank < 1:
            raise ValueError(f"{item_type} {item_id}: invalid source rank")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{item_type} {item_id}: invalid TOPSIS score")
        if type(row[no_evidence_key]) is not bool:
            raise ValueError(f"{item_type} {item_id}: invalid evidence flag")
        no_support = row[no_evidence_key] if item_type == "anno" else not row[no_evidence_key]

        entry = {"item_type": item_type, **row}
        entry["source_rank"] = entry.pop("rank")
        entry["source_topsis_score"] = entry.pop("topsis_score")
        (no_evidence if no_support else evidence).append(entry)

    evidence.sort(key=lambda entry: (-entry["source_topsis_score"], entry[id_key]))
    # These items have no meaningful within-group order; ID only fixes display order.
    no_evidence.sort(key=lambda entry: entry[id_key])
    return evidence, no_evidence


def add_within_group_percentiles(evidence):
    """Assign the same midrank percentile to exactly tied TOPSIS scores."""
    total = len(evidence)
    start = 0
    while start < total:
        end = start + 1
        score = evidence[start]["source_topsis_score"]
        while end < total and evidence[end]["source_topsis_score"] == score:
            end += 1
        # Positions are start+1, ..., end. Their mean (minus 0.5) / total.
        percentile = (start + end) / (2 * total)
        for entry in evidence[start:end]:
            entry["group_rank"] = start + 1
            entry["group_percentile"] = percentile
        start = end


def merge_evidence(annos, images):
    """Sort both groups together by percentile; smaller is more suspicious."""
    add_within_group_percentiles(annos)
    add_within_group_percentiles(images)
    merged = sorted(
        annos + images,
        key=lambda entry: (
            entry["group_percentile"],
            0 if entry["item_type"] == "anno" else 1,
            entry["anno_id"] if entry["item_type"] == "anno" else entry["image_name"],
        ),
    )
    previous_percentile = None
    tied_rank = 0
    for position, entry in enumerate(merged, start=1):
        percentile = entry["group_percentile"]
        if previous_percentile is None or percentile != previous_percentile:
            tied_rank = position
            previous_percentile = percentile
        entry["merged_rank"] = tied_rank
    return merged


def build_merged_rank(anno_source, image_source):
    annos, no_overlap_annos = split_ranking(
        anno_source, "anno", "anno_id", "never_overlapped"
    )
    images, no_candidate_images = split_ranking(
        image_source, "image", "image_name", "has_candidate"
    )
    if "no_candidate_images" in image_source:
        reported = image_source["no_candidate_images"]
        if (not isinstance(reported, list) or len(reported) != len(set(reported))
                or set(reported) != {entry["image_name"] for entry in no_candidate_images}):
            raise ValueError("Image no_candidate_images disagrees with ranking flags")

    merged = merge_evidence(annos, images)
    all_ranking = no_overlap_annos + merged + no_candidate_images
    for position, entry in enumerate(all_ranking, start=1):
        entry["total_rank"] = position
    return {
        "merge_rule": {
            "description": "Sort evidence-backed items by within-group TOPSIS midrank percentile, ascending",
            "percentile_formula": "(first_tied_position + last_tied_position - 1) / (2 * group_size)",
            "group_rank_direction": "1 is most suspicious within its own group",
            "percentile_direction": "smaller is more suspicious",
            "cross_group_ties": "share merged_rank; anno is displayed first",
            "total_rank_order": "never_overlapped_annos, merged_ranking, no_candidate_images",
            "note": "Percentiles express relative group position, not fault probabilities.",
        },
        "counts": {
            "evidence_annos": len(annos),
            "evidence_images": len(images),
            "merged_evidence_items": len(merged),
            "never_overlapped_annos": len(no_overlap_annos),
            "no_candidate_images": len(no_candidate_images),
            "total_items": len(all_ranking),
        },
        "merged_ranking": merged,
        "never_overlapped_annos": no_overlap_annos,
        "no_candidate_images": no_candidate_images,
        "all_ranking": all_ranking,
    }


def main():
    result = build_merged_rank(load_json(ANNO_RANK_PATH), load_json(IMAGE_RANK_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, separators=(",", ":"))
    print(f"Merged {result['counts']['merged_evidence_items']} evidence items to {OUTPUT_PATH}")
    print(f"Never-overlapped annos: {result['counts']['never_overlapped_annos']}")
    print(f"Images without candidate tracks: {result['counts']['no_candidate_images']}")


if __name__ == "__main__":
    main()
