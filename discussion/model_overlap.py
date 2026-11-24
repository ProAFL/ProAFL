'''
讨论下ours,在3个数据集上不同模型上的overlap情况
'''
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils.common import read_json,read_yaml

def get_missfault_imgname_set(anno_json)->set[str]:
    # 带有.后缀名
    images = anno_json["images"]
    imgId2imgname = {image["id"]:image["file_name"] for image in images}    
    annos = anno_json["annotations"]
    missfault_imgname_set = set()
    for anno in annos:
        if int(anno["fault_type"]) == 4:
            missfault_imgname_set.add(imgId2imgname[anno["image_id"]] )
    return missfault_imgname_set

def get_fault_items(anno_json):
    missfault_imgname_set = get_missfault_imgname_set(anno_json)
    annos = anno_json["annotations"]
    anno_fault_nomiss = set()
    for anno in annos:
        aid = int(anno["id"])
        fault_type = int(anno["fault_type"])
        if fault_type in [1,2,3]:
            anno_fault_nomiss.add(aid)
    fault_items = anno_fault_nomiss | missfault_imgname_set
    return fault_items


def get_rank_by_model_name(model_name):
    rank_json_path = root_dir/"dual_branch"/"process_features"/dataset_name/model_name/f"{fault_ratio}_repeat"/f"repeat_{repeat_id}"/"merged_rank.json"
    with open(rank_json_path, encoding="utf-8") as file:
        ranking = json.load(file)["all_ranking"]
    ranked_items = []
    for item in ranking:
        if item["item_type"] == "anno":
            ranked_items.append(item["anno_id"])
        elif item["item_type"] == "image":
            ranked_items.append(item["image_name"])
        else:
            raise ValueError(f"Unsupported item_type: {item['item_type']}")
    ranked_set = set(ranked_items)
    if len(ranked_items) != len(ranked_set):
        raise ValueError("The total ranking contains duplicate items")
    return ranked_items
     

def vis_venn(yolo_rank,rtdetr_rank,frcnn_rank,fault_items,cut_off=0.4):
    """Plot overlap of fault items appearing in each model's top-ranked cutoff."""
    if not 0 <= cut_off <= 1:
        raise ValueError("cut_off must be between 0 and 1")

    fault_items = set(fault_items)

    def top_faults(rank):
        # cutoff is a fraction of the complete ranking; floor to whole items.
        cutoff_count = int(len(rank) * cut_off)
        return set(rank[:cutoff_count]) & fault_items
        # return set(rank[:cutoff_count])

    yolo_faults = top_faults(yolo_rank)
    rtdetr_faults = top_faults(rtdetr_rank)
    frcnn_faults = top_faults(frcnn_rank)

    # merge_set = set(yolo_faults) | set(rtdetr_faults) | set(frcnn_faults)

    # Counts for the seven mutually exclusive regions of a three-set Venn.
    regions = {
        "100": len(yolo_faults - rtdetr_faults - frcnn_faults),
        "010": len(rtdetr_faults - yolo_faults - frcnn_faults),
        "001": len(frcnn_faults - yolo_faults - rtdetr_faults),
        "110": len((yolo_faults & rtdetr_faults) - frcnn_faults),
        "101": len((yolo_faults & frcnn_faults) - rtdetr_faults),
        "011": len((rtdetr_faults & frcnn_faults) - yolo_faults),
        "111": len(yolo_faults & rtdetr_faults & frcnn_faults),
    }

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    circles = (
        ((0.40, 0.58), "#4C78A8", "YOLOv7"),
        ((0.60, 0.58), "#F58518", "RT-DETR"),
        ((0.50, 0.39), "#54A24B", "Faster R-CNN"),
    )
    for (x, y), color, _ in circles:
        ax.add_patch(Circle((x, y), 0.25, facecolor=color, edgecolor=color,
                            alpha=0.35, linewidth=1.5))

    # Region label positions in axes coordinates.
    positions = {
        "100": (0.31, 0.67), "010": (0.69, 0.67), "001": (0.50, 0.27),
        "110": (0.50, 0.70), "101": (0.40, 0.43), "011": (0.60, 0.43),
        "111": (0.50, 0.53),
    }
    for region, count in regions.items():
        percentage = count / len(fault_items) * 100 if fault_items else 0
        # percentage = count / len(merge_set) * 100
        ax.text(*positions[region], f"{percentage:.1f}%", ha="center", va="center",
                fontsize=11)

    legend_handles = [
        Circle((0, 0), radius=0.12, facecolor=color, edgecolor=color, alpha=0.45,
               label=label)
        for _, color, label in circles
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.03),
              ncol=3, frameon=False, fontsize=9, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.0)
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.12, 0.84)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0.05, top=1)
    fig.savefig(output_file_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    anno_withmiss_json = read_json(annotation_withmiss_json_path)
    fault_items = get_fault_items(anno_withmiss_json)
    
    yolo_rank = get_rank_by_model_name("yolov7")
    rtdetr_rank = get_rank_by_model_name("rtdetr")
    frcnn_rank = get_rank_by_model_name("frcnn")
    vis_venn(yolo_rank,rtdetr_rank,frcnn_rank,fault_items,cut_off=0.4)

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "voc" # voc|kitti|visdrone
    fault_ratio = 0.1
    repeat_id = 1
    
    annotation_withmiss_json_path = root_dir/f"fault_inject/{fault_ratio}/{dataset_name}/coco_format/annotations_with_miss.json"
    output_file_path = f"discussion/overlap_imgs/{dataset_name}_toponly.pdf"
    main()
    
