import matplotlib.pyplot as plt
import json
from pathlib import Path
from utils.common import read_yaml

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

def group_fault(anno_json):
    annos = anno_json["annotations"]
    fault_group = {
        "class_fault":set[int](),
        "loc_fault":set[int](),
        "redundancy_fault":set[int](),
        "miss_fault":get_missfault_imgname_set(anno_json)
    }
    for anno in annos:
        aid = int(anno["id"])
        fault_type = int(anno["fault_type"])
        if fault_type == 1: # class fault
            fault_group["class_fault"].add(aid)
        elif fault_type == 2: # loc fault
            fault_group["loc_fault"].add(aid)
        elif fault_type == 3: # redundancy fault
            fault_group["redundancy_fault"].add(aid)
    return fault_group



def vis_diversity(rank_items,fault_group):
    """Plot the cumulative diversity of detected faults along the ranking."""
    

    group_names = ("class_fault", "loc_fault", "redundancy_fault", "miss_fault")
    labels = ("Cls Fault", "Loc Fault", "Red Fault", "Mis Fault")
    colors = ("#348abd", "#36a13b", "#ff851b", "#e6373b")

    groups = [set(fault_group[name]) for name in group_names]
    total_faults = sum(len(group) for group in groups)
    if not rank_items:
        raise ValueError("rank_items must not be empty")
    if total_faults == 0:
        raise ValueError("fault_group must contain at least one fault")

    # The areas are shares of all fault instances, so their sum reaches 1
    # when the ranking has inspected every fault item.
    x = [i / len(rank_items) for i in range(len(rank_items) + 1)]
    cumulative = [[0.0] for _ in groups]
    for item in rank_items:
        for index, group in enumerate(groups):
            cumulative[index].append(
                cumulative[index][-1] + (1 / total_faults if item in group else 0)
            )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    # Match the reference PDF page size (250.109 × 181.825 pt).
    fig, ax = plt.subplots(figsize=(250.109 / 72, 181.825 / 72))
    ax.stackplot(x, *cumulative, labels=labels, colors=colors, linewidth=0.35,
                 edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Ratio of inspected instances", fontsize=9, labelpad=2)
    ax.set_ylabel("Ratio of detected faults", fontsize=9, labelpad=2)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                   labelsize=8, pad=2)
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(False)
    ax.legend(loc="upper left", fontsize=7, framealpha=1, borderpad=0.25,
              labelspacing=0.15, handlelength=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.15, top=0.99)
    fig.savefig(out_file_path, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

def main():
    with open(annotation_withmiss_json_path, encoding="utf-8") as file:
        anno_withmiss_json = json.load(file)
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
    fault_group = group_fault(anno_withmiss_json)
    vis_diversity(ranked_items, fault_group)



    


if __name__ == "__main__":
    config = read_yaml("config.yaml")
    root_dir = Path(config["exp_data_dir"])

    dataset_name = "voc"
    model_name = "yolov7"
    fault_ratio = 0.1
    repeat_id = 1
    rank_json_path = root_dir/"dual_branch"/"process_features"/dataset_name/model_name/f"{fault_ratio}_repeat"/f"repeat_{repeat_id}"/"merged_rank.json"
    annotation_withmiss_json_path = root_dir/f"fault_inject/{fault_ratio}/{dataset_name}/coco_format/annotations_with_miss.json"
    out_file_path = f"{dataset_name}_diversity.pdf"

    main()
