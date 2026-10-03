
import os
import re
from numbers import Integral
import joblib
from pycocotools.coco import COCO
import pandas as pd
from utils.common import read_json

def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name


def get_missed_img_name_set(annotations_with_miss_json):
    miss_img_name_set = set()
    imgId_to_imgName = get_image_id_to_image_name_for_coco(annotations_with_miss_json)
    annos = annotations_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgId_to_imgName[anno["image_id"]]
            miss_img_name_set.add(img_name)
    return miss_img_name_set

def get_error_ann_id_set(coco:COCO):
    anns = coco.loadAnns(coco.getAnnIds())
    error_ann_id_set = set()
    for ann in anns:
        if ann["fault_type"] in [1,2,3]:              
            error_ann_id_set.add(ann["id"])
    return error_ann_id_set


def vis(fp_idd_list,fn_idd_list,anno_with_miss_json):
    """Visualize up to five cases per group over the last five saved epochs."""
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle

    collected_predict_boxes_dir = os.path.join(exp_root,"collection_process_info",dataset_name,
                 model_name,"collected_predict_boxes",f"inject_{inject_ratio}")
    images_dir = os.path.join(exp_root, "datasets", f"{dataset_name}-yolo", "origin", "train", "images")
    output_dir = os.path.join(exp_root, "rank", "ours", dataset_name, model_name,
                              str(inject_ratio), "hard_sample_vis")

    images_by_id = {image["id"]: image for image in anno_with_miss_json["images"]}
    images_by_name = {image["file_name"]: image for image in anno_with_miss_json["images"]}
    anns_by_id = {ann["id"]: ann for ann in anno_with_miss_json["annotations"]}
    anns_by_image = {}
    for ann in anno_with_miss_json["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    groups = {
        "fp_annoid": [idd for idd in fp_idd_list if isinstance(idd, Integral)][:5],
        "fp_img": [idd for idd in fp_idd_list if isinstance(idd, str)][:5],
        "fn_annoid": [idd for idd in reversed(fn_idd_list) if isinstance(idd, Integral)][:5],
        "fn_img": [idd for idd in reversed(fn_idd_list) if isinstance(idd, str)][:5],
    }
    selected = []
    for group, ids in groups.items():
        if not ids:
            print(f"{group}: 没有可视化样本")
        for position, idd in enumerate(ids, start=1):
            if isinstance(idd, Integral):
                ann = anns_by_id[int(idd)]
                image_info = images_by_id[ann["image_id"]]
                gt_anns = [ann]
            else:
                image_info = images_by_name[idd]
                gt_anns = anns_by_image.get(image_info["id"], [])
                if group == "fn_img":
                    gt_anns = [ann for ann in gt_anns if ann["fault_type"] == 4]
            selected.append((group, position, idd, image_info["file_name"], gt_anns))
    if not selected:
        return []

    epoch_files = []
    for filename in os.listdir(collected_predict_boxes_dir):
        match = re.fullmatch(r"epoch_(\d+)_predicted_bboxs\.json", filename)
        if match:
            epoch_files.append((int(match.group(1)), filename))
    epoch_files = sorted(epoch_files)[-5:]
    if not epoch_files:
        raise FileNotFoundError(f"没有轮次预测文件: {collected_predict_boxes_dir}")
    selected_names = {item[3] for item in selected}
    epoch_predictions = {}
    for epoch, filename in epoch_files:
        predictions = read_json(os.path.join(collected_predict_boxes_dir, filename))
        epoch_predictions[epoch] = {
            name: (predictions.get(name, {}).get("predicted_bboxs") or [])
            for name in selected_names
        }
    print(f"可视化轮次: {[epoch for epoch, _ in epoch_files]}（预测类别 ID 保持原文件编号）")

    saved_paths = []
    for group, position, idd, image_name, gt_anns in selected:
        with Image.open(os.path.join(images_dir, image_name)) as source:
            image = source.convert("RGB")
        gt_color = "#00cc00" if group.startswith("fp") else "#ff3333"
        rows = (len(epoch_files) + 2) // 3
        figure = Figure(figsize=(18, rows * 6))
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, 3, squeeze=False).ravel()
        for ax, (epoch, _) in zip(axes, epoch_files):
            ax.imshow(image)
            preds = epoch_predictions[epoch][image_name]
            for pred in preds:
                x1, y1, x2, y2 = pred["bbox"]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="#0066ff", linewidth=1.2))
                ax.text(x1, y2, f"pred cls={pred['predicted_cls']} conf={pred['conf']:.3f}", color="#0066ff",
                        fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.7, "pad": 1})
            for ann in gt_anns:
                x, y, width, height = ann["bbox"]
                ax.add_patch(Rectangle((x, y), width, height,
                                       fill=False, edgecolor=gt_color, linewidth=2.5))
                ax.text(x, y, f"GT ann={ann['id']} cls={ann['category_id']}", color=gt_color,
                        fontsize=9, va="top", bbox={"facecolor": "white", "alpha": 0.8, "pad": 1})
            ax.set_title(f"epoch {epoch} | GT: {len(gt_anns)} | predictions: {len(preds)}")
            ax.set_xlim(0, image.width)
            ax.set_ylim(image.height, 0)
            ax.axis("off")
        for ax in axes[len(epoch_files):]:
            ax.axis("off")
        figure.suptitle(f"{group} | {idd} | {image_name} | GT: {'green' if group.startswith('fp') else 'red'}, predictions: blue")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        group_dir = os.path.join(output_dir, group)
        os.makedirs(group_dir, exist_ok=True)
        case_name = str(int(idd)) if isinstance(idd, Integral) else os.path.splitext(os.path.basename(idd))[0]
        save_path = os.path.join(group_dir, f"{position:02d}_{case_name}.png")
        figure.savefig(save_path, dpi=150)
        figure.clear()
        saved_paths.append(save_path)
        print(f"可视化已保存: {save_path}")
    return saved_paths

def main():
    rank_path =os.path.join(exp_root,"rank","ours",dataset_name,model_name,str(inject_ratio),"rank.joblib")
    rank = joblib.load(rank_path)

    coco = COCO(anno_with_miss_json_path)
    fault_annid_set = get_error_ann_id_set(coco)
    anno_with_miss_json = read_json(anno_with_miss_json_path) 
    miss_imgname_set = get_missed_img_name_set(anno_with_miss_json)
    fault_idd = fault_annid_set | miss_imgname_set

    k = 0.1 
    cut = int(len(rank)*k)

    top_rank = rank[:cut]
    last_rank = rank[-cut:] if cut else []

    fp_idd_list = []
    for idd in top_rank:
        if idd not in fault_idd:
            fp_idd_list.append(idd)
    fn_idd_list = []
    for idd in last_rank:
        if idd in fault_idd:
            fn_idd_list.append(idd)

    vis(fp_idd_list,fn_idd_list,anno_with_miss_json)
    '''
    annoid_features_csv_path = os.path.join(exp_root,"rank","ours",dataset_name,model_name,str(inject_ratio),"annoid_features.csv")
    img_features_csv_path = os.path.join(exp_root,"rank","ours",dataset_name,model_name,str(inject_ratio),"img_features.csv")
    annoid_features_df = pd.read_csv(annoid_features_csv_path)
    img_features_df = pd.read_csv(img_features_csv_path)

    annoid_features_df = annoid_features_df.set_index("annoid", verify_integrity=True)
    img_features_df = img_features_df.set_index("imgname", verify_integrity=True)
    feature_tables = {
        "annoid": annoid_features_df,
        "imgname": img_features_df,
    }
    rank_position = {idd: position for position, idd in enumerate(rank, start=1)}
    output_dir = os.path.dirname(rank_path)
    summary_rows = []

    for kind, table in feature_tables.items():
        is_this_kind = (lambda idd: isinstance(idd, int)) if kind == "annoid" else (lambda idd: isinstance(idd, str))
        file_kind = "annoid" if kind == "annoid" else "img"
        feature_columns = [column for column in table.columns if column != "topsis_score"]
        detail_frames = []

        for case, hard_ids, reference_ids in (
            ("fp", fp_idd_list, [idd for idd in top_rank if idd in fault_idd]),
            ("fn", fn_idd_list, [idd for idd in last_rank if idd not in fault_idd]),
        ):
            hard_ids = [idd for idd in hard_ids if is_this_kind(idd)]
            reference_ids = [idd for idd in reference_ids if is_this_kind(idd)]
            missing_ids = set(hard_ids + reference_ids) - set(table.index)
            if missing_ids:
                raise ValueError(f"{kind} 特征 CSV 缺少 {len(missing_ids)} 个排名对象，例如 {next(iter(missing_ids))}")

            reference_case = "tp" if case == "fp" else "tn"
            for label, ids in ((case, hard_ids), (reference_case, reference_ids)):
                subset_path = os.path.join(output_dir, f"{label}_{file_kind}.csv")
                table.loc[ids].reset_index().to_csv(subset_path, index=False)
                print(f"{label}_{file_kind}: {len(ids)} 个对象，已保存: {subset_path}")

            if not hard_ids:
                continue

            hard_features = table.loc[hard_ids]
            detail = hard_features.reset_index()
            detail.insert(0, "case", case)
            detail.insert(2, "rank_position", [rank_position[idd] for idd in hard_ids])
            detail_frames.append(detail)

            if not reference_ids:
                print(f"{kind} {case}: {len(hard_ids)} 个难例；同区间没有可比较的样本")
                continue

            reference_features = table.loc[reference_ids]
            print(f"{kind} {case}: {len(hard_ids)} 个难例，对照 {len(reference_ids)} 个同类型样本")
            for column in [*feature_columns, "topsis_score"]:
                hard_mean = hard_features[column].mean()
                reference_mean = reference_features[column].mean()
                std = table[column].std()
                summary_rows.append({
                    "case": case,
                    "kind": kind,
                    "feature": column,
                    "hard_count": len(hard_ids),
                    "reference_count": len(reference_ids),
                    "hard_mean": hard_mean,
                    "reference_mean": reference_mean,
                    "hard_median": hard_features[column].median(),
                    "reference_median": reference_features[column].median(),
                    "hard_zero_fraction": hard_features[column].eq(0).mean(),
                    "reference_zero_fraction": reference_features[column].eq(0).mean(),
                    "hard_unique_values": hard_features[column].nunique(),
                    "reference_unique_values": reference_features[column].nunique(),
                    "mean_difference": hard_mean - reference_mean,
                    "standardized_difference": (hard_mean - reference_mean) / std if std > 0 else 0.0,
                })

        if detail_frames:
            detail_path = os.path.join(output_dir, f"hard_{kind}_samples.csv")
            pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
            print(f"难例明细已保存: {detail_path}")

    summary_path = os.path.join(output_dir, "hard_sample_feature_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"特征对比已保存: {summary_path}")
    if summary_df.empty:
        return
    for (kind, case), group in summary_df.groupby(["kind", "case"]):
        strongest = group[group["feature"] != "topsis_score"].sort_values(
            "standardized_difference", key=lambda values: values.abs(), ascending=False
        ).head(4)
        if strongest["mean_difference"].eq(0).all():
            if group.loc[group["feature"] != "topsis_score", ["hard_zero_fraction", "reference_zero_fraction"]].eq(1).all().all():
                print(f"{kind} {case}: 两组所有特征均为 0，当前特征无法区分这些难例")
            elif group.loc[group["feature"] != "topsis_score", ["hard_unique_values", "reference_unique_values"]].eq(1).all().all():
                print(f"{kind} {case}: 两组所有特征取值相同，当前特征无法区分这些难例")
            else:
                print(f"{kind} {case}: 两组特征均值相同")
            continue
        print(f"{kind} {case} 差异最大的特征（难例均值 - 对照均值）：")
        for row in strongest.itertuples():
            print(f"  {row.feature}: {row.mean_difference:+.4f} ({row.standardized_difference:+.2f} 个总体标准差)")

    '''


if __name__ == "__main__":
    exp_root = "/data/mml/data_debugging_data/ProAFL_data"
    dataset_name = "voc"
    model_name = "yolov7"
    inject_ratio = 0.1
    anno_with_miss_json_path = os.path.join(exp_root,"fault_inject",str(inject_ratio),dataset_name,
                                       "coco_format","annotations_with_miss.json")
    main()
