import os
from ours.small_utils import read_json
from collections import defaultdict
from ours.rank.img_rank import get_img_to_clusters_by_unifind
import cv2
from ours.small_utils import save_json_file
from ours.data_provider import get_all_miss_error_img_name_set
from ours.base_data_manager import get_annotations_with_miss_json_path
import matplotlib.pyplot as plt
from ours.data_organization_tools import get_img_name_to_ann_ids,get_annoId_to_anno
def select_imagename():
    misserror_imgname_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)
    return misserror_imgname_set



def vis_clusters(img_path, clusters,annos, save_path:str):
    # 读取图像（注意 cv2 是 BGR，需要转 RGB）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')
    for cluster in clusters:
        for pbox in cluster:
            pbbox = pbox["bbox"] # x1y1x2y2
            pw,ph = pbbox[2] - pbbox[0], pbbox[3] - pbbox[1]
            rect = plt.Rectangle((int(pbbox[0]), int(pbbox[1])), pw, ph, linewidth=0.5, edgecolor="blue", facecolor='none')
            ax.add_patch(rect)

    for _anno in annos:
        x, y, w, h = _anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        # 绘制边界框
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def calu_iou(bbox1,bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def measure_clusters(clusters):
    cluster_id = 0
    for cluster in clusters:
        cluster_id += 1
        print(f"cluster ID:{cluster_id}")
        confi_list = []
        for pbox in cluster:
            confi_list.append(pbox["conf"])
            print(pbox)
        print("cluster_confi_mean", round(sum(confi_list)/len(confi_list),3))
        iou_list = []
        for pbox1 in cluster:
            for pbox2 in cluster:
                iou = calu_iou(pbox1["bbox"],pbox2["bbox"])
                print(pbox1["epoch"],pbox2["epoch"], round(iou,3))
                iou_list.append(iou)
        print("cluster_iou_mean", round(sum(iou_list)/len(iou_list),3))
    print()

def main():
    json_data = read_json(img_to_nomatched_pboxs_json_path)
    img_to_p_boxs = defaultdict(list) # 这些图像中的unmatched p
    all_img_name_set = set()
    no_clusters_image_name_set = set() # 这些图像集合没有簇形成
    for img_name,info in json_data.items():
        all_img_name_set.add(img_name)
        if info["No_matched_p_box_list"] == []:
            no_clusters_image_name_set.add(img_name)
        img_to_p_boxs[img_name] = info["No_matched_p_box_list"]
    img_to_clusters = get_img_to_clusters_by_unifind(img_to_p_boxs,iou_thre=0.6)



    savepath = os.path.join(save_dir,"imgtoclusters.json")
    save_json_file(img_to_clusters,savepath)
    print(savepath)
    
    
    anno_json = read_json(anno_json_path)
    annoId_to_anno = get_annoId_to_anno(anno_json)
    imgname_to_annids = get_img_name_to_ann_ids(anno_json)
    
    imgnameset = select_imagename()
    for imgname in imgnameset:
        imgname = "000821.png"
        imgpath = os.path.join(imgs_dir,imgname)

        clusters = img_to_clusters[imgname]
        measure_clusters(clusters)
        annoIds = imgname_to_annids[imgname]
        anno_list = []
        for annoId in annoIds:
            anno = annoId_to_anno[annoId]
            anno_list.append(anno)

        save_path = os.path.join(save_dir,"pics" f"{imgname}_clusters.pdf")
        vis_clusters(imgpath,clusters,anno_list,save_path)
        

        print(f"保存in:{save_path}")





if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8"
    model_name = "YOLOv7"
    imgs_dir = f"{exp_root_dir}/datasets/{dataset_name}-yolo/origin/train/images"
    save_dir = os.path.join(f"{exp_root_dir}/case/{dataset_name}/{model_name}/imglevel2")
    os.makedirs(save_dir,exist_ok=True)
    epochs = 50
    img_to_nomatched_pboxs_json_path = os.path.join(
        exp_root_dir,"collection_bbox_level",
        dataset_name,model_name,"img_to_nomatched_pboxs.json")
    
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)

    anno_json_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-coco",
                                            "train",f"_annotations.coco_error.json")
    main()


    