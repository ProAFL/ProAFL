
import cv2
import os
from ours.small_utils import read_json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np


def vis_bbox(img_path, gbox, pbox, epoch, save_path:str):
    # 读取图像（注意 cv2 是 BGR，需要转 RGB）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f'Epoch:{epoch}', fontsize=14, color='black')
    ax.axis('off')
    gbbox = gbox["gt_bbox"] # x1y1x2y2
    pbbox = pbox["bbox"] # x1y1x2y2
    gw,gh = gbbox[2] - gbbox[0], gbbox[3] - gbbox[1]
    pw,ph = pbbox[2] - pbbox[0], pbbox[3] - pbbox[1]
    
    rect = plt.Rectangle((int(gbbox[0]), int(gbbox[1])), gw, gh, linewidth=0.5, edgecolor="green", facecolor='none')
    ax.add_patch(rect)

    rect = plt.Rectangle((int(pbbox[0]), int(pbbox[1])), pw, ph, linewidth=0.5, edgecolor="blue", facecolor='none')
    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def draw_iou_line(iou_list,save_path):
    iou_arr = np.asarray(iou_list, dtype=float)
    # epochs = np.arange(len(iou_arr))
    epochs = list(range(1,len(iou_list)+1))

    with plt.style.context(["science", "ieee", "no-latex"]):
        plt.rcParams.update({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
        fig, ax = plt.subplots(figsize=(3.5, 1.2), constrained_layout=True)
        ax.plot(
            epochs,
            iou_arr,
            color="#2f6fbb",
            linewidth=1.2,
            marker="o",
            markersize=2.2,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label="IoU",
        )
        ax.set_xlabel("Epoch", labelpad=4)
        ax.set_ylabel("IoU", labelpad=1)
        ax.set_xlim(0, max(len(iou_arr) - 1, 1))
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
        ax.legend(loc="center right", frameon=False)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    match_info = []
    for gid_str in match_json:
        if int(gid_str) == gid:
            # 找到了，赶紧提取gid信息
            match_info = match_json[gid_str]
    if match_info == []:
        raise Exception("没找到gid信息")
    # 拿到gbox
    match_item = match_info[0]
    gbox = match_item["g_box"]

    pbox_list = [None]*50 # 50个epoch
    iou_list = [0]*50

    # 拿到pbox_list
    for match_item in match_info:
        epoch = match_item["epoch"]
        pbox = match_item["p_box"]
        iou = match_item["iou_val"]
        pbox_list[epoch] = pbox
        iou_list[epoch] = iou
    # 绘制iou_list折线图
    save_path = os.path.join(save_dir,"iouline" f"iou.pdf")
    draw_iou_line(iou_list,save_path)
    print(f"iou save in:{save_path}")
    imgpath = os.path.join(imgs_dir,gbox["img_name"])
    for epoch,pbox in enumerate(pbox_list):
        save_path = os.path.join(save_dir,f"epoch_{epoch}.pdf")
        if pbox is not None:
            vis_bbox(imgpath, gbox, pbox, epoch, save_path)
            print(f"epoch:{epoch}绘框完毕,保存在:{save_path}")




def select_gid(early_ratio=0.4, eps=1e-6):
    def gradual_rising_score(y):
        y = np.asarray(y, dtype=float)

        if len(y) < 5 or np.any(np.isnan(y)):
            return -np.inf

        n = len(y)
        split = max(2, int(n * early_ratio))

        early_y = y[:split]
        late_y = y[split:]

        # 前期不能出现 0（若只要求第一个值非 0，可改成 y[0] <= eps）
        if np.any(early_y <= eps):
            return -np.inf

        early_diff = np.diff(early_y)
        late_diff = np.diff(late_y)

        # 前低后高
        early_mean = np.mean(early_y)
        late_mean = np.mean(late_y)
        stage_gain = late_mean - early_mean

        # 前、后期都要总体上升
        early_gain = early_y[-1] - early_y[0]
        late_gain = late_y[-1] - late_y[0]

        if stage_gain <= 0 or early_gain <= 0 or late_gain <= 0:
            return -np.inf

        # 两个阶段中“上升步”的比例
        early_rise_ratio = np.mean(early_diff > 0)
        late_rise_ratio = np.mean(late_diff > 0)

        # 回退总量
        drop_penalty = (
            -np.sum(early_diff[early_diff < 0])
            -np.sum(late_diff[late_diff < 0])
        )

        # 步长波动：越小代表越缓慢、越稳定
        step_std = np.std(np.diff(y))

        # 总增长幅度
        total_gain = y[-1] - y[0]

        score = (
            2.0 * stage_gain
            + 1.0 * total_gain
            + 0.8 * early_rise_ratio
            + 0.8 * late_rise_ratio
            - 2.0 * drop_penalty
            - 1.0 * step_std
        )

        return score
    gid_score_list = []
    for item in metric_json:
        gid = item["g_box_id"]
        iou_list = item["iou_list"]
        score = gradual_rising_score(iou_list)
        gid_score_list.append((gid,score))
    # ssd 越小越平滑，因此越靠前
    gid_score_list.sort(key=lambda x: x[1],reverse=True) # 越大优先级越高
    priority_gid_list = [
        gid for gid, score in gid_score_list
        if np.isfinite(score)
    ]
    priority_gid_list = [gid for gid, _ in gid_score_list]
    return priority_gid_list

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8"
    
    model_name = "YOLOv7"
    

    imgs_dir = f"{exp_root_dir}/datasets/{dataset_name}-yolo/origin/train/images"
    match_json = read_json(f"{exp_root_dir}/collection_bbox_level/{dataset_name}/{model_name}/gp_box_match/match_v2.json")
    metric_json = read_json(f"{exp_root_dir}/collection_bbox_level/{dataset_name}/{model_name}/collection_metric/collection_metrics_v2.json")
    
    
    # select_gid()

    gid = 7663
    save_dir = os.path.join(f"{exp_root_dir}/case",dataset_name,model_name,f"gid_{gid}")
    os.makedirs(save_dir,exist_ok=True)
    main()
