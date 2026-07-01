'''
检查一下repair结果是否正确
'''
import os
import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ours.base_data_manager import (exp_data_root_dir,get_correct_ann_file_path,get_error_ann_file_path,
                                    get_annotations_with_miss_json_path,get_collected_gt_box_json_path)
from ours.small_utils import read_json
from ours.data_organization_tools import (get_g_id_to_g_box, get_gid_to_anno_id,
                                          get_imgname_to_imgid, get_cls_id_to_name, get_annoid_to_imgname,
                                          get_all_miss_img_name_list,conver_ours_rank,conver_datactive_rank,
                                          get_img_name_to_missed_annids,get_img_name_to_ann_ids,get_annoId_to_anno)




def get_repaired_info(rank_res:list,g_boxes_json:dict, gid_to_annoid:dict, anno_with_miss_json:dict):
    repair_cutoff_rate = 0.4
    repair_cutoff_point = int(len(rank_res) * repair_cutoff_rate)
    repair_idds = rank_res[:repair_cutoff_point]
    gid_to_gbox = get_g_id_to_g_box(g_boxes_json)
    all_miss_img_name_list = get_all_miss_img_name_list(anno_with_miss_json)
    repaired_info = {
        "no_fault":[],
        "miss_fault":[],
        "cls_fault":[],
        "loc_falut":[],
        "redun_fault":[]
    }
    for idd in repair_idds:
        if type(idd) is str:
            img_name = idd
            if img_name in all_miss_img_name_list:
                repaired_info["miss_fault"].append(img_name)
        else:
            gid = idd
            gbox = gid_to_gbox[gid]
            fault_type = gbox["fault_type"]
            annoid = gid_to_annoid[gid]
            if fault_type == 1:
                repaired_info["cls_fault"].append(annoid)
            elif fault_type == 2:
                repaired_info["loc_falut"].append(annoid)
            elif fault_type == 3:
                repaired_info["redun_fault"].append(annoid)
            elif fault_type == 0:
                repaired_info["no_fault"].append(annoid)
            else:
                raise Exception("fault_type 错误")
    return repaired_info




def get_annos_based_img_name(anno_json:dict,img_name):
    anno_list = []
    imgname_to_imgid = get_imgname_to_imgid(anno_json)
    imgid = imgname_to_imgid[img_name]
    annos = anno_json["annotations"]
    for anno in annos:
        if anno["image_id"] == imgid:
            anno_list.append(anno)
    return anno_list


def visual_img(img_name:str,class_id_to_name:dict, correct_annos:list[dict],error_annos:list[dict],repair_annos:list[dict]):
    """
    在一行3列的子图中显示图像的标注信息：
    1. 完全正确的标注框
    2. 错误的标注框（区分miss fault、redun fault、cls fault和loc fault）
    3. 修复后的标注框
    """
    
    # 获得图像路径并读取图像
    img_path = os.path.join(exp_data_root_dir, "datasets", f"{dataset_name}-coco", "train", img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"无法读取图像: {img_path}")
    
    # 转换为RGB格式用于matplotlib显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 计算标注集合
    correct_anno_id_set = set([anno["id"] for anno in correct_annos])
    error_anno_id_set = set([anno["id"] for anno in error_annos])

    no_fault_ann_id_set = set()
    cls_fault_ann_id_set = set()
    loc_fault_ann_id_set = set()
    redun_fault_ann_id_set = set()
    missed_anno_id_set = set()
    
    for error_anno in error_annos:
        if error_anno["fault_type"] == 0:
            no_fault_ann_id_set.add(error_anno["id"])
        elif error_anno["fault_type"] == 1:
            cls_fault_ann_id_set.add(error_anno["id"])
        elif error_anno["fault_type"] == 2:
            loc_fault_ann_id_set.add(error_anno["id"])
        elif error_anno["fault_type"] == 3:
            redun_fault_ann_id_set.add(error_anno["id"])
    missed_anno_id_set = correct_anno_id_set - error_anno_id_set  # 正确中有，错误中没有
    print("total error nums:",len(error_annos))
    print("no_fault nums:",len(no_fault_ann_id_set))
    print("cls_fault nums:",len(cls_fault_ann_id_set))
    print("loc_fault nums:",len(loc_fault_ann_id_set))
    print("redun_fault nums:",len(redun_fault_ann_id_set))
    print("missed_anno nums:",len(missed_anno_id_set))

    assert len(no_fault_ann_id_set) + len(cls_fault_ann_id_set) + len(loc_fault_ann_id_set) + \
    len(redun_fault_ann_id_set) == len(error_annos), "错误标签不对"

    # 设置颜色
    correct_color = (0, 255, 0)      # 绿色 - 正确标注
    miss_color = (255, 0, 0)         # 红色 - 缺失标注（miss fault）
    redun_color = (0, 0, 255)        # 蓝色 - 冗余标注（redun fault）
    cls_color = (255, 255, 0)        # 黄色 - 类别错误标注（cls fault）
    loc_color = (0, 0, 0)         # 黑色 - loc错误标注（loc fault）
    repair_color = (128, 0, 128)     # 紫色 - 修复标注
    
    # 创建3列子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Image: {img_name}', fontsize=16)
    
    # 1. 显示完全正确的标注框
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    ax1.set_title('Correct Annotations', fontsize=14, color='green')
    ax1.axis('off')
    
    for anno in correct_annos:
        x, y, w, h = anno['bbox']

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = anno['category_id']
        class_name = class_id_to_name[class_id]
        
        # 绘制边界框
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        
        # 添加类别标签
        ax1.text(x1, y1 - 5, f'{class_name}', fontsize=10, color='green', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 2. 显示错误的标注框（区分miss fault、redun fault和cls fault）
    ax2 = axes[1]
    ax2.imshow(img_rgb)
    ax2.set_title('Error Annotations', fontsize=14, color='red')
    ax2.axis('off')
    
    # 绘制错误标注中的冗余标注（redun fault）
    for anno in error_annos:
        if anno["fault_type"] == 3: # redun fault box
            x, y, w, h = anno['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = anno['category_id']
            class_name = class_name = class_id_to_name[class_id]
            
            # 绘制蓝色边界框表示冗余标注
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='blue', facecolor='none')
            ax2.add_patch(rect)
            
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{class_name} (redun)', fontsize=10, color='blue', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 绘制错误标注中缺失的部分（miss fault）- 用虚线框表示
    for anno in correct_annos:
        if anno['id'] in missed_anno_id_set:
            x, y, w, h = anno['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = anno['category_id']
            class_name = class_name = class_id_to_name[class_id]
            
            # 绘制红色虚线框表示缺失标注
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', 
                               linestyle='--', facecolor='none')
            ax2.add_patch(rect)
            
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{class_name} (miss)', fontsize=10, color='red', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 绘制类别错误标注（cls fault）
    for error_anno in error_annos:
        if error_anno["fault_type"] == 1: # cls fault box
            x, y, w, h = error_anno['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            error_class_id = error_anno['category_id']
            error_class_name = class_id_to_name[error_class_id]
            
            # 绘制黄色边界框表示类别错误标注
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
            ax2.add_patch(rect)
            
            # 添加类别标签，显示正确的类别和错误的类别
            ax2.text(x1, y1 - 5, f'{error_class_name} (cls)', fontsize=10, color='yellow', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 绘制loc错误标注（loc fault）
    for error_anno in error_annos:
        if error_anno["fault_type"] == 2: # loc fault box
            x, y, w, h = error_anno['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            error_class_id = error_anno['category_id']
            error_class_name = class_id_to_name[error_class_id]
            
            # 绘制黄色边界框表示类别错误标注
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='black', facecolor='none')
            ax2.add_patch(rect)
            
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{error_class_name} (loc)', fontsize=10, color='black', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
     # 绘制错误标注中正确的（no fault）
    for error_anno in error_annos:
        if error_anno["fault_type"] == 0: # loc fault box
            x, y, w, h = error_anno['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            error_class_id = error_anno['category_id']
            error_class_name = class_id_to_name[error_class_id]
            
            # 绘制黄色边界框表示类别错误标注
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{error_class_name} (no fault)', fontsize=10, color='green',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    
    # 3. 显示修复后的标注框
    ax3 = axes[2]
    ax3.imshow(img_rgb)
    ax3.set_title('Repaired Annotations', fontsize=14, color='purple')
    ax3.axis('off')
    
    for anno in repair_annos:
        x, y, w, h = anno['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = anno['category_id']
        class_name = class_id_to_name[class_id]
        
        # 绘制紫色边界框表示修复后的标注
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='purple', facecolor='none')
        ax3.add_patch(rect)
        
        # 添加类别标签
        ax3.text(x1, y1 - 5, f'{class_name}', fontsize=10, color='purple', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(exp_data_root_dir, "visualization_results", img_name.replace('.jpg', '_comparison.jpg'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")


def main():
    anno_error_json = read_json(anno_error_json_path)
    anno_correct_json = read_json(anno_correct_json_path)
    anno_repair_json = read_json(anno_repair_json_path)
    anno_with_miss_json = read_json(anno_with_miss_json_path)
    g_boxes_json = read_json(g_boxes_json_path)
    cls_id_to_name = get_cls_id_to_name(anno_correct_json)

    gid_to_anno_id = get_gid_to_anno_id(g_boxes_json, anno_error_json)
    repaired_info = get_repaired_info(rank_res,g_boxes_json,gid_to_anno_id,anno_with_miss_json)

    miss_img_name_list = repaired_info["miss_fault"][:4] # 4张带有miss fault的图像name
    cls_annoid_list = repaired_info["cls_fault"][:4]
    loc_annoid_list = repaired_info["loc_falut"][:4]
    redun_annoid_list = repaired_info["redun_fault"][:4]

    # 看下miss的修复结果
    for img_name in miss_img_name_list:
        # 得到这张图像的所有正确标注
        correct_annos = get_annos_based_img_name(anno_correct_json,img_name)
        # 得到这张图像的miss错误标注
        error_annos = get_annos_based_img_name(anno_error_json, img_name)
        # 得到这张图像的修复后的标注
        repair_annos = get_annos_based_img_name(anno_repair_json, img_name)
        visual_img(img_name,cls_id_to_name,correct_annos,error_annos,repair_annos)
    

    # error_annoid_to_imgname = get_annoid_to_imgname(anno_error_json)

    # # 看下cls fault的修复结果
    # for ann_id in cls_annoid_list:
    #     img_name = error_annoid_to_imgname[ann_id]
    #     # 得到这张图像的所有正确标注
    #     correct_annos = get_annos_based_img_name(anno_correct_json,img_name)
    #     # 得到这张图像的miss错误标注
    #     error_annos = get_annos_based_img_name(anno_error_json, img_name)
    #     # 得到这张图像的修复后的标注
    #     repair_annos = get_annos_based_img_name(anno_repair_json, img_name)
    #     visual_img(img_name,cls_id_to_name,correct_annos,error_annos,repair_annos)
    

    
    # # 看下loc fault的修复结果
    # for ann_id in loc_annoid_list:
    #     img_name = error_annoid_to_imgname[ann_id]
    #     # 得到这张图像的所有正确标注
    #     correct_annos = get_annos_based_img_name(anno_correct_json,img_name)
    #     # 得到这张图像的miss错误标注
    #     error_annos = get_annos_based_img_name(anno_error_json, img_name)
    #     # 得到这张图像的修复后的标注
    #     repair_annos = get_annos_based_img_name(anno_repair_json, img_name)
    #     visual_img(img_name,cls_id_to_name,correct_annos,error_annos,repair_annos)

    
    # # 看下redun fault的修复结果
    # for ann_id in redun_annoid_list:
    #     img_name = error_annoid_to_imgname[ann_id]
    #     # 得到这张图像的所有正确标注
    #     correct_annos = get_annos_based_img_name(anno_correct_json,img_name)
    #     # 得到这张图像的miss错误标注
    #     error_annos = get_annos_based_img_name(anno_error_json, img_name)
    #     # 得到这张图像的修复后的标注
    #     repair_annos = get_annos_based_img_name(anno_repair_json, img_name)
    #     visual_img(img_name,cls_id_to_name,correct_annos,error_annos,repair_annos)

def check_miss(gt_missed_annoids, repairAnnids, errorAnnids, correctAnnids):
    assert set(correctAnnids) - set(errorAnnids) == set(gt_missed_annoids), "没通过"
    for missed_annid in gt_missed_annoids:
        assert missed_annid in repairAnnids, "没修复"


def detail_check():
    '''
    仔细比对repair anno json与correct anno json异同
    '''
    anno_correct_json = read_json(anno_correct_json_path)
    anno_repair_json = read_json(anno_repair_json_path)
    anno_error_json = read_json(anno_error_json_path)

    repair_annoId_to_anno = get_annoId_to_anno(anno_repair_json)
    correct_annoId_to_anno =  get_annoId_to_anno(anno_correct_json)
    correct_id_list = []
    repair_id_list = []
    for anno in anno_correct_json["annotations"]:
        correct_id_list.append(anno["id"])
    for anno in anno_repair_json["annotations"]:
        repair_id_list.append(anno["id"])
    common_id_set = set(correct_id_list) & set(repair_id_list)

    residue_miss_id_set = set(correct_id_list) - set(repair_id_list) # 残留的miss fault anno id set
    repaired_miss_id_set = set() # 修复的miss fault anno id set

    repaired_cls_id_set = set() # 修复的cls fault anno id set
    residue_cls_id_set = set() # 残留的cls fault anno id set

    repaired_loc_id_set = set() # 修复的loc fault anno id set
    residue_loc_id_set = set() # 残留的loc fault anno id set

    residue_redunc_id_set = set(repair_id_list) - set(correct_id_list) # 残留的redunc fault anno id set
    redunc_fault_id_set = set()
    for error_anno in anno_error_json["annotations"]:
        if error_anno["fault_type"] == 3:
            redunc_fault_id_set.add(error_anno["id"])
    repaired_redunc_id_set = redunc_fault_id_set - residue_redunc_id_set # 修复的redunc fault anno id set


    irrelevant_id_set = set() # 无关的anno id set
    for c_id in common_id_set:
        repair_anno = repair_annoId_to_anno[c_id]
        correct_anno = correct_annoId_to_anno[c_id]
        if "fault_type" in repair_anno:
            fault_type = repair_anno["fault_type"]
            if fault_type == 0:
                # 没被篡改过的anno
                irrelevant_id_set.add(c_id)
            elif fault_type == 1:
                # cls fault anno, 接着判断是否修复
                correct_label = correct_anno["category_id"]
                repair_label = repair_anno["category_id"]
                if repair_label == correct_label:
                    # cls 被修复了
                    repaired_cls_id_set.add(c_id)
                else:
                    residue_cls_id_set.add(c_id)
            elif fault_type == 2:
                # loc fault anno, 接着判断是否修复
                correct_bbox = correct_anno["bbox"]
                repair_bbox = repair_anno["bbox"]
                if correct_bbox == repair_bbox:
                    # loc 被修复了
                    repaired_loc_id_set.add(c_id)
                else:
                    residue_loc_id_set.add(c_id)
        else:
            # 从 correct anno json迁移过来的anno,即修复了的miss fault anno
            repaired_miss_id_set.add(c_id)

    print("无关的anno数量:", len(irrelevant_id_set))
    print("修复的cls fault数量:", len(repaired_cls_id_set))
    print("残留的cls fault数量:", len(residue_cls_id_set))
    print("修复的loc fault数量:", len(repaired_loc_id_set))
    print("残留的loc fault数量:", len(residue_loc_id_set))
    print("修复的redunc fault数量:", len(repaired_redunc_id_set))
    print("残留的redunc fault数量:", len(residue_redunc_id_set))
    print("修复的miss fault数量:", len(repaired_miss_id_set))
    print("残留的miss fault数量:", len(residue_miss_id_set))


if __name__ == '__main__':
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    method_name = "datactive" # ours|datactive
    exp_id = "exp_02"

    anno_correct_json_path = get_correct_ann_file_path(dataset_name,"train")
    anno_error_json_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    
    # repaired anno json path
    anno_repair_json_path = os.path.join(exp_data_root_dir,"Results",method_name,dataset_name,model_name,exp_id,
                                         "repair","_annotations.coco_repair.json")
    

    
    # 排序结果
    rank_res = joblib.load(os.path.join(exp_data_root_dir,"Results",method_name,dataset_name,model_name,exp_id,
                                        "rank","rank.joblib"))
    # 收集的gboxs
    g_boxes_json_path = get_collected_gt_box_json_path(dataset_name)
    detail_check()
