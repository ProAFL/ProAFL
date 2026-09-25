'''
分析下our method rank结果的误报情况
'''
import json
import os
import joblib
from helper.data_organization_tools import get_all_miss_error_img_name_set,get_all_error_clean_set,get_all_positive_idd_set,get_all_negtive_idd_set
from helper.base_data_manager import get_annotations_with_miss_json_path,get_origin_tainimgs_dir,get_all_img_name
from ours.small_utils import read_json


def stat_img_obj_nums(idd_set):
    imgname_set = set()
    annoid_set = set()
    for idd in idd_set:
        if type(idd) is str:
            imgname_set.add(idd)
        if type(idd) is int:
            annoid_set.add(idd)
    print(f"\tImg数量:{len(imgname_set)}/{len(idd_set)}")
    print(f"\tObj数量:{len(annoid_set)}/{len(idd_set)}")
    return len(imgname_set),len(annoid_set)


def fp_analyse(converted_rank:list,top_percent:float,positive_idd_set:set,negtive_idd_set:set):
    top_point = int(len(converted_rank)*top_percent)
    top_rank = converted_rank[:top_point]
    print(f"FP({top_percent*100}%):")
    fp_idd_set = set(top_rank) & negtive_idd_set
    print(f"{len(fp_idd_set)}/{top_point}")
    fp_imgnum, fp_objnum = stat_img_obj_nums(fp_idd_set)
    fp_num = len(fp_idd_set)
    negtive_img_num, nagtive_obj_num = stat_img_obj_nums(negtive_idd_set)
    fpr_img = round(fp_imgnum / negtive_img_num,4)
    fpr_obj = round(fp_objnum / nagtive_obj_num,4)
    fpr = round(fp_num/len(negtive_idd_set),4)
    return fpr_img,fpr_obj,fpr
    # print("TP:")
    # tp_idd_set = set(top_rank) & positive_idd_set
    # print(f"{len(tp_idd_set)}/{top_point}")
    # stat_img_obj_nums(tp_idd_set)
    

def main():
    all_imgname_list = get_all_img_name(trainimgs_dir)
    print(f"总共包含的图像数量:{len(all_imgname_list)}")
    converted_rank = joblib.load(os.path.join(root_dir,method_name,dataset_name,model_name,"rank","converted_rank.joblib"))
    print(f"总的排序长度:{len(converted_rank)}")

    ranked_img_list = []
    ranked_annoId_list = []
    for idd in converted_rank:
        if type(idd) is str:
            ranked_img_list.append(idd)
        elif type(idd) is int:
            ranked_annoId_list.append(idd)
        else:
            raise Exception("rank中的element有误")
    print(f"\t图像数量:{len(ranked_img_list)}")
    print(f"\tObj数量:{len(ranked_annoId_list)}")

    annotations_with_miss = read_json(annotations_with_miss_json_path)
    print("Positive:")
    positive_idd_set = get_all_positive_idd_set(annotations_with_miss)
    print(f"Totalnum:{len(positive_idd_set)}")
    stat_img_obj_nums(positive_idd_set)

    print("Negtive:")
    negtive_idd_set = get_all_negtive_idd_set(annotations_with_miss)
    print(f"Totalnum:{len(negtive_idd_set)}")
    stat_img_obj_nums(negtive_idd_set)

    res = {}
    for top_percent in [0.1,0.2,0.3,0.4,0.5]:
        fpr_img,fpr_obj,fpr = fp_analyse(converted_rank,top_percent,positive_idd_set,negtive_idd_set)
        res[top_percent] = {"fpr":fpr,"fpr_img":fpr_img,"fpr_obj":fpr_obj}
    with open("top_percent_fpr.json", "w", encoding="utf-8") as f:
        json.dump(res,f)
    
if __name__ == "__main__":
    root_dir = "/data/mml/data_debugging_data/ProAFL_data"
    method_name = "ours"
    dataset_name = "voc"
    model_name = "yolov7"
    annotations_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    trainimgs_dir = get_origin_tainimgs_dir(dataset_name)
    main()

