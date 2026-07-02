
from ours.base_data_manager import  get_all_img_name
from collections import defaultdict
from pycocotools.coco import COCO
import json
def get_g_id_to_g_box(g_boxes_json:dict) -> dict:
    g_id_to_g_box = {}
    for img_name, g_boxes in g_boxes_json.items():
        for g_box in g_boxes:
            g_id_to_g_box[g_box["box_id"]] = g_box
    return g_id_to_g_box

def get_imgid_to_imgname(anno:dict) -> dict:
    images = anno["images"]
    img_id_to_img_name = {}
    for image in images:
        img_id = image["id"]
        img_name = image["file_name"]
        img_id_to_img_name[img_id] = img_name
    return img_id_to_img_name

def get_imgname_to_imgid(anno:dict) -> dict:
    images = anno["images"]
    img_name_to_img_id = {}
    for image in images:
        img_id = image["id"]
        img_name = image["file_name"]
        img_name_to_img_id[img_name] = img_id
    return img_name_to_img_id

def get_cls_id_to_name(anno:dict) -> dict:
    categories = anno["categories"]
    cls_id_to_name = {}
    for category in categories:
        cls_id_to_name[category["id"]] = category["name"]
    return cls_id_to_name

def get_img_name_to_ann_ids(anno:dict) -> dict:
    imgname_to_annids = defaultdict(list)
    img_id_to_img_name = get_imgid_to_imgname(anno)
    annos = anno["annotations"]
    for anno in annos:
        anno_id = anno["id"]
        image_id = anno["image_id"]
        img_name = img_id_to_img_name[image_id]
        imgname_to_annids[img_name].append(anno_id)
    return imgname_to_annids

def get_img_name_to_missed_annids(anno_with_miss:dict) -> dict:
    imgname_to_missedannids = defaultdict(list)
    img_id_to_img_name = get_imgid_to_imgname(anno_with_miss)
    annos = anno_with_miss["annotations"]
    for anno in annos:
        anno_id = anno["id"]
        image_id = anno["image_id"]
        img_name = img_id_to_img_name[image_id]
        fault_type = anno["fault_type"]
        if fault_type == 4:
            imgname_to_missedannids[img_name].append(anno_id)
    return imgname_to_missedannids

def get_gid_to_img_and_line(g_boxes_json:dict):
    res = {}
    for img_name,g_box_list in g_boxes_json.items():
        line_no = 0
        for g_box in g_box_list:
            gid = g_box["box_id"]
            res[gid] = {
                "img_name":img_name,
                "line_no":line_no
            }
            line_no += 1
    return res

def get_gid_to_anno_id(g_boxes_json:dict,anno:dict)->dict:
    '''我们收集的gboxs和coco风格的anno json的box进行对应，即gid to annoid'''
    gid_to_anno_id = {}
    gid_to_img_and_line =get_gid_to_img_and_line(g_boxes_json)
    img_name_to_ann_ids = get_img_name_to_ann_ids(anno)
    for gid in gid_to_img_and_line.keys():
        img_name = gid_to_img_and_line[gid]["img_name"]
        line_no = gid_to_img_and_line[gid]["line_no"]
        anno_id = img_name_to_ann_ids[img_name][line_no]
        gid_to_anno_id[gid] = anno_id
    return gid_to_anno_id

def get_annoid_to_imgname(anno_json:dict) -> dict:
    imgid_to_imgname = get_imgid_to_imgname(anno_json)
    annoid_to_imgname = {}
    annos = anno_json["annotations"]
    for anno in annos:
        img_name = imgid_to_imgname[anno["image_id"]]
        annoid_to_imgname[anno["id"]] = img_name
    return annoid_to_imgname

def get_error_annoid_set(anno_json:dict) -> set:
    error_annoid_set = set()
    annos = anno_json["annotations"]
    for anno in annos:
         if anno["fault_type"] != 0:
            error_annoid_set.add(anno["id"])
    return error_annoid_set

def get_all_miss_img_name_list(anno_with_miss_json:dict) -> list:
    imgid_to_imgname = get_imgid_to_imgname(anno_with_miss_json)
    miss_img_name_list = []
    annos = anno_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgid_to_imgname[anno["image_id"]]
            miss_img_name_list.append(img_name)
    return miss_img_name_list

def get_annoId_to_anno(anno_json:dict)->dict:
    annoId_to_anno = {}
    annos = anno_json["annotations"]
    for anno in annos:
        annoId_to_anno[anno["id"]] = anno
    return annoId_to_anno

def get_all_error_annoids(anno_error_with_miss:dict) -> list:
    all_error_annoids = []
    annos = anno_error_with_miss["annotations"]
    for anno in annos:
        if anno["fault_type"] != 0:
            all_error_annoids.append(anno["id"])
    return all_error_annoids

def get_all_error_imgset(anno_error_with_miss:dict)->set:
    error_imgset = set()
    all_error_annoids = get_all_error_annoids(anno_error_with_miss)
    annoid2imgname = get_annoid_to_imgname(anno_error_with_miss)
    for error_annoid in all_error_annoids:
        imgname = annoid2imgname[error_annoid]
        error_imgset.add(imgname)
    return error_imgset


def get_all_error_idd_set(anno_error_with_miss:dict)->set:
    error_annoid_set = set()
    annos = anno_error_with_miss["annotations"]
    for anno in annos:
         if anno["fault_type"] in [1,2,3]:
            error_annoid_set.add(anno["id"])
    missfault_imgname_set = get_all_miss_error_img_name_set(anno_error_with_miss)
    error_idd_set = error_annoid_set | missfault_imgname_set
    return error_idd_set

def get_all_annoids_detail(anno_error_with_miss:dict) -> list:
    all_annoids_detail = {
        "class_fault":[],
        "loc_fault":[],
        "redun_fault":[],
        "miss_fault":[],
        "clean":[]
    }
    annos = anno_error_with_miss["annotations"]
    for anno in annos:
        if anno["fault_type"] == 0:
            all_annoids_detail["clean"].append(anno["id"])
        elif anno["fault_type"] == 1:
            all_annoids_detail["class_fault"].append(anno["id"])
        elif anno["fault_type"] == 2:
            all_annoids_detail["loc_fault"].append(anno["id"])
        elif anno["fault_type"] == 3:
            all_annoids_detail["redun_fault"].append(anno["id"])
        elif anno["fault_type"] == 4:
            all_annoids_detail["miss_fault"].append(anno["id"])
        else:
            raise Exception("fault_type异常")
    return all_annoids_detail

def get_all_error_clean_set(anno_error_with_miss:dict) -> dict:
    all_miss_img_name_set = set()
    cls_error_annoid_set = set()
    loc_error_annoid_set =set()
    redun_error_annoid_set = set()
    clean_annoid_set = set()
    imgId_to_imgName = get_imgid_to_imgname(anno_error_with_miss)
    annos = anno_error_with_miss["annotations"]
    for anno in annos:
        if anno["fault_type"] == 0:
            clean_annoid_set.add(anno["id"])
        elif anno["fault_type"] == 1:
            cls_error_annoid_set.add(anno["id"])
        elif anno["fault_type"] == 2:
            loc_error_annoid_set.add(anno["id"])
        elif anno["fault_type"] == 3:
            redun_error_annoid_set.add(anno["id"])
        elif anno["fault_type"] == 4:
            image_id = anno["image_id"]
            image_name = imgId_to_imgName[image_id]
            all_miss_img_name_set.add(image_name)
    all_error_clean_set = {
        "miss_set":all_miss_img_name_set,
        "cls_set":cls_error_annoid_set,
        "loc_set":loc_error_annoid_set,
        "redun_set":redun_error_annoid_set,
        "clean_set":clean_annoid_set
    }
    return all_error_clean_set

def conver_ours_rank(ours_rank:list,g_boxes_json:dict,anno_error_json:dict) -> list:
    '''
    把我们方法的到的混排（imgname or gid）转换为（imgname or anno_id）
    '''
    gid_to_anno_id = get_gid_to_anno_id(g_boxes_json, anno_error_json)
    converted_ours_rank = []
    for idd in ours_rank:
        if type(idd) is str:
            img_name = idd
            converted_ours_rank.append(img_name)
        else:
            gid = idd # 我们方法rank的idd是（imgname or gid）
            annoid = gid_to_anno_id[gid]
            converted_ours_rank.append(annoid)
    return converted_ours_rank # imgname or gid 的排序

def conver_datactive_rank(datactive_rank:list, bg_catId:int) -> list:
    '''
    把datactive方法得到的序（instances）转换为统一的（imgname or anno_id）序
    '''
    converted_rank_list = []
    for instance in datactive_rank:
        gt_category_id = instance["gt_category_id"] # 背景类实例其实就是被怀疑有miss fault的img
        if gt_category_id == bg_catId:
            converted_rank_list.append(instance["image_name"])
        else:
            converted_rank_list.append(instance["anno_id"])
    return converted_rank_list

def get_all_image_name_list(anno_json:dict)->list:
    all_img_name_list = []
    images = anno_json["images"]
    all_img_name_list = [img["file_name"] for img in images]
    assert len(set(all_img_name_list)) == len(all_img_name_list), "具有重复元素"
    return all_img_name_list

def get_name_id_map(ANN_FILE):
    
    coco = COCO(ANN_FILE)

    # 1) file_name -> img_id（假设 file_name 唯一）
    name2id = {img_info["file_name"]: img_id for img_id, img_info in coco.imgs.items()}

    # 2) 反向：img_id -> file_name
    id2name = {img_id: img_info["file_name"] for img_id, img_info in coco.imgs.items()}

    return name2id,id2name

def get_all_errored_g_box_id_set(gt_json:dict) -> set[int]:
    '''
    基于我们收集的g_boxs，获得fault g box id set
    '''

    all_errored_g_box_id_set = set()
    for img_name,g_boxs in gt_json.items():
        for g_box in g_boxs:
            if g_box["fault_type"] != 0:
                all_errored_g_box_id_set.add(g_box["box_id"])
    return all_errored_g_box_id_set

def get_all_correct_g_box_id_set(gt_json:dict) -> set[int]:
    '''
    基于我们收集的g_boxs，获得fault g box id set
    '''

    all_correct_g_box_id_set = set()
    for img_name,g_boxs in gt_json.items():
        for g_box in g_boxs:
            if g_box["fault_type"] == 0:
                all_correct_g_box_id_set.add(g_box["box_id"])
    return all_correct_g_box_id_set

def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name

def get_all_miss_error_img_name_set(annos_with_miss_json_path:str) -> set[str]:
    '''
    获得所有具有miss fault的 img name set
    '''
    if type(annos_with_miss_json_path) is str:
        with open(annos_with_miss_json_path, "r") as f:
            annos_with_miss_json = json.load(f)
    elif type(annos_with_miss_json_path) is dict:
        annos_with_miss_json = annos_with_miss_json_path
    else:
        raise Exception("参数类型错误")
    
    imageid_2_imagename = get_image_id_to_image_name_for_coco(annos_with_miss_json)
    anns = annos_with_miss_json["annotations"]
    all_miss_error_img_name_set = set()
    for ann in anns:
        if ann["fault_type"] == 4:
            image_name = imageid_2_imagename[ann["image_id"]]
            all_miss_error_img_name_set.add(image_name)
    return all_miss_error_img_name_set


def get_all_gids(gt_json:dict) -> list[int]:
    '''
    得到所有的g_box_id_list
    
    参数
    ----
    gt_json : dict
        数据格式：
        {
            image_name:[g_box_1,g_box_2],
            ...
        }
    返回
    ---
    all_g_box_id_list : list[int]
        提取出的所有的g_box_id_list
    '''
    all_g_box_id_list = []
    for img_name, g_box_list in gt_json.items():
        for g_box in g_box_list:
            all_g_box_id_list.append(g_box["box_id"])
    return all_g_box_id_list

def get_g_id_to_metric(metric_json_path):
    '''
    提供每个gid对应的metric(conf_list和iou_list)
    '''
    with open(metric_json_path, "r", encoding="utf-8") as f:
        gt_box_metric_collection_list = json.load(f)
    print(f"matched gt_box数量:{len(gt_box_metric_collection_list)}")

    g_box_id_to_metric = {}

    for collection in gt_box_metric_collection_list:
        g_box_id = collection["g_box_id"]
        conf_list = collection["conf_list"]
        iou_list = collection["iou_list"]
        g_box_id_to_metric[g_box_id] = {
            "conf_list":conf_list,
            "iou_list":iou_list,
        }
    return g_box_id_to_metric

def split_img_miss_no_miss(imgs_dir,annos_with_miss_json_path:str):
    all_img_name_list = get_all_img_name(imgs_dir)
    with_miss_fault_img_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)
    no_miss_fault_img_set = set(all_img_name_list) - with_miss_fault_img_set
    return with_miss_fault_img_set,no_miss_fault_img_set

def get_annos_by_img_name(anno_json:dict,img_name)->list:
    anno_list = []
    imgname_to_imgid = get_imgname_to_imgid(anno_json)
    imgid = imgname_to_imgid[img_name]
    annos = anno_json["annotations"]
    for anno in annos:
        if anno["image_id"] == imgid:
            anno_list.append(anno)
    return anno_list