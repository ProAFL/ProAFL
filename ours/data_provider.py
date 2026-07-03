'''
负责提供一些高层数据
'''
from ours.small_utils import read_json
from helper.data_organization_tools import get_imgid_to_imgname
def get_all_miss_error_img_name_set(annos_with_miss_json_path:str) -> set[str]:
    '''
    获得所有具有miss fault的 img name set
    '''
    annos_with_miss_json = read_json(annos_with_miss_json_path)
    imageid_2_imagename = get_imgid_to_imgname(annos_with_miss_json)
    
    anns = annos_with_miss_json["annotations"]
    all_miss_error_img_name_set = set()
    for ann in anns:
        if ann["fault_type"] == 4:
            image_name = imageid_2_imagename[ann["image_id"]]
            all_miss_error_img_name_set.add(image_name)
    
    return all_miss_error_img_name_set
