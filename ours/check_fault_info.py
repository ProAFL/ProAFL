'''
textfaulttext
'''
from helper.base_data_manager import get_annotations_with_miss_json_path
from ours.data_provider import get_all_miss_error_img_name_set


def main():

    all_miss_error_img_name_set = get_all_miss_error_img_name_set(error_anno_file_path)
    print(f"textmiss faulttextCount:{len(all_miss_error_img_name_set)}")


if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8"                           
    error_anno_file_path = get_annotations_with_miss_json_path(dataset_name)

    main()


