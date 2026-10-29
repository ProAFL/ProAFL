"""统一实验配置：切换实验时修改这里，五个流程脚本共用。"""

from pathlib import Path

from ours.small_utils import read_yaml

DATA_ROOT = Path(read_yaml("config.yaml")["exp_data_dir"])
DATASETNAME = "visdrone"    # voc | kitti | visdrone
MODEL_NAME = "yolov7"     # yolov7 | rtdetr | frcnn | fcos
FAULT_RATIO = 0.1
REPEAT_ID = 1
'''
EXPERIMENT_IDS = {
    ("voc", "yolov7"): 1791094449,
    ("kitti", "yolov7"): 1792056154,
    ("visdrone", "yolov7"): 1791968956,
    ("voc", "rtdetr"): 1792398239,
    ("kitti", "rtdetr"): 1792556872,
    ("visdrone", "rtdetr"): 1792504350,
    ("voc", "frcnn"): 1792570209,
    ("kitti", "frcnn"): 1792642708,
    ("visdrone", "frcnn"): 1792917048, # 1792748834,1792917048(500per)，1792933936
    ("visdrone", "fcos"):1793025492 
}
EXPID = EXPERIMENT_IDS[(DATASETNAME, MODEL_NAME)]
'''
# EPOCHS = 100 if (DATASETNAME, MODEL_NAME) == ("voc", "rtdetr") else 50
EPOCHS = 50
LATE_EPOCHS = 10

if DATASETNAME == "visdrone":
    CONF_THRESHOLD = 0.1
else:
    CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 图片分支保持原有规则，不增加类别冲突特征。
MIN_CANDIDATE_CONF = 0.25
MAX_ANNO_IOU = 0.2 # 这个预测框与GTbox的IOU阈值不能超过0.2
MAX_ANNO_COVERAGE = 0.5
DUPLICATE_IOU = 0.97
TRACK_IOU = 0.5
# IMAGE_PERCENTILE_SCALE = 0.1 if (DATASETNAME, MODEL_NAME) == ("visdrone", "yolov7") else 1.0
if DATASETNAME == "visdrone":
    if MODEL_NAME == "yolov7":
        IMAGE_PERCENTILE_SCALE = 0.1
    else:
        IMAGE_PERCENTILE_SCALE = 0.1
else:
    IMAGE_PERCENTILE_SCALE = 1.0

ANNOTATIONS_PATH = DATA_ROOT / f"fault_inject/{FAULT_RATIO}/{DATASETNAME}/coco_format/annotations_no_miss.json"

'''
PREDICTIONS_DIR = DATA_ROOT / (
    f"collection_process_info/{DATASETNAME}/{MODEL_NAME}/collected_predict_boxes/"
    f"inject_{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}_{EXPID}"
)
'''
# 当前 VisDrone/YOLOv7 的既有过程度量对应此采集实验；其他实验可显式指定目录。
PREDICTION_EXPERIMENT_ID = (
    1791968956 if (DATASETNAME, MODEL_NAME, REPEAT_ID) == ("visdrone", "yolov7", 1) else None
)
PREDICTION_REPEAT_DIR = f"repeat_{REPEAT_ID}"
if PREDICTION_EXPERIMENT_ID is not None:
    PREDICTION_REPEAT_DIR += f"_{PREDICTION_EXPERIMENT_ID}"
PREDICTIONS_DIR = DATA_ROOT / (
    f"collection_process_info/{DATASETNAME}/{MODEL_NAME}/collected_predict_boxes/"
    f"inject_{FAULT_RATIO}_repeat/{PREDICTION_REPEAT_DIR}"
)

# 所有新结果放入独立目录，避免覆盖原流程数据。
# EXPERIMENT_PATH = f"{DATASETNAME}/{MODEL_NAME}/{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}_{EXPID}"
EXPERIMENT_PATH = f"{DATASETNAME}/{MODEL_NAME}/{FAULT_RATIO}_repeat/repeat_{REPEAT_ID}"
METRICS_DIR = DATA_ROOT / "dual_branch/process_metrics" / EXPERIMENT_PATH
FEATURES_DIR = DATA_ROOT / "dual_branch/process_features" / EXPERIMENT_PATH
