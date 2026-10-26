# FCOS–ResNet50–FPN

`fcos_pipeline.py` 通过文件顶部的 Python 常量配置实验，直接运行即可，不需要命令行参数。所有采集结果保存为普通 `.json` 文件。

## 配置和运行

先修改脚本顶部的常量：

```python
MODE = "train"          # 训练；训练完成后改为 "collect" 再运行
DATASET_NAME = "visdrone"
DEVICE = "cuda:1"        # 按可用 GPU 修改；仅支持 CUDA GPU
BATCH_SIZE = 4
NUM_WORKERS = 4
TRAIN_EPOCHS = 50
```

从仓库根目录运行：

```bash
conda run --no-capture-output -n data_debugging python -B frcnn/fcos_pipeline.py
```

也可以在 `frcnn` 目录中直接运行 `python fcos_pipeline.py`。训练和采集始终使用同一条命令，由 `MODE` 决定执行内容。

## 训练设置

默认配置：

```python
MODE = "train"
PRETRAINED_WEIGHTS = "coco"
RESUME_PATH = None
LEARNING_RATE = 0.005
TRAIN_EPOCHS = 50
```

默认从 COCO 预训练权重开始，保留主干、分类特征层、框回归和 centerness 权重，仅替换类别输出卷积。权重缓存不存在时，Torchvision 会下载权重。将 `PRETRAINED_WEIGHTS` 设为 `"none"` 可从随机初始化开始，适合离线功能检查。

训练使用普通 FP32 精度和原版 FCOS 损失计算，仅使用 CUDA GPU，不自动回退 CPU。

训练使用 SGD：momentum=0.9、weight decay=0.0005。第 0 轮做学习率预热，总轮数的 60%、80% 处衰减。输入尺寸由 `MIN_SIZE=800`、`MAX_SIZE=1333` 控制。只做颜色扰动，归一化由模型内部完成。

数据根目录由 `DATA_ROOT` 指定，默认读取：

```text
/data/mml/data_debugging_data/ProAFL_data/datasets/visdrone-coco/train
/data/mml/data_debugging_data/ProAFL_data/fault_inject/0.1/visdrone/coco_format/annotations_no_miss.json
```

权重保存到：

```text
models/visdrone/fcos/fault_train/0.1/repeat_1/
├── run_config.json
├── epoch_0.pth
├── epoch_1.pth
├── ...
├── training_state.pth
└── train_log.jsonl
```

`RESUME_PATH=None` 时从头训练，直接覆盖同名权重和配置，并清空旧训练日志。断点续训时，把 `RESUME_PATH` 设为 `training_state.pth` 的路径，`MODE` 保持 `"train"`，已有训练日志会保留。`TRAIN_EPOCHS` 表示总轮数；续训应保持原有数据、分辨率、batch size 和总轮数。

## 采集设置

训练完成后修改：

```python
MODE = "collect"
COLLECT_EPOCHS = None      # 采集全部已有 epoch 权重；或填写 [0, 10, 25, 49]
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.65
MAX_DETECTIONS = 300      # 每图最终最多保存的预测框数
TOPK_CANDIDATES = 1000    # 每个特征层的候选上限
```

仍运行相同的 Python 命令。`EXPERIMENT_ID=None` 时自动使用时间戳生成实验编号，默认输出：

```text
collection_process_info/visdrone/fcos/collected_predict_boxes/inject_0.1_repeat/repeat_1_<实验编号>/
├── collection_config.json
├── epoch_0_predicted_bboxs.json
└── ...
```

每个文件按图片流式写入，完成后才将临时文件改成正式文件名。图片无预测框时也会保留空列表。预测框字段与现有流程一致：`predicted_box_id`、`img_name`、`predicted_cls`、`conf`、`bbox`、`probs`。预测框 ID 在每轮内全局唯一。

FCOS 类别直接使用 0～9，不加 FRCNN 的背景偏移。完整类别得分定义为：

```text
probs[c] = sqrt(sigmoid(class_logit[c]) * sigmoid(centerness_logit))
conf = probs[predicted_cls]
```

这些是独立的检测得分，不是和为 1 的 softmax 概率。FCOS 和 FRCNN 分数含义不同，同一个分数阈值不代表相同的筛选强度。

## 自定义路径和功能检查

`ANNOTATIONS_PATH`、`IMAGE_DIR`、`CHECKPOINT_DIR`、`COLLECTION_OUTPUT_DIR` 可分别指定路径；设为 `None` 时使用自动生成的默认路径。

只检查少量数据时，可以临时设置：

```python
DEVICE = "cuda:0"
PRETRAINED_WEIGHTS = "none"
NUM_WORKERS = 0
BATCH_SIZE = 2
TRAIN_EPOCHS = 1
MAX_IMAGES = 2
MAX_TRAIN_STEPS = 1
MIN_SIZE = 128
MAX_SIZE = 192
CHECKPOINT_DIR = Path("/tmp/fcos_check/checkpoints")
COLLECTION_OUTPUT_DIR = Path("/tmp/fcos_check/predictions")
```

先以 `MODE="train"` 运行，再改成 `MODE="collect"` 运行。功能检查结果不能作为正式实验结果；正式运行应恢复 `MAX_IMAGES=0`、`MAX_TRAIN_STEPS=0` 及正常的分辨率、轮数和预训练设置。

## 后续 ProAFL 排序

采集格式兼容现有标注和图片过程度量。完整采集后，在 `ours/dual_branch/config.py` 中把 `MODEL_NAME` 设为 `"fcos"`，并在 `EXPERIMENT_IDS` 字典中添加实际实验编号对应的 `("visdrone", "fcos")` 条目，再运行原有过程度量、特征和排序流程。评估时同步设置 `ours/eval_rank_for_ours.py` 的模型名和实验编号。

完整 FCOS 实验需要单独训练，不能直接加载 FRCNN 的权重。脚本不会自动修改当前 FRCNN 实验配置。
