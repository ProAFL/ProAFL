# SSD512–VGG16 训练与采集

入口文件为 `ssd512_pipline.py`，按要求保留这个文件名。通过顶部 Python 常量配置，不使用命令行参数。仅使用 GPU、普通 FP32 训练，逐轮保存权重，采集普通 JSON，每个 epoch 结束后打印汇总。

## 运行

修改文件顶部：

```python
MODE = "train"
DEVICE = "cuda:0"
BATCH_SIZE = 16
TRAIN_EPOCHS = 50
PRETRAINED_WEIGHTS = "coco_transfer"
```

在仓库根目录运行：

```bash
conda run --no-capture-output -n data_debugging python -B frcnn/ssd512_pipline.py
```

在 `frcnn` 目录中也可直接运行 `python ssd512_pipline.py`。训练完成后把 `MODE` 改成 `"collect"`，运行同一条命令。

## SSD512 结构

使用 Torchvision 的 VGG 特征提取器、SSD 检测头、损失及默认框生成器组装 SSD512。固定输入 512×512，七个检测特征层的尺寸为：

```text
64×64、32×32、16×16、8×8、4×4、2×2、1×1
```

每个位置的默认框数为 `[4, 6, 6, 6, 6, 4, 4]`，总计 **24,564** 个。anchor 步长为 `[8, 16, 32, 64, 128, 256, 512]`，尺度为 `[0.04, 0.10, 0.26, 0.42, 0.58, 0.74, 0.90, 1.06]`。结构参照 SSD512 的七层设计及 COCO 尺度设置，源码参考：

- [MMDetection SSD512 配置](https://github.com/open-mmlab/mmdetection/blob/main/configs/ssd/ssd512_coco.py)
- [MMDetection SSD anchor 生成器](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py)
- [Torchvision SSD 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py)

这不是给 `ssd300_vgg16` 传入 `size=512`：该构造器固定为 SSD300。脚本额外修正了 SSD512 中间特征层的 stride/padding，并增加末层、anchor 和预测头。

## 初始化与训练

Torchvision 没有现成的 SSD512 COCO 权重。本脚本的默认初始化为 **SSD300 COCO 权重的部分迁移**：

- 迁移名称和形状兼容的 VGG/额外特征层及框回归参数。
- 不迁移 COCO 分类头；当前类别头、新增层、anchor 数变化的回归层重新初始化。
- 初始化来源及迁移张量数写入 `run_config.json` 和训练状态，不冒充原生 SSD512 预训练权重。

首次使用 `PRETRAINED_WEIGHTS="coco_transfer"` 且缓存不存在时，会下载 Torchvision SSD300 COCO 权重。设成 `"none"` 则从随机初始化训练。采集和续训直接读取本实验权重，不再下载预训练参数。

默认 SGD 学习率 0.002、momentum=0.9、weight decay=0.0005，首轮预热，总轮数的 60%、80% 处衰减。训练默认开启颜色扰动、扩图、随机 IoU 裁剪、水平翻转，框与标签同步变换；可通过 `ENABLE_SSD_AUGMENTATION` 关闭。这是项目中的训练配置，不声称完全复现原论文的训练配方或检测精度。

背景占训练类别 0，原始 VisDrone 类别 0～9 在训练时转换为 1～10；采集输出恢复为 0～9。`probs` 是完整 softmax 去掉背景列后的十维概率，不再重新归一化，因此：

```text
conf = probs[predicted_cls]
sum(probs) <= 1
```

权重保存到独立目录：

```text
models/visdrone/ssd512/fault_train/0.1/repeat_1/
```

每轮生成 `epoch_N.pth`，并更新含优化器和学习率调度器的 `training_state.pth`。`RESUME_PATH=None` 表示从头训练，覆盖同名配置、权重并重置训练日志；设置为 `training_state.pth` 路径则续训，`TRAIN_EPOCHS` 始终表示总轮数。续训需保持数据、增强和训练轮数设置一致。

## 采集

```python
MODE = "collect"
COLLECT_EPOCHS = None       # 全部已有 epoch 权重；也可填写 [0, 10, 25, 49]
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 200        # 每图最终输出上限
TOPK_CANDIDATES = 400       # 每个前景类别的 NMS 前上限
```

每个 epoch 输出普通 `epoch_N_predicted_bboxs.json`，默认目录：

```text
collection_process_info/visdrone/ssd512/collected_predict_boxes/inject_0.1_repeat/repeat_1_<实验编号>/
```

JSON 与现有流程字段一致：`predicted_box_id`、`img_name`、`predicted_cls`、`conf`、`bbox`、`probs`。按图片流式写入，避免整轮预测常驻内存；ID 每轮内全局唯一，无预测图片也保留空列表。

需要调整覆盖和文件大小时，对比固定样本上的阈值和每图框数上限。512 输入仍可能丢失微小目标细节，不能保证相对 FCOS/FRCNN 的精度或未重叠数量。

## 路径与后续排序

默认使用 VisDrone、错误比例 0.1、repeat 1，与原流程读取相同的 `annotations_no_miss.json` 和图片。`ANNOTATIONS_PATH`、`IMAGE_DIR`、`CHECKPOINT_DIR`、`COLLECTION_OUTPUT_DIR` 可指定自定义路径。

完整采集后，在 `ours/dual_branch/config.py` 中将 `MODEL_NAME` 改为 `"ssd512"`，并在 `EXPERIMENT_IDS` 中添加实际采集编号对应的 `("visdrone", "ssd512")` 条目。评估时同步修改 `ours/eval_rank_for_ours.py`。脚本不会自动切换你正在使用的 FCOS/FRCNN 实验。

`MAX_IMAGES`、`MAX_TRAIN_STEPS` 只用于少量图片/步数的功能检查。正式实验设为 0，训练和采集均执行全量流程。
