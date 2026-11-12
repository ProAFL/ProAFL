"""顺序执行 fault 标签重复实验：python repeat_train.py。"""

import gc
from pathlib import Path

import torch
import yaml

import train as training


# 实验配置：直接修改这里即可。
exp_data_root = "/data/mml/data_debugging_data/ProAFL_data"
dataset_names = ["voc", "kitti", "visdrone"]
repeat_ids = range(1, 11)
inject_ratio = 0.1
is_save_each_epoch = True
batch_size = 128
workers = 16
gpu_id = 0
epochs = 50
weights = "yolov7.pt"


def run_training(dataset_name, repeat_id):
    """重新创建参数和训练状态，先置换标签，再完成一次训练。"""
    opt = training.parse_opt([], dataset_name=dataset_name, batch_size=batch_size,
                             workers=workers, gpu_id=gpu_id)
    opt.epochs = epochs
    opt.weights = weights
    opt.resume = False
    opt.seed = repeat_id
    opt.save_each_epoch = is_save_each_epoch
    opt.world_size = 1
    opt.global_rank = -1
    opt.total_batch_size = opt.batch_size
    opt.name = f"{dataset_name}_repeat_{repeat_id}"
    opt.save_dir = str(Path(exp_data_root) / "models" / dataset_name / "yolov7" /
                       "fault_train" / str(inject_ratio) / f"repeat_{repeat_id}")
    opt.data = training.check_file(opt.data)
    opt.cfg = training.check_file(opt.cfg)
    opt.hyp = training.check_file(opt.hyp)
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    labels_root = Path(exp_data_root) / "fault_inject" / str(inject_ratio) / dataset_name / "yolo_format"
    training.label_replace(dataset_name, str(labels_root / "labels_train"),
                           str(labels_root / "labels_val"), data_root=exp_data_root)

    training.set_logging(opt.global_rank)
    device = training.select_device(opt.device, batch_size=opt.batch_size)
    writer = training.SummaryWriter(opt.save_dir)
    try:
        return training.train(hyp, opt, device, writer)
    finally:
        writer.close()


def main():
    for dataset_name in dataset_names:
        for repeat_id in repeat_ids:
            print(f"开始训练: dataset={dataset_name}, repeat={repeat_id}", flush=True)
            try:
                run_training(dataset_name, repeat_id)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"训练完成: dataset={dataset_name}, repeat={repeat_id}", flush=True)


if __name__ == "__main__":
    main()
