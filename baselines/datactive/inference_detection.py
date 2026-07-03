import json
import time

import torch
from tqdm import tqdm

from modelsCodes.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn_v2
from UTILS.mydataset import DetectionDataSet
from UTILS import presets
from UTILS.engine import evaluate
from UTILS.cal_voc_map import cal_voc_map

datatype = 'KITTI'
modeltype = 'ssd300'
runtype = 'train'
model, root_path, layer_num = None, None, None

model_path = './autodl-tmp/models/' + modeltype + 'dirty0.1_' + ('resnet50' if modeltype == 'frcnn' else 'vgg16') + '_' + datatype + '_epoch_25.pth'
results_save_path = './autodl-tmp/' + modeltype + 'dirty0.1_' + datatype + runtype + '_inferences.json'

print(model_path)
print(results_save_path)
if datatype == 'VOC':
    root_path = './autodl-tmp/dataset/VOCdevkit/VOC2012'
    layer_num = 21

elif datatype == 'VisDrone':
    root_path = './autodl-tmp/dataset'
    layer_num = 12

elif datatype == 'COCO':

    root_path = './autodl-tmp/dataset/COCO'
    layer_num = 91

elif datatype == 'KITTI':
    root_path = './autodl-tmp/dataset/KITTI'
    layer_num = 8


# load model
modelState = torch.load(model_path, map_location="cpu")
if modeltype == 'ssd300':
    model = ssd300_vgg16(num_classes=layer_num)
elif modeltype == 'frcnn':
    model = fasterrcnn_resnet50_fpn_v2(num_classes=layer_num)

model.load_state_dict(modelState["model"])

val_dataset = DetectionDataSet(root=root_path, runtype=runtype,
                               transforms=presets.DetectionPresetEval(),
                               datatype=datatype,
                               traintype='clean')

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# inference

results = []
model.eval()

start_time = time.time()

with torch.no_grad():
    for image, targets in tqdm(val_dataloader):

        image = list(img.to(device) for img in image)

        outputs = model(image)
        # instances results
        for i, prediction in enumerate(outputs):
            cat_ids = prediction["labels"].cpu()
            bboxs = prediction["boxes"].cpu().numpy().tolist()
            scores = prediction['scores'].cpu()
            full_score = prediction['full_scores'].cpu().numpy().tolist()
            for j in range(prediction["labels"].shape[0]):
                assert full_score[j][cat_ids[j]] == scores[j]
                content_dic = {
                    "image_name": targets[i]["image_name"],
                    "image_id": int(targets[i]["image_id"].numpy()[0]),
                    "category_id": int(cat_ids[j]),
                    "bbox": bboxs[j],
                    "score": float(scores[j]),
                    "full_scores": full_score[j],
                }
                results.append(content_dic)

    json_str = json.dumps(results, indent=4)

    end_time = time.time()
    print("inference time: ", end_time - start_time)

    with open(results_save_path, 'w') as json_file:
        json_file.write(json_str)

# load results

with open(results_save_path, 'r') as f:
    results = json.load(f)

# check max(full_score) == score
# for i in range(len(results)):
#     print(results[i]["score"],max(results[i]["full_scores"][1:]))
# assert results[i]["score"] == max(results[i]["full_scores"][1:]), "score != max(full_score)"

cal_voc_map(model, results, val_dataset)
