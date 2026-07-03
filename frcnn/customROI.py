import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops


class RoIHeadsCustom(RoIHeads):
    def postprocess_detections(
        self,
        class_logits,   # type: torch.Tensor
        box_regression, # type: torch.Tensor
        proposals,      # type: list[torch.Tensor]
        image_shapes,   # type: list[tuple[int, int]]
    ):
        # type: (...) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)  # [sum_boxes, num_classes]

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores_full = []  # 每个检测的全类别概率
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # boxes: [B, num_classes, 4] 或 [B, 4, ...]，最后一维是坐标
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # 为每个预测生成 label 索引
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)  # [B, num_classes]

            # 去掉 background 类（index=0）
            boxes = boxes[:, 1:]     # [B, num_classes-1, 4]
            scores = scores[:, 1:]   # [B, num_classes-1]
            labels = labels[:, 1:]   # [B, num_classes-1]

            # 为之后的每个“展开的检测框”保存整条概率向量
            # 思路：对每个 proposal，它的 scores[i, :] 是全类概率；
            # 展开时会变成 B*(num_classes-1) 个“检测”，我们简单重复对应行。
            scores_full = scores.detach().clone()                  # [B, C]
            scores_full = scores_full.unsqueeze(1).repeat(1, scores.size(1), 1)
            # 现在 scores_full 形状是 [B, C, C]，展开后与 boxes/scores 对齐
            scores_full = scores_full.reshape(-1, scores.size(1))  # [B*C, C]

            # 展开 boxes / scores / labels
            boxes = boxes.reshape(-1, 4)       # [B*C, 4]
            scores_flat = scores.reshape(-1)   # [B*C]
            labels_flat = labels.reshape(-1)   # [B*C]

            # 阈值过滤
            inds = torch.where(scores_flat > self.score_thresh)[0]
            boxes, scores_flat, labels_flat, scores_full = \
                boxes[inds], scores_flat[inds], labels_flat[inds], scores_full[inds]

            # 去掉过小的框
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores_flat, labels_flat, scores_full = \
                boxes[keep], scores_flat[keep], labels_flat[keep], scores_full[keep]

            # 按类别做 batched NMS
            keep = box_ops.batched_nms(boxes, scores_flat, labels_flat, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            boxes, scores_full, labels_flat = \
                boxes[keep], scores_full[keep], labels_flat[keep]

            all_boxes.append(boxes)
            all_scores_full.append(scores_full)  # 每个检测 [num_classes-1] 的概率向量
            all_labels.append(labels_flat)

        return all_boxes, all_scores_full, all_labels