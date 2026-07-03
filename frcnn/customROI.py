import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import boxes as box_ops


class RoIHeadsCustom(RoIHeads):
    def postprocess_detections(
        self,
        class_logits,                       
        box_regression,                     
        proposals,                                
        image_shapes,                                
    ):
                                                                                          
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)                            

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores_full = []                                
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
                                                              
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

                             
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)                    

                                           
            boxes = boxes[:, 1:]                            
            scores = scores[:, 1:]                       
            labels = labels[:, 1:]                       

                            
                                                        
                                                     
            scores_full = scores.detach().clone()                          
            scores_full = scores_full.unsqueeze(1).repeat(1, scores.size(1), 1)
                                                                    
            scores_full = scores_full.reshape(-1, scores.size(1))            

                                          
            boxes = boxes.reshape(-1, 4)                 
            scores_flat = scores.reshape(-1)          
            labels_flat = labels.reshape(-1)          

                  
            inds = torch.where(scores_flat > self.score_thresh)[0]
            boxes, scores_flat, labels_flat, scores_full =\
                boxes[inds], scores_flat[inds], labels_flat[inds], scores_full[inds]

                  
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores_flat, labels_flat, scores_full =\
                boxes[keep], scores_flat[keep], labels_flat[keep], scores_full[keep]

                              
            keep = box_ops.batched_nms(boxes, scores_flat, labels_flat, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            boxes, scores_full, labels_flat =\
                boxes[keep], scores_full[keep], labels_flat[keep]

            all_boxes.append(boxes)
            all_scores_full.append(scores_full)                             
            all_labels.append(labels_flat)

        return all_boxes, all_scores_full, all_labels