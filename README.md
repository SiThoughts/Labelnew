# Prepare ROI boxes - MultiScaleRoIAlign expects list of tensors
roi_boxes_list = []  # List of tensors, one per image
gt_labels_list = []
gt_is_defect_list = []

for batch_idx, (pred_boxes, target) in enumerate(zip(pred_boxes_list, targets)):
    if len(pred_boxes) == 0:
        # Add empty tensor for this image
        roi_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32, device=self.device))
        continue
    
    pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32, device=self.device)
    gt_boxes = target['boxes']
    gt_labels = target['labels']
    
    # Match predicted boxes to ground truth
    matched_labels, is_defect = match_boxes_to_gt(pred_boxes_tensor, gt_boxes, gt_labels)
    
    # Add boxes for this image (format: [x1, y1, x2, y2])
    roi_boxes_list.append(pred_boxes_tensor)
    
    gt_labels_list.append(matched_labels)
    gt_is_defect_list.append(is_defect)

# Skip if no boxes in entire batch
total_boxes = sum(len(boxes) for boxes in roi_boxes_list)
if total_boxes == 0:
    continue

gt_labels_all = torch.cat(gt_labels_list, dim=0)
gt_is_defect_all = torch.cat(gt_is_defect_list, dim=0)

# ROI pooling - pass list of tensors
feature_dict = {str(i): feat for i, feat in enumerate(adapted_features)}
image_shapes = [images.shape[2:] for _ in range(images.shape[0])]
roi_features = self.model.roi_pool(feature_dict, roi_boxes_list, image_shapes)
