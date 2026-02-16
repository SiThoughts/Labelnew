"""
Export Ensemble Model (YOLOv8-NAS + Faster R-CNN) to ONNX with Weighted Boxes Fusion
This creates a composite ONNX model that runs both models and fuses predictions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import onnx
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.common import load_config


class EnsembleModelONNX(nn.Module):
    """
    Ensemble model wrapper for ONNX export
    Combines YOLOv8 and Faster R-CNN predictions with Weighted Boxes Fusion
    """
    
    def __init__(self, yolo_model, frcnn_model, config):
        super().__init__()
        self.yolo_model = yolo_model.model  # Get underlying PyTorch model
        self.frcnn_model = frcnn_model
        
        # Ensemble parameters
        self.yolo_weight = config['ensemble']['models'][0]['weight']
        self.frcnn_weight = config['ensemble']['models'][1]['weight']
        self.iou_threshold = config['ensemble']['wbf_iou_threshold']
        self.conf_threshold = config['recall_optimization']['conf_threshold_inference']
        
        # Label mapping: chip (0→2), check (1→1)
        self.label_mapping = torch.tensor([2, 1], dtype=torch.long)
        
        # Set models to eval mode
        self.yolo_model.eval()
        self.frcnn_model.eval()
    
    def forward(self, x):
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor (1, 3, H, W)
            
        Returns:
            boxes: (N, 4) - [x1, y1, x2, y2]
            labels: (N,) - class IDs (remapped)
            scores: (N,) - confidence scores
        """
        # Get YOLO predictions
        yolo_output = self.yolo_model(x)
        
        # Get FRCNN predictions
        frcnn_output = self.frcnn_model(x)
        
        # Parse YOLO output
        # YOLOv8 output format: [batch, num_predictions, 4+num_classes]
        yolo_pred = yolo_output[0]  # Get first output
        
        # Extract boxes, scores, labels from YOLO
        yolo_boxes, yolo_scores, yolo_labels = self.parse_yolo_output(yolo_pred, x.shape[2:])
        
        # Extract boxes, scores, labels from FRCNN
        frcnn_boxes = frcnn_output[0]['boxes']
        frcnn_scores = frcnn_output[0]['scores']
        frcnn_labels = frcnn_output[0]['labels'] - 1  # Convert to 0-indexed
        
        # Filter by confidence
        yolo_mask = yolo_scores >= self.conf_threshold
        frcnn_mask = frcnn_scores >= self.conf_threshold
        
        yolo_boxes = yolo_boxes[yolo_mask]
        yolo_scores = yolo_scores[yolo_mask]
        yolo_labels = yolo_labels[yolo_mask]
        
        frcnn_boxes = frcnn_boxes[frcnn_mask]
        frcnn_scores = frcnn_scores[frcnn_mask]
        frcnn_labels = frcnn_labels[frcnn_mask]
        
        # Combine predictions with weighted boxes fusion
        boxes, scores, labels = self.weighted_boxes_fusion(
            yolo_boxes, yolo_scores, yolo_labels,
            frcnn_boxes, frcnn_scores, frcnn_labels,
            x.shape[2:]
        )
        
        # Apply label mapping
        if len(labels) > 0:
            labels = self.label_mapping[labels]
        
        return boxes, labels, scores
    
    def parse_yolo_output(self, yolo_pred, img_shape):
        """Parse YOLO output to boxes, scores, labels"""
        # YOLOv8 format: [x, y, w, h, class0_score, class1_score, ...]
        
        if len(yolo_pred.shape) == 3:
            yolo_pred = yolo_pred[0]  # Remove batch dimension
        
        # Get class scores
        class_scores = yolo_pred[:, 4:]
        scores, labels = torch.max(class_scores, dim=1)
        
        # Get boxes
        boxes_xywh = yolo_pred[:, :4]
        
        # Convert to xyxy
        boxes = torch.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        
        return boxes, scores, labels.long()
    
    def weighted_boxes_fusion(self, boxes1, scores1, labels1, boxes2, scores2, labels2, img_shape):
        """
        High-Recall Weighted Boxes Fusion in PyTorch
        Strategy: Keep ALL boxes from either model, only merge when both detect same object
        This maximizes recall by ensuring no detections are lost
        """
        # Combine all boxes
        if len(boxes1) == 0 and len(boxes2) == 0:
            return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        
        if len(boxes1) == 0:
            return boxes2, scores2, labels2
        
        if len(boxes2) == 0:
            return boxes1, scores1, labels1
        
        # Normalize boxes to [0, 1]
        h, w = img_shape
        boxes1_norm = boxes1.clone()
        boxes1_norm[:, [0, 2]] /= w
        boxes1_norm[:, [1, 3]] /= h
        
        boxes2_norm = boxes2.clone()
        boxes2_norm[:, [0, 2]] /= w
        boxes2_norm[:, [1, 3]] /= h
        
        # Weight scores
        scores1_weighted = scores1 * self.yolo_weight
        scores2_weighted = scores2 * self.frcnn_weight
        
        # HIGH RECALL STRATEGY: Use very high IoU threshold for fusion
        # Only merge boxes that are almost identical, keep everything else
        high_iou_threshold = 0.8  # Much higher than normal to preserve all detections
        
        # Find matching boxes between models
        matched_boxes = []
        matched_scores = []
        matched_labels = []
        used_boxes2 = set()
        
        for i in range(len(boxes1_norm)):
            box1 = boxes1_norm[i]
            label1 = labels1[i]
            score1 = scores1_weighted[i]
            
            best_iou = 0
            best_j = -1
            
            # Find best matching box from model 2 with same label
            for j in range(len(boxes2_norm)):
                if j in used_boxes2:
                    continue
                
                if labels2[j] != label1:
                    continue
                
                iou = self.box_iou(box1.unsqueeze(0), boxes2_norm[j].unsqueeze(0))[0, 0]
                
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            # If high IoU match found, merge them
            if best_iou >= high_iou_threshold and best_j >= 0:
                # Average the boxes weighted by scores
                total_score = score1 + scores2_weighted[best_j]
                merged_box = (box1 * score1 + boxes2_norm[best_j] * scores2_weighted[best_j]) / total_score
                
                matched_boxes.append(merged_box)
                matched_scores.append(total_score)  # Sum scores for merged detections
                matched_labels.append(label1)
                used_boxes2.add(best_j)
            else:
                # No match - keep box from model 1
                matched_boxes.append(box1)
                matched_scores.append(score1)
                matched_labels.append(label1)
        
        # Add all unmatched boxes from model 2
        for j in range(len(boxes2_norm)):
            if j not in used_boxes2:
                matched_boxes.append(boxes2_norm[j])
                matched_scores.append(scores2_weighted[j])
                matched_labels.append(labels2[j])
        
        # Convert to tensors
        if len(matched_boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        
        fused_boxes = torch.stack(matched_boxes)
        fused_scores = torch.stack(matched_scores)
        fused_labels = torch.stack(matched_labels)
        
        # Apply very light NMS only to remove exact duplicates (IoU > 0.95)
        # This preserves overlapping detections for maximum recall
        keep_indices = self.soft_nms_fusion(fused_boxes, fused_scores, fused_labels, 0.95)
        
        # Denormalize boxes
        final_boxes = fused_boxes[keep_indices]
        final_boxes[:, [0, 2]] *= w
        final_boxes[:, [1, 3]] *= h
        
        return final_boxes, fused_scores[keep_indices], fused_labels[keep_indices]
    
    def soft_nms_fusion(self, boxes, scores, labels, iou_threshold):
        """
        Very soft NMS for fusion - only removes near-exact duplicates
        Preserves overlapping detections for maximum recall
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # Keep highest scoring box
            idx = sorted_indices[0]
            keep.append(idx.item())
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            ious = self.box_iou(boxes[idx].unsqueeze(0), boxes[sorted_indices[1:]])
            
            # Only remove if BOTH high IoU AND same label (very strict)
            # This keeps overlapping detections from different models
            mask = (ious[0] < iou_threshold) | (labels[sorted_indices[1:]] != labels[idx])
            sorted_indices = sorted_indices[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def box_iou(self, box1, boxes2):
        """Calculate IoU between box1 and boxes2"""
        # box1: (1, 4), boxes2: (N, 4)
        x1 = torch.max(box1[:, 0].unsqueeze(1), boxes2[:, 0])
        y1 = torch.max(box1[:, 1].unsqueeze(1), boxes2[:, 1])
        x2 = torch.min(box1[:, 2].unsqueeze(1), boxes2[:, 2])
        y2 = torch.min(box1[:, 3].unsqueeze(1), boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1.unsqueeze(1) + area2 - intersection
        
        return intersection / union


def export_ensemble_to_onnx():
    """Export ensemble model to ONNX"""
    
    print("=" * 80)
    print("Exporting Ensemble Model to ONNX")
    print("YOLOv8-NAS + Faster R-CNN with Weighted Boxes Fusion")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load YOLO model
    yolo_path = 'runs/ensemble/yolov8n/weights/best.pt'
    print(f"\nLoading YOLOv8-NAS from: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    
    # Load Faster R-CNN model
    frcnn_path = 'runs/ensemble/frcnn_resnet101/best.pth'
    print(f"Loading Faster R-CNN from: {frcnn_path}")
    
    frcnn_model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
    num_classes = config['dataset']['nc'] + 1
    frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(frcnn_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        frcnn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        frcnn_model.load_state_dict(checkpoint)
    
    frcnn_model.eval()
    
    # Create ensemble model
    print("\nCreating ensemble model...")
    ensemble_model = EnsembleModelONNX(yolo_model, frcnn_model, config)
    ensemble_model.eval()
    
    # Create dummy input
    imgsz = config['export']['imgsz']
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    
    # Output path
    output_path = 'models/ensemble_defect_detection.onnx'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to ONNX...")
    print(f"Output: {output_path}")
    print(f"Input size: {imgsz}x{imgsz}")
    print(f"Opset: 11")
    
    # Export to ONNX
    torch.onnx.export(
        ensemble_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'boxes': {0: 'num_boxes'},
            'labels': {0: 'num_boxes'},
            'scores': {0: 'num_boxes'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"\n✓ ONNX export completed!")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
    except Exception as e:
        print(f"⚠ Warning: ONNX validation failed: {e}")
    
    # Test inference
    print("\nTesting ONNX inference...")
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Get input/output details
        input_name = session.get_inputs()[0].name
        print(f"Input name: {input_name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        
        for i, output in enumerate(session.get_outputs()):
            print(f"Output {i}: {output.name}, shape: {output.shape}")
        
        # Run test inference
        test_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})
        
        print(f"\n✓ Inference test successful!")
        print(f"Number of detections: {len(outputs[0])}")
        
    except Exception as e:
        print(f"⚠ Warning: Inference test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {output_path}")
    print("\nLabel Mapping:")
    print("  - chip (YOLO 0) → ONNX output 2")
    print("  - check (YOLO 1) → ONNX output 1")
    print("\nUsage:")
    print(f"  python inference/onnx_inference.py --model {output_path} --image <image_path>")
    print("=" * 80)


if __name__ == '__main__':
    export_ensemble_to_onnx()
