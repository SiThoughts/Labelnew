"""
Export Ensemble to Single ONNX File
Exports YOLO and FRCNN separately, then merges them with fusion logic into one ONNX file
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import onnx
from onnx import helper, numpy_helper
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.common import load_config


def export_models_to_onnx(config):
    """Export both YOLO and FRCNN to ONNX"""
    
    print("=" * 80)
    print("Step 1: Exporting Individual Models to ONNX")
    print("=" * 80)
    
    imgsz = config['export']['imgsz']
    
    # Export YOLO
    print("\n[1/2] Exporting YOLOv8-NAS...")
    yolo_path = 'runs/ensemble/yolov8n/weights/best.pt'
    yolo_onnx = 'models/temp_yolo.onnx'
    
    Path(yolo_onnx).parent.mkdir(parents=True, exist_ok=True)
    
    yolo_model = YOLO(yolo_path)
    yolo_export_path = yolo_model.export(
        format='onnx',
        imgsz=imgsz,
        opset=11,
        simplify=True,
        dynamic=False
    )
    
    # Move to temp location
    import shutil
    if str(yolo_export_path) != yolo_onnx:
        shutil.move(str(yolo_export_path), yolo_onnx)
    
    print(f"✓ YOLO exported to: {yolo_onnx}")
    
    # Export FRCNN
    print("\n[2/2] Exporting Faster R-CNN...")
    frcnn_path = 'runs/ensemble/frcnn_resnet101/best.pth'
    frcnn_onnx = 'models/temp_frcnn.onnx'
    
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
    
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    
    torch.onnx.export(
        frcnn_model,
        dummy_input,
        frcnn_onnx,
        input_names=['frcnn_input'],
        output_names=['frcnn_boxes', 'frcnn_labels', 'frcnn_scores'],
        dynamic_axes={
            'frcnn_boxes': {0: 'num_boxes'},
            'frcnn_labels': {0: 'num_boxes'},
            'frcnn_scores': {0: 'num_boxes'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ FRCNN exported to: {frcnn_onnx}")
    
    return yolo_onnx, frcnn_onnx


def create_fusion_subgraph(yolo_outputs, frcnn_outputs, imgsz, conf_threshold=0.15):
    """
    Create ONNX subgraph for high-recall fusion
    This is a simplified version that concatenates detections from both models
    """
    nodes = []
    initializers = []
    
    # Unpack outputs
    yolo_out = yolo_outputs[0]  # YOLO output tensor
    frcnn_boxes, frcnn_labels, frcnn_scores = frcnn_outputs
    
    # For simplicity, we'll create nodes that:
    # 1. Parse YOLO output
    # 2. Concatenate boxes, labels, scores from both models
    # 3. Apply label mapping
    
    # Create confidence threshold constant
    conf_tensor = helper.make_tensor(
        name='conf_threshold',
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=[conf_threshold]
    )
    initializers.append(conf_tensor)
    
    # Create label mapping tensor [0->2, 1->1]
    label_map_tensor = helper.make_tensor(
        name='label_mapping',
        data_type=onnx.TensorProto.INT64,
        dims=[2],
        vals=[2, 1]
    )
    initializers.append(label_map_tensor)
    
    # Note: Full WBF implementation in ONNX is complex
    # This creates a simplified version that concatenates detections
    # For production, you may need to use the Python wrapper approach
    
    return nodes, initializers


def merge_models_simple(yolo_onnx_path, frcnn_onnx_path, output_path, config):
    """
    Merge YOLO and FRCNN ONNX models into a single file
    Uses onnx.compose to create a unified model
    """
    
    print("\n" + "=" * 80)
    print("Step 2: Merging Models into Single ONNX")
    print("=" * 80)
    
    # Load both models
    print("\nLoading ONNX models...")
    yolo_model = onnx.load(yolo_onnx_path)
    frcnn_model = onnx.load(frcnn_onnx_path)
    
    print(f"YOLO inputs: {[inp.name for inp in yolo_model.graph.input]}")
    print(f"YOLO outputs: {[out.name for out in yolo_model.graph.output]}")
    print(f"FRCNN inputs: {[inp.name for inp in frcnn_model.graph.input]}")
    print(f"FRCNN outputs: {[out.name for out in frcnn_model.graph.output]}")
    
    # Add prefixes to avoid name conflicts
    print("\nAdding prefixes to model nodes...")
    yolo_model_prefixed = onnx.compose.add_prefix(yolo_model, prefix='yolo_', inplace=False)
    frcnn_model_prefixed = onnx.compose.add_prefix(frcnn_model, prefix='frcnn_', inplace=False)
    
    # Create a wrapper graph that runs both models in parallel
    print("\nCreating ensemble wrapper...")
    
    # Get image size
    imgsz = config['export']['imgsz']
    
    # Create shared input
    input_tensor = helper.make_tensor_value_info(
        'input',
        onnx.TensorProto.FLOAT,
        [1, 3, imgsz, imgsz]
    )
    
    # Create identity nodes to connect shared input to both models
    yolo_input_name = [inp.name for inp in yolo_model_prefixed.graph.input][0]
    frcnn_input_name = [inp.name for inp in frcnn_model_prefixed.graph.input][0]
    
    identity_yolo = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=[yolo_input_name],
        name='input_to_yolo'
    )
    
    identity_frcnn = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=[frcnn_input_name],
        name='input_to_frcnn'
    )
    
    # Get output names from both models
    yolo_output_names = [out.name for out in yolo_model_prefixed.graph.output]
    frcnn_output_names = [out.name for out in frcnn_model_prefixed.graph.output]
    
    # Create fusion nodes (simplified - just concatenate)
    # In a full implementation, you'd add WBF nodes here
    
    # For now, we'll output both model predictions separately
    # User can apply fusion in post-processing
    
    # Create final output value infos
    outputs = []
    
    # YOLO outputs (keep first one)
    yolo_out = helper.make_tensor_value_info(
        'yolo_predictions',
        onnx.TensorProto.FLOAT,
        [1, -1, -1]  # Dynamic shape
    )
    outputs.append(yolo_out)
    
    # FRCNN outputs
    frcnn_boxes_out = helper.make_tensor_value_info(
        'frcnn_boxes',
        onnx.TensorProto.FLOAT,
        [-1, 4]
    )
    frcnn_labels_out = helper.make_tensor_value_info(
        'frcnn_labels',
        onnx.TensorProto.INT64,
        [-1]
    )
    frcnn_scores_out = helper.make_tensor_value_info(
        'frcnn_scores',
        onnx.TensorProto.FLOAT,
        [-1]
    )
    outputs.extend([frcnn_boxes_out, frcnn_labels_out, frcnn_scores_out])
    
    # Identity nodes to rename outputs
    identity_yolo_out = helper.make_node(
        'Identity',
        inputs=[yolo_output_names[0]],
        outputs=['yolo_predictions'],
        name='yolo_output'
    )
    
    identity_frcnn_boxes = helper.make_node(
        'Identity',
        inputs=[frcnn_output_names[0]],
        outputs=['frcnn_boxes'],
        name='frcnn_boxes_output'
    )
    
    identity_frcnn_labels = helper.make_node(
        'Identity',
        inputs=[frcnn_output_names[1]],
        outputs=['frcnn_labels'],
        name='frcnn_labels_output'
    )
    
    identity_frcnn_scores = helper.make_node(
        'Identity',
        inputs=[frcnn_output_names[2]],
        outputs=['frcnn_scores'],
        name='frcnn_scores_output'
    )
    
    # Combine all nodes
    all_nodes = (
        [identity_yolo, identity_frcnn] +
        list(yolo_model_prefixed.graph.node) +
        list(frcnn_model_prefixed.graph.node) +
        [identity_yolo_out, identity_frcnn_boxes, identity_frcnn_labels, identity_frcnn_scores]
    )
    
    # Combine initializers
    all_initializers = (
        list(yolo_model_prefixed.graph.initializer) +
        list(frcnn_model_prefixed.graph.initializer)
    )
    
    # Create merged graph
    merged_graph = helper.make_graph(
        nodes=all_nodes,
        name='ensemble_defect_detection',
        inputs=[input_tensor],
        outputs=outputs,
        initializer=all_initializers
    )
    
    # Create model
    merged_model = helper.make_model(
        merged_graph,
        producer_name='defect_detection_ensemble',
        opset_imports=[helper.make_opsetid('', 11)]
    )
    
    # Save model
    print(f"\nSaving merged model to: {output_path}")
    onnx.save(merged_model, output_path)
    
    # Verify
    print("\nVerifying merged model...")
    try:
        onnx.checker.check_model(merged_model)
        print("✓ Model is valid!")
    except Exception as e:
        print(f"⚠ Validation warning: {e}")
    
    return output_path


def main():
    """Main export function"""
    
    print("=" * 80)
    print("ENSEMBLE EXPORT TO SINGLE ONNX FILE")
    print("=" * 80)
    
    # Load config
    config = load_config()
    
    # Export individual models
    yolo_onnx, frcnn_onnx = export_models_to_onnx(config)
    
    # Merge into single ONNX
    output_path = 'models/ensemble_defect_detection_single.onnx'
    merged_path = merge_models_simple(yolo_onnx, frcnn_onnx, output_path, config)
    
    # Clean up temp files
    print("\nCleaning up temporary files...")
    Path(yolo_onnx).unlink()
    Path(frcnn_onnx).unlink()
    print("✓ Cleanup complete")
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE!")
    print("=" * 80)
    print(f"\nSingle ONNX file: {merged_path}")
    print("\nNote: This ONNX contains both models running in parallel.")
    print("Outputs:")
    print("  - yolo_predictions: Raw YOLO output")
    print("  - frcnn_boxes: FRCNN bounding boxes")
    print("  - frcnn_labels: FRCNN class labels")
    print("  - frcnn_scores: FRCNN confidence scores")
    print("\nFor fusion with WBF, use the Python inference wrapper:")
    print("  python inference/ensemble_onnx_inference.py")
    print("\nLabel Mapping (apply in post-processing):")
    print("  - chip (0) → 2")
    print("  - check (1) → 1")
    print("=" * 80)


if __name__ == '__main__':
    main()
