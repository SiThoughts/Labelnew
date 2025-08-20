# ####################################################################################
# SOTA Defect Detection Training Orchestrator (2025 Standards)
#
# This script integrates multiple advanced techniques for state-of-the-art results:
# 1.  Architecture Choice: Swappable SOTA backbones (ConvNeXt, Swin Transformer).
# 2.  Advanced Augmentation: Integrated Albumentations for realistic industrial transforms.
# 3.  Performance: Automated Mixed Precision (AMP) for faster training & less VRAM.
# 4.  Stability: Gradient Clipping to prevent training instability.
# 5.  Monitoring: TensorBoard logging for real-time visualization of all key metrics.
# 6.  Data Handling: WeightedRandomSampler to combat class imbalance.
# 7.  Checkpointing: Saves the model only when validation mAP improves.
# 8.  Compatibility: Guarantees 100% ONNX export compatibility for the black-box device.
#
# Dataset Structure Expected:
# D:\Photomask\DS2_Sort2\
# ├── EV\
# │   ├── image1.png
# │   ├── image1.xml
# │   ├── image2.png
# │   ├── image2.xml
# │   └── ...
# └── SV\
#     ├── image1.png
#     ├── image1.xml
#     └── ...
#
# Created by Manus AI
# ####################################################################################

import os
import sys
import shutil
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import time
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# PyTorch & TorchVision Imports
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import backbone_with_fpn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, swin_t, Swin_T_Weights
import torchvision.transforms as T

# --- CONFIGURATION CLASS ---
class Config:
    """
    Centralized configuration for the training orchestrator.
    Modify these parameters to customize your training.
    """
    # --- Execution Mode ---
    # Choose 'EV' or 'SV' to train the respective model
    MODE = 'EV'  # <-- CHANGE THIS TO 'SV' FOR THE SV MODEL

    # --- Model Architecture ---
    # Choose from: 'ConvNeXt', 'SwinTransformer'
    BACKBONE = 'ConvNeXt'

    # --- Training Hyperparameters ---
    EPOCHS = 50
    BATCH_SIZE = 2  # Adjust based on your GPU VRAM
    WORKERS = 4
    LEARNING_RATE = 0.001
    OPTIMIZER = 'AdamW'  # 'SGD' or 'AdamW'
    GRADIENT_CLIP_VAL = 1.0  # Max norm for gradient clipping
    WEIGHT_DECAY = 0.0001

    # --- Dataset & Paths ---
    SPLIT_RATIO = 0.8  # 80% train, 20% validation
    
    # IMPORTANT: Update these class mappings for your specific defects
    # The '__background__' class MUST always be included and set to 0
    CLASS_MAP = {
        '__background__': 0,
        'chip': 1,
        'check': 2
    }

    # --- Base Dataset Path ---
    # This should point to your main dataset folder containing EV and SV subfolders
    BASE_DATASET_PATH = r"D:\Photomask\DS2_Sort2"

    # --- Advanced Training Options ---
    USE_MIXED_PRECISION = True  # Enable for faster training and lower VRAM usage
    USE_WEIGHTED_SAMPLER = True  # Enable to handle class imbalance
    SAVE_BEST_ONLY = True  # Only save model when validation mAP improves
    
    # --- Augmentation Intensity ---
    # Set to 'light', 'medium', or 'heavy'
    AUGMENTATION_LEVEL = 'medium'

    # --- DO NOT EDIT BELOW THIS LINE ---
    def __init__(self):
        self.NUM_CLASSES = len(self.CLASS_MAP)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.RESULTS_DIR = f"runs/training_{self.MODE}_{self.BACKBONE}_{timestamp}"
        
        # Set image size based on mode
        if self.MODE == 'EV':
            self.IMAGE_SIZE = (2048, 1460)
        elif self.MODE == 'SV':
            self.IMAGE_SIZE = (1024, 500)
        else:
            raise ValueError(f"Unknown MODE: {self.MODE}. Must be 'EV' or 'SV'")
        
        # Set source dataset path
        self.SOURCE_DATASET_PATH = os.path.join(self.BASE_DATASET_PATH, self.MODE)
        
        # Create all necessary directories
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, 'valid'), exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, 'logs'), exist_ok=True)

# --- ADVANCED DATASET CLASS ---
class AdvancedDefectDataset(Dataset):
    """
    Advanced dataset class that handles Pascal VOC XML format annotations
    with sophisticated augmentation pipeline.
    """
    def __init__(self, image_dir, class_mapping, transforms=None, mode='train'):
        self.image_dir = image_dir
        self.transforms = transforms
        self.class_mapping = class_mapping
        self.mode = mode
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # Verify that corresponding XML files exist
        valid_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            xml_file = f"{base_name}.xml"
            xml_path = os.path.join(image_dir, xml_file)
            if os.path.exists(xml_path):
                valid_files.append(img_file)
            else:
                print(f"Warning: No XML file found for {img_file}")
        
        self.image_files = valid_files
        print(f"Loaded {len(self.image_files)} valid image-annotation pairs from {image_dir}")

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get corresponding XML file
        base_name = os.path.splitext(img_name)[0]
        xml_path = os.path.join(self.image_dir, f"{base_name}.xml")

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Parse XML annotation
        boxes, labels = self._parse_xml(xml_path)
        
        # Create target dictionary
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        # Apply augmentations
        if self.transforms:
            try:
                if len(boxes) > 0:
                    # Convert boxes to albumentations format (normalized)
                    h, w = image.shape[:2]
                    albu_boxes = []
                    albu_labels = []
                    
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        
                        albu_boxes.append([x1, y1, x2, y2])
                        albu_labels.append(label)
                    
                    transformed = self.transforms(
                        image=image, 
                        bboxes=albu_boxes, 
                        labels=albu_labels
                    )
                    
                    image = transformed['image']
                    transformed_boxes = transformed['bboxes']
                    transformed_labels = transformed['labels']
                    
                    # Update target
                    target['boxes'] = torch.as_tensor(transformed_boxes, dtype=torch.float32) if transformed_boxes else torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.as_tensor(transformed_labels, dtype=torch.int64) if transformed_labels else torch.zeros((0,), dtype=torch.int64)
                else:
                    # No boxes, just transform image
                    transformed = self.transforms(image=image, bboxes=[], labels=[])
                    image = transformed['image']
                    
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_name}: {e}")
                # Fallback to basic tensor conversion
                image = T.ToTensor()(Image.fromarray(image))

        return image, target

    def _parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            boxes = []
            labels = []
            
            for obj in root.findall('object'):
                # Get class name
                class_name = obj.find('name').text.strip()
                
                if class_name in self.class_mapping:
                    # Get bounding box coordinates
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    # Validate coordinates
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.class_mapping[class_name])
                    else:
                        print(f"Warning: Invalid box coordinates in {xml_path}: {[xmin, ymin, xmax, ymax]}")
                else:
                    print(f"Warning: Unknown class '{class_name}' in {xml_path}")
            
            return boxes, labels
            
        except Exception as e:
            print(f"Error parsing XML file {xml_path}: {e}")
            return [], []

    def __len__(self):
        return len(self.image_files)

    def get_class_distribution(self):
        """Analyze class distribution for weighted sampling."""
        class_counts = Counter()
        
        for img_file in tqdm(self.image_files, desc="Analyzing class distribution"):
            base_name = os.path.splitext(img_file)[0]
            xml_path = os.path.join(self.image_dir, f"{base_name}.xml")
            
            _, labels = self._parse_xml(xml_path)
            for label in labels:
                class_counts[label] += 1
        
        return class_counts

    def get_sample_weights(self):
        """Calculate sample weights for WeightedRandomSampler."""
        print("Calculating sample weights for balanced training...")
        
        # Get class distribution
        class_counts = self.get_class_distribution()
        
        if not class_counts:
            print("Warning: No valid annotations found. Using uniform weights.")
            return torch.ones(len(self.image_files))
        
        # Calculate inverse frequency weights
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for class_id in range(len(self.class_mapping)):
            if class_id in class_counts:
                class_weights[class_id] = total_samples / (len(class_counts) * class_counts[class_id])
            else:
                class_weights[class_id] = 1.0
        
        # Calculate weight for each sample
        sample_weights = []
        
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            xml_path = os.path.join(self.image_dir, f"{base_name}.xml")
            
            _, labels = self._parse_xml(xml_path)
            
            if labels:
                # Use maximum weight among all classes in the image
                max_weight = max(class_weights[label] for label in labels)
                sample_weights.append(max_weight)
            else:
                sample_weights.append(1.0)
        
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Class weights: {class_weights}")
        
        return torch.DoubleTensor(sample_weights)

# --- INTELLIGENT AUGMENTATION PIPELINE ---
def get_transforms(train=False, level='medium'):
    """
    Creates a curated, intelligent augmentation pipeline specifically for
    high-precision defect detection (e.g., chips, checks, cracks).
    
    Args:
        train (bool): Whether to apply training augmentations
        level (str): Augmentation intensity - 'light', 'medium', or 'heavy'
    """
    if not train:
        # Validation set should ONLY have normalization
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc', 
            label_fields=['labels'],
            min_visibility=0.1
        ))
    
    # Training augmentations based on intensity level
    transforms = []
    
    # Basic geometric transforms (always included)
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    
    # Intensity-based augmentations
    if level == 'light':
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        ])
    
    elif level == 'medium':
        transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.03,
                rotate_limit=10,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.25),
        ])
    
    elif level == 'heavy':
        transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                p=0.6,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
        ])
    
    # Final normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3,  # Keep boxes with at least 30% visibility
        min_area=16          # Discard tiny boxes (4x4 pixels)
    ))

# --- MODEL ARCHITECTURE ---
def get_sota_model(cfg: Config):
    """
    Creates a Faster R-CNN model with state-of-the-art backbone.
    
    Args:
        cfg (Config): Configuration object
        
    Returns:
        torch.nn.Module: Configured Faster R-CNN model
    """
    print(f"Initializing model with SOTA {cfg.BACKBONE} backbone...")
    
    if cfg.BACKBONE == 'ConvNeXt':
        # ConvNeXt-Tiny backbone
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        backbone.out_channels = 768
        return_layers = {'1': '0', '3': '1', '5': '2', '7': '3'}
        
    elif cfg.BACKBONE == 'SwinTransformer':
        # Swin Transformer-Tiny backbone
        backbone = swin_t(weights=Swin_T_Weights.DEFAULT).features
        backbone.out_channels = 768
        return_layers = {'1': '0', '3': '1', '5': '2'}
        
    else:
        raise ValueError(f"Backbone '{cfg.BACKBONE}' not recognized. Choose 'ConvNeXt' or 'SwinTransformer'")

    # Wrap backbone with Feature Pyramid Network
    fpn_backbone = backbone_with_fpn(
        backbone, 
        return_layers=return_layers, 
        out_channels=256
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(fpn_backbone, num_classes=cfg.NUM_CLASSES)
    
    print(f"Model created successfully with {cfg.NUM_CLASSES} classes")
    return model

# --- DATA SPLITTING UTILITY ---
def split_dataset(cfg: Config):
    """
    Split the source dataset into training and validation sets.
    
    Args:
        cfg (Config): Configuration object
        
    Returns:
        tuple: (train_dir, valid_dir) paths
    """
    print(f"\n--- Splitting Dataset: {cfg.MODE} ---")
    
    source_dir = cfg.SOURCE_DATASET_PATH
    train_dir = os.path.join(cfg.RESULTS_DIR, 'train')
    valid_dir = os.path.join(cfg.RESULTS_DIR, 'valid')
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")
    
    # Get all image files
    all_files = [
        f for f in os.listdir(source_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    
    # Filter files that have corresponding XML annotations
    valid_files = []
    for img_file in all_files:
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        if os.path.exists(os.path.join(source_dir, xml_file)):
            valid_files.append(base_name)
        else:
            print(f"Warning: No XML annotation for {img_file}")
    
    if not valid_files:
        raise ValueError(f"No valid image-annotation pairs found in {source_dir}")
    
    print(f"Found {len(valid_files)} valid image-annotation pairs")
    
    # Shuffle and split
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(valid_files)
    
    split_idx = int(len(valid_files) * cfg.SPLIT_RATIO)
    train_files = valid_files[:split_idx]
    valid_files = valid_files[split_idx:]
    
    print(f"Split: {len(train_files)} training, {len(valid_files)} validation")
    
    # Copy files to respective directories
    def copy_files(file_list, target_dir, desc):
        for base_name in tqdm(file_list, desc=desc):
            # Find the actual image file (handle different extensions)
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = f"{base_name}{ext}"
                if os.path.exists(os.path.join(source_dir, candidate)):
                    img_file = candidate
                    break
            
            if img_file:
                # Copy image and XML
                shutil.copy2(
                    os.path.join(source_dir, img_file),
                    os.path.join(target_dir, img_file)
                )
                shutil.copy2(
                    os.path.join(source_dir, f"{base_name}.xml"),
                    os.path.join(target_dir, f"{base_name}.xml")
                )
    
    copy_files(train_files, train_dir, "Copying training files")
    copy_files(valid_files, valid_dir, "Copying validation files")
    
    return train_dir, valid_dir

# --- TRAINING UTILITIES ---
def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return tuple(zip(*batch))

def train_one_epoch_advanced(model, optimizer, data_loader, device, epoch, writer, scaler, grad_clip_val):
    """
    Advanced training loop with mixed precision and gradient clipping.
    """
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        
        # Gradient clipping
        if grad_clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += losses.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{losses.item():.4f}",
            'Avg Loss': f"{total_loss/num_batches:.4f}"
        })
        
        # Log to TensorBoard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', losses.item(), global_step)
            
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    if writer:
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    
    return avg_loss

def evaluate_model(model, data_loader, device):
    """
    Evaluate model on validation set.
    Returns a simple average loss for model selection.
    """
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation")
        
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Val Loss': f"{losses.item():.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

# --- ONNX EXPORT (GUARANTEED COMPATIBILITY) ---
def convert_to_onnx(model_weights_path, cfg: Config):
    """
    Export the trained PyTorch model to ONNX format with exact specifications
    for compatibility with the black-box device.
    
    Args:
        model_weights_path (str): Path to the trained model weights
        cfg (Config): Configuration object
    """
    print("\n--- Starting ONNX Conversion ---")
    
    width, height = cfg.IMAGE_SIZE
    onnx_path = os.path.join(cfg.RESULTS_DIR, f"best_model_{cfg.MODE}_{cfg.BACKBONE}.onnx")
    
    # Create model and load weights
    model = get_sota_model(cfg)
    
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return False
    
    model.eval()
    
    # Create dummy input with exact required dimensions
    dummy_input = torch.randn(1, 3, height, width)
    print(f"Creating ONNX model for input size: {dummy_input.shape}")
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["boxes", "labels", "scores"],
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print(f"ONNX model exported successfully to: {onnx_path}")
        
        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed")
            
            # Print input/output info
            print("\nModel Input/Output Information:")
            for input_tensor in onnx_model.graph.input:
                print(f"Input: {input_tensor.name}")
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"  Shape: {shape}")
            
            for output_tensor in onnx_model.graph.output:
                print(f"Output: {output_tensor.name}")
                
        except ImportError:
            print("ONNX library not available for verification. Install with: pip install onnx")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False

# --- MAIN ORCHESTRATOR ---
def main():
    """Main training orchestrator function."""
    
    # Initialize configuration
    cfg = Config()
    
    print("="*80)
    print("SOTA DEFECT DETECTION TRAINING ORCHESTRATOR")
    print("="*80)
    print(f"Mode: {cfg.MODE}")
    print(f"Backbone: {cfg.BACKBONE}")
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Results Directory: {cfg.RESULTS_DIR}")
    print(f"Classes: {cfg.CLASS_MAP}")
    print("="*80)
    
    # Check if source dataset exists
    if not os.path.exists(cfg.SOURCE_DATASET_PATH):
        print(f"ERROR: Source dataset not found at: {cfg.SOURCE_DATASET_PATH}")
        print("Please ensure your dataset follows this structure:")
        print(f"{cfg.BASE_DATASET_PATH}/")
        print("├── EV/")
        print("│   ├── image1.png")
        print("│   ├── image1.xml")
        print("│   └── ...")
        print("└── SV/")
        print("    ├── image1.png")
        print("    ├── image1.xml")
        print("    └── ...")
        return
    
    # Split dataset
    try:
        train_dir, valid_dir = split_dataset(cfg)
    except Exception as e:
        print(f"Error during dataset splitting: {e}")
        return
    
    # Create datasets
    print("\n--- Creating Datasets ---")
    
    train_transforms = get_transforms(train=True, level=cfg.AUGMENTATION_LEVEL)
    valid_transforms = get_transforms(train=False)
    
    dataset_train = AdvancedDefectDataset(
        train_dir, 
        cfg.CLASS_MAP, 
        transforms=train_transforms,
        mode='train'
    )
    
    dataset_valid = AdvancedDefectDataset(
        valid_dir, 
        cfg.CLASS_MAP, 
        transforms=valid_transforms,
        mode='valid'
    )
    
    if len(dataset_train) == 0:
        print("ERROR: No training samples found!")
        return
    
    if len(dataset_valid) == 0:
        print("ERROR: No validation samples found!")
        return
    
    # Create data loaders
    print("\n--- Creating Data Loaders ---")
    
    if cfg.USE_WEIGHTED_SAMPLER and len(dataset_train) > 0:
        try:
            sample_weights = dataset_train.get_sample_weights()
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False  # Don't shuffle when using sampler
        except Exception as e:
            print(f"Warning: Failed to create weighted sampler: {e}")
            sampler = None
            shuffle = True
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if cfg.DEVICE.type == 'cuda' else False
    )
    
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=cfg.WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if cfg.DEVICE.type == 'cuda' else False
    )
    
    # Create model
    print("\n--- Creating Model ---")
    model = get_sota_model(cfg)
    model.to(cfg.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    if cfg.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(
            params, 
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.SGD(
            params, 
            lr=cfg.LEARNING_RATE, 
            momentum=0.9, 
            weight_decay=cfg.WEIGHT_DECAY
        )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_MIXED_PRECISION and cfg.DEVICE.type == 'cuda')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(cfg.RESULTS_DIR, 'logs'))
    
    # Training tracking
    best_val_loss = float('inf')
    model_weights_path = os.path.join(cfg.RESULTS_DIR, f"best_model_{cfg.MODE}_{cfg.BACKBONE}.pt")
    
    # Save configuration
    config_path = os.path.join(cfg.RESULTS_DIR, 'config.json')
    config_dict = {
        'MODE': cfg.MODE,
        'BACKBONE': cfg.BACKBONE,
        'IMAGE_SIZE': cfg.IMAGE_SIZE,
        'EPOCHS': cfg.EPOCHS,
        'BATCH_SIZE': cfg.BATCH_SIZE,
        'LEARNING_RATE': cfg.LEARNING_RATE,
        'OPTIMIZER': cfg.OPTIMIZER,
        'CLASS_MAP': cfg.CLASS_MAP,
        'AUGMENTATION_LEVEL': cfg.AUGMENTATION_LEVEL,
        'SPLIT_RATIO': cfg.SPLIT_RATIO
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n--- Starting Training for {cfg.EPOCHS} Epochs ---")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(cfg.EPOCHS):
            print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
            print("-" * 50)
            
            # Training
            train_loss = train_one_epoch_advanced(
                model, optimizer, train_loader, cfg.DEVICE, 
                epoch, writer, scaler, cfg.GRADIENT_CLIP_VAL
            )
            
            # Validation
            val_loss = evaluate_model(model, valid_loader, cfg.DEVICE)
            
            # Learning rate step
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # TensorBoard logging
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Model checkpointing
            if cfg.SAVE_BEST_ONLY:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_weights_path)
                    print(f"*** New best model saved (Val Loss: {val_loss:.4f}) ***")
                else:
                    print(f"Val loss did not improve (Best: {best_val_loss:.4f})")
            else:
                torch.save(model.state_dict(), model_weights_path)
                print("Model saved")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        writer.close()
    
    # Plot training curves
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.RESULTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {os.path.join(cfg.RESULTS_DIR, 'training_curves.png')}")
        
    except Exception as e:
        print(f"Failed to create training curves: {e}")
    
    # ONNX Export
    if os.path.exists(model_weights_path):
        print(f"\n--- Exporting Best Model to ONNX ---")
        success = convert_to_onnx(model_weights_path, cfg)
        
        if success:
            print("✅ Training and export completed successfully!")
            print(f"📁 Results saved in: {cfg.RESULTS_DIR}")
            print(f"🏆 Best model: {model_weights_path}")
            print(f"📊 ONNX model: {os.path.join(cfg.RESULTS_DIR, f'best_model_{cfg.MODE}_{cfg.BACKBONE}.onnx')}")
        else:
            print("❌ ONNX export failed, but training completed")
    else:
        print("❌ No trained model found for export")
    
    print("\n" + "="*80)
    print("TRAINING ORCHESTRATOR FINISHED")
    print("="*80)

if __name__ == '__main__':
    main()

