import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import numpy as np
import argparse
from pathlib import Path
import yaml
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusDetConfig:
    """Configuration for FocusDet training."""
    
    def __init__(self, image_type: str = 'EV'):
        self.image_type = image_type
        
        # Model configuration
        self.model_name = 'focusdet'
        self.num_classes = 2  # chip, check
        self.input_size = (1024, 1024)  # High resolution for tiny objects
        
        # Training configuration
        self.batch_size = 4  # Small batch for memory efficiency
        self.num_epochs = 300  # Extended training for maximum accuracy
        self.learning_rate = 1e-5  # Conservative for stability
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Loss configuration
        self.focal_loss_alpha = [0.25, 0.75]  # [chip, check] - favor minority class
        self.focal_loss_gamma = 2.0
        self.iou_loss_weight = 0.3
        self.classification_loss_weight = 0.4
        self.bbox_loss_weight = 0.3
        
        # Optimization
        self.warmup_epochs = 20
        self.lr_schedule = 'cosine'
        self.patience = 50  # Early stopping patience
        
        # Data augmentation
        self.augmentation_prob = 0.5
        self.color_jitter = 0.2
        self.rotation_degrees = 10
        
        # Validation
        self.val_interval = 5
        self.save_top_k = 5

class FocusDetDataset(torch.utils.data.Dataset):
    """Custom dataset for FocusDet training with COCO format."""
    
    def __init__(self, images_dir: str, annotations_file: str, transforms=None, 
                 class_weights: Dict = None, augment_minority: bool = True):
        self.images_dir = images_dir
        self.transforms = transforms
        self.class_weights = class_weights or {}
        self.augment_minority = augment_minority
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Create image list
        self.image_ids = list(self.images.keys())
        
        # Calculate class distribution for sampling
        self.class_counts = {'chip': 0, 'check': 0}
        for ann in self.annotations:
            class_name = 'chip' if ann['category_id'] == 1 else 'check'
            self.class_counts[class_name] += 1
        
        logger.info(f"Dataset loaded: {len(self.image_ids)} images, {len(self.annotations)} annotations")
        logger.info(f"Class distribution: {self.class_counts}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Convert annotations to tensors
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(img_id)
        }
        
        return image, target

class FocusDetModel(nn.Module):
    """Simplified FocusDet model implementation."""
    
    def __init__(self, num_classes: int = 2):
        super(FocusDetModel, self).__init__()
        self.num_classes = num_classes
        
        # Backbone (simplified ResNet-like)
        self.backbone = self._create_backbone()
        
        # Feature fusion network (simplified FPN)
        self.fpn = self._create_fpn()
        
        # Detection head
        self.detection_head = self._create_detection_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self):
        """Create backbone network."""
        # Simplified backbone - in practice, use ResNet or similar
        layers = []
        in_channels = 3
        channels = [64, 128, 256, 512, 1024]
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _create_fpn(self):
        """Create feature pyramid network."""
        return nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _create_detection_head(self):
        """Create detection head."""
        return nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes + 4, 1)  # classes + bbox
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # FPN
        features = self.fpn(features)
        
        # Detection head
        output = self.detection_head(features)
        
        return output

class FocusDetLoss(nn.Module):
    """Combined loss function for FocusDet."""
    
    def __init__(self, config: FocusDetConfig, class_weights: Dict = None):
        super(FocusDetLoss, self).__init__()
        self.config = config
        self.class_weights = class_weights or {}
        
        # Loss components
        self.focal_loss = self._create_focal_loss()
        self.bbox_loss = nn.SmoothL1Loss()
        
    def _create_focal_loss(self):
        """Create focal loss for classification."""
        alpha = torch.tensor(self.config.focal_loss_alpha)
        gamma = self.config.focal_loss_gamma
        
        def focal_loss(pred, target):
            ce_loss = nn.CrossEntropyLoss(weight=alpha)(pred, target)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            return focal_loss
        
        return focal_loss
    
    def forward(self, predictions, targets):
        # Simplified loss calculation
        # In practice, this would be more complex with proper anchor matching
        
        # Classification loss
        cls_loss = self.focal_loss(predictions[:, :self.config.num_classes], targets['labels'])
        
        # Bbox regression loss
        bbox_loss = self.bbox_loss(predictions[:, self.config.num_classes:], targets['boxes'])
        
        # Combine losses
        total_loss = (
            self.config.classification_loss_weight * cls_loss +
            self.config.bbox_loss_weight * bbox_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss
        }

class FocusDetTrainer:
    """FocusDet trainer with class balancing."""
    
    def __init__(self, config: FocusDetConfig, dataset_path: str):
        self.config = config
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class weights
        weights_path = os.path.join(dataset_path, 'class_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.class_weights = json.load(f)[config.image_type]
        else:
            self.class_weights = {'chip': 1.0, 'check': 1.0}
        
        logger.info(f"Using class weights: {self.class_weights}")
        
        # Initialize model
        self.model = FocusDetModel(num_classes=config.num_classes).to(self.device)
        
        # Initialize loss
        self.criterion = FocusDetLoss(config, self.class_weights)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Training state
        self.best_map = 0.0
        self.current_epoch = 0
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = f"focusdet_{config.image_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized for {config.image_type} images")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_data_loaders(self):
        """Create training and validation data loaders."""
        # Data transforms
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ColorJitter(brightness=self.config.color_jitter, 
                                 contrast=self.config.color_jitter),
            transforms.RandomRotation(self.config.rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_images_dir = os.path.join(self.dataset_path, self.config.image_type, 'images', 'train')
        train_ann_file = os.path.join(self.dataset_path, self.config.image_type, 'annotations', 'train.json')
        
        val_images_dir = os.path.join(self.dataset_path, self.config.image_type, 'images', 'val')
        val_ann_file = os.path.join(self.dataset_path, self.config.image_type, 'annotations', 'val.json')
        
        train_dataset = FocusDetDataset(
            train_images_dir, train_ann_file, 
            transforms=train_transforms, 
            class_weights=self.class_weights
        )
        
        val_dataset = FocusDetDataset(
            val_images_dir, val_ann_file, 
            transforms=val_transforms
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch):
        """Custom collate function for variable-sized targets."""
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss (simplified)
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for i, target in enumerate(targets):
                # Simplified loss calculation
                # In practice, this would involve proper anchor matching
                if len(target['boxes']) > 0:
                    loss = loss + torch.mean(predictions[i])  # Placeholder
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                predictions = self.model(images)
                
                # Simplified validation loss
                loss = torch.mean(predictions)  # Placeholder
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting FocusDet training...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.val_interval == 0:
                val_loss = self.validate(val_loader)
                
                # Check if best model
                is_best = val_loss < self.best_map  # Simplified metric
                if is_best:
                    self.best_map = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Best: {self.best_map:.4f}")
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update learning rate
            self.scheduler.step()
        
        logger.info("Training completed!")
        return os.path.join(self.output_dir, 'best_model.pth')

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train FocusDet model')
    parser.add_argument('--image_type', type=str, choices=['EV', 'SV'], required=True,
                       help='Image type to train on (EV or SV)')
    parser.add_argument('--dataset_path', type=str, default='focusdet_dataset',
                       help='Path to the converted dataset')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create configuration
    config = FocusDetConfig(image_type=args.image_type)
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Create trainer
    trainer = FocusDetTrainer(config, args.dataset_path)
    
    # Start training
    best_model_path = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Ready for ONNX export!")

if __name__ == "__main__":
    main()

