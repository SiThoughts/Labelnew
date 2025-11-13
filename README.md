"""
PyTorch dataset for defect detection with support for normal (defect-free) images.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from src.xml_parser import parse_xml


class DefectDetectionDataset(Dataset):
    """
    Dataset for defect detection that handles both defective and normal images.
    """
    
    def __init__(self, data_dir, class_map, image_size=(640, 640), 
                 train=True, augment_config=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images and XML annotations
            class_map: Dictionary mapping class IDs to class names
            image_size: Target image size (width, height)
            train: Whether this is training data
            augment_config: Augmentation configuration
        """
        self.data_dir = data_dir
        self.class_map = class_map
        self.image_size = image_size
        self.train = train
        
        # Build reverse class map (name -> id)
        self.class_name_to_id = {v.lower(): k for k, v in class_map.items()}
        
        # Find all images
        self.image_paths = []
        self.xml_paths = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    xml_path = os.path.splitext(img_path)[0] + '.xml'
                    
                    if os.path.exists(xml_path):
                        self.image_paths.append(img_path)
                        self.xml_paths.append(xml_path)
        
        # Setup augmentations
        self.transform = self._build_transforms(augment_config if train else None)
    
    def _build_transforms(self, augment_config):
        """Build augmentation pipeline."""
        transforms = []
        
        if augment_config:
            # Add augmentations
            h_flip = augment_config.get('horizontal_flip', 0.0)
            v_flip = augment_config.get('vertical_flip', 0.0)
            rotation = augment_config.get('rotation', 0)
            brightness = augment_config.get('brightness', 0.0)
            contrast = augment_config.get('contrast', 0.0)
            
            if h_flip > 0:
                transforms.append(A.HorizontalFlip(p=h_flip))
            if v_flip > 0:
                transforms.append(A.VerticalFlip(p=v_flip))
            if rotation > 0:
                transforms.append(A.Rotate(limit=rotation, p=0.5))
            if brightness > 0 or contrast > 0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.5
                ))
        
        # Always resize and normalize
        transforms.extend([
            A.Resize(self.image_size[1], self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            )
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single item."""
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Parse XML annotation
            xml_path = self.xml_paths[idx]
            annotation = parse_xml(xml_path)
            
            # Extract boxes and labels
            boxes = []
            labels = []
            
            for obj in annotation['objects']:
                # Get class name and map to ID
                class_name = obj['name'].lower().strip()
                
                # Handle variations (chip/chips, check/checks)
                if class_name.startswith('chip'):
                    class_name = 'chip'
                elif class_name.startswith('check'):
                    class_name = 'check'
                
                # Get class ID
                class_id = self.class_name_to_id.get(class_name)
                if class_id is None:
                    print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                    continue
                
                # Get bounding box
                bbox = obj['bndbox']
                boxes.append([
                    bbox['xmin'],
                    bbox['ymin'],
                    bbox['xmax'],
                    bbox['ymax']
                ])
                labels.append(class_id)
            
            # Handle normal images (no defects)
            has_defects = len(boxes) > 0
            
            if not has_defects:
                # Create a dummy box for normal images
                # This is required by Faster R-CNN
                boxes = [[0, 0, 1, 1]]
                labels = [0]  # Background class
            
            # Apply augmentations
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
            
            # Handle case where all boxes were removed by augmentation
            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]
                labels = [0]
                has_defects = False
            
            # Create target dictionary
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'has_defects': torch.tensor([1.0 if has_defects else 0.0], dtype=torch.float32)
            }
            
            # Add image_id for evaluation
            target['image_id'] = torch.tensor([idx])
            
            return image, target
            
        except Exception as e:
            print(f"Error loading image {idx} ({self.image_paths[idx]}): {e}")
            # Return next image
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable number of objects per image.
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets
