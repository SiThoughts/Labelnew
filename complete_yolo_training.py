#!/usr/bin/env python3
"""
Complete YOLO Training Script
- Tiles images into smaller patches
- Combines train/val datasets (4:1 split)
- Applies rotation augmentation
- Creates YAML config
- Trains YOLOv11n model
"""

import os
import cv2
import numpy as np
import yaml
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import subprocess
import sys

def get_user_input():
    """Get dataset path from user"""
    print("=" * 60)
    print("COMPLETE YOLO TRAINING SCRIPT")
    print("=" * 60)
    
    dataset_path = input("\nEnter the complete path to your dataset folder: ").strip()
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Path '{dataset_path}' does not exist!")
        sys.exit(1)
    
    # Check if it has the expected structure
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"ERROR: '{dataset_path}' must contain 'images' and 'labels' folders!")
        sys.exit(1)
    
    return dataset_path

def tile_image_and_labels(image_path, label_path, tile_size=640, overlap=0.1):
    """
    Tile an image and adjust corresponding YOLO labels
    Returns list of (tiled_image, tiled_labels) tuples
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    h, w = image.shape[:2]
    
    # Read labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
    
    # Calculate step size with overlap
    step = int(tile_size * (1 - overlap))
    
    tiles = []
    
    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            # Extract tile
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Convert absolute coordinates for this tile
            tile_labels = []
            for label in labels:
                class_id, x_center, y_center, bbox_w, bbox_h = label
                
                # Convert normalized coordinates to absolute
                abs_x_center = x_center * w
                abs_y_center = y_center * h
                abs_bbox_w = bbox_w * w
                abs_bbox_h = bbox_h * h
                
                # Check if bbox center is within this tile
                if (x <= abs_x_center <= x + tile_size and 
                    y <= abs_y_center <= y + tile_size):
                    
                    # Convert to tile-relative coordinates
                    new_x_center = (abs_x_center - x) / tile_size
                    new_y_center = (abs_y_center - y) / tile_size
                    new_bbox_w = abs_bbox_w / tile_size
                    new_bbox_h = abs_bbox_h / tile_size
                    
                    # Clamp to [0, 1] range
                    new_x_center = max(0, min(1, new_x_center))
                    new_y_center = max(0, min(1, new_y_center))
                    new_bbox_w = max(0, min(1, new_bbox_w))
                    new_bbox_h = max(0, min(1, new_bbox_h))
                    
                    tile_labels.append([class_id, new_x_center, new_y_center, new_bbox_w, new_bbox_h])
            
            tiles.append((tile, tile_labels))
    
    return tiles

def rotate_image_and_labels(image, labels, angle):
    """
    Rotate image and adjust YOLO labels accordingly
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Rotate labels
    rotated_labels = []
    for label in labels:
        class_id, x_center, y_center, bbox_w, bbox_h = label
        
        # Convert to absolute coordinates
        abs_x = x_center * w
        abs_y = y_center * h
        
        # Apply rotation
        rotated_point = np.dot(M, np.array([abs_x, abs_y, 1]))
        new_abs_x, new_abs_y = rotated_point
        
        # Convert back to normalized
        new_x_center = new_abs_x / w
        new_y_center = new_abs_y / h
        
        # Clamp to [0, 1] range
        new_x_center = max(0, min(1, new_x_center))
        new_y_center = max(0, min(1, new_y_center))
        
        rotated_labels.append([class_id, new_x_center, new_y_center, bbox_w, bbox_h])
    
    return rotated, rotated_labels

def process_dataset(dataset_path):
    """
    Main processing function
    """
    print("\n" + "=" * 60)
    print("STEP 1: PROCESSING DATASET")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(dataset_path), 'processed_dataset')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))
    os.makedirs(os.path.join(output_dir, 'labels'))
    
    # Collect all images from train and val
    all_images = []
    
    for subset in ['train', 'val']:
        img_dir = os.path.join(dataset_path, 'images', subset)
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(img_dir, img_file)
                    label_path = os.path.join(dataset_path, 'labels', subset, 
                                            os.path.splitext(img_file)[0] + '.txt')
                    all_images.append((img_path, label_path, img_file))
    
    print(f"Found {len(all_images)} images to process")
    
    # Process each image
    processed_count = 0
    rotation_angles = [0, 90, 180, 270]  # Only rotations as requested
    
    for img_path, label_path, img_file in all_images:
        print(f"Processing: {img_file}")
        
        # Tile the image
        tiles = tile_image_and_labels(img_path, label_path)
        
        for tile_idx, (tile_img, tile_labels) in enumerate(tiles):
            # Apply rotations
            for angle in rotation_angles:
                if angle == 0:
                    final_img = tile_img
                    final_labels = tile_labels
                else:
                    final_img, final_labels = rotate_image_and_labels(tile_img, tile_labels, angle)
                
                # Save processed image and labels
                base_name = os.path.splitext(img_file)[0]
                output_name = f"{base_name}_tile{tile_idx}_rot{angle}"
                
                # Save image
                img_output_path = os.path.join(output_dir, 'images', f"{output_name}.jpg")
                cv2.imwrite(img_output_path, final_img)
                
                # Save labels
                label_output_path = os.path.join(output_dir, 'labels', f"{output_name}.txt")
                with open(label_output_path, 'w') as f:
                    for label in final_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                
                processed_count += 1
    
    print(f"Created {processed_count} processed images")
    return output_dir

def create_train_val_split(processed_dir):
    """
    Split processed data into train (80%) and val (20%)
    """
    print("\n" + "=" * 60)
    print("STEP 2: CREATING TRAIN/VAL SPLIT")
    print("=" * 60)
    
    # Get all processed images
    images_dir = os.path.join(processed_dir, 'images')
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Split 80/20
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    
    # Create train/val directories
    for subset, files in [('train', train_files), ('val', val_files)]:
        subset_img_dir = os.path.join(processed_dir, 'images', subset)
        subset_label_dir = os.path.join(processed_dir, 'labels', subset)
        
        os.makedirs(subset_img_dir, exist_ok=True)
        os.makedirs(subset_label_dir, exist_ok=True)
        
        for file in files:
            # Move image
            src_img = os.path.join(images_dir, file)
            dst_img = os.path.join(subset_img_dir, file)
            shutil.move(src_img, dst_img)
            
            # Move label
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(processed_dir, 'labels', label_file)
            dst_label = os.path.join(subset_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
    
    return processed_dir

def create_yaml_config(processed_dir, original_dataset_path):
    """
    Create YAML configuration file
    """
    print("\n" + "=" * 60)
    print("STEP 3: CREATING YAML CONFIG")
    print("=" * 60)
    
    # Try to read original data.yaml for class names
    original_yaml = os.path.join(original_dataset_path, 'data.yaml')
    class_names = ['defect']  # default
    nc = 1
    
    if os.path.exists(original_yaml):
        try:
            with open(original_yaml, 'r') as f:
                original_data = yaml.safe_load(f)
                if 'names' in original_data:
                    class_names = original_data['names']
                    nc = len(class_names)
        except:
            print("Warning: Could not read original data.yaml, using default class names")
    
    # Create new YAML
    yaml_content = {
        'path': os.path.abspath(processed_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': nc,
        'names': class_names
    }
    
    yaml_path = os.path.join(processed_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Created YAML config: {yaml_path}")
    print(f"Classes: {class_names}")
    
    return yaml_path

def train_model(yaml_path):
    """
    Train YOLOv11n model
    """
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MODEL")
    print("=" * 60)
    
    # Training command
    cmd = [
        'yolo', 'train',
        f'model=yolov11n.pt',
        f'data={yaml_path}',
        'epochs=100',
        'imgsz=640',
        'batch=16',
        'project=runs/train',
        'name=complete_training'
    ]
    
    print("Starting training with command:")
    print(" ".join(cmd))
    print("\nThis will take a while...")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Results saved in: runs/train/complete_training")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

def main():
    """
    Main function
    """
    # Get dataset path from user
    dataset_path = get_user_input()
    
    # Process dataset (tiling + augmentation)
    processed_dir = process_dataset(dataset_path)
    
    # Create train/val split
    create_train_val_split(processed_dir)
    
    # Create YAML config
    yaml_path = create_yaml_config(processed_dir, dataset_path)
    
    # Train model
    train_model(yaml_path)
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED!")
    print("=" * 60)
    print(f"Processed dataset: {processed_dir}")
    print("Training results: runs/train/complete_training")

if __name__ == "__main__":
    main()

