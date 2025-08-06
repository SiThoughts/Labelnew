import os
import sys
import shutil
import random
import yaml
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict

class SAHIDataProcessor:
    """SAHI data processor for existing YOLO datasets to optimize tiny defect detection."""
    
    def __init__(self, source_dataset_path: str, output_base_path: str = "sahi_datasets"):
        self.source_dataset_path = source_dataset_path
        self.output_base_path = output_base_path
        
        # SAHI slicing parameters optimized for tiny defects
        self.slice_configs = {
            'EV': {
                'slice_height': 640,
                'slice_width': 640,
                'overlap_height_ratio': 0.3,  # 30% overlap for better boundary handling
                'overlap_width_ratio': 0.3,
                'min_area_ratio': 0.1,  # Include objects with >10% visible area
                'target_size': (2048, 1460)  # Expected EV image size
            },
            'SV': {
                'slice_height': 512,
                'slice_width': 512,
                'overlap_height_ratio': 0.3,
                'overlap_width_ratio': 0.3,
                'min_area_ratio': 0.1,
                'target_size': (1024, 500)  # Expected SV image size
            }
        }
        
        print(f"SAHI Data Processor initialized")
        print(f"Source: {source_dataset_path}")
        print(f"Output: {output_base_path}")
    
    def parse_yolo_annotation(self, label_path: str, img_width: int, img_height: int) -> List[Dict]:
        """Parse YOLO format annotation file."""
        annotations = []
        
        if not os.path.exists(label_path):
            return annotations
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from YOLO format to absolute coordinates
                abs_center_x = center_x * img_width
                abs_center_y = center_y * img_height
                abs_width = width * img_width
                abs_height = height * img_height
                
                xmin = abs_center_x - abs_width / 2
                ymin = abs_center_y - abs_height / 2
                xmax = abs_center_x + abs_width / 2
                ymax = abs_center_y + abs_height / 2
                
                annotations.append({
                    'class_id': class_id,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'center': [abs_center_x, abs_center_y],
                    'size': [abs_width, abs_height]
                })
        
        except Exception as e:
            print(f"Error parsing {label_path}: {e}")
        
        return annotations
    
    def slice_image_with_annotations(self, image_path: str, annotations: List[Dict], 
                                   slice_config: Dict, model_type: str) -> List[Dict]:
        """Slice image and annotations using SAHI approach."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return []
        
        img_height, img_width = image.shape[:2]
        
        slice_height = slice_config['slice_height']
        slice_width = slice_config['slice_width']
        overlap_height_ratio = slice_config['overlap_height_ratio']
        overlap_width_ratio = slice_config['overlap_width_ratio']
        min_area_ratio = slice_config['min_area_ratio']
        
        # Calculate step sizes
        step_height = int(slice_height * (1 - overlap_height_ratio))
        step_width = int(slice_width * (1 - overlap_width_ratio))
        
        slices = []
        slice_id = 0
        
        # Generate slices
        y_positions = list(range(0, img_height, step_height))
        x_positions = list(range(0, img_width, step_width))
        
        # Ensure we cover the entire image
        if y_positions[-1] + slice_height < img_height:
            y_positions.append(img_height - slice_height)
        if x_positions[-1] + slice_width < img_width:
            x_positions.append(img_width - slice_width)
        
        for y in y_positions:
            for x in x_positions:
                # Ensure slice doesn't go beyond image boundaries
                y = max(0, min(y, img_height - slice_height))
                x = max(0, min(x, img_width - slice_width))
                
                x_end = x + slice_width
                y_end = y + slice_height
                
                # Extract slice
                slice_img = image[y:y_end, x:x_end]
                
                # Find annotations that overlap with this slice
                slice_annotations = []
                for ann in annotations:
                    bbox = ann['bbox']
                    xmin, ymin, xmax, ymax = bbox
                    
                    # Check if annotation overlaps with slice
                    if (xmax > x and xmin < x_end and ymax > y and ymin < y_end):
                        # Calculate intersection
                        intersect_xmin = max(xmin, x)
                        intersect_ymin = max(ymin, y)
                        intersect_xmax = min(xmax, x_end)
                        intersect_ymax = min(ymax, y_end)
                        
                        # Calculate area ratios
                        original_area = (xmax - xmin) * (ymax - ymin)
                        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
                        
                        if original_area > 0:
                            area_ratio = intersect_area / original_area
                        else:
                            area_ratio = 0
                        
                        # Only include if significant overlap
                        if area_ratio >= min_area_ratio:
                            # Convert to slice coordinates
                            slice_xmin = intersect_xmin - x
                            slice_ymin = intersect_ymin - y
                            slice_xmax = intersect_xmax - x
                            slice_ymax = intersect_ymax - y
                            
                            # Ensure coordinates are within slice bounds
                            slice_xmin = max(0, slice_xmin)
                            slice_ymin = max(0, slice_ymin)
                            slice_xmax = min(slice_width, slice_xmax)
                            slice_ymax = min(slice_height, slice_ymax)
                            
                            # Only add if the resulting box is valid
                            if slice_xmax > slice_xmin and slice_ymax > slice_ymin:
                                slice_annotations.append({
                                    'class_id': ann['class_id'],
                                    'bbox': [slice_xmin, slice_ymin, slice_xmax, slice_ymax],
                                    'area_ratio': area_ratio
                                })
                
                # Create slice info
                slice_info = {
                    'slice_id': slice_id,
                    'image': slice_img,
                    'annotations': slice_annotations,
                    'slice_bbox': [x, y, x_end, y_end],
                    'original_size': (img_width, img_height),
                    'slice_size': (slice_width, slice_height),
                    'has_objects': len(slice_annotations) > 0
                }
                
                slices.append(slice_info)
                slice_id += 1
        
        return slices
    
    def convert_to_yolo_format(self, annotations: List[Dict], slice_width: int, slice_height: int) -> List[str]:
        """Convert annotations to YOLO format."""
        yolo_annotations = []
        
        for ann in annotations:
            bbox = ann['bbox']
            xmin, ymin, xmax, ymax = bbox
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = (xmin + xmax) / 2.0 / slice_width
            center_y = (ymin + ymax) / 2.0 / slice_height
            width = (xmax - xmin) / slice_width
            height = (ymax - ymin) / slice_height
            
            # Ensure values are within [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Only add if the box is valid
            if width > 0 and height > 0:
                yolo_annotations.append(f"{ann['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def process_dataset_split(self, dataset_path: str, model_type: str, split_name: str) -> Dict:
        """Process a dataset split (train/val/test) with SAHI slicing."""
        images_dir = os.path.join(dataset_path, 'images', split_name)
        labels_dir = os.path.join(dataset_path, 'labels', split_name)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: {split_name} split not found for {model_type}")
            return {'slices': [], 'stats': {}}
        
        slice_config = self.slice_configs[model_type]
        all_slices = []
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        print(f"Processing {len(image_files)} images in {model_type}/{split_name}...")
        
        processed_count = 0
        total_slices = 0
        total_objects = 0
        slices_with_objects = 0
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Parse annotations
            annotations = self.parse_yolo_annotation(label_path, img_width, img_height)
            
            # Slice image and annotations
            slices = self.slice_image_with_annotations(image_path, annotations, slice_config, model_type)
            
            # Add metadata to slices
            for slice_info in slices:
                base_name = os.path.splitext(image_file)[0]
                slice_filename = f"{base_name}_slice_{slice_info['slice_id']:04d}"
                
                slice_info.update({
                    'original_image_path': image_path,
                    'slice_filename': slice_filename,
                    'split_name': split_name
                })
                
                all_slices.append(slice_info)
                total_slices += 1
                total_objects += len(slice_info['annotations'])
                
                if slice_info['has_objects']:
                    slices_with_objects += 1
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(image_files)} images, "
                      f"generated {total_slices} slices")
        
        stats = {
            'original_images': len(image_files),
            'processed_images': processed_count,
            'total_slices': total_slices,
            'slices_with_objects': slices_with_objects,
            'total_objects': total_objects,
            'avg_objects_per_slice': total_objects / max(1, total_slices)
        }
        
        print(f"  Completed {model_type}/{split_name}: {stats}")
        
        return {'slices': all_slices, 'stats': stats}
    
    def save_sliced_dataset(self, slices: List[Dict], output_dataset_path: str, split_name: str) -> int:
        """Save sliced dataset to disk."""
        images_dir = os.path.join(output_dataset_path, 'images', split_name)
        labels_dir = os.path.join(output_dataset_path, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        saved_count = 0
        
        for slice_info in slices:
            slice_filename = slice_info['slice_filename']
            
            # Save image
            image_path = os.path.join(images_dir, f"{slice_filename}.jpg")
            success = cv2.imwrite(image_path, slice_info['image'])
            
            if not success:
                print(f"Failed to save image: {image_path}")
                continue
            
            # Save labels
            label_path = os.path.join(labels_dir, f"{slice_filename}.txt")
            
            if slice_info['annotations']:
                slice_height, slice_width = slice_info['slice_size']
                yolo_annotations = self.convert_to_yolo_format(
                    slice_info['annotations'], slice_width, slice_height
                )
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            else:
                # Create empty label file for slices without annotations
                with open(label_path, 'w') as f:
                    pass
            
            saved_count += 1
        
        return saved_count
    
    def create_dataset_yaml(self, output_dataset_path: str, model_type: str):
        """Create dataset.yaml file for YOLO training."""
        yaml_content = {
            'path': os.path.abspath(output_dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,
            'names': ['chip', 'check'],
            
            # Add metadata for tiny defect detection
            'model_type': model_type,
            'slice_config': self.slice_configs[model_type],
            'optimized_for': 'tiny_defect_detection',
            'sahi_processed': True
        }
        
        yaml_path = os.path.join(output_dataset_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created dataset.yaml at {yaml_path}")
        return yaml_path
    
    def process_model_type(self, model_type: str):
        """Process a specific model type (EV or SV) with SAHI slicing."""
        source_dataset = os.path.join(self.source_dataset_path, f'{model_type}_dataset')
        
        if not os.path.exists(source_dataset):
            print(f"Source dataset not found: {source_dataset}")
            return False
        
        print(f"\nProcessing {model_type} dataset with SAHI slicing...")
        print("=" * 60)
        
        # Create output directory
        output_dataset = os.path.join(self.output_base_path, f'{model_type}_sahi_dataset')
        os.makedirs(output_dataset, exist_ok=True)
        
        # Process each split
        all_stats = {}
        
        for split_name in ['train', 'val', 'test']:
            print(f"\nProcessing {split_name} split...")
            
            result = self.process_dataset_split(source_dataset, model_type, split_name)
            slices = result['slices']
            stats = result['stats']
            
            if slices:
                saved_count = self.save_sliced_dataset(slices, output_dataset, split_name)
                stats['saved_slices'] = saved_count
                print(f"  Saved {saved_count} slices to {split_name}")
            
            all_stats[split_name] = stats
        
        # Create dataset.yaml
        self.create_dataset_yaml(output_dataset, model_type)
        
        # Save processing statistics
        stats_path = os.path.join(output_dataset, 'processing_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"\n{model_type} SAHI dataset created successfully!")
        print(f"Output: {output_dataset}")
        print(f"Statistics saved: {stats_path}")
        
        # Print summary
        print(f"\nSUMMARY for {model_type}:")
        for split_name, stats in all_stats.items():
            if stats:
                print(f"  {split_name}: {stats['original_images']} images → "
                      f"{stats['total_slices']} slices ({stats['slices_with_objects']} with objects)")
        
        return True
    
    def process_all(self):
        """Process both EV and SV datasets."""
        print("SAHI Data Processing for Tiny Defect Detection")
        print("=" * 60)
        print(f"Source: {self.source_dataset_path}")
        print(f"Output: {self.output_base_path}")
        
        # Create output directory
        os.makedirs(self.output_base_path, exist_ok=True)
        
        success_count = 0
        
        for model_type in ['EV', 'SV']:
            if self.process_model_type(model_type):
                success_count += 1
        
        print(f"\nProcessing completed! {success_count}/2 datasets processed successfully.")
        print(f"SAHI datasets saved to: {os.path.abspath(self.output_base_path)}")
        
        return success_count == 2

def main():
    """Main function to run SAHI data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAHI Data Processing for Tiny Defect Detection')
    parser.add_argument('--source', type=str, default='defect_detection_datasets_fixed',
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, default='sahi_datasets',
                       help='Output directory for SAHI datasets')
    parser.add_argument('--model_type', type=str, choices=['EV', 'SV', 'BOTH'], default='BOTH',
                       help='Model type to process')
    
    args = parser.parse_args()
    
    print("SAHI Data Processor for Tiny Defect Detection")
    print("=" * 50)
    print("This script applies SAHI slicing to existing YOLO datasets")
    print("to optimize them for tiny defect detection.")
    print()
    
    # Initialize processor
    processor = SAHIDataProcessor(args.source, args.output)
    
    # Process datasets
    if args.model_type == 'BOTH':
        processor.process_all()
    else:
        processor.process_model_type(args.model_type)

if __name__ == "__main__":
    main()

