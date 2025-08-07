import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from datetime import datetime

class XMLToCOCOConverter:
    """Convert XML annotations to COCO format for FocusDet training."""
    
    def __init__(self, source_base_path: str = r"D:\Photomask", output_path: str = "focusdet_dataset"):
        self.source_base_path = source_base_path
        self.output_path = output_path
        
        # Data source configuration
        self.training_sources = ["DS0", "DS2_Sort2"]
        self.validation_source = "MSA_Sort3"
        
        # Class mapping
        self.class_mapping = {'chip': 1, 'check': 2}  # COCO format starts from 1
        self.class_names = ['chip', 'check']
        
        # Image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics
        self.stats = {
            'EV': {'train': {'images': 0, 'annotations': 0, 'chip': 0, 'check': 0},
                   'val': {'images': 0, 'annotations': 0, 'chip': 0, 'check': 0}},
            'SV': {'train': {'images': 0, 'annotations': 0, 'chip': 0, 'check': 0},
                   'val': {'images': 0, 'annotations': 0, 'chip': 0, 'check': 0}}
        }
        
    def find_matching_image(self, xml_path: str) -> str:
        """Find the image file that matches an XML annotation."""
        xml_dir = os.path.dirname(xml_path)
        xml_basename = os.path.splitext(os.path.basename(xml_path))[0]
        
        # Try different extensions
        for ext in self.image_extensions:
            candidate = os.path.join(xml_dir, xml_basename + ext)
            if os.path.exists(candidate):
                return candidate
                
        # Try with variations (dots/underscores)
        variations = [
            xml_basename.replace('.', '_'),
            xml_basename.replace('_', '.'),
        ]
        
        for base in variations:
            for ext in self.image_extensions:
                candidate = os.path.join(xml_dir, base + ext)
                if os.path.exists(candidate):
                    return candidate
        
        return None
    
    def parse_xml_annotation(self, xml_path: str) -> Tuple[List[Dict], int, int]:
        """Parse XML annotation file and return COCO format annotations."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size_elem = root.find('size')
            if size_elem is not None:
                width = int(size_elem.find('width').text)
                height = int(size_elem.find('height').text)
            else:
                # Fallback: read image to get dimensions
                img_path = self.find_matching_image(xml_path)
                if img_path:
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                else:
                    print(f"Warning: Could not determine dimensions for {xml_path}")
                    return [], 0, 0
            
            annotations = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                    
                class_name = name_elem.text.lower()
                if class_name not in self.class_mapping:
                    continue
                
                bbox_elem = obj.find('bndbox')
                if bbox_elem is None:
                    continue
                
                # Parse bounding box
                xmin = float(bbox_elem.find('xmin').text)
                ymin = float(bbox_elem.find('ymin').text)
                xmax = float(bbox_elem.find('xmax').text)
                ymax = float(bbox_elem.find('ymax').text)
                
                # Convert to COCO format (x, y, width, height)
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height
                
                annotation = {
                    'bbox': [xmin, ymin, bbox_width, bbox_height],
                    'area': area,
                    'category_id': self.class_mapping[class_name],
                    'iscrowd': 0
                }
                
                annotations.append(annotation)
            
            return annotations, width, height
            
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return [], 0, 0
    
    def create_coco_dataset(self, image_type: str, split: str) -> Dict:
        """Create COCO format dataset structure."""
        dataset = {
            'info': {
                'description': f'FocusDet Dataset - {image_type} {split}',
                'version': '1.0',
                'year': 2025,
                'contributor': 'Defect Detection System',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Custom License',
                    'url': ''
                }
            ],
            'categories': [
                {'id': 1, 'name': 'chip', 'supercategory': 'defect'},
                {'id': 2, 'name': 'check', 'supercategory': 'defect'}
            ],
            'images': [],
            'annotations': []
        }
        
        return dataset
    
    def process_folder(self, folder_path: str, image_type: str, split: str, 
                      output_images_dir: str, dataset: Dict) -> None:
        """Process a folder and add images/annotations to dataset."""
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return
        
        image_id = len(dataset['images']) + 1
        annotation_id = len(dataset['annotations']) + 1
        
        # Get all XML files
        xml_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(folder_path, xml_file)
            img_path = self.find_matching_image(xml_path)
            
            if not img_path:
                print(f"Warning: No matching image for {xml_path}")
                continue
            
            # Parse annotations
            annotations, width, height = self.parse_xml_annotation(xml_path)
            if not annotations:
                continue
            
            # Copy image to output directory
            img_filename = f"{image_id:06d}_{os.path.basename(img_path)}"
            output_img_path = os.path.join(output_images_dir, img_filename)
            shutil.copy2(img_path, output_img_path)
            
            # Add image info
            image_info = {
                'id': image_id,
                'file_name': img_filename,
                'width': width,
                'height': height,
                'license': 1
            }
            dataset['images'].append(image_info)
            
            # Add annotations
            for ann in annotations:
                ann['id'] = annotation_id
                ann['image_id'] = image_id
                dataset['annotations'].append(ann)
                
                # Update statistics
                class_name = self.class_names[ann['category_id'] - 1]
                self.stats[image_type][split][class_name] += 1
                self.stats[image_type][split]['annotations'] += 1
                annotation_id += 1
            
            self.stats[image_type][split]['images'] += 1
            image_id += 1
    
    def convert_dataset(self):
        """Convert the entire dataset from XML to COCO format."""
        print("Starting XML to COCO conversion for FocusDet...")
        
        # Create output directory structure
        for image_type in ['EV', 'SV']:
            for split in ['train', 'val']:
                images_dir = os.path.join(self.output_path, image_type, 'images', split)
                os.makedirs(images_dir, exist_ok=True)
            
            annotations_dir = os.path.join(self.output_path, image_type, 'annotations')
            os.makedirs(annotations_dir, exist_ok=True)
        
        # Process each image type (EV/SV)
        for image_type in ['EV', 'SV']:
            print(f"\nProcessing {image_type} images...")
            
            # Create datasets for train and val
            train_dataset = self.create_coco_dataset(image_type, 'train')
            val_dataset = self.create_coco_dataset(image_type, 'val')
            
            # Process training data (DS0 + DS2_Sort2)
            for source in self.training_sources:
                folder_path = os.path.join(self.source_base_path, source, image_type)
                output_images_dir = os.path.join(self.output_path, image_type, 'images', 'train')
                print(f"  Processing training data from {folder_path}")
                self.process_folder(folder_path, image_type, 'train', output_images_dir, train_dataset)
            
            # Process validation data (MSA_Sort3)
            folder_path = os.path.join(self.source_base_path, self.validation_source, image_type)
            output_images_dir = os.path.join(self.output_path, image_type, 'images', 'val')
            print(f"  Processing validation data from {folder_path}")
            self.process_folder(folder_path, image_type, 'val', output_images_dir, val_dataset)
            
            # Save COCO annotation files
            train_ann_path = os.path.join(self.output_path, image_type, 'annotations', 'train.json')
            val_ann_path = os.path.join(self.output_path, image_type, 'annotations', 'val.json')
            
            with open(train_ann_path, 'w') as f:
                json.dump(train_dataset, f, indent=2)
            
            with open(val_ann_path, 'w') as f:
                json.dump(val_dataset, f, indent=2)
            
            print(f"  Saved annotations to {train_ann_path} and {val_ann_path}")
    
    def calculate_class_weights(self) -> Dict:
        """Calculate class weights for handling imbalance."""
        weights = {}
        
        for image_type in ['EV', 'SV']:
            total_chips = self.stats[image_type]['train']['chip']
            total_checks = self.stats[image_type]['train']['check']
            total_annotations = total_chips + total_checks
            
            if total_annotations > 0:
                chip_weight = total_annotations / (2 * total_chips) if total_chips > 0 else 1.0
                check_weight = total_annotations / (2 * total_checks) if total_checks > 0 else 1.0
                
                weights[image_type] = {
                    'chip': chip_weight,
                    'check': check_weight,
                    'imbalance_ratio': total_chips / total_checks if total_checks > 0 else float('inf')
                }
            else:
                weights[image_type] = {'chip': 1.0, 'check': 1.0, 'imbalance_ratio': 1.0}
        
        return weights
    
    def print_statistics(self):
        """Print dataset statistics and class weights."""
        print("\n" + "="*60)
        print("DATASET CONVERSION COMPLETE")
        print("="*60)
        
        for image_type in ['EV', 'SV']:
            print(f"\n{image_type} Dataset Statistics:")
            print("-" * 30)
            
            for split in ['train', 'val']:
                stats = self.stats[image_type][split]
                print(f"{split.upper()}:")
                print(f"  Images: {stats['images']}")
                print(f"  Total annotations: {stats['annotations']}")
                print(f"  Chips: {stats['chip']}")
                print(f"  Checks: {stats['check']}")
                
                if stats['chip'] > 0 and stats['check'] > 0:
                    ratio = stats['chip'] / stats['check']
                    print(f"  Chip/Check ratio: {ratio:.2f}")
        
        # Print class weights
        weights = self.calculate_class_weights()
        print(f"\nRecommended Class Weights:")
        print("-" * 30)
        
        for image_type in ['EV', 'SV']:
            w = weights[image_type]
            print(f"{image_type}:")
            print(f"  Chip weight: {w['chip']:.3f}")
            print(f"  Check weight: {w['check']:.3f}")
            print(f"  Imbalance ratio: {w['imbalance_ratio']:.2f}")
        
        # Save weights to file
        weights_path = os.path.join(self.output_path, 'class_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        print(f"\nClass weights saved to: {weights_path}")
        
        print(f"\nDataset ready for FocusDet training!")
        print(f"Output directory: {os.path.abspath(self.output_path)}")

def main():
    """Main conversion function."""
    converter = XMLToCOCOConverter()
    converter.convert_dataset()
    converter.print_statistics()

if __name__ == "__main__":
    main()

