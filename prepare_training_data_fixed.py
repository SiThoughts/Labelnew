import os
import shutil
from pathlib import Path
from xml_parser import AnnotationParser
from typing import List, Dict, Tuple
import random
import glob

class TrainingDataPreparator:
    """Prepares training data for YOLO model training - FIXED VERSION with robust filename matching."""
    
    def __init__(self):
        self.parser = AnnotationParser()
        self.class_mapping = {'chip': 0, 'check': 1}
        
        # Data source paths
        self.training_sources = [
            r"D:\Photomask\DS0",
            r"D:\Photomask\DS2_Sort2"
        ]
        
        self.validation_source = r"D:\Photomask\MSA_Sort3"
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    def find_matching_image(self, xml_path: str) -> str:
        """
        Find the actual image file that matches an XML file, regardless of extension.
        Handles filename mismatches and extension differences.
        """
        xml_dir = os.path.dirname(xml_path)
        xml_basename = os.path.splitext(os.path.basename(xml_path))[0]
        
        # Method 1: Direct extension replacement
        for ext in self.image_extensions:
            candidate = os.path.join(xml_dir, xml_basename + ext)
            if os.path.exists(candidate):
                return candidate
        
        # Method 2: Search for files with similar base names (handle dots and variations)
        # Remove common suffixes and try again
        base_variations = [
            xml_basename,
            xml_basename.replace('.', '_'),  # Replace dots with underscores
            xml_basename.replace('_', '.'),  # Replace underscores with dots
        ]
        
        for base in base_variations:
            for ext in self.image_extensions:
                candidate = os.path.join(xml_dir, base + ext)
                if os.path.exists(candidate):
                    return candidate
        
        # Method 3: Fuzzy matching - find files that start with the same prefix
        xml_prefix = xml_basename.split('_')[0] if '_' in xml_basename else xml_basename[:10]
        
        for file in os.listdir(xml_dir):
            if file.startswith(xml_prefix) and any(file.lower().endswith(ext) for ext in self.image_extensions):
                # Additional check: ensure it's not too different
                file_base = os.path.splitext(file)[0]
                if len(file_base) > 0 and abs(len(file_base) - len(xml_basename)) < 10:
                    return os.path.join(xml_dir, file)
        
        return None
    
    def get_all_xml_files(self, folder_path: str) -> List[str]:
        """Get all XML files in a folder."""
        if not os.path.exists(folder_path):
            return []
        
        xml_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith('.xml'):
                xml_files.append(os.path.join(folder_path, file))
        
        return sorted(xml_files)
    
    def collect_data_from_folder(self, folder_path: str, subfolder: str) -> List[Tuple[str, str, bool]]:
        """
        Collect ALL images from a folder using robust filename matching.
        
        Returns:
            List of (image_path, xml_path, has_defects) tuples
        """
        data_pairs = []
        target_folder = os.path.join(folder_path, subfolder)
        
        if not os.path.exists(target_folder):
            print(f"Warning: Folder {target_folder} does not exist")
            return data_pairs
        
        # Get ALL XML files first
        xml_files = self.get_all_xml_files(target_folder)
        print(f"Found {len(xml_files)} XML files in {target_folder}")
        
        matched_images = 0
        missing_images = 0
        defect_count = 0
        clean_count = 0
        
        # Process each XML file
        for xml_path in xml_files:
            # Find matching image file using robust matching
            image_path = self.find_matching_image(xml_path)
            
            if image_path:
                matched_images += 1
                # Check if XML has defects
                annotation = self.parser.parse_xml_file(xml_path)
                has_defects = annotation is not None and len(annotation['objects']) > 0
                
                if has_defects:
                    defect_count += len(annotation['objects'])
                    
                data_pairs.append((image_path, xml_path, has_defects))
                
                if has_defects:
                    pass  # Image with defects
                else:
                    clean_count += 1
            else:
                missing_images += 1
                print(f"Warning: No matching image found for {os.path.basename(xml_path)}")
        
        # Also find images without XML files (additional clean images)
        all_images = set()
        for file in os.listdir(target_folder):
            if any(file.lower().endswith(ext) for ext in self.image_extensions):
                all_images.add(os.path.join(target_folder, file))
        
        # Find images that don't have corresponding XML files
        xml_matched_images = {pair[0] for pair in data_pairs}
        orphan_images = all_images - xml_matched_images
        
        for image_path in orphan_images:
            data_pairs.append((image_path, None, False))
            clean_count += 1
        
        print(f"  - XML files processed: {len(xml_files)}")
        print(f"  - Images matched to XML: {matched_images}")
        print(f"  - Missing images: {missing_images}")
        print(f"  - Orphan images (no XML): {len(orphan_images)}")
        print(f"  - Total defect instances: {defect_count}")
        print(f"  - Clean images: {clean_count}")
        print(f"  - Total image pairs: {len(data_pairs)}")
        
        return data_pairs
    
    def prepare_yolo_dataset(self, data_pairs: List[Tuple[str, str, bool]], output_dir: str, split_name: str):
        """
        Prepare YOLO format dataset from ALL image pairs.
        """
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        successful_pairs = 0
        defect_images = 0
        clean_images = 0
        total_defect_instances = 0
        
        for image_path, xml_path, has_defects in data_pairs:
            try:
                # Copy image file
                image_filename = os.path.basename(image_path)
                target_image_path = os.path.join(images_dir, image_filename)
                shutil.copy2(image_path, target_image_path)
                
                # Create label file
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_filename)
                
                if has_defects and xml_path:
                    # Image with defects - create proper labels
                    annotation = self.parser.parse_xml_file(xml_path)
                    if annotation and annotation['objects']:
                        yolo_lines = self.parser.convert_to_yolo_format(annotation, self.class_mapping)
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_lines))
                        defect_images += 1
                        total_defect_instances += len(annotation['objects'])
                    else:
                        # Empty label file for clean image
                        with open(label_path, 'w') as f:
                            pass  # Empty file
                        clean_images += 1
                else:
                    # Clean image - create empty label file
                    with open(label_path, 'w') as f:
                        pass  # Empty file
                    clean_images += 1
                
                successful_pairs += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Successfully processed {successful_pairs} images for {split_name} split")
        print(f"  - Images with defects: {defect_images}")
        print(f"  - Total defect instances: {total_defect_instances}")
        print(f"  - Clean images: {clean_images}")
        
        return successful_pairs, total_defect_instances
    
    def create_dataset_yaml(self, output_dir: str, model_type: str):
        """Create YAML configuration file for the dataset."""
        yaml_content = f"""# Dataset configuration for {model_type} defect detection
path: {output_dir}
train: images/train
val: images/val
test: images/test

# Classes
nc: 2  # number of classes
names: ['chip', 'check']  # class names
"""
        
        yaml_path = os.path.join(output_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created dataset configuration: {yaml_path}")
    
    def prepare_model_data(self, model_type: str, output_base_dir: str):
        """Prepare complete dataset for a specific model type using FIXED robust matching."""
        print(f"\n=== Preparing {model_type} Model Data (FIXED VERSION) ===")
        
        output_dir = os.path.join(output_base_dir, f"{model_type}_dataset")
        
        # Collect training data from multiple sources
        all_training_pairs = []
        for source_path in self.training_sources:
            pairs = self.collect_data_from_folder(source_path, model_type)
            all_training_pairs.extend(pairs)
        
        print(f"Total training pairs for {model_type}: {len(all_training_pairs)}")
        
        # Collect validation/test data
        validation_pairs = self.collect_data_from_folder(self.validation_source, model_type)
        print(f"Total validation pairs for {model_type}: {len(validation_pairs)}")
        
        if not all_training_pairs:
            print(f"Error: No training data found for {model_type} model")
            return
        
        # Split validation data into val and test (50-50 split)
        random.shuffle(validation_pairs)
        split_point = len(validation_pairs) // 2
        val_pairs = validation_pairs[:split_point]
        test_pairs = validation_pairs[split_point:]
        
        # Prepare YOLO datasets
        train_count, train_defects = self.prepare_yolo_dataset(all_training_pairs, output_dir, 'train')
        val_count, val_defects = self.prepare_yolo_dataset(val_pairs, output_dir, 'val') if val_pairs else (0, 0)
        test_count, test_defects = self.prepare_yolo_dataset(test_pairs, output_dir, 'test') if test_pairs else (0, 0)
        
        # Create dataset configuration
        self.create_dataset_yaml(output_dir, model_type)
        
        print(f"\n{model_type} Dataset Summary:")
        print(f"  Training samples: {train_count} (defect instances: {train_defects})")
        print(f"  Validation samples: {val_count} (defect instances: {val_defects})")
        print(f"  Test samples: {test_count} (defect instances: {test_defects})")
        print(f"  Total defect instances: {train_defects + val_defects + test_defects}")
        print(f"  Dataset location: {output_dir}")
        
        return output_dir

def main():
    """Main function to prepare both EV and SV datasets using FIXED robust matching."""
    print("Defect Detection Dataset Preparation (FIXED VERSION)")
    print("=" * 60)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create output directory
    output_base = "defect_detection_datasets_fixed"
    os.makedirs(output_base, exist_ok=True)
    
    preparator = TrainingDataPreparator()
    
    # Prepare both datasets
    ev_dataset_path = preparator.prepare_model_data('EV', output_base)
    sv_dataset_path = preparator.prepare_model_data('SV', output_base)
    
    print("\n" + "=" * 60)
    print("FIXED Dataset preparation completed!")
    print(f"EV dataset: {ev_dataset_path}")
    print(f"SV dataset: {sv_dataset_path}")
    print("\nThis version should find ALL your images and defects!")
    print("Expected: 3000+ training images with 5000+ defect instances")

if __name__ == "__main__":
    main()

