import os
import shutil
from pathlib import Path
from xml_parser import AnnotationParser
from typing import List, Dict, Tuple
import random

class TrainingDataPreparator:
    """Prepares training data for YOLO model training - includes ALL images."""
    
    def __init__(self):
        self.parser = AnnotationParser()
        self.class_mapping = {'chip': 0, 'check': 1}
        
        # Data source paths
        self.training_sources = [
            r"D:\Photomask\DS0",
            r"D:\Photomask\DS2_Sort2"
        ]
        
        self.validation_source = r"D:\Photomask\MSA_Sort3"
    
    def get_all_images_in_folder(self, folder_path: str) -> List[str]:
        """Get all image files in a folder."""
        if not os.path.exists(folder_path):
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        return sorted(image_files)
    
    def collect_data_from_folder(self, folder_path: str, subfolder: str) -> List[Tuple[str, str, bool]]:
        """
        Collect ALL images from a folder, with or without annotations.
        
        Returns:
            List of (image_path, xml_path, has_defects) tuples
        """
        data_pairs = []
        target_folder = os.path.join(folder_path, subfolder)
        
        if not os.path.exists(target_folder):
            print(f"Warning: Folder {target_folder} does not exist")
            return data_pairs
        
        # Get ALL image files
        all_images = self.get_all_images_in_folder(target_folder)
        print(f"Found {len(all_images)} total images in {target_folder}")
        
        # Process each image
        for image_path in all_images:
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            xml_path = os.path.join(target_folder, image_basename + '.xml')
            
            if os.path.exists(xml_path):
                # Check if XML has defects
                annotation = self.parser.parse_xml_file(xml_path)
                has_defects = annotation is not None and len(annotation['objects']) > 0
                data_pairs.append((image_path, xml_path, has_defects))
            else:
                # Image without XML - treat as clean image
                data_pairs.append((image_path, None, False))
        
        defect_count = sum(1 for _, _, has_defects in data_pairs if has_defects)
        clean_count = len(data_pairs) - defect_count
        
        print(f"  - Images with defects: {defect_count}")
        print(f"  - Clean images: {clean_count}")
        print(f"  - Total images: {len(data_pairs)}")
        
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
        
        for image_path, xml_path, has_defects in data_pairs:
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
        
        print(f"Successfully processed {successful_pairs} images for {split_name} split")
        print(f"  - With defects: {defect_images}")
        print(f"  - Clean images: {clean_images}")
        
        return successful_pairs
    
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
        """Prepare complete dataset for a specific model type using ALL images."""
        print(f"\n=== Preparing {model_type} Model Data (ALL IMAGES) ===")
        
        output_dir = os.path.join(output_base_dir, f"{model_type}_dataset")
        
        # Collect training data from multiple sources
        all_training_pairs = []
        for source_path in self.training_sources:
            pairs = self.collect_data_from_folder(source_path, model_type)
            all_training_pairs.extend(pairs)
        
        print(f"Total training images for {model_type}: {len(all_training_pairs)}")
        
        # Collect validation/test data
        validation_pairs = self.collect_data_from_folder(self.validation_source, model_type)
        print(f"Total validation images for {model_type}: {len(validation_pairs)}")
        
        if not all_training_pairs:
            print(f"Error: No training data found for {model_type} model")
            return
        
        # Split validation data into val and test (50-50 split)
        random.shuffle(validation_pairs)
        split_point = len(validation_pairs) // 2
        val_pairs = validation_pairs[:split_point]
        test_pairs = validation_pairs[split_point:]
        
        # Prepare YOLO datasets
        train_count = self.prepare_yolo_dataset(all_training_pairs, output_dir, 'train')
        val_count = self.prepare_yolo_dataset(val_pairs, output_dir, 'val') if val_pairs else 0
        test_count = self.prepare_yolo_dataset(test_pairs, output_dir, 'test') if test_pairs else 0
        
        # Create dataset configuration
        self.create_dataset_yaml(output_dir, model_type)
        
        print(f"\n{model_type} Dataset Summary:")
        print(f"  Training samples: {train_count}")
        print(f"  Validation samples: {val_count}")
        print(f"  Test samples: {test_count}")
        print(f"  Dataset location: {output_dir}")
        
        return output_dir

def main():
    """Main function to prepare both EV and SV datasets using ALL images."""
    print("Defect Detection Dataset Preparation (ALL IMAGES)")
    print("=" * 50)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create output directory
    output_base = "defect_detection_datasets_full"
    os.makedirs(output_base, exist_ok=True)
    
    preparator = TrainingDataPreparator()
    
    # Prepare both datasets
    ev_dataset_path = preparator.prepare_model_data('EV', output_base)
    sv_dataset_path = preparator.prepare_model_data('SV', output_base)
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print(f"EV dataset: {ev_dataset_path}")
    print(f"SV dataset: {sv_dataset_path}")
    print("\nNow you're using ALL your images for training!")

if __name__ == "__main__":
    main()

