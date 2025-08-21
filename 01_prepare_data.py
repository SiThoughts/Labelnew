# File: 01_prepare_data.py
# Data preparation script optimized for Windows paths and defect detection
import os
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
# Main project directory (Windows format)
PROJECT_ROOT = r"D:\Photomask\final_dataset"
SOURCE_FOLDERS = ["SV_dataset", "EV_dataset"]
OUTPUT_SUFFIX = "_processed"
VALIDATION_SPLIT_RATIO = 0.20  # 20% for validation, 80% for training

# Set random seed for reproducible splits
random.seed(42)

def process_dataset(source_root, val_split_ratio):
    """
    Process a single dataset (SV or EV):
    1. Combine train/test/val splits
    2. Shuffle all images randomly
    3. Create new 80/20 train/val split
    4. Copy images and labels to new structure
    """
    output_root = source_root + OUTPUT_SUFFIX
    print(f"--- Processing: {source_root} -> {output_root} ---")

    # Create new directory structure
    dirs = {
        'train_img': os.path.join(output_root, "images", "train"),
        'train_lbl': os.path.join(output_root, "labels", "train"),
        'val_img': os.path.join(output_root, "images", "val"),
        'val_lbl': os.path.join(output_root, "labels", "val")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Find all images across train/test/val subfolders
    source_img_dir = os.path.join(source_root, "images")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    image_paths = []
    for ext in valid_extensions:
        pattern = os.path.join(source_img_dir, "**", f"*{ext}")
        image_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(source_img_dir, "**", f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    if not image_paths:
        print(f"ERROR: No images found in {source_img_dir}")
        return False
    
    # Remove duplicates and shuffle
    image_paths = list(set(image_paths))
    random.shuffle(image_paths)
    print(f"Found and shuffled {len(image_paths)} total images.")

    # Split into training and validation sets
    split_index = int(len(image_paths) * (1 - val_split_ratio))
    train_files = image_paths[:split_index]
    val_files = image_paths[split_index:]
    print(f"Split: {len(train_files)} training, {len(val_files)} validation")

    # Copy files to new structure
    def copy_files_with_labels(file_list, dest_img_dir, dest_lbl_dir, split_name):
        copied_images = 0
        copied_labels = 0
        missing_labels = 0
        
        for img_path_str in tqdm(file_list, desc=f"Copying {split_name} files"):
            img_path = Path(img_path_str)
            base_filename = img_path.stem
            label_filename = base_filename + ".txt"
            
            # Find the original split folder (train, test, or val)
            original_split_folder = img_path.parent.name
            
            # Construct the source label path
            source_lbl_path = Path(source_root) / "labels" / original_split_folder / label_filename

            # Copy image
            dest_img_path = os.path.join(dest_img_dir, img_path.name)
            shutil.copy2(img_path, dest_img_path)
            copied_images += 1
            
            # Copy label if it exists (some images might be clean/empty)
            if source_lbl_path.exists():
                dest_lbl_path = os.path.join(dest_lbl_dir, label_filename)
                shutil.copy2(source_lbl_path, dest_lbl_path)
                copied_labels += 1
            else:
                # Create empty label file for clean images
                empty_label_path = os.path.join(dest_lbl_dir, label_filename)
                with open(empty_label_path, 'w') as f:
                    pass  # Empty file
                missing_labels += 1
        
        print(f"  {split_name}: {copied_images} images, {copied_labels} labels, {missing_labels} clean images")
        return copied_images, copied_labels

    # Process training and validation splits
    copy_files_with_labels(train_files, dirs['train_img'], dirs['train_lbl'], "Training")
    copy_files_with_labels(val_files, dirs['val_img'], dirs['val_lbl'], "Validation")
    
    print(f"--- Finished processing {source_root} ---\n")
    return True

def main():
    print("=" * 60)
    print("DEFECT DETECTION DATA PREPARATION")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Looking for datasets: {SOURCE_FOLDERS}")
    print(f"Validation split: {VALIDATION_SPLIT_RATIO * 100:.0f}%")
    print("=" * 60)
    
    # Change to project directory
    if not os.path.exists(PROJECT_ROOT):
        print(f"ERROR: Project root directory not found: {PROJECT_ROOT}")
        print("Please check the path in the script configuration.")
        return False
    
    os.chdir(PROJECT_ROOT)
    
    success_count = 0
    for folder in SOURCE_FOLDERS:
        if os.path.isdir(folder):
            if process_dataset(folder, VALIDATION_SPLIT_RATIO):
                success_count += 1
        else:
            print(f"WARNING: Source directory not found, skipping: {folder}")
    
    print("=" * 60)
    if success_count == len(SOURCE_FOLDERS):
        print("✓ DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print(f"✓ Processed {success_count} datasets")
        print("✓ Ready for COCO conversion")
    else:
        print(f"⚠ PARTIAL SUCCESS: {success_count}/{len(SOURCE_FOLDERS)} datasets processed")
    print("=" * 60)
    
    return success_count > 0

if __name__ == "__main__":
    main()

