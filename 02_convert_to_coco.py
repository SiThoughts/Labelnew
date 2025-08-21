# File: 02_convert_to_coco.py
# Convert YOLO format labels to COCO format for MMDetection
import json
import os
import glob
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# === CONFIGURATION ===
PROJECT_ROOT = r"D:\Photomask\final_dataset"
PROCESSED_FOLDERS = ["SV_dataset_processed", "EV_dataset_processed"]
# YOLO class mapping: 0 -> chip, 1 -> check
CLASSES = ["chip", "check"]

def yolo_to_coco(root_dir, split):
    """
    Convert YOLO format annotations to COCO format for a specific split (train/val)
    
    Args:
        root_dir: Path to processed dataset (e.g., "SV_dataset_processed")
        split: Either "train" or "val"
    """
    img_dir = os.path.join(root_dir, "images", split)
    lbl_dir = os.path.join(root_dir, "labels", split)
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"ERROR: Missing directories in {root_dir} for {split} split")
        return False
    
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    
    # Statistics tracking
    total_chips = 0
    total_checks = 0
    clean_images = 0
    
    print(f"Converting {split} split in {root_dir}...")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    for img_path in tqdm(sorted(image_files), desc=f"Processing {split} images"):
        try:
            # Load image to get dimensions
            with Image.open(img_path) as im:
                width, height = im.size
            
            filename = os.path.basename(img_path)
            
            # Add image info to COCO format
            images.append({
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height
            })
            
            # Look for corresponding label file
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(lbl_dir, base_name + ".txt")
            
            image_has_annotations = False
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"WARNING: Invalid annotation format in {label_path}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # Convert YOLO format (center, normalized) to COCO format (top-left, absolute)
                        x = (x_center - bbox_width / 2.0) * width
                        y = (y_center - bbox_height / 2.0) * height
                        w = bbox_width * width
                        h = bbox_height * height
                        
                        # Ensure bounding box is within image bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        # Skip invalid boxes
                        if w <= 0 or h <= 0:
                            print(f"WARNING: Invalid bbox dimensions in {label_path}: {line}")
                            continue
                        
                        # Add annotation to COCO format
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id + 1,  # COCO categories start from 1
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        
                        # Update statistics
                        if class_id == 0:  # chip
                            total_chips += 1
                        elif class_id == 1:  # check
                            total_checks += 1
                        
                        ann_id += 1
                        image_has_annotations = True
                        
                    except ValueError as e:
                        print(f"WARNING: Could not parse annotation in {label_path}: {line} - {e}")
                        continue
            
            if not image_has_annotations:
                clean_images += 1
            
            img_id += 1
            
        except Exception as e:
            print(f"ERROR: Could not process image {img_path}: {e}")
            continue
    
    # Create COCO categories
    categories = []
    for i, class_name in enumerate(CLASSES):
        categories.append({
            "id": i + 1,  # COCO categories start from 1
            "name": class_name,
            "supercategory": "defect"
        })
    
    # Create final COCO dataset
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {
            "description": f"Defect Detection Dataset - {split.upper()} split",
            "version": "1.0",
            "year": 2024,
            "contributor": "Photomask Defect Detection Pipeline"
        }
    }
    
    # Create annotations directory and save
    ann_dir = os.path.join(root_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    output_path = os.path.join(ann_dir, f"instances_{split}.json")
    
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    # Print statistics
    print(f"✓ Saved: {output_path}")
    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Chips: {total_chips} ({total_chips/(total_chips+total_checks)*100:.1f}%)" if (total_chips+total_checks) > 0 else "  Chips: 0")
    print(f"  Checks: {total_checks} ({total_checks/(total_chips+total_checks)*100:.1f}%)" if (total_chips+total_checks) > 0 else "  Checks: 0")
    print(f"  Clean images: {clean_images}")
    print()
    
    return True

def main():
    print("=" * 60)
    print("YOLO TO COCO CONVERSION")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Class mapping: {dict(enumerate(CLASSES))}")
    print("=" * 60)
    
    # Change to project directory
    if not os.path.exists(PROJECT_ROOT):
        print(f"ERROR: Project root directory not found: {PROJECT_ROOT}")
        return False
    
    os.chdir(PROJECT_ROOT)
    
    success_count = 0
    total_conversions = 0
    
    for folder in PROCESSED_FOLDERS:
        if os.path.isdir(folder):
            print(f"\n--- Processing {folder} ---")
            
            # Convert both train and val splits
            for split in ["train", "val"]:
                if yolo_to_coco(folder, split):
                    success_count += 1
                total_conversions += 1
        else:
            print(f"WARNING: Processed directory not found, skipping: {folder}")
    
    print("=" * 60)
    if success_count == total_conversions:
        print("✓ COCO CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"✓ Converted {success_count} dataset splits")
        print("✓ Ready for model training")
    else:
        print(f"⚠ PARTIAL SUCCESS: {success_count}/{total_conversions} conversions completed")
    print("=" * 60)
    
    return success_count > 0

if __name__ == "__main__":
    main()

