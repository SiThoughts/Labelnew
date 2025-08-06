#!/usr/bin/env python3
"""
Reduce SAHI dataset size by keeping all slices with defects and sampling clean slices.
This will make training much faster while maintaining good performance.
"""

import os
import glob
import random
import shutil
from pathlib import Path

def reduce_sahi_dataset(dataset_path, clean_slice_ratio=0.3):
    """
    Reduce SAHI dataset size by keeping all defect slices and sampling clean slices.
    
    Args:
        dataset_path: Path to SAHI dataset (e.g., 'sahi_datasets/EV_sahi_dataset')
        clean_slice_ratio: Ratio of clean slices to keep (0.3 = keep 30% of clean slices)
    """
    
    print(f"Reducing SAHI dataset: {dataset_path}")
    print(f"Clean slice ratio: {clean_slice_ratio}")
    print("=" * 60)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_path, 'labels', split)
        images_dir = os.path.join(dataset_path, 'images', split)
        
        if not os.path.exists(labels_dir):
            print(f"Skipping {split} - directory not found")
            continue
            
        print(f"\nProcessing {split} split...")
        
        # Get all label files
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        defect_slices = []
        clean_slices = []
        
        # Categorize slices
        for label_file in label_files:
            if os.path.getsize(label_file) > 0:
                # File has content (defects)
                defect_slices.append(label_file)
            else:
                # Empty file (no defects)
                clean_slices.append(label_file)
        
        print(f"  Found {len(defect_slices)} slices with defects")
        print(f"  Found {len(clean_slices)} clean slices")
        
        # Keep all defect slices
        keep_defect = defect_slices
        
        # Sample clean slices
        num_clean_to_keep = int(len(clean_slices) * clean_slice_ratio)
        keep_clean = random.sample(clean_slices, min(num_clean_to_keep, len(clean_slices)))
        
        # Files to remove
        remove_clean = [f for f in clean_slices if f not in keep_clean]
        
        print(f"  Keeping {len(keep_defect)} defect slices")
        print(f"  Keeping {len(keep_clean)} clean slices")
        print(f"  Removing {len(remove_clean)} clean slices")
        
        # Remove unwanted clean slices
        removed_count = 0
        for label_file in remove_clean:
            try:
                # Remove label file
                os.remove(label_file)
                
                # Remove corresponding image file
                base_name = os.path.splitext(os.path.basename(label_file))[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_file = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(image_file):
                        os.remove(image_file)
                        break
                
                removed_count += 1
                
            except Exception as e:
                print(f"    Error removing {label_file}: {e}")
        
        print(f"  Successfully removed {removed_count} clean slices")
        
        # Final count
        remaining_labels = len(glob.glob(os.path.join(labels_dir, "*.txt")))
        remaining_images = len(glob.glob(os.path.join(images_dir, "*.*")))
        print(f"  Final count: {remaining_labels} labels, {remaining_images} images")

def main():
    """Main function to reduce both EV and SV SAHI datasets."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Reduce EV dataset
    ev_dataset = "sahi_datasets/EV_sahi_dataset"
    if os.path.exists(ev_dataset):
        reduce_sahi_dataset(ev_dataset, clean_slice_ratio=0.25)  # Keep 25% of clean slices
    else:
        print(f"EV dataset not found: {ev_dataset}")
    
    print("\n" + "=" * 60)
    
    # Reduce SV dataset
    sv_dataset = "sahi_datasets/SV_sahi_dataset"
    if os.path.exists(sv_dataset):
        reduce_sahi_dataset(sv_dataset, clean_slice_ratio=0.25)  # Keep 25% of clean slices
    else:
        print(f"SV dataset not found: {sv_dataset}")
    
    print("\n" + "=" * 60)
    print("Dataset reduction completed!")
    print("\nRecommended next steps:")
    print("1. Increase batch size to 8 or 16 for faster training")
    print("2. Start training with reduced dataset")
    print("3. Monitor training speed and adjust as needed")

if __name__ == "__main__":
    main()

