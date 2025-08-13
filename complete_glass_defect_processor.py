#!/usr/bin/env python3
"""
Complete Glass Defect Dataset Processor
Reads XML annotations, tiles images, creates SAM2 segmentation masks, saves in YOLOv11 format
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
import urllib.request
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Setup logging
def setup_logging(output_dir):
    """Setup comprehensive logging"""
    log_file = os.path.join(output_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def download_sam2_checkpoint(checkpoint_path, model_type):
    """Download SAM2 checkpoint if it doesn't exist"""
    if os.path.exists(checkpoint_path):
        return
    
    logging.info(f"Downloading SAM2 checkpoint: {checkpoint_path}")
    
    checkpoint_urls = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
        'vit_s': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt'
    }
    
    if model_type not in checkpoint_urls:
        raise ValueError(f"Unknown model type: {model_type}")
    
    url = checkpoint_urls[model_type]
    dir_path = os.path.dirname(checkpoint_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, checkpoint_path)
        logging.info(f"Downloaded checkpoint to {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error downloading checkpoint: {e}")
        raise

def parse_xml_annotation(xml_path):
    """Parse Pascal VOC XML annotation file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Parse objects
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            
            # Convert class names to IDs
            if name == 'chip':
                class_id = 0
            elif name == 'check':
                class_id = 1
            else:
                logging.warning(f"Unknown class name: {name}, skipping")
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            objects.append({
                'class_id': class_id,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return img_width, img_height, objects
    
    except Exception as e:
        logging.error(f"Error parsing XML {xml_path}: {e}")
        return None, None, []

def clip_bbox_to_tile(bbox, tile_x, tile_y, tile_size, min_area_frac=0.0002):
    """Clip bounding box to tile boundaries and check minimum area"""
    xmin, ymin, xmax, ymax = bbox
    
    # Clip to tile boundaries
    xmin_clipped = max(0, xmin - tile_x)
    ymin_clipped = max(0, ymin - tile_y)
    xmax_clipped = min(tile_size, xmax - tile_x)
    ymax_clipped = min(tile_size, ymax - tile_y)
    
    # Check if box is valid after clipping
    if xmax_clipped <= xmin_clipped or ymax_clipped <= ymin_clipped:
        return None
    
    # Check minimum area requirement
    clipped_area = (xmax_clipped - xmin_clipped) * (ymax_clipped - ymin_clipped)
    tile_area = tile_size * tile_size
    
    if clipped_area / tile_area < min_area_frac:
        return None
    
    return [xmin_clipped, ymin_clipped, xmax_clipped, ymax_clipped]

def tile_image_and_annotations(img_path, xml_path, tile_size=1024, overlap=0.1):
    """Tile image and adjust annotations for each tile"""
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Could not load image: {img_path}")
        return []
    
    img_height, img_width = img.shape[:2]
    logging.info(f"Processing image {os.path.basename(img_path)} - Size: {img_width}x{img_height}")
    
    # Parse XML annotations
    xml_width, xml_height, objects = parse_xml_annotation(xml_path)
    if xml_width is None:
        logging.error(f"Failed to parse XML: {xml_path}")
        return []
    
    logging.info(f"Found {len(objects)} objects in annotation")
    
    # Calculate tile positions
    step_size = int(tile_size * (1 - overlap))
    tiles = []
    tile_count = 0
    
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            # Calculate tile boundaries
            tile_x1 = x
            tile_y1 = y
            tile_x2 = min(x + tile_size, img_width)
            tile_y2 = min(y + tile_size, img_height)
            
            # Skip if tile is too small
            if (tile_x2 - tile_x1) < tile_size * 0.5 or (tile_y2 - tile_y1) < tile_size * 0.5:
                continue
            
            # Extract tile
            tile = img[tile_y1:tile_y2, tile_x1:tile_x2]
            
            # Pad tile if necessary
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            # Process annotations for this tile
            tile_objects = []
            for obj in objects:
                clipped_bbox = clip_bbox_to_tile(obj['bbox'], tile_x1, tile_y1, tile_size)
                if clipped_bbox is not None:
                    tile_objects.append({
                        'class_id': obj['class_id'],
                        'bbox': clipped_bbox
                    })
            
            if tile_objects:  # Only save tiles with objects
                tiles.append({
                    'image': tile,
                    'objects': tile_objects,
                    'tile_id': tile_count
                })
                tile_count += 1
    
    logging.info(f"Created {len(tiles)} tiles with objects from {os.path.basename(img_path)}")
    return tiles

def mask_to_polygon(mask):
    """Convert binary mask to polygon points"""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of points
    points = simplified_contour.reshape(-1, 2)
    
    return points.flatten()

def dilate_mask_for_checks(mask, class_id, dilation_pixels=2):
    """Apply dilation to masks for 'checks' class (class 1)"""
    if class_id == 1:  # checks class
        kernel = np.ones((dilation_pixels * 2 + 1, dilation_pixels * 2 + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return mask

def save_yolo_segmentation(label_path, segmentations, img_width, img_height):
    """Save YOLO format segmentation labels"""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    with open(label_path, 'w') as f:
        for class_id, polygon in segmentations:
            if len(polygon) >= 6:  # At least 3 points (6 coordinates)
                # Normalize coordinates
                normalized_polygon = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / img_width
                    y = polygon[i + 1] / img_height
                    normalized_polygon.extend([x, y])
                
                # Write to file
                polygon_str = ' '.join([f"{coord:.6f}" for coord in normalized_polygon])
                f.write(f"{class_id} {polygon_str}\n")

def process_dataset(input_dir, output_dir, sam_checkpoint="sam2_h.pth", model_type="vit_h", 
                   tile_size=1024, overlap=0.1):
    """Process entire dataset"""
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("STARTING GLASS DEFECT DATASET PROCESSING")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tile size: {tile_size}x{tile_size}")
    logger.info(f"Overlap: {overlap*100:.1f}%")
    logger.info(f"SAM2 model: {model_type}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    total_images = len(image_files)
    logger.info(f"Found {total_images} images to process")
    
    if total_images == 0:
        logger.error("No images found in input directory!")
        return
    
    # Initialize SAM2
    logger.info("="*50)
    logger.info("STEP 1: INITIALIZING SAM2 MODEL")
    logger.info("="*50)
    
    download_sam2_checkpoint(sam_checkpoint, model_type)
    
    try:
        sam2_model = build_sam2(model_type, sam_checkpoint)
        predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SAM2 model: {e}")
        return
    
    # Process each image
    logger.info("="*50)
    logger.info("STEP 2: PROCESSING IMAGES")
    logger.info("="*50)
    
    processed_images = 0
    total_tiles_created = 0
    total_masks_created = 0
    
    for img_idx, img_path in enumerate(image_files, 1):
        logger.info(f"\n[{img_idx}/{total_images}] Processing: {img_path.name}")
        logger.info(f"Remaining: {total_images - img_idx} images")
        
        # Find corresponding XML file
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.exists():
            logger.warning(f"No XML annotation found for {img_path.name}, skipping")
            continue
        
        # Tile image and annotations
        logger.info(f"  → Tiling image...")
        tiles = tile_image_and_annotations(str(img_path), str(xml_path), tile_size, overlap)
        
        if not tiles:
            logger.warning(f"  → No valid tiles created for {img_path.name}")
            continue
        
        logger.info(f"  → Created {len(tiles)} tiles")
        total_tiles_created += len(tiles)
        
        # Process each tile with SAM2
        for tile_idx, tile_data in enumerate(tiles):
            tile_name = f"{img_path.stem}_tile_{tile_data['tile_id']:04d}"
            
            logger.info(f"  → Processing tile {tile_idx+1}/{len(tiles)}: {tile_name}")
            
            # Save tile image
            tile_img_path = os.path.join(output_images_dir, f"{tile_name}.jpg")
            cv2.imwrite(tile_img_path, tile_data['image'])
            
            # Set image for SAM2
            image_rgb = cv2.cvtColor(tile_data['image'], cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            
            # Process each object in tile
            segmentations = []
            objects_in_tile = len(tile_data['objects'])
            
            for obj_idx, obj in enumerate(tile_data['objects']):
                logger.info(f"    → Object {obj_idx+1}/{objects_in_tile} (class {obj['class_id']})")
                
                # Create input box for SAM2
                xmin, ymin, xmax, ymax = obj['bbox']
                input_box = np.array([xmin, ymin, xmax, ymax])
                
                try:
                    # Predict mask
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    
                    if len(masks) > 0:
                        mask = masks[0]
                        
                        # Apply dilation for checks class
                        mask = dilate_mask_for_checks(mask, obj['class_id'], dilation_pixels=2)
                        
                        # Convert mask to polygon
                        polygon = mask_to_polygon(mask)
                        
                        if len(polygon) >= 6:  # Valid polygon
                            segmentations.append((obj['class_id'], polygon))
                            total_masks_created += 1
                            logger.info(f"      ✓ Mask created successfully")
                        else:
                            logger.warning(f"      ✗ Invalid polygon generated")
                
                except Exception as e:
                    logger.error(f"      ✗ SAM2 failed: {e}")
                    continue
            
            # Save segmentation labels
            tile_label_path = os.path.join(output_labels_dir, f"{tile_name}.txt")
            save_yolo_segmentation(tile_label_path, segmentations, tile_size, tile_size)
            
            logger.info(f"    → Saved {len(segmentations)} segmentation masks")
        
        processed_images += 1
        logger.info(f"  ✓ Completed {img_path.name}")
        logger.info(f"  → Progress: {processed_images}/{total_images} images processed")
    
    # Final summary
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total images processed: {processed_images}/{total_images}")
    logger.info(f"Total tiles created: {total_tiles_created}")
    logger.info(f"Total segmentation masks created: {total_masks_created}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Images saved to: {output_images_dir}")
    logger.info(f"Labels saved to: {output_labels_dir}")
    
    if processed_images < total_images:
        logger.warning(f"Warning: {total_images - processed_images} images were skipped")

def main():
    parser = argparse.ArgumentParser(description='Complete Glass Defect Dataset Processor')
    parser.add_argument('--input_dir', default=r'D:\Photomask\DS2_Sort2', 
                       help='Input directory containing images and XML annotations')
    parser.add_argument('--output_dir', default=r'D:\Photomask\DS2_Sort2_processed', 
                       help='Output directory for processed dataset')
    parser.add_argument('--tile_size', type=int, default=1024, 
                       help='Tile size (default: 1024)')
    parser.add_argument('--overlap', type=float, default=0.1, 
                       help='Overlap fraction (default: 0.1)')
    parser.add_argument('--sam_checkpoint', default='sam2_h.pth', 
                       help='SAM2 checkpoint path')
    parser.add_argument('--model_type', default='vit_h', 
                       choices=['vit_h', 'vit_l', 'vit_b', 'vit_s'],
                       help='SAM2 model type')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        tile_size=args.tile_size,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main()

