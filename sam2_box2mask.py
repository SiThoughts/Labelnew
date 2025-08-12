import os
import argparse
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def process_dataset_with_sam(src, sam_checkpoint, model_type):
    # Load the SAM model
    print(f"Loading SAM model type '{model_type}' from checkpoint '{sam_checkpoint}'...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor = SamPredictor(sam)

    # Walk through dataset folders (train, val, test)
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(src, 'images', split)
        label_dir = os.path.join(src, 'labels', split)
        mask_out_dir = os.path.join(src, 'masks', split)

        if not os.path.exists(img_dir):
            print(f"Skipping missing split: {img_dir}")
            continue

        os.makedirs(mask_out_dir, exist_ok=True)

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if not os.path.exists(label_path):
                continue

            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            with open(label_path, 'r') as f:
                boxes = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x_center, y_center, w, h = map(float, parts)
                    h_img, w_img = image.shape[:2]
                    x_center *= w_img
                    y_center *= h_img
                    w *= w_img
                    h *= h_img
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2
                    boxes.append([x_min, y_min, x_max, y_max])

            if not boxes:
                continue

            input_boxes = torch.tensor(boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

            for i, mask in enumerate(masks):
                mask_img = (mask[0].cpu().numpy() * 255).astype(np.uint8)
                mask_filename = os.path.join(mask_out_dir, f"{os.path.splitext(img_file)[0]}_{i}.png")
                cv2.imwrite(mask_filename, mask_img)

    print("SAM/SAM2 conversion complete!")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO box labels to masks using SAM")
    parser.add_argument('--src', required=True, help='Path to YOLO dataset root')
    parser.add_argument('--sam_checkpoint', required=True, help='Path to SAM .pth checkpoint')
    parser.add_argument('--model_type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                        help='Model type for SAM')
    args = parser.parse_args()

    process_dataset_with_sam(args.src, args.sam_checkpoint, args.model_type)

if __name__ == '__main__':
    main()
