import os

def make_empty_labels(root_dir):
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = []
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in img_exts:
                images.append(os.path.join(dirpath, file))

    for img_path in images:
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                pass
            print(f"Created empty label: {label_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python make_empty_labels_for_missing.py <root_dataset_dir>")
        exit(1)
    make_empty_labels(sys.argv[1])
