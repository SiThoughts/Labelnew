import os

def validate_labels(root_dir):
    total_files = 0
    poly_files = 0
    box_lines = 0
    empty_files = 0
    malformed_files = 0

    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                total_files += 1
                path = os.path.join(dirpath, file)
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    empty_files += 1
                    continue

                has_poly = False
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        box_lines += 1
                    elif len(parts) >= 6:
                        has_poly = True
                    else:
                        malformed_files += 1

                if has_poly:
                    poly_files += 1

    print(f"Total label files: {total_files}")
    print(f"Files with polygons: {poly_files}")
    print(f"Files with any 5-number box lines: {box_lines}")
    print(f"Empty label files: {empty_files}")
    print(f"Malformed lines in files: {malformed_files}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate_seg_dataset.py <root_dataset_dir>")
        exit(1)
    validate_labels(sys.argv[1])
