import os

def remove_box_lines(root_dir):
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(dirpath, file)
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                new_lines = [ln for ln in lines if len(ln.split()) != 5]
                if len(new_lines) != len(lines):
                    with open(path, 'w') as f:
                        f.write("\n".join(new_lines) + "\n")
                    print(f"Cleaned: {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python remove_box_lines.py <root_dataset_dir>")
        exit(1)
    remove_box_lines(sys.argv[1])
