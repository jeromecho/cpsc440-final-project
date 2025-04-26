import os
import shutil
from PIL import Image
import numpy as np
import cv2

def is_high_texture(img_path, std_thresh=10, lap_var_thresh=20):
    image = Image.open(img_path).convert("L")  # grayscale
    gray = np.array(image)

    stddev = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return stddev > std_thresh and laplacian_var > lap_var_thresh

def filter_by_texture(input_dir, output_dir, std_thresh=10, lap_var_thresh=20):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    kept = 0
    for fname in image_files:
        in_path = os.path.join(input_dir, fname)
        if is_high_texture(in_path, std_thresh, lap_var_thresh):
            shutil.copy(in_path, os.path.join(output_dir, fname))
            kept += 1

    print(f"[INFO] Kept {kept}/{len(image_files)} images with high texture.")

# Example usage
if __name__ == "__main__":
    input_dir = "output_tiles/normal"   # or "output_tiles/normal"
    output_dir = "filtered_tiles/normal"
    filter_by_texture(input_dir, output_dir)

