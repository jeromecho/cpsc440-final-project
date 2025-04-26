import os
from multiprocessing import Pool
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np

def crop_single_tiff_rasterio(input_path, output_dir, tile_size=(384, 384)):
    try:
        print(f"[START] {input_path}")
        with rasterio.open(input_path) as dataset:
            width = dataset.width
            height = dataset.height
            tile_w, tile_h = tile_size

            for top in range(0, height, tile_h):
                for left in range(0, width, tile_w):
                    bottom = min(top + tile_h, height)
                    right = min(left + tile_w, width)

                    win = Window(left, top, right - left, bottom - top)
                    tile = dataset.read(window=win)  # shape: (bands, tile_h, tile_w)

                    if tile.shape[0] == 1:
                        tile_img = Image.fromarray(tile[0])
                    else:
                        tile_img = Image.fromarray(np.moveaxis(tile, 0, -1))  # (H, W, C)

                    tile_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_{left}_{top}.png"
                    tile_img.save(os.path.join(output_dir, tile_name))

        print(f"[DONE] {input_path}")
    except Exception as e:
        print(f"[ERROR] {input_path}: {e}")

def crop_tiffs_from_dir(input_dir, output_dir, tile_size=(384, 384)):
    os.makedirs(output_dir, exist_ok=True)
    input_paths = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(('.tif', '.tiff'))
    ]
    with Pool(processes=4) as pool:
        pool.starmap(crop_single_tiff_rasterio, [(p, output_dir, tile_size) for p in input_paths])

if __name__ == "__main__":
    tumor_dir = "./tumor 2"
    normal_dir = "./normal 2"
    tumor_out_dir = "output_tiles/tumor"
    normal_out_dir = "output_tiles/normal"
    tile_size = (384, 384)

    # crop_tiffs_from_dir(tumor_dir, tumor_out_dir, tile_size)
    crop_tiffs_from_dir(normal_dir, normal_out_dir, tile_size)

