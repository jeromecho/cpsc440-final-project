from torch.utils.data import Dataset
from PIL import Image
import os
import random


class PretrainingDataset(Dataset):
    def __init__(self, img_dir, transform=None, crop_size=None):
        self.img_dir = img_dir
        self.transform = transform
        self.crop_size = crop_size

        all_image_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ]

        # Filter out corrupted or unreadable images
        self.image_paths = []
        for path in all_image_paths:
            try:
                with Image.open(path) as img:
                    img.verify()  # validate image file
                self.image_paths.append(path)
            except Exception as e:
                print(f"Skipping corrupted image: {path} ({e})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"img_path {img_path}")

        with Image.open(img_path) as img:

            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

