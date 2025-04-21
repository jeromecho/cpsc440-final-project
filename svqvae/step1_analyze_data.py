from torch.utils.data import Dataset
from PIL import Image
import os
import random

# For CAM16 TIFFs
Image.MAX_IMAGE_PIXELS = None

class PretrainingDataset(Dataset):
    def __init__(self, img_dir, transform=None, crop_size=None):
        self.img_dir = img_dir
        self.transform = transform
        self.crop_size = crop_size

        self.image_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ][:1]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"img_path {img_path}")

        with Image.open(img_path) as img:
            width, height = img.size
            image = None

            if self.crop_size is not None:
                crop_w, crop_h = self.crop_size
                if width >= crop_w and height >= crop_h:
                    # 🎯 Random top-left coordinates for cropping
                    left = random.randint(0, width - crop_w)
                    upper = random.randint(0, height - crop_h)
                    crop_box = (left, upper, left + crop_w, upper + crop_h)
                    image = img.crop(crop_box).convert('RGB')
                else:
                    print(f"WARNING: Skipping crop for image {img_path} (too small for crop size)")
                    image = img.convert('RGB')  # Fallback: use full image

            else:
                image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

