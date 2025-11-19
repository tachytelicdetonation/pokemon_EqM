import os
from PIL import Image
from torch.utils.data import Dataset
from glob import glob

class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob(os.path.join(root_dir, "*.png"))
        self.image_paths += glob(os.path.join(root_dir, "*.jpg"))
        self.image_paths += glob(os.path.join(root_dir, "*.jpeg"))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or the next one
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # Return label 0 for all images as we are doing unconditional/single-class generation for now
        label = 0
        return image, label
