import os
import torch
import random
from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms

class MixedLatentMasking:
    """
    Applies a mix of large structural masks (Cutout-style) and 
    many small masks (Dropout-style) to the latent tensor.
    """
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, x):
        if random.random() > self.prob:
            return x
            
        c, h, w = x.shape
        mask = torch.ones((1, h, w), device=x.device)
        
        # Randomly choose a masking strategy:
        # 0: Large Patches Only (Structural)
        # 1: Small Patches Only (Texture)
        # 2: Mixed (Both)
        strategy = random.randint(0, 2)
        
        # 1. Large Patches (Structural removal)
        if strategy == 0 or strategy == 2:
            num_large = random.randint(1, 3)
            for _ in range(num_large):
                ph = random.randint(h // 4, h // 2)
                pw = random.randint(w // 4, w // 2)
                
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                mask[:, top:top+ph, left:left+pw] = 0
            
        # 2. Small Patches (Texture/Noise removal)
        if strategy == 1 or strategy == 2:
            num_small = random.randint(4, 16)
            for _ in range(num_small):
                ph = random.randint(1, h // 8)
                pw = random.randint(1, w // 8)
                
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                mask[:, top:top+ph, left:left+pw] = 0
            
        return x * mask

class LatentDataset(Dataset):
    def __init__(self, root_dir, load_to_memory=True):
        self.root_dir = root_dir
        self.latent_paths = sorted(glob(os.path.join(root_dir, "*.pt")))
        self.load_to_memory = load_to_memory
        
        print(f"Found {len(self.latent_paths)} latents in {root_dir}")
        
        self.latents = []
        if self.load_to_memory:
            print("Loading latents into memory...")
            for path in self.latent_paths:
                self.latents.append(torch.load(path))
            print("Latents loaded.")
            
        # Latent space augmentation
        # 1. Random Horizontal Flip
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        
        # 2. Mixed Masking (Small & Large patches)
        self.masking = MixedLatentMasking(prob=0.5)

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        if self.load_to_memory:
            latent = self.latents[idx]
        else:
            latent = torch.load(self.latent_paths[idx])
            
        # Apply augmentations
        latent = self.flip(latent)
        latent = self.masking(latent)
            
        # Return label 0 for all images
        label = 0
        return latent, label
