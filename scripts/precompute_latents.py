import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import PokemonDataset
from utils.vae import load_vae, encode_latents
from args import get_args

def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents")
    parser.add_argument("--data_path", type=str, default="data/raw", help="Path to raw images")
    parser.add_argument("--output_dir", type=str, default="data/latents", help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--vae_path", type=str, default="qwen_image_vae.safetensors", help="Path to VAE")
    parser.add_argument("--vae_type", type=str, default="ema", help="VAE type")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE
    vae = load_vae(args.vae_path, args.vae_type, device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    # Data
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = PokemonDataset(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Found {len(dataset)} images. Precomputing latents...")
    
    count = 0
    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            latents = encode_latents(vae, x, device)
            
            # Save each latent individually to match dataset structure
            # This allows for shuffling and random access during training
            for i in range(latents.size(0)):
                latent = latents[i].cpu()
                save_path = os.path.join(args.output_dir, f"{count:05d}.pt")
                torch.save(latent, save_path)
                count += 1
                
    print(f"Saved {count} latents to {args.output_dir}")

if __name__ == "__main__":
    main()
