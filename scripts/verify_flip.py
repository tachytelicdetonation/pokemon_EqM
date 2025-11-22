import torch
import os
import sys
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vae import load_vae, encode_latents, decode_latents

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE
    vae = load_vae("qwen_image_vae.safetensors", "ema", device)
    vae.eval()

    # Load an image (find one in data/raw)
    import glob
    img_paths = glob.glob("data/raw/*.jpg") + glob.glob("data/raw/*.png")
    if not img_paths:
        print("No images found in data/raw")
        return
    
    img_path = img_paths[0]
    print(f"Testing with image: {img_path}")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Encode original
        z = encode_latents(vae, img_tensor, device)
        
        # 2. Flip latent
        z_flipped = torch.flip(z, dims=[-1]) # Flip width dimension
        
        # 3. Decode flipped latent
        img_from_flipped_latent = decode_latents(vae, z_flipped, device)
        
        # 4. Flip original image (reference)
        img_flipped_ref = torch.flip(img_tensor, dims=[-1])
        
        # Save comparison
        # Denormalize for saving
        def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
        
        comparison = torch.cat([
            denorm(img_tensor), 
            denorm(img_from_flipped_latent), 
            denorm(img_flipped_ref)
        ], dim=0)
        
        save_image(comparison, "flip_verification.png")
        print("Saved flip_verification.png. Left: Original, Middle: Decoded from Flipped Latent, Right: Flipped Image")

if __name__ == "__main__":
    main()
