import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_pokemon_data(output_dir="data/raw"):
    """
    Downloads the Pokemon dataset from Hugging Face and saves images to the output directory.
    """
    print(f"Downloading Pokemon dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
    
    print(f"Found {len(dataset)} images. Saving to disk...")
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        image = item["image"]
        # Save as PNG
        image.save(os.path.join(output_dir, f"pokemon_{i:04d}.png"))
        
    print("Download complete!")

if __name__ == "__main__":
    download_pokemon_data()
