
import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EqM_models
import generate_pokemon # This verifies syntax of generate_pokemon.py

def test_model_forward():
    print("Testing EqM model forward pass...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    # Test XL model to check for parameter count issues
    model = EqM_models['EqM-XL/2'](input_size=4, num_classes=1).to(device)
    
    B = 2
    C = 4 # Latent channels
    H = 4 # Latent size (input_size)
    W = 4
    
    x = torch.randn(B, C, H, W).to(device)
    t = torch.rand(B).to(device)
    y = torch.zeros(B, dtype=torch.long).to(device)
    
    # Forward pass
    try:
        out = model(x, t, y)
        print(f"Forward pass successful. Output shape: {out.shape}")
        assert out.shape == (B, C, H, W) # forward returns only mean even if learn_sigma=True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

    print("Test passed!")

if __name__ == "__main__":
    test_model_forward()
