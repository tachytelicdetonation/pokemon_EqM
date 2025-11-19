import torch
import sys
import os
import argparse
import logging
from PIL import Image
import numpy as np
from copy import deepcopy

# Add ref directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ref')))

from models import EqM_models
from diffusers.models import AutoencoderKL

def main(args):
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    # Load checkpoint
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "ema" in checkpoint:
            model.load_state_dict(checkpoint["ema"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    
    # VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Sampling
    os.makedirs(args.output_dir, exist_ok=True)
    
    n = args.num_samples
    # Split into batches
    batch_size = args.batch_size
    num_batches = (n + batch_size - 1) // batch_size
    
    print(f"Generating {n} samples in {num_batches} batches...")
    
    total_generated = 0
    for i in range(num_batches):
        current_batch_size = min(batch_size, n - total_generated)
        
        z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
        # Label 0 for pokemon
        y = torch.zeros(current_batch_size, dtype=torch.long, device=device)
        t = torch.ones(current_batch_size, device=device)
        
        if args.cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * current_batch_size, device=device) # Assuming args.num_classes is null class
            y = torch.cat([y, y_null], 0)
            t = torch.cat([t, t], 0)
            model_fn = model.forward_with_cfg
        else:
            model_fn = model.forward

        xt = z
        m = torch.zeros_like(xt).to(xt)
        
        # Detach to prevent graph accumulation
        xt = xt.detach()
        if args.sampler == 'ngd':
            m = m.detach()

        # Sampling loop (Gradient Descent)
        # Only use no_grad if we are not using an EBM (which requires gradients)
        context = torch.no_grad() if args.ebm == 'none' else torch.enable_grad()
        with context:
            for step in range(args.num_sampling_steps):
                if args.sampler == 'gd':
                    out = model_fn(xt, t, y, args.cfg_scale) if args.cfg_scale > 1.0 else model_fn(xt, t, y)
                    if isinstance(out, tuple): out = out[0]
                elif args.sampler == 'ngd':
                    x_ = xt + args.stepsize * m * args.mu
                    out = model_fn(x_, t, y, args.cfg_scale) if args.cfg_scale > 1.0 else model_fn(x_, t, y)
                    if isinstance(out, tuple): out = out[0]
                    m = out
                
                xt = xt + out * args.stepsize
                # t += args.stepsize # Time conditioning might be fixed or evolving? In EqM paper, it's equilibrium, so t might not matter or be fixed.
                # Ref code: t += args.stepsize. But wait, EqM is equilibrium, so maybe t is just a dummy or used for annealing?
                # Ref code: t += args.stepsize
                # Let's follow ref code.
                t += args.stepsize
                
                # Detach intermediate results to save memory if not needing gradients for next step (usually true for sampling)
                # Even with EBM, we usually take a step and then detach for the next iteration unless we are doing BPTT through time (unlikely here)
                xt = xt.detach()
                if args.sampler == 'ngd':
                    m = m.detach()

        if args.cfg_scale > 1.0:
            xt, _ = xt.chunk(2, dim=0)
            
        # Decode
        with torch.no_grad():
            samples = vae.decode(xt / 0.18215).sample
        
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        for j, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{args.output_dir}/{total_generated + j:05d}.png")
        
        total_generated += current_batch_size
        print(f"Generated {total_generated}/{n} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default="generated_samples")
    parser.add_argument("--sampler", type=str, default='gd', choices=['gd', 'ngd'])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--uncond", type=bool, default=False)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    
    args = parser.parse_args()
    main(args)
