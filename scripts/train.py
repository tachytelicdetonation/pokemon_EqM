# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM (Single GPU/CPU).
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pokemon_eqm.models import EqM_models
from pokemon_eqm.utils.download import find_model
from pokemon_eqm.transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from pokemon_eqm.utils import wandb as wandb_utils
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F

class CenterCrop:
    def __init__(self, image_size):
        self.image_size = image_size
    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0 # Dummy label

# Try to import utils.vae if available, otherwise fallback
try:
    from pokemon_eqm.utils.vae import load_vae, encode_latents, decode_latents
    USE_UTILS_VAE = True
except ImportError:
    USE_UTILS_VAE = False


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def compute_grad_stats(model, device):
    """
    Compute gradient statistics (GSNR, norm, etc.) for monitoring.
    """
    total_norms = []
    nonzero = 0
    total = 0
    
    sum_grads = 0.0
    sum_sq_grads = 0.0
    count_grads = 0

    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.data
            param_norm = g.norm(2).item()
            total_norms.append(param_norm)

            nonzero += torch.count_nonzero(g).item()
            total += g.numel()

            # Accumulate statistics for GSNR (memory efficient)
            sum_grads += g.sum().item()
            sum_sq_grads += (g ** 2).sum().item()
            count_grads += g.numel()

    if not total_norms:
        return {
            "total_norm": 0.0,
            "nonzero": 0,
            "total": 0,
            "max_norm": 0.0,
            "gsnr": 0.0,
            "grad_mean": 0.0,
            "grad_var": 0.0,
        }

    total_norm_tensor = torch.tensor(total_norms, device=device)
    total_norm = torch.norm(total_norm_tensor, 2).item()
    max_norm = max(total_norms)

    grad_mean = sum_grads / count_grads if count_grads > 0 else 0.0
    grad_var = (sum_sq_grads / count_grads - grad_mean ** 2) if count_grads > 0 else 0.0

    epsilon = 1e-10
    gsnr = (grad_mean ** 2) / (grad_var + epsilon)

    return {
        "total_norm": total_norm,
        "nonzero": nonzero,
        "total": total,
        "max_norm": max_norm,
        "gsnr": gsnr,
        "grad_mean": grad_mean,
        "grad_var": grad_var,
    }


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new EqM model.
    """
    # Setup Device
    # Setup Device
    device_name = getattr(args, "device", None)
    if device_name:
        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested via config/args but not available.")
            device = torch.device("cuda")
        elif device_name == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested via config/args but not available.")
            device = torch.device("mps")
        else:
            device = torch.device(device_name)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set seed
    seed = args.global_seed
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
    experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                    f"{args.path_type}-{args.prediction}-{args.loss_weight}"
    experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = f"{experiment_dir}/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    entity = os.environ.get("ENTITY")
    project = os.environ.get("PROJECT", "pokemon-eqm")
    if args.wandb:
        wandb_utils.initialize(args, entity, experiment_name, project)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
        use_liere=args.use_liere,
        liere_jitter_std=args.liere_jitter_std,
        liere_jitter_mode=getattr(args, 'liere_jitter_mode', 'gaussian'),
        liere_pos_embed_shift=getattr(args, 'liere_pos_embed_shift', None),
        liere_pos_embed_jitter=getattr(args, 'liere_pos_embed_jitter', None),
        liere_pos_embed_rescale=getattr(args, 'liere_pos_embed_rescale', 2.0)
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Load checkpoint if provided (not via args but maybe hardcoded or future feature)
    # For now, we only support resume via manual code change or if we added it to config
    # But user asked to remove args, so we rely on config.
    
    requires_grad(ema, False)
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    
    # Load VAE
    if USE_UTILS_VAE:
        vae = load_vae(None, args.vae, device) # vae_path is None in config usually
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose([
        CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    try:
        dataset = ImageFolder(args.data_path, transform=transform)
    except:
        # Fallback for flat directory
        dataset = FlatFolderDataset(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(args.num_classes, size=(args.batch_size,), device=device)
    
    # Fixed noise for consistent sampling
    fixed_noise = torch.randn(args.sample_batch_size, 4, latent_size, latent_size, device=device)
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                if USE_UTILS_VAE:
                    x = encode_latents(vae, x, device)
                else:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            model_kwargs = dict(y=y, return_act=False, train=True)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # Compute gradient stats for GSNR
            grad_stats = compute_grad_stats(model, device)
            
            # Log to wandb every step
            if args.wandb:
                wandb_utils.log({
                    "train/loss": loss.item(),
                    "grad/gsnr": grad_stats["gsnr"],
                    "grad/norm": grad_stats["total_norm"],
                    "grad/mean": grad_stats["grad_mean"],
                    "grad/var": grad_stats["grad_var"],
                }, step=train_steps)

            if train_steps % args.log_every == 0:
                # Measure training speed:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train/avg_loss": avg_loss, "train/steps_per_sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save EqM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Sampling (Fixed and Random)
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info(f"Generating samples at step {train_steps}...")
                model.eval()
                
                # Helper to sample and log
                def generate_and_log(noise, prefix):
                    # Use ODE sampler (dopri5) for better quality
                    sampler_fn = transport_sampler.sample_ode(
                        sampling_method="dopri5", 
                        num_steps=50
                    )
                    
                    # Create labels
                    sample_y = torch.zeros(noise.shape[0], dtype=torch.long, device=device)
                    
                    # CFG
                    if args.cfg_scale > 1.0:
                        noise_in = torch.cat([noise, noise], 0)
                        y_null = torch.tensor([args.num_classes] * noise.shape[0], device=device)
                        y_in = torch.cat([sample_y, y_null], 0)
                        
                        model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)
                        model_fn = ema.forward_with_cfg
                    else:
                        noise_in = noise
                        model_kwargs = dict(y=sample_y)
                        model_fn = ema.forward
                        
                    with torch.no_grad():
                        samples = sampler_fn(noise_in, model_fn, **model_kwargs)[-1]
                        if args.cfg_scale > 1.0:
                            samples, _ = samples.chunk(2, dim=0)
                            
                        # Decode
                        if USE_UTILS_VAE:
                            imgs = decode_latents(vae, samples, device)
                        else:
                            imgs = vae.decode(samples / 0.18215).sample
                            
                        imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                        
                        # Save images
                        prefix_dir = f"{sample_dir}/step_{train_steps:07d}/{prefix}"
                        os.makedirs(prefix_dir, exist_ok=True)
                        
                        wandb_images = []
                        for i, img in enumerate(imgs):
                            Image.fromarray(img).save(f"{prefix_dir}/{i:05d}.png")
                            if i < 8:
                                wandb_images.append(wandb_utils.wandb.Image(img, caption=f"{prefix}_{i}"))
                        
                        if args.wandb:
                            wandb_utils.log({f"{prefix}/samples": wandb_images}, step=train_steps)

                # Fixed noise
                generate_and_log(fixed_noise, "fixed")
                
                # Random noise
                random_noise = torch.randn_like(fixed_noise)
                generate_and_log(random_noise, "random")
                
                model.train()
                
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--run-id", type=str, default=None, help="WandB run ID for resuming")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    args = parser.parse_args()
    
    # Load config from JSON
    config_path = args.config
    if not config_path.endswith(".json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "configs", f"{config_path}.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Update args with config values
            for key, value in config.items():
                setattr(args, key, value)
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")

    main(args)
