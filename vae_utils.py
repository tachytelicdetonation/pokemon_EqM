import torch
import logging
from diffusers import AutoencoderKL, AutoencoderKLWan

logger = logging.getLogger(__name__)

def load_vae(vae_path=None, vae_type="ema", device="cpu"):
    """
    Load VAE model from local path or HuggingFace hub.
    """
    if vae_path:
        logger.info(f"Loading VAE from {vae_path}")
        try:
            vae = AutoencoderKLWan.from_single_file(vae_path).to(device)
            logger.info("Loaded Wan/Qwen VAE")
            return vae
        except Exception:
            logger.info("Failed to load as Wan VAE, trying standard AutoencoderKL")
            vae = AutoencoderKL.from_single_file(vae_path).to(device)
            return vae
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
        return vae

def encode_latents(vae, x, device):
    """
    Encode images to latents, handling different VAE architectures.
    """
    if isinstance(vae, AutoencoderKLWan):
        # Wan VAE expects (B, C, T, H, W)
        x_in = x.unsqueeze(2)
        posterior = vae.encode(x_in).latent_dist
        # Apply scaling factor 0.661 to normalize to unit variance (measured std ~1.51)
        latents = posterior.sample().squeeze(2).mul_(0.661)
    else:
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample().mul_(0.18215)
    return latents

def decode_latents(vae, latents, device):
    """
    Decode latents to images, handling different VAE architectures.
    """
    if isinstance(vae, AutoencoderKLWan):
        # Wan VAE expects (B, C, T, H, W)
        latents_in = latents.unsqueeze(2)
        # Apply inverse scaling factor (1/0.661)
        samples = vae.decode(latents_in / 0.661).sample
        samples = samples.squeeze(2) # (B, C, H, W)
    else:
        samples = vae.decode(latents / 0.18215).sample
    return samples
