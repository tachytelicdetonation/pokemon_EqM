# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM (Single GPU/CPU).
"""

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import math
from pathlib import Path

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import to_pil_image

from pokemon_eqm.models import EqM_models
from pokemon_eqm.transport import Sampler, create_transport
from pokemon_eqm.utils import wandb as wandb_utils
from pokemon_eqm.utils.download import find_model
from pokemon_eqm.utils.visualization import AttentionVisualizer


def compute_attention_metrics_only(
    model, sample_input, sample_t, sample_y, num_registers=0, use_diff_attn=False
):
    """
    Compute attention quality metrics (lightweight, no visualization).

    Args:
        model: EqM model
        sample_input: Sample input tensor [B, C, H, W]
        sample_t: Timestep tensor [B]
        sample_y: Class label tensor [B]
        num_registers: Number of register tokens
        use_diff_attn: Whether using differential attention

    Returns:
        Dict of scalar metrics
    """
    from pokemon_eqm.utils.attention_metrics import (
        compute_all_metrics,
        get_scalar_metrics,
    )

    was_training = model.training
    model.eval()

    with torch.no_grad():
        # Get attention from middle layer
        _, attn_weights = model(
            sample_input,
            sample_t,
            sample_y,
            return_attention=True,
            attention_layer_idx=len(model.blocks) // 2,
        )

        # Handle differential attention format
        if use_diff_attn and isinstance(attn_weights, dict):
            attn_for_metrics = attn_weights["attn1"]  # [B, num_heads, N, N]
        else:
            if attn_weights.dim() == 3:
                attn_for_metrics = attn_weights.unsqueeze(0)
            else:
                attn_for_metrics = attn_weights

        N = attn_for_metrics.shape[-1]
        num_patches_total = N - num_registers if num_registers > 0 else N
        spatial_size = int(num_patches_total**0.5)

        # Compute metrics
        metrics = compute_all_metrics(
            attn=attn_for_metrics,
            spatial_size=spatial_size,
            num_registers=num_registers,
        )

        scalar_metrics = get_scalar_metrics(metrics)

    if was_training:
        model.train()

    return scalar_metrics


class CenterCrop:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA").convert("RGB")


class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = pil_loader(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label


# Try to import utils.vae if available, otherwise fallback
try:
    from pokemon_eqm.utils.vae import decode_latents, encode_latents, load_vae

    USE_UTILS_VAE = True
except ImportError:
    USE_UTILS_VAE = False


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):  # 0.9999 is the actual value
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
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


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
            sum_sq_grads += (g**2).sum().item()
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
    grad_var = (sum_sq_grads / count_grads - grad_mean**2) if count_grads > 0 else 0.0

    epsilon = 1e-10
    gsnr = (grad_mean**2) / (grad_var + epsilon)

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
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Setup an experiment folder:
    os.makedirs(
        args.results_dir, exist_ok=True
    )  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace(
        "/", "-"
    )  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
    experiment_name = (
        f"{experiment_index:03d}-{model_string_name}-"
        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
    )
    experiment_dir = (
        f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
    )
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = f"{experiment_dir}/samples"
    os.makedirs(sample_dir, exist_ok=True)
    viz_dir = f"{experiment_dir}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Initialize attention visualizer for GIF generation
    visualizer = AttentionVisualizer(save_dir=viz_dir, max_frames=500)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    entity = os.environ.get("ENTITY")
    project = os.environ.get("PROJECT", "pokemon-eqm")
    if args.wandb:
        wandb_utils.initialize(args, entity, experiment_name, project)

    # Create model:
    assert args.image_size % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder)."
    )
    latent_size = args.image_size // 8

    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
        use_liere=args.use_liere,
        liere_jitter_std=args.liere_jitter_std,
        liere_jitter_mode=getattr(args, "liere_jitter_mode", "gaussian"),
        liere_pos_embed_shift=getattr(args, "liere_pos_embed_shift", None),
        liere_pos_embed_jitter=getattr(args, "liere_pos_embed_jitter", None),
        liere_pos_embed_rescale=getattr(args, "liere_pos_embed_rescale", 2.0),
        num_registers=getattr(
            args, "num_registers", 0
        ),  # Register tokens for attention sinks
        use_diff_attn=getattr(args, "use_diff_attn", False),  # Differential attention
        use_spatial_decay=getattr(
            args, "use_spatial_decay", False
        ),  # Spatial attention decay
        spatial_decay_rate=getattr(args, "spatial_decay_rate", 0.1),  # Base decay rate
        spatial_decay_radius=getattr(
            args, "spatial_decay_radius", 0.25
        ),  # Base radius in normalized space
        use_per_head_decay=getattr(
            args, "use_per_head_decay", True
        ),  # ALiBi-style per-head decay
        decay_type=getattr(
            args, "decay_type", "exponential"
        ),  # 'exponential' or 'linear' decay
        distance_type=getattr(
            args, "distance_type", "l2"
        ),  # 'l2' (Euclidean) or 'l1' (Manhattan)
        use_content_gate=getattr(
            args, "use_content_gate", True
        ),  # Content-aware gating (SDT-style)
        # M-DGSA parameters
        use_matrix_lambda=getattr(args, "use_matrix_lambda", False),
        matrix_lambda_scale=getattr(args, "matrix_lambda_scale", 1.0),
        matrix_lambda_use_qk=getattr(args, "matrix_lambda_use_qk", True),
        # G1 output gating for attention sink elimination
        use_output_gate=getattr(args, "use_output_gate", False),
        output_gate_type=getattr(args, "output_gate_type", "per_token"),
        output_gate_init_bias=getattr(args, "output_gate_init_bias", -2.0),
        # σReparam for entropy collapse prevention
        use_sigma_reparam=getattr(args, "use_sigma_reparam", False),
        sigma_target=getattr(args, "sigma_target", 1.0),
        # Complexity bias for anti-curriculum attention
        use_complexity_bias=getattr(args, "use_complexity_bias", False),
        complexity_method=getattr(args, "complexity_method", "variance"),
        complexity_bias_scale=getattr(args, "complexity_bias_scale", 1.0),
        # Head routing for complexity specialization
        use_head_routing=getattr(args, "use_head_routing", False),
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    import prodigyopt

    opt = prodigyopt.Prodigy(
        model.parameters(),
        lr=1.0,
        weight_decay=0.01,  # recommended for diffusion models
        safeguard_warmup=True,
        use_bias_correction=True,  # recommended for diffusion models
        betas=(0.9, 0.99),  # helps with diffusion training
    )

    # Setup AMP
    mixed_precision = getattr(args, "mixed_precision", "no")
    # Handle boolean legacy config
    if isinstance(mixed_precision, bool):
        mixed_precision = "fp16" if mixed_precision else "no"

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    amp_dtype = torch.float32
    if mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        amp_dtype = torch.float16
    elif mixed_precision == "no":
        amp_dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed_precision value: {mixed_precision}")

    logger.info(f"Using mixed precision: {mixed_precision} (dtype: {amp_dtype})")

    # GradScaler is only needed for fp16
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == "fp16"))

    # Load checkpoint if provided (not via args but maybe hardcoded or future feature)
    # For now, we only support resume via manual code change or if we added it to config
    # But user asked to remove args, so we rely on config.

    requires_grad(ema, False)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        # Model architecture info
        num_registers=getattr(args, "num_registers", 0),
        # SIGReg parameters
        use_sigreg=getattr(args, "use_sigreg", False),
        sigreg_lambda=getattr(args, "sigreg_lambda", 0.05),
        sigreg_num_slices=getattr(args, "sigreg_num_slices", 1024),
        # Auxiliary attention losses (2025 research)
        use_aux_losses=getattr(args, "use_aux_losses", False),
        aux_entropy_floor_threshold=getattr(args, "aux_entropy_floor_threshold", 0.3),
        aux_entropy_floor_weight=getattr(args, "aux_entropy_floor_weight", 0.02),
        aux_entropy_ceiling_threshold=getattr(
            args, "aux_entropy_ceiling_threshold", 0.85
        ),
        aux_entropy_ceiling_weight=getattr(args, "aux_entropy_ceiling_weight", 0.02),
        aux_gate_entropy_weight=getattr(args, "aux_gate_entropy_weight", 0.01),
        aux_gate_sparsity_weight=getattr(args, "aux_gate_sparsity_weight", 0.005),
        aux_hsic_weight=getattr(args, "aux_hsic_weight", 0.01),
        aux_position_disagreement_weight=getattr(
            args, "aux_position_disagreement_weight", 0.005
        ),
        aux_lambda_smoothness_weight=getattr(
            args, "aux_lambda_smoothness_weight", 0.001
        ),
        aux_lambda_entropy_weight=getattr(args, "aux_lambda_entropy_weight", 0.01),
        aux_warmup_steps=getattr(args, "aux_warmup_steps", 1000),
        # Head specialization losses (2024-2025 research)
        aux_hard_focus_weight=getattr(args, "aux_hard_focus_weight", 0.02),
        aux_complexity_diversity_weight=getattr(args, "aux_complexity_diversity_weight", 0.01),
        aux_complexity_ortho_weight=getattr(args, "aux_complexity_ortho_weight", 0.01),
        aux_load_balance_weight=getattr(args, "aux_load_balance_weight", 0.005),
    )  # default: velocity;
    transport_sampler = Sampler(transport)

    # Load VAE
    if USE_UTILS_VAE:
        vae = load_vae(None, args.vae, device)  # vae_path is None in config usually
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(
            device
        )

    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    transform = transforms.Compose(
        [
            CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    try:
        dataset = ImageFolder(args.data_path, transform=transform, loader=pil_loader)
    except:
        # Fallback for flat directory
        dataset = FlatFolderDataset(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Resume logic
    start_epoch = 0
    train_steps = 0
    resume_path = None
    if args.run_id:
        # If run_id is provided, try to find the checkpoint
        # This is a bit specific to the user's setup, but let's look for 'latest.pt' in the experiment dir
        # We might need to search for the experiment dir based on run_id if we were using wandb run paths,
        # but here we assume we might be resuming from the same directory structure.
        # For now, let's look for a 'latest.pt' in the current experiment_dir if it exists,
        # or allow a specific --resume argument.
        pass  # Placeholder if we want complex logic

    # Check for latest.pt in the created experiment_dir (if we are restarting the same run)
    # Or if the user wants to resume from a specific path.
    # Let's add a simple check: if experiment_dir has latest.pt, load it.
    if os.path.exists(f"{checkpoint_dir}/latest.pt"):
        resume_path = f"{checkpoint_dir}/latest.pt"

    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        if "scaler" in checkpoint and mixed_precision:
            scaler.load_state_dict(checkpoint["scaler"])
        train_steps = checkpoint["train_steps"]
        start_epoch = train_steps // len(loader)
        logger.info(f"Resumed at step {train_steps}, epoch {start_epoch}")

    # Initialize Metrics
    calculate_fid = getattr(args, "calculate_fid", False)
    if calculate_fid:
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        logger.info("FID metric initialized.")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps is already set if resumed
    log_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(args.num_classes, size=(args.batch_size,), device=device)

    # Fixed noise for consistent sampling
    fixed_noise = torch.randn(
        args.sample_batch_size, 4, latent_size, latent_size, device=device
    )

    logger.info(f"Training for {args.epochs} epochs...")
    best_fid = float("inf")

    for epoch in range(start_epoch, args.epochs):
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

            # Enable return_act for dispersive loss (improves FID by ~20%)
            use_disp = getattr(args, "use_disp", True)
            model_kwargs = dict(y=y, return_act=use_disp, train=True)

            opt.zero_grad()

            with torch.amp.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=(mixed_precision != "no"),
            ):
                loss_dict = transport.training_losses(
                    model, x, model_kwargs, train_step=train_steps
                )
                loss = loss_dict["loss"].mean()

            scaler.scale(loss).backward()

            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Compute gradient stats for GSNR
            grad_stats = compute_grad_stats(model, device)

            # Calculate instantaneous steps/sec for WandB
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_end_time = time()
            # Avoid division by zero
            step_duration = (
                step_end_time - start_time
                if log_steps == 1
                else step_end_time - last_step_time
            )
            # For the very first step of a log interval, this might be slightly off if we don't track last_step_time globally
            # Let's just use a simple delta from the top of the loop?
            # Actually, the previous code used `start_time` which was reset every log_every steps.
            # To get per-step speed, we need to track time per step.

            # Let's use a simpler approach:
            # We need to capture time at the start of the step.
            # Since I'm editing a block in the middle, I can't easily insert at the top of the loop without a larger edit.
            # However, I can use `time()` here and compare to `last_step_time`.

            current_time = time()
            if "last_step_time" not in locals():
                last_step_time = start_time

            step_duration = current_time - last_step_time
            last_step_time = current_time

            # If step_duration is 0 (too fast), cap it
            if step_duration < 1e-6:
                step_duration = 1e-6

            current_steps_per_sec = 1.0 / step_duration

            # Log to wandb every step
            if args.wandb:
                log_dict = {
                    "train/loss": loss.item(),
                    "grad/gsnr": grad_stats["gsnr"],
                    "grad/norm": grad_stats["total_norm"],
                    "grad/mean": grad_stats["grad_mean"],
                    "grad/var": grad_stats["grad_var"],
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/scale": scaler.get_scale(),
                    "train/steps_per_sec": current_steps_per_sec,
                }
                # Add SIGReg loss if available
                if "sigreg_loss" in loss_dict:
                    sigreg_val = loss_dict["sigreg_loss"]
                    log_dict["train/sigreg_loss"] = (
                        sigreg_val.item() if torch.is_tensor(sigreg_val) else sigreg_val
                    )
                if "disp_loss" in loss_dict:
                    disp_val = loss_dict["disp_loss"]
                    log_dict["train/disp_loss"] = (
                        disp_val.item() if torch.is_tensor(disp_val) else disp_val
                    )

                # Add auxiliary attention losses if available (2025 research)
                aux_loss_keys = [
                    "aux_entropy_floor",
                    "aux_entropy_ceiling",
                    "aux_gate_entropy",
                    "aux_gate_sparsity",
                    "aux_hsic",
                    "aux_position_disagreement",
                    "aux_lambda_smoothness",
                    "aux_lambda_entropy",
                    "aux_total",
                ]
                for key in aux_loss_keys:
                    if key in loss_dict:
                        val = loss_dict[key]
                        log_dict[f"train/{key}"] = (
                            val.item() if torch.is_tensor(val) else val
                        )

                # Compute attention metrics (every step) and store viz data (periodically)
                attention_viz_every = getattr(
                    args, "attention_viz_every", 50
                )  # Store viz data every N steps
                if getattr(args, "compute_attention_metrics", True):
                    sample_x_attn = x[:1]  # Use first sample from batch
                    sample_t_attn = torch.rand(1, device=device)
                    sample_y_attn = y[:1]

                    # Get attention weights with full metrics
                    from pokemon_eqm.utils.attention_metrics import (
                        compute_all_metrics,
                        get_scalar_metrics,
                    )

                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        _, attn_weights = model(
                            sample_x_attn,
                            sample_t_attn,
                            sample_y_attn,
                            return_attention=True,
                            attention_layer_idx=len(model.blocks) // 2,
                        )

                        # Handle differential attention format
                        use_diff_attn = getattr(args, "use_diff_attn", False)
                        num_registers = getattr(args, "num_registers", 0)

                        if use_diff_attn and isinstance(attn_weights, dict):
                            attn_for_metrics = attn_weights["attn1"]
                        else:
                            if attn_weights.dim() == 3:
                                attn_for_metrics = attn_weights.unsqueeze(0)
                            else:
                                attn_for_metrics = attn_weights

                        N = attn_for_metrics.shape[-1]
                        num_patches_total = (
                            N - num_registers if num_registers > 0 else N
                        )
                        spatial_size = int(num_patches_total**0.5)

                        # Compute all metrics (every step - lightweight)
                        metrics = compute_all_metrics(
                            attn=attn_for_metrics,
                            spatial_size=spatial_size,
                            num_registers=num_registers,
                        )

                        # Add scalar metrics to log dict (logged every step)
                        scalar_metrics = get_scalar_metrics(metrics)
                        for k, v in scalar_metrics.items():
                            log_dict[f"attention/{k}"] = v

                        # Store visualization data periodically (not every step)
                        if train_steps % attention_viz_every == 0:
                            visualizer.add_metrics(train_steps, scalar_metrics)

                            if use_diff_attn and isinstance(attn_weights, dict):
                                attn1_viz = attn_weights["attn1"][0]
                                attn2_viz = attn_weights["attn2"][0]
                                # Store both: regular attention grid AND differential comparison
                                visualizer.store_attention_data(
                                    attn1_viz, train_steps, num_registers
                                )
                                visualizer.store_diff_attention_data(
                                    attn1_viz, attn2_viz, train_steps, num_registers
                                )
                            else:
                                attn_for_viz = (
                                    attn_weights[0]
                                    if attn_weights.dim() == 4
                                    else attn_weights
                                )
                                visualizer.store_attention_data(
                                    attn_for_viz, train_steps, num_registers
                                )

                            if "entropy/per_head" in metrics:
                                visualizer.store_entropy_data(
                                    metrics["entropy/per_head"], train_steps
                                )
                            if "diversity/similarity_matrix" in metrics:
                                visualizer.store_similarity_data(
                                    metrics["diversity/similarity_matrix"], train_steps
                                )

                    if was_training:
                        model.train()

                wandb_utils.log(log_dict, step=train_steps)

            if train_steps % args.log_every == 0:
                # Measure training speed (averaged):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train/avg_loss": avg_loss,
                            "train/avg_steps_per_sec": steps_per_sec,
                        },
                        step=train_steps,
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
                    "args": args,
                    "train_steps": train_steps,
                    "scaler": scaler.state_dict(),
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)

                # Save latest
                latest_path = f"{checkpoint_dir}/latest.pt"
                torch.save(checkpoint, latest_path)

                logger.info(f"Saved checkpoint to {checkpoint_path} and {latest_path}")

                # Generate and upload attention visualization GIFs at checkpoint
                if args.wandb:
                    logger.info(
                        f"Generating attention visualization GIFs at step {train_steps}..."
                    )
                    try:
                        gifs = visualizer.generate_all_gifs(
                            viz_dir, step_suffix=f"_step_{train_steps:07d}"
                        )
                        if gifs:
                            logger.info(
                                f"Generated {len(gifs)} GIFs: {list(gifs.keys())}"
                            )
                            visualizer.upload_gifs_to_wandb(
                                wandb_utils, gifs, train_steps
                            )
                            logger.info("Uploaded GIFs to wandb")
                        # Clear stored data to free memory (keep samples and metrics for final GIF)
                        visualizer.clear_data()
                    except Exception as e:
                        logger.warning(f"Failed to generate GIFs at checkpoint: {e}")

            # Sampling (Fixed and Random)
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info(f"Generating samples at step {train_steps}...")
                model.eval()

                # Helper to sample and log
                def generate_and_log(
                    noise,
                    prefix,
                    sampling_method="gd",
                    num_steps=250,
                    stepsize=0.0017,
                    mu=0.3,
                ):
                    # Create labels
                    n = noise.shape[0]
                    if "fixed" in prefix:
                        sample_y = torch.arange(n, device=device) % args.num_classes
                    else:
                        sample_y = torch.randint(
                            args.num_classes, size=(n,), device=device
                        )

                    # Setup classifier-free guidance (following reference implementation)
                    use_cfg = args.cfg_scale > 1.0
                    if use_cfg:
                        # Double the noise and labels for CFG
                        xt = torch.cat([noise, noise], 0)
                        t = torch.ones((n * 2,), device=device)
                        y_null = torch.tensor([args.num_classes] * n, device=device)
                        y = torch.cat([sample_y, y_null], 0)
                        model_fn = ema.forward_with_cfg
                    else:
                        xt = noise.clone()
                        t = torch.ones((n,), device=device)
                        y = sample_y
                        model_fn = ema.forward

                    # Sampling Loop (GD / NAG-GD) - following reference implementation
                    if sampling_method in ["gd", "ngd"]:
                        m = torch.zeros_like(xt)
                        with torch.no_grad():
                            for i in range(num_steps - 1):
                                if sampling_method == "gd":
                                    out = (
                                        model_fn(xt, t, y, args.cfg_scale)
                                        if use_cfg
                                        else model_fn(xt, t, y)
                                    )
                                    if not torch.is_tensor(out):
                                        out = out[0]
                                elif sampling_method == "ngd":
                                    x_ = xt + stepsize * m * mu
                                    out = (
                                        model_fn(x_, t, y, args.cfg_scale)
                                        if use_cfg
                                        else model_fn(x_, t, y)
                                    )
                                    if not torch.is_tensor(out):
                                        out = out[0]
                                    m = out

                                xt = xt + out * stepsize
                                t += stepsize

                            # Extract first half after sampling completes (reference implementation approach)
                            if use_cfg:
                                xt, _ = xt.chunk(2, dim=0)

                            samples = xt
                    else:
                        # Fallback to ODE/SDE if needed (legacy)
                        sampler_fn = transport_sampler.sample_ode(
                            sampling_method=sampling_method, num_steps=num_steps
                        )
                        if args.cfg_scale > 1.0:
                            noise_in = torch.cat([noise, noise], 0)
                            y_null = torch.tensor(
                                [args.num_classes] * noise.shape[0], device=device
                            )
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

                    imgs = (
                        torch.clamp(127.5 * imgs + 128.0, 0, 255)
                        .permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    # Save images
                    prefix_dir = f"{sample_dir}/step_{train_steps:07d}/{prefix}"
                    os.makedirs(prefix_dir, exist_ok=True)

                    wandb_images = []
                    for i, img in enumerate(imgs):
                        Image.fromarray(img).save(f"{prefix_dir}/{i:05d}.png")
                        if i < 8:
                            wandb_images.append(
                                wandb_utils.wandb.Image(img, caption=f"{prefix}_{i}")
                            )

                    if args.wandb:
                        wandb_utils.log(
                            {f"{prefix}/samples": wandb_images}, step=train_steps
                        )

                    return imgs

                # Fixed noise (Visuals - default sampler)
                fixed_samples = generate_and_log(fixed_noise, "fixed", "gd", 250)

                # Store samples for GIF generation
                if fixed_samples is not None:
                    visualizer.store_sample_data(fixed_samples, train_steps)

                # FID Sampler Configurations
                fid_configs = [
                    {
                        "method": "gd",
                        "steps": 250,
                        "name": "gd_250",
                        "stepsize": 0.0017,
                    },
                    {
                        "method": "ngd",
                        "steps": 250,
                        "name": "ngd_250",
                        "stepsize": 0.0017,
                        "mu": 0.3,
                    },
                ]

                # Generate random noise once for fair comparison
                random_noise = torch.randn_like(fixed_noise)

                # Pre-load real images for FID if needed
                real_images_tensor = None
                if calculate_fid:
                    try:
                        num_fid_samples = random_noise.shape[0]
                        logger.info(
                            f"Processing {num_fid_samples} real images for FID..."
                        )

                        # Create a temporary loader for one batch
                        fid_loader = DataLoader(
                            dataset,
                            batch_size=num_fid_samples,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False,
                        )

                        # Get one batch
                        x, _ = next(iter(fid_loader))
                        # Transform to uint8 [0, 255]
                        x = (x * 0.5 + 0.5) * 255
                        x = torch.clamp(x, 0, 255).to(torch.uint8)

                        if x.shape[0] > num_fid_samples:
                            x = x[:num_fid_samples]

                        real_images_tensor = x.to(device)
                    except Exception as e:
                        logger.warning(f"Failed to load real images for FID: {e}")

                # Loop over samplers
                for config in fid_configs:
                    method = config["method"]
                    steps = config["steps"]
                    name = config["name"]

                    stepsize = config.get("stepsize", 0.0017)
                    mu = config.get("mu", 0.3)

                    logger.info(
                        f"Generating samples for {name} ({method}, {steps} steps)..."
                    )
                    fake_imgs_np = generate_and_log(
                        random_noise, f"random_{name}", method, steps, stepsize, mu
                    )

                    # FID and IS Calculation
                    if calculate_fid and real_images_tensor is not None:
                        logger.info(f"Calculating FID for {name}...")

                        try:
                            # Convert numpy uint8 [N, H, W, 3] -> tensor uint8 [N, 3, H, W]
                            fake_imgs = (
                                torch.from_numpy(fake_imgs_np)
                                .permute(0, 3, 1, 2)
                                .to(device)
                            )

                            # Update metrics
                            fid_metric.update(fake_imgs, real=False)

                            # Update with real images
                            fid_metric.update(real_images_tensor, real=True)

                            # Compute FID
                            fid_score = fid_metric.compute().item()

                            logger.info(f"[{name}] FID: {fid_score:.4f}")

                            if args.wandb:
                                wandb_utils.log(
                                    {
                                        f"metrics/fid_{name}": fid_score,
                                    },
                                    step=train_steps,
                                )

                            # Save best FID (only for default gd_250)
                            if name == "gd_250" and fid_score < best_fid:
                                best_fid = fid_score
                                checkpoint = {
                                    "model": model.state_dict(),
                                    "ema": ema.state_dict(),
                                    "opt": opt.state_dict(),
                                    "args": args,
                                    "train_steps": train_steps,
                                    "scaler": scaler.state_dict(),
                                    "fid": fid_score,
                                }
                                torch.save(checkpoint, f"{checkpoint_dir}/best_fid.pt")
                                logger.info(
                                    f"New best FID: {best_fid:.4f}. Saved to best_fid.pt"
                                )

                        except Exception as e:
                            logger.warning(f"Failed to calculate FID for {name}: {e}")

                        # Reset metrics
                        fid_metric.reset()

                model.train()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Generate and upload final summary GIFs at end of training (samples evolution)
    logger.info("Generating final summary GIFs...")
    try:
        gifs = visualizer.generate_all_gifs(viz_dir, step_suffix="_final")
        if gifs:
            logger.info(f"Generated {len(gifs)} GIFs: {list(gifs.keys())}")
            if args.wandb:
                visualizer.upload_gifs_to_wandb(wandb_utils, gifs, train_steps)
                logger.info("Uploaded final GIFs to wandb")
    except Exception as e:
        logger.warning(f"Failed to generate final GIFs: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="WandB run ID for resuming"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    args = parser.parse_args()

    # Load config from JSON
    config_path = args.config
    if not config_path.endswith(".json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "configs", f"{config_path}.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            # Update args with config values
            for key, value in config.items():
                setattr(args, key, value)
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")

    main(args)
