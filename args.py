import argparse
import json
import os
from models import EqM_models

def get_args():
    parser = argparse.ArgumentParser(description="Pokemon EqM Training Arguments")
    parser.add_argument("--config", type=str, default="test", help="Path to JSON config file or name of config (production/test)")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint for generation/resuming")
    parser.add_argument("--run_id", type=str, help="Run ID to resume from (or directory name)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling/training. Omit for stochastic runs.")
    
    # Overrides
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--log_every", type=int, help="Override log frequency")
    parser.add_argument("--cache_latents", action="store_true", help="Enable latent caching")
    
    args = parser.parse_args()

    # Default configuration
    config = {
        "data_path": "data/raw",
        "results_dir": "results",
        "model": "EqM-XL/2",
        "image_size": 256,
        "num_classes": 1,
        "epochs": 1000,
        "batch_size": 32,
        "lr_warmup_steps": 500,
        "lr": 1e-4,
        "weight_decay": 0,
        "num_workers": 4,
        "log_every": 100,
        "ckpt_every": 1000,
        "run_id": None,
        "ckpt": None,
        "vae": "ema",
        "vae_path": "qwen_image_vae.safetensors",
        "vae_channels": 16,
        "cfg_scale": 1.5,
        "cfg_scale_cap": 1.5,
        "uncond": False,
        "energy_head": "implicit",
        "path_type": "Linear",
        "prediction": "velocity",
        "loss_weight": "velocity",
        "sample_eps": None,
        "train_eps": None,
        "num_samples": 5,
        # Sampling step size: leave None to auto-span [sample_eps, 1] over num_sampling_steps.
        # Override in JSON config if you need a custom fixed dt.
        "stepsize": None,
        "num_sampling_steps": 250,
        "sampler": "ngd",
        "mu": 0.3,
        "stepsize_mult": 1.0,
        "sample_batch_size": 8,
        "output_dir": "generated_samples",
        # Default seed for deterministic sampling/training; override via config or --seed
        "seed": 42,
        "log_images": True,
        "watch_grads": False,
        # Gradient clipping for training stability (common for diffusion models: 0.1-1.0)
        # Set to 0 to disable, or positive value to enable
        "grad_clip": 1.0,
        "log_layer_grads": True,  # Log per-layer gradient stats at checkpoint time
        "log_layer_histogram": False,  # Log histogram of layer gradient norms (expensive)
        # Adaptive early-stop for sampling is off by default to avoid zero-step samples on fresh models.
        "adaptive": False,
        "min_adaptive_steps": 10,
        "grad_thresh": 10.0,
        "sigreg_lambda": 0.1,
        "noised_input_path": None,
        "metrics_subset_size": 16,
        # Training optimizations (H100/H200 optimized defaults)
        "mixed_precision": "fp8",  # Options: "fp8", "bf16", "fp16", or False/None for fp32
        "compile": True,  # Enable torch.compile for faster execution
        "compile_mode": "max-autotune",  # Options: "default", "reduce-overhead", "max-autotune"
        "cache_latents": True,  # Cache VAE-encoded latents (requires preprocessing)
        # Positional Encodings: RoPE or LieRE (cannot use both simultaneously)
        "use_rope": False,  # Enable 2D Axial RoPE (fixed rotations for H and W dimensions)
        "rope_base": 10000,  # Base frequency for RoPE (higher = better long-range modeling)
        "use_liere": True,  # Enable LieRE (learnable rotation matrices via Lie algebra)
                           # LieRE is now the default - it learns optimal positional encodings
                           # Reference: https://arxiv.org/abs/2406.10322 (ICML 2025)
    }

    # Load config from JSON
    config_path = args.config
    if not config_path.endswith(".json"):
        # Assume it's a config name in the configs directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "configs", f"{config_path}.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            json_config = json.load(f)
            config.update(json_config)
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")

    # Backward compatibility: accept legacy 'ebm' key
    if 'ebm' in config and 'energy_head' not in config:
        config['energy_head'] = config.pop('ebm')
    else:
        config.pop('ebm', None)

    # Override with runtime args if provided
    if args.ckpt:
        config['ckpt'] = args.ckpt
    if args.run_id:
        config['run_id'] = args.run_id
    if args.seed is not None:
        config['seed'] = args.seed
        
    # Apply overrides
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.log_every is not None:
        config['log_every'] = args.log_every
    if args.cache_latents:
        config['cache_latents'] = True
    # Default to stochastic behavior if seed is absent in config
    if 'seed' not in config:
        config['seed'] = None
    # Treat negative seed values as a signal for stochastic runs
    if isinstance(config.get('seed'), int) and config['seed'] < 0:
        config['seed'] = None

    # Normalize legacy sentinel
    if str(config.get('energy_head', 'implicit')).lower() == 'none':
        config['energy_head'] = 'implicit'

    # Add config name for logging
    config['config'] = args.config

    # For backward compatibility downstream
    config['ebm'] = config.get('energy_head', 'implicit')

    # Convert to Namespace for backward compatibility
    return argparse.Namespace(**config)
