import argparse
import json
import os
from models import EqM_models

def get_args():
    parser = argparse.ArgumentParser(description="Pokemon EqM Training Arguments")
    parser.add_argument("--config", type=str, default="test", help="Path to JSON config file or name of config (production/test)")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint for generation/resuming")
    parser.add_argument("--run_id", type=str, help="Run ID to resume from (or directory name)")
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
        "log_every": 10,
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
        "stepsize": 0.0017,
        "num_sampling_steps": 250,
        "sampler": "ngd",
        "mu": 0.3,
        "stepsize_mult": 1.0,
        "sample_batch_size": 8,
        "output_dir": "generated_samples",
        "seed": 42,
        "log_images": True,
        "watch_grads": False,
        "adaptive": True,
        "grad_thresh": 10.0,
        "sigreg_lambda": 0.1,
        "noised_input_path": None,
        "metrics_subset_size": 16,
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

    # Normalize legacy sentinel
    if str(config.get('energy_head', 'implicit')).lower() == 'none':
        config['energy_head'] = 'implicit'

    # Add config name for logging
    config['config'] = args.config

    # For backward compatibility downstream
    config['ebm'] = config.get('energy_head', 'implicit')

    # Convert to Namespace for backward compatibility
    return argparse.Namespace(**config)
