import wandb
import datetime
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
import subprocess


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def _safe_run(cmd):
    """Run a shell command and return stdout or None on failure."""
    try:
        out = subprocess.run(
            cmd,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


def _requirements_hash():
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    req_path = os.path.abspath(req_path)
    if not os.path.exists(req_path):
        return None
    with open(req_path, "rb") as f:
        contents = f.read()
    return hashlib.sha256(contents).hexdigest()[:12]


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    # Lightweight provenance (no files uploaded)
    git_commit = _safe_run("git rev-parse --short HEAD")
    git_status = _safe_run("git status --short")
    req_hash = _requirements_hash()
    config_dict.update(
        {
            "git_commit": git_commit,
            "git_status": git_status,
            "requirements_hash": req_hash,
        }
    )
    if "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])
    
    # Append timestamp to exp_name to ensure uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.run_id:
        run_id = args.run_id
        # If resuming, we might want to keep the old name or just use the run_id as name to avoid confusion
        # But wandb.init(id=...) will resume the run with that ID.
        # We can try to fetch the run name if we wanted, but for now let's just use the ID or a resume indicator
        run_name = f"{exp_name}_resume_{run_id}" 
        # Note: The actual run name in UI might not change if we resume, which is fine.
    else:
        run_name = f"{exp_name}_{timestamp}"
        run_id = generate_run_id(run_name)

    if args.config == 'testing':
        return run_id, run_name
    
    seed_tag = None
    if getattr(args, "seed", None) is not None:
        seed_tag = f"seed_{getattr(args, 'seed')}"

    tags = [
        str(getattr(args, "model", "")),
        f"size_{getattr(args, 'image_size', '')}",
        seed_tag,
        f"config_{getattr(args, 'config', '')}",
        os.path.basename(getattr(args, "data_path", "")) if getattr(args, "data_path", None) else None,
    ]
    tags = [t for t in tags if t]

    wandb.init(
        entity=entity,
        project=project_name,
        name=run_name if not args.run_id else None, # Don't overwrite name if resuming, or let wandb handle it
        config=config_dict,
        id=run_id,
        resume="allow",
        tags=tags,
        settings=wandb.Settings(code_dir="."),
    )
    # Metric hygiene
    wandb.define_metric("train_step")
    wandb.define_metric("*", step_metric="train_step")
    return run_id, run_name

def log(stats, step=None):
    if is_main_process() and wandb.run is not None:
        payload = {k: v for k, v in stats.items()}
        if step is not None and "train_step" not in payload:
            payload["train_step"] = step
        wandb.log(payload, step=step)


def log_image(sample, step=None):
    if is_main_process() and wandb.run is not None:
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def log_samples_table(rows, step=None, preview_image=None, preview_images=None):
    """
    rows: list of dicts with keys train_step, index, seed, label, file_path, sampler, num_steps, cfg_scale
    preview_image: optional single wandb.Image to log (deprecated, use preview_images)
    preview_images: optional list of wandb.Image to log separately
    """
    if not (is_main_process() and wandb.run is not None):
        return

    # Determine columns based on what's in the rows
    base_columns = ["train_step", "index", "seed", "label", "file_path"]
    extra_columns = ["sampler", "num_steps", "cfg_scale"]
    columns = base_columns + [col for col in extra_columns if any(col in row for row in rows)]

    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*(row.get(col) for col in columns))
    to_log = {"sample_trace": table}

    # Handle preview images
    if preview_images is not None:
        to_log["generated_samples"] = preview_images
    elif preview_image is not None:
        to_log["generated_sample"] = preview_image

    wandb.log(to_log, step=step)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
