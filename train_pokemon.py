import torch
import sys
import os
import argparse
import logging
from time import time
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from copy import deepcopy



from models import EqM_models
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
import wandb_utils

from dataset import PokemonDataset
from generate_pokemon import sample_pokemon
import glob

def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def main(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_name = f"pokemon-eqm-{args.model}"
    
    # Initialize WandB
    run_id = None
    run_name = None
    try:
        run_id, run_name = wandb_utils.initialize(args, entity=None, exp_name=experiment_name, project_name="pokemon-eqm")
    except Exception as e:
        logger.warning(f"WandB initialization failed: {e}")
        # Fallback if wandb fails or is not used
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment_name}_{timestamp}"
        run_id = run_name # Use name as ID if wandb fails

    if args.run_id:
        run_id = args.run_id
        # If resuming, we might want to fetch the run name associated with this ID if possible, 
        # or just assume the directory structure matches.
        # For simplicity, let's assume if run_id is provided, we look for that specific directory.
        # But wait, the directory structure is results/experiment_name/checkpoints
        # It doesn't seem to use run_id in the path in the original code.
        # We need to change the directory structure to include run_id or run_name.
        pass

    # Update experiment dir to include run_name/id to avoid collisions
    # If args.run_id is provided, we try to find the existing directory.
    
    if args.run_id:
        # Search for directory containing run_id
        # Assuming structure: results/run_name (where run_name contains timestamp or is unique)
        # Or results/experiment_name/run_id
        # Let's stick to a cleaner structure: results/run_name
        # But run_name is generated inside initialize.
        
        # If resuming, we expect the user to provide the run_id (which might be the wandb run id).
        # But our directory might be named after run_name.
        # Let's assume the user provides the full run_name or we search for it.
        # Actually, let's change the structure to results/run_name
        
        # If args.run_id is passed, we assume it's the run_name (folder name) or we search for it.
        potential_dirs = glob.glob(os.path.join(args.results_dir, f"*{args.run_id}*"))
        if potential_dirs:
            experiment_dir = potential_dirs[0]
            logger.info(f"Resuming from directory: {experiment_dir}")
        else:
            logger.warning(f"Could not find directory for run_id {args.run_id}. Creating new one.")
            experiment_dir = os.path.join(args.results_dir, run_name)
    else:
        experiment_dir = os.path.join(args.results_dir, run_name)

    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_dir = os.path.join(experiment_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)



    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    # Initialize model
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    start_epoch = 0
    resume_step = 0
    
    # Resume logic
    if args.run_id:
        # Find latest checkpoint
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            logger.info(f"Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            opt.load_state_dict(checkpoint["opt"])
            # args = checkpoint["args"] # Don't overwrite args, might want to change some
            
            # Extract step from filename if possible (0000100.pt)
            try:
                resume_step = int(os.path.basename(latest_ckpt).split('.')[0])
                start_epoch = resume_step // (len(dataset) // args.batch_size) # Approx
            except:
                pass
        else:
            logger.warning("No checkpoints found in directory, starting from scratch.")

    # Transport
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    requires_grad(vae, False)

    # Data
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = PokemonDataset(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset contains {len(dataset)} images")

    model.train()
    ema.eval()

    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    
    # Generate initial samples
    logger.info("Generating initial samples...")
    # Temporarily override batch_size for sampling to avoid OOM
    sample_args = deepcopy(args)
    sample_args.batch_size = args.sample_batch_size
    sample_pokemon(ema, vae, sample_args, device, num_samples=5, output_dir=os.path.join(sample_dir, f"step_{train_steps:07d}"))

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # VAE encode expects (B, C, H, W)
                posterior = vae.encode(x).latent_dist
                x = posterior.sample().mul_(0.18215)
            
            model_kwargs = dict(y=y, return_act=False, train=True)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                
                wandb_utils.log({"train_loss": avg_loss, "steps_per_sec": steps_per_sec}, step=train_steps)
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Generate samples
                logger.info(f"Generating samples at step {train_steps}...")
                # Temporarily override batch_size for sampling to avoid OOM
                sample_args = deepcopy(args)
                sample_args.batch_size = args.sample_batch_size
                sample_pokemon(ema, vae, sample_args, device, num_samples=5, output_dir=os.path.join(sample_dir, f"step_{train_steps:07d}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2") # Default to XL model
    parser.add_argument("--image-size", type=int, default=256) # Smaller image size for Pokemon
    parser.add_argument("--num-classes", type=int, default=1) # Single class for now
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--uncond", type=bool, default=False) # We pass labels (0), so not purely uncond in model sense, but effectively yes.
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    
    # Transport args
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    parser.add_argument("--loss-weight", type=str, default="velocity", choices=[None, "velocity", "likelihood"])
    parser.add_argument("--sample-eps", type=float)
    parser.add_argument("--train-eps", type=float)
    parser.add_argument("--run_id", type=str, help="Run ID to resume from (or directory name)")
    
    # Sampling args for training loop
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default='gd', choices=['gd', 'ngd', 'ode_dopri5', 'ode_euler', 'ode_heun'])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--sample-batch-size", type=int, default=1, help="Batch size for sample generation (smaller to avoid OOM)")
    parser.add_argument("--output-dir", type=str, default="generated_samples") # Used by sample_pokemon but overridden in loop

    args = parser.parse_args()
    main(args)
