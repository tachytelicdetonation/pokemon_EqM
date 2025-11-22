import torch
import logging
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms
import schedulefree

from models import EqM_models
from transport import create_transport
from utils.vae import load_vae
from datasets import PokemonDataset
from datasets.latents import LatentDataset
from args import get_args
from engine import Trainer

# Mixed precision support
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    GradScaler = None

# Float8 training support
try:
    from torchao.float8 import convert_to_float8_training
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    convert_to_float8_training = None

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def main():
    args = get_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Setup logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.vae_channels,
        uncond=args.uncond,
        energy_head=args.energy_head,
        use_rope=getattr(args, 'use_rope', True),
        rope_base=getattr(args, 'rope_base', 10000),
        use_liere=getattr(args, 'use_liere', False)
    ).to(device)

    # Apply FP8 conversion before compilation (if enabled)
    use_torchao_fp8 = False
    if getattr(args, 'mixed_precision', None) == 'fp8':
        if TORCHAO_AVAILABLE:
            from torchao.float8 import Float8LinearConfig, ScalingType

            # Use rowwise scaling for better accuracy (recommended best practice)
            # Options: "tensorwise" (fastest), "rowwise" (best accuracy/perf), "rowwise_with_gw_hp" (most accurate)
            config = Float8LinearConfig(
                scaling_type_input=ScalingType.DYNAMIC,
                scaling_type_weight=ScalingType.DYNAMIC,
                scaling_type_grad_output=ScalingType.DYNAMIC,
            )

            logger.info("Converting model to float8 training with torchao (rowwise scaling)...")
            convert_to_float8_training(model, config=config)
            use_torchao_fp8 = True
            logger.info("Float8 conversion complete")
        else:
            logger.warning("torchao not available, falling back to bfloat16")
            args.mixed_precision = 'bf16'

    # Compile model for faster execution (PyTorch 2.0+)
    if getattr(args, 'compile', False):
        logger.info(f"Compiling model with torch.compile (mode={getattr(args, 'compile_mode', 'default')})...")
        compile_mode = getattr(args, 'compile_mode', 'default')
        model = torch.compile(model, mode=compile_mode)
        logger.info("Model compilation complete")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    opt = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.lr_warmup_steps
    )

    # Transport
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )

    # VAE
    vae = load_vae(args.vae_path, args.vae, device)
    requires_grad(vae, False)

    # Data
    if getattr(args, 'cache_latents', False):
        import os
        latent_dir = os.path.join("data", "latents")
        if not os.path.exists(latent_dir) or not os.listdir(latent_dir):
            logger.info("Latents not found. Running precomputation...")
            # We could run the script here, or just error out. 
            # For robustness, let's run the logic inline or call the script.
            # Calling the script via subprocess is safer to avoid import issues/state pollution
            import subprocess
            import sys
            subprocess.run([sys.executable, "scripts/precompute_latents.py", 
                            "--data_path", args.data_path, 
                            "--output_dir", latent_dir,
                            "--image_size", str(args.image_size),
                            "--vae_path", args.vae_path], check=True)
            
        dataset = LatentDataset(latent_dir, load_to_memory=True)
        logger.info(f"Using cached latents from {latent_dir}")
    else:
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = PokemonDataset(args.data_path, transform=transform)
        logger.info("Using raw images (on-the-fly encoding)")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset contains {len(dataset)} items")

    # Scheduler
    lr_scheduler = None

    # Setup mixed precision training
    scaler = None
    amp_dtype = None
    use_amp = getattr(args, 'mixed_precision', None)

    if use_amp and device.type == 'cuda':
        # For FP8 with torchao, we don't use autocast (torchao handles it internally)
        if use_torchao_fp8:
            amp_dtype = None  # No autocast for torchao FP8
            logger.info("Using float8 training with torchao (no autocast)")
        else:
            # Map precision string to dtype for autocast
            dtype_map = {
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            amp_dtype = dtype_map.get(use_amp, torch.bfloat16)

            # GradScaler only needed for fp16
            if use_amp == 'fp16' and GradScaler is not None:
                scaler = GradScaler()
                logger.info("Using mixed precision training: fp16 with GradScaler")
            else:
                logger.info(f"Using mixed precision training: {use_amp} (dtype={amp_dtype})")

    model.train()
    opt.train()
    ema.eval()
    # Trainer
    trainer = Trainer(model, ema, opt, transport, vae, loader, lr_scheduler, device, args,
                     scaler=scaler, amp_dtype=amp_dtype)
    
    # Resume logic
    start_epoch = 0
    if args.run_id:
        # Try to find checkpoint
        import glob
        import os
        ckpts = glob.glob(os.path.join(trainer.checkpoint_dir, "*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            resume_step = trainer.load_checkpoint(latest_ckpt)
            steps_per_epoch = len(loader) # Ensure steps_per_epoch is defined for resume logic
            start_epoch = resume_step // steps_per_epoch
        else:
            logger.warning("No checkpoints found, starting from scratch.")

    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()
