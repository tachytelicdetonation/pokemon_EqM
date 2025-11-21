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
from args import get_args
from engine import Trainer

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def main():
    args = get_args()
    
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

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.vae_channels,
        uncond=args.uncond,
        energy_head=args.energy_head
    ).to(device)

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

    # Scheduler
    lr_scheduler = None

    model.train()
    opt.train()
    ema.eval()
    # Trainer
    trainer = Trainer(model, ema, opt, transport, vae, loader, lr_scheduler, device, args)
    
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
