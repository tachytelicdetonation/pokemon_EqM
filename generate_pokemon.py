import torch
import sys
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms



from models import EqM_models
from transport import create_transport, Sampler
from utils.vae import load_vae, decode_latents, encode_latents
from args import get_args

def _load_noised_latents(path: str, vae, args, device) -> torch.Tensor:
    """Load partially noised inputs (image files or .npy latents) and encode to latents."""
    p = Path(path)
    assert p.exists(), f"noised_input_path not found: {path}"

    def _prep_image(img_path: Path) -> torch.Tensor:
        tfm = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            latents = encode_latents(vae, x, device)
        return latents

    latents = []
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        tensor = torch.tensor(arr, device=device, dtype=torch.float32)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        latents.append(tensor)
    elif p.is_file():
        latents.append(_prep_image(p))
    else:
        files = sorted([f for f in p.iterdir() if f.is_file()])
        for f in files:
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                latents.append(_prep_image(f))
            elif f.suffix.lower() == ".npy":
                arr = np.load(f)
                tensor = torch.tensor(arr, device=device, dtype=torch.float32)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                latents.append(tensor)

    assert len(latents) > 0, f"Found no valid image/latent files in {path}"
    return torch.cat(latents, dim=0)


def sample_pokemon(model, vae, args, device, num_samples=None, output_dir=None, initial_noise=None, return_stats: bool=False):
    if num_samples is None:
        num_samples = args.num_samples
    if output_dir is None:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Transport setup (if needed for ODE)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )

    # Optional partial-noise path
    if initial_noise is None and getattr(args, "noised_input_path", None):
        initial_noise = _load_noised_latents(args.noised_input_path, vae, args, device)

    n = num_samples
    batch_size = args.batch_size
    num_batches = (n + batch_size - 1) // batch_size

    cfg_scale = min(args.cfg_scale, getattr(args, "cfg_scale_cap", args.cfg_scale))
    sample_eps = float(getattr(args, "sample_eps", 0.0) or 0.0)
    # If the user didn't tune stepsize explicitly, traverse [sample_eps, 1] evenly.
    user_stepsize = getattr(args, "stepsize", None)
    base_dt = (1.0 - sample_eps) / max(1, args.num_sampling_steps) if user_stepsize is None else user_stepsize
    stepsize = base_dt * getattr(args, "stepsize_mult", 1.0)
    energy_head = getattr(args, "energy_head", getattr(args, "ebm", "implicit"))
    adaptive = bool(getattr(args, "adaptive", False))
    min_adaptive_steps = int(getattr(args, "min_adaptive_steps", 0) or 0)
    grad_thresh = float(getattr(args, "grad_thresh", 10.0))
    collect_energy = energy_head != "implicit"

    print(f"Generating {n} samples in {num_batches} batches (sampler={args.sampler}, stepsize={stepsize:.5f}, mu={args.mu}, adaptive={adaptive})...")

    total_generated = 0

    grad_norm_sums = torch.zeros(args.num_sampling_steps, dtype=torch.float64)
    grad_norm_counts = torch.zeros(args.num_sampling_steps, dtype=torch.float64)
    all_step_counts = []
    all_energy = []
    sample_tensors = []

    for _ in range(num_batches):
        current_batch_size = min(batch_size, n - total_generated)

        # Determine latent channels from model or args
        in_channels = model.in_channels if hasattr(model, 'in_channels') else 4

        if initial_noise is not None:
            start_idx = total_generated
            end_idx = total_generated + current_batch_size
            idx_end = min(end_idx, initial_noise.shape[0])
            z = initial_noise[start_idx:idx_end].to(device)
            if z.shape[0] < current_batch_size:
                repeat = initial_noise[:current_batch_size - z.shape[0]].to(device)
                z = torch.cat([z, repeat], dim=0)
        else:
            z = torch.randn(current_batch_size, in_channels, args.image_size // 8, args.image_size // 8, device=device)

        y = torch.zeros(current_batch_size, dtype=torch.long, device=device)
        # Start integration at sample_eps (0 by default) and march forward to t=1.
        t = torch.full((current_batch_size,), sample_eps, device=device)

        cfg_on = cfg_scale > 1.0
        if cfg_on:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * current_batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            t = torch.cat([t, t], 0)

        xt = z.detach()
        m = torch.zeros_like(xt).to(xt)

        def model_forward(x_in, t_in):
            if cfg_on:
                return model.forward_with_cfg(x_in, t_in, y, cfg_scale, get_energy=collect_energy)
            return model.forward(x_in, t_in, y, get_energy=collect_energy)

        if args.sampler in ['gd', 'ngd']:
            context = torch.no_grad() if energy_head == 'implicit' else torch.enable_grad()
            with context:
                step_counts = torch.zeros(xt.shape[0], device=device, dtype=torch.int)
                finished = torch.zeros(xt.shape[0], device=device, dtype=torch.bool)
                last_energy = None

                for step in range(args.num_sampling_steps):
                    active = ~finished
                    if not active.any():
                        break

                    if args.sampler == 'gd':
                        out_obj = model_forward(xt, t)
                    else:  # ngd
                        x_pred = xt
                        if active.any():
                            x_pred = xt.clone()
                            x_pred[active] = xt[active] + stepsize * m[active] * args.mu
                        out_obj = model_forward(x_pred, t)

                    energy_batch = None
                    out = out_obj
                    if isinstance(out_obj, tuple):
                        out, energy_batch = out_obj
                        last_energy = energy_batch

                    grad_norm = out.view(out.size(0), -1).norm(dim=1)
                    grad_norm_sums[step] += grad_norm.detach().cpu().mean()
                    grad_norm_counts[step] += 1

                    if adaptive:
                        step_counts[active] += 1
                        if step + 1 >= min_adaptive_steps:
                            newly_finished = grad_norm < grad_thresh
                            finished = finished | newly_finished
                    else:
                        step_counts += 1

                    # Clamp dt to avoid overshooting t>1
                    dt = torch.full_like(t, stepsize)
                    remaining = (1.0 - t).clamp(min=0.0)
                    dt = torch.minimum(dt, remaining)

                    out = out * active.view(-1, 1, 1, 1)
                    xt = xt + out * dt.view(-1, 1, 1, 1)
                    t = t + active.float() * dt

                    xt = xt.detach()
                    if args.sampler == 'ngd':
                        m = out.detach()

                    # Mark samples that have reached t >= 1
                    finished = finished | (t >= 1.0 - 1e-6)

                per_sample_counts = step_counts[:current_batch_size] if cfg_on else step_counts
                all_step_counts.append(per_sample_counts.cpu())
                if last_energy is not None:
                    energy_main = last_energy[:current_batch_size] if cfg_on else last_energy
                    all_energy.append(energy_main.detach().cpu())

        elif args.sampler.startswith('ode'):
            sampler = Sampler(transport)

            if args.sampler == 'ode_dopri5':
                sample_fn = sampler.sample_ode(sampling_method='dopri5', num_steps=args.num_sampling_steps)
            elif args.sampler == 'ode_euler':
                sample_fn = sampler.sample_ode(sampling_method='euler', num_steps=args.num_sampling_steps)
            elif args.sampler == 'ode_heun':
                sample_fn = sampler.sample_ode(sampling_method='heun', num_steps=args.num_sampling_steps)
            else:
                raise ValueError(f"Unknown ODE sampler: {args.sampler}")

            model_kwargs = dict(y=y)
            if cfg_on:
                model_kwargs['cfg_scale'] = cfg_scale
                model_fn = model.forward_with_cfg
            else:
                model_fn = model.forward

            traj = sample_fn(z, model_fn, **model_kwargs)
            xt = traj[-1]
            all_step_counts.append(torch.full((current_batch_size,), args.num_sampling_steps, dtype=torch.int))

        if cfg_on:
            xt, _ = xt.chunk(2, dim=0)

        with torch.no_grad():
            samples = decode_latents(vae, xt, device)

        if return_stats:
            sample_tensors.append(samples.detach().cpu())

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        for j, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{output_dir}/{total_generated + j:05d}.png")

        total_generated += current_batch_size
        print(f"Generated {total_generated}/{n} samples.")

    images = [Image.open(f"{output_dir}/{i:05d}.png") for i in range(n)]

    if not return_stats:
        return images

    grad_trace = (grad_norm_sums / torch.clamp(grad_norm_counts, min=1)).tolist()
    step_counts_tensor = torch.cat(all_step_counts, dim=0) if all_step_counts else torch.zeros(0)
    energy_tensor = torch.cat(all_energy, dim=0) if all_energy else None
    stats = {
        "grad_norm_trace": grad_trace,
        "mean_steps": step_counts_tensor.float().mean().item() if step_counts_tensor.numel() else 0.0,
        "max_steps": step_counts_tensor.max().item() if step_counts_tensor.numel() else 0,
        "median_steps": step_counts_tensor.median().item() if step_counts_tensor.numel() else 0.0,
        "step_counts": step_counts_tensor,
        "energy": energy_tensor,
    }
    if sample_tensors:
        stats["samples_tensor"] = torch.cat(sample_tensors, dim=0)
    return images, stats


def main():
    args = get_args()
    
    # Set seed
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
        in_channels=args.vae_channels,
        uncond=args.uncond,
        energy_head=args.energy_head
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
    vae = load_vae(args.vae_path, args.vae, device)

    # Sampling
    sample_pokemon(model, vae, args, device)

if __name__ == "__main__":
    main()
