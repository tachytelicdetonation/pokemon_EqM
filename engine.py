import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time

import torch

import wandb
from generate_pokemon import sample_pokemon
from utils import wandb as wandb_utils
from utils.metrics import MetricsHelper
from utils.sigreg import sig_isotropic_gaussian_loss
from utils.vae import encode_latents

logger = logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)


class Trainer:
    def __init__(
        self,
        model,
        ema,
        opt,
        transport,
        vae,
        loader,
        lr_scheduler,
        device,
        args,
        scaler=None,
        amp_dtype=None,
    ):
        self.model = model
        self.ema = ema
        self.opt = opt
        self.transport = transport
        self.vae = vae
        self.loader = loader
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.args = args
        self.scaler = scaler
        self.amp_dtype = amp_dtype
        self.use_amp = amp_dtype is not None

        self.train_steps = 0
        self.start_time = time()
        self.last_log_time = time()
        self.steps_since_last_log = 0
        self.last_grad_stats = None  # Store last grad stats for per-layer logging
        # Lower decay for faster EMA updates with smaller dataset
        self.ema_decay = 0.999  # Changed from 0.9999 to 0.999 (10x faster)
        self.sigreg_enabled = getattr(self.args, "sigreg_lambda", 0) > 0
        self.sigreg_loss_sum = 0.0
        self.sigreg_var_sum = 0.0
        self.sigreg_var_count = 0

        # Fixed noise for consistent sampling
        # Determine latent channels from model or args
        in_channels = model.in_channels if hasattr(model, "in_channels") else 4
        # We generate fixed noise for the preview samples (use sampling batch size)
        # Using a fixed generator for reproducibility of this noise
        g = torch.Generator()
        seed = getattr(args, "seed", None)
        # If no (or negative) seed is provided, fall back to stochastic generator state
        if isinstance(seed, int) and seed >= 0:
            g.manual_seed(seed)
        else:
            g.seed()
        preview_n = max(1, self.args.sample_batch_size)
        self.fixed_noise = torch.randn(
            preview_n,
            in_channels,
            args.image_size // 8,
            args.image_size // 8,
            generator=g,
        )

        # Validation reference for metrics
        self.metrics = MetricsHelper(device)
        self.val_reference = None
        if self.metrics.available:
            try:
                batch, _ = next(iter(loader))
                take = min(getattr(self.args, "metrics_subset_size", 16), batch.size(0))
                self.val_reference = batch[:take].to(device)
            except Exception as e:
                logger.warning(f"Unable to grab validation batch for metrics: {e}")

        self.setup_directories()

        if getattr(self.args, "watch_grads", False) and wandb.run is not None:
            try:
                # Coarse histogram cadence aligned to checkpoints, clamped to [500, 2000]
                log_freq = min(2000, max(500, getattr(self.args, "ckpt_every", 1000)))
                wandb.watch(
                    self.model, log="gradients", log_freq=log_freq, log_graph=False
                )
            except Exception as e:
                logger.warning(f"WandB gradient watch failed: {e}")

    def setup_directories(self):
        os.makedirs(self.args.results_dir, exist_ok=True)
        experiment_name = f"pokemon-eqm-{self.args.model}"

        run_id = None
        run_name = None
        try:
            run_id, run_name = wandb_utils.initialize(
                self.args,
                entity=None,
                exp_name=experiment_name,
                project_name="pokemon-eqm",
            )
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_name}_{timestamp}"
            run_id = run_name

        if self.args.run_id:
            # Logic to find existing directory could be improved here, but keeping simple for now
            self.experiment_dir = os.path.join(
                self.args.results_dir, self.args.run_id
            )  # Assuming run_id is the folder name
            if not os.path.exists(self.experiment_dir):
                # Fallback search
                import glob

                potential_dirs = glob.glob(
                    os.path.join(self.args.results_dir, f"*{self.args.run_id}*")
                )
                if potential_dirs:
                    self.experiment_dir = potential_dirs[0]
                else:
                    self.experiment_dir = os.path.join(self.args.results_dir, run_name)
        else:
            self.experiment_dir = os.path.join(self.args.results_dir, run_name)

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.sample_dir = os.path.join(self.experiment_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.opt.load_state_dict(checkpoint["opt"])

        try:
            resume_step = int(os.path.basename(checkpoint_path).split(".")[0])
            self.train_steps = resume_step
            return resume_step
        except:
            return 0

    def train(self, start_epoch=0):
        self.model.train()
        self.ema.eval()

        logger.info(f"Training for {self.args.epochs} epochs...")

        # Initial samples
        if self.train_steps == 0:
            self.generate_samples()

        for epoch in range(start_epoch, self.args.epochs):
            logger.info(f"Beginning epoch {epoch}...")
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # If using cached latents, x is already a latent tensor.
                # Otherwise, encode it.
                if not getattr(self.args, "cache_latents", False):
                    with torch.no_grad():
                        x = encode_latents(self.vae, x, self.device)

                # Mixed precision training context
                amp_context = (
                    torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if self.use_amp
                    else torch.amp.autocast(device_type="cuda", enabled=False)
                )

                with amp_context:
                    model_kwargs = dict(
                        y=y,
                        return_act=self.sigreg_enabled,
                        apply_disp_loss=False,
                        train=True,
                    )
                    loss_dict = self.transport.training_losses(
                        self.model, x, model_kwargs
                    )
                    loss = loss_dict["loss"].mean()

                    # SIGReg auxiliary regularizer on penultimate embeddings
                    if self.sigreg_enabled:
                        penultimate = loss_dict.get("penultimate")
                        if penultimate is not None:
                            sig_features = self.model.sigreg_embedding(penultimate)
                            sig_loss, proj_var = sig_isotropic_gaussian_loss(
                                sig_features
                            )
                            loss = loss + self.args.sigreg_lambda * sig_loss
                            self.sigreg_loss_sum += sig_loss.item()
                            self.sigreg_var_sum += proj_var.mean().item()
                            self.sigreg_var_count += 1
                        else:
                            logger.warning(
                                "SIGReg enabled but penultimate activations were not returned; skipping SIGReg for this step."
                            )

                self.opt.zero_grad()
                # Scale loss if using fp16 with GradScaler
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Only compute expensive stats and log if it's a logging step
                is_log_step = (self.train_steps % self.args.log_every == 0)
                
                grad_stats = None
                was_clipped = False
                
                if is_log_step:
                    # Compute gradient statistics before clipping
                    grad_stats = self.compute_grad_stats()
                    self.last_grad_stats = grad_stats  # Store for per-layer logging

                # SIGReg loss tracking
                sigreg_loss_value = None
                sigreg_var_value = None
                if self.sigreg_enabled:
                    penultimate = loss_dict.get("penultimate")
                    if penultimate is not None:
                        sigreg_loss_value = sig_loss.item()
                        sigreg_var_value = proj_var.mean().item()

                # Gradient clipping (common for diffusion models: 0.1-1.0)
                clip_value = getattr(self.args, "grad_clip", 1.0)
                
                if clip_value > 0:
                    # If we didn't compute stats yet but need to clip, we might need to compute norm
                    # But clip_grad_norm_ computes it anyway.
                    if self.scaler is not None:
                        self.scaler.unscale_(self.opt)
                    
                    # We can capture the norm from clip_grad_norm_
                    unclipped_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                    
                    if is_log_step and grad_stats is not None:
                         # Update total_norm with the one computed by clip_grad_norm_ if we want exact consistency
                         # or just rely on our pre-computed one. 
                         # Actually, clip_grad_norm_ returns the norm BEFORE clipping.
                         if unclipped_norm > clip_value:
                             was_clipped = True

                # Optimizer step with mixed precision support
                if self.scaler is not None:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                update_ema(self.ema, self.model, decay=self.ema_decay)

                self.train_steps += 1
                self.steps_since_last_log += 1

                # Log metrics to wandb only on log steps
                if is_log_step:
                    self.log_step_metrics(loss.item(), grad_stats, sigreg_loss_value, sigreg_var_value, was_clipped)
                    self.print_terminal_metrics()

                if (
                    self.train_steps % self.args.ckpt_every == 0
                    and self.train_steps > 0
                ):
                    self.save_checkpoint()
                    self.generate_samples()

    def log_step_metrics(self, loss_value, grad_stats, sigreg_loss_value, sigreg_var_value, was_clipped):
        """Log metrics to wandb immediately after each training step."""
        # Get current learning rate
        if self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.get_last_lr()[0]
        else:
            current_lr = self.opt.param_groups[0].get("lr", 0.0)

        # Compute gradient fraction nonzero
        frac_nonzero = (
            grad_stats["nonzero"] / grad_stats["total"]
            if grad_stats["total"] > 0
            else 0.0
        )

        # CUDA memory
        cuda_mem_alloc = (
            torch.cuda.memory_allocated(self.device)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0
        )
        cuda_mem_reserved = (
            torch.cuda.memory_reserved(self.device)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0
        )

        # Build metrics dict with industry-standard gradient health metrics only
        stats = {
            "train/loss": loss_value,
            "train/learning_rate": current_lr,
            "train/ema_decay": self.ema_decay,
            # Core gradient health metrics (industry standard)
            "grad/norm": grad_stats["total_norm"],
            "grad/max_norm": grad_stats["max_norm"],
            "grad/fraction_nonzero": frac_nonzero,
            "grad/was_clipped": 1.0 if was_clipped else 0.0,
            # System metrics
            "system/cuda_mem_alloc_bytes": float(cuda_mem_alloc),
            "system/cuda_mem_reserved_bytes": float(cuda_mem_reserved),
        }

        # Add SIGReg metrics if enabled
        if sigreg_loss_value is not None:
            stats["train/sigreg_loss"] = sigreg_loss_value
            stats["train/sigreg_proj_var_mean"] = sigreg_var_value

        # Log to wandb
        wandb_utils.log(stats, step=self.train_steps)

    def print_terminal_metrics(self):
        """Print summary metrics to terminal at log_every interval."""
        end_time = time()
        elapsed = end_time - self.last_log_time
        steps_per_sec = self.steps_since_last_log / elapsed if elapsed > 0 else 0.0

        logger.info(
            f"(step={self.train_steps:07d}) Steps/Sec: {steps_per_sec:.2f}"
        )

        # Reset counters
        self.last_log_time = time()
        self.steps_since_last_log = 0

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "args": self.args,
        }
        checkpoint_path = f"{self.checkpoint_dir}/{self.train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def generate_samples(self):
        logger.info(f"Generating samples at step {self.train_steps}...")

        # Log per-layer gradient statistics (infrequent, at checkpoint time)
        if getattr(self.args, "log_layer_grads", True):
            self.log_layer_gradients()

        sample_args = deepcopy(self.args)
        sample_args.batch_size = self.args.sample_batch_size

        # Generate FIXED noise samples (for consistency tracking)
        logger.info("Generating fixed noise samples...")
        self._generate_and_log_samples(
            sample_args,
            initial_noise=self.fixed_noise,
            prefix="fixed",
            subdirectory="fixed"
        )

        # Generate RANDOM noise samples (for diversity assessment)
        logger.info("Generating random noise samples...")
        random_noise = torch.randn_like(self.fixed_noise)
        self._generate_and_log_samples(
            sample_args,
            initial_noise=random_noise,
            prefix="random",
            subdirectory="random"
        )

    def _generate_and_log_samples(self, sample_args, initial_noise, prefix, subdirectory):
        """Helper method to generate samples and log to wandb with given prefix."""
        output_dir = os.path.join(
            self.sample_dir,
            f"step_{self.train_steps:07d}",
            subdirectory
        )

        result = sample_pokemon(
            self.ema,
            self.vae,
            sample_args,
            self.device,
            num_samples=self.args.sample_batch_size,
            output_dir=output_dir,
            initial_noise=initial_noise,
            return_stats=True,
        )

        if isinstance(result, tuple):
            samples, sample_stats = result
        else:
            samples, sample_stats = result, {}

        rows = []
        for idx, _ in enumerate(samples):
            file_path = os.path.join(output_dir, f"{idx:05d}.png")
            rows.append(
                {
                    "train_step": self.train_steps,
                    "index": idx,
                    "seed": getattr(self.args, "seed", None) if prefix == "fixed" else "random",
                    "label": 0,
                    "file_path": file_path,
                    "sampler": getattr(self.args, "sampler", "unknown"),
                    "num_steps": sample_stats.get("mean_steps", 0)
                    if sample_stats
                    else 0,
                    "cfg_scale": getattr(self.args, "cfg_scale", 1.0),
                    "type": prefix,
                }
            )

        # Create preview grid from all samples (up to 8)
        preview_images = None
        if (
            hasattr(self.args, "log_images")
            and self.args.log_images
            and len(samples) > 0
        ):
            preview_count = min(8, len(samples))
            preview_images = [
                wandb.Image(samples[i], caption=f"{prefix.capitalize()} {i}")
                for i in range(preview_count)
            ]

        # Log samples table with prefix
        wandb_utils.log(
            {f"{prefix}/generated_samples": preview_images} if preview_images else {},
            step=self.train_steps
        )

        # Sampling diagnostics with prefix
        if sample_stats:
            diag = {
                f"{prefix}/sampling/mean_steps": sample_stats.get("mean_steps", 0.0),
                f"{prefix}/sampling/median_steps": sample_stats.get("median_steps", 0.0),
                f"{prefix}/sampling/max_steps": sample_stats.get("max_steps", 0),
            }
            trace = sample_stats.get("grad_norm_trace", [])
            if trace:
                diag[f"{prefix}/sampling/grad_norm_first"] = trace[0]
                diag[f"{prefix}/sampling/grad_norm_last"] = trace[-1]
            energy = sample_stats.get("energy")
            if energy is not None:
                diag[f"{prefix}/energy/mean"] = energy.mean().item()
            wandb_utils.log(diag, step=self.train_steps)

        # Lightweight FID/IS hooks on fixed reference with prefix
        sample_tensor = sample_stats.get("samples_tensor") if sample_stats else None
        if (
            sample_tensor is not None
            and self.metrics.available
            and self.val_reference is not None
        ):
            real = (self.val_reference * 0.5 + 0.5).clamp(0, 1)
            fake = (sample_tensor.to(self.device) * 0.5 + 0.5).clamp(0, 1)
            metric_vals = self.metrics.compute(real, fake)
            if metric_vals:
                wandb_utils.log(
                    {f"{prefix}/metric/{k}": v for k, v in metric_vals.items()},
                    step=self.train_steps,
                )

            # Compute LPIPS diversity (perceptual distance between samples)
            lpips_vals = self.metrics.compute_lpips_diversity(sample_tensor)
            if lpips_vals:
                wandb_utils.log(
                    {f"{prefix}/metric/{k}": v for k, v in lpips_vals.items()},
                    step=self.train_steps,
                )

        # free tensor to keep memory low
        if sample_tensor is not None:
            del sample_tensor

    def log_layer_gradients(self):
        """Log per-layer gradient statistics to wandb (expensive, call infrequently)."""
        if self.last_grad_stats is None or not self.last_grad_stats.get("layer_stats"):
            return

        layer_stats = self.last_grad_stats["layer_stats"]

        # Log top layers by gradient norm (most problematic)
        sorted_layers = sorted(
            layer_stats.items(), key=lambda x: x[1]["norm"], reverse=True
        )

        # Log top 10 layers with highest gradients
        top_layers = {}
        for i, (name, stats) in enumerate(sorted_layers[:10]):
            # Shorten layer names for readability
            short_name = name.replace("module.", "").replace("model.", "")
            if len(short_name) > 50:
                short_name = short_name[:47] + "..."

            top_layers[f"layer_grad/{i:02d}_{short_name}_norm"] = stats["norm"]
            top_layers[f"layer_grad/{i:02d}_{short_name}_max"] = stats["max"]

        if top_layers:
            wandb_utils.log(top_layers, step=self.train_steps)

        # Create a histogram table of layer gradient norms
        if getattr(self.args, "log_layer_histogram", False):
            layer_names = []
            layer_norms = []
            for name, stats in sorted_layers[:20]:  # Top 20 for histogram
                short_name = name.replace("module.", "").replace("model.", "")
                layer_names.append(short_name[:30])  # Truncate for display
                layer_norms.append(stats["norm"])

            wandb_utils.log(
                {"layer_grad/norm_distribution": wandb.Histogram(layer_norms)},
                step=self.train_steps,
            )

    def compute_grad_stats(self):
        """Industry-standard gradient statistics for training health monitoring."""
        total_norms = []
        nonzero = 0
        total = 0
        layer_stats = {}  # Per-layer tracking

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.data
                param_norm = g.norm(2).item()
                total_norms.append(param_norm)

                nonzero += torch.count_nonzero(g).item()
                total += g.numel()

                # Per-layer stats for detailed monitoring (checkpoint time only)
                layer_stats[name] = {
                    "norm": param_norm,
                    "mean": g.mean().item(),
                    "std": g.std().item(),
                    "max": g.abs().max().item(),
                }

        if not total_norms:
            return {
                "total_norm": 0.0,
                "nonzero": 0,
                "total": 0,
                "max_norm": 0.0,
                "layer_stats": {},
            }

        # Total gradient norm (L2 norm across all parameters)
        total_norm_tensor = torch.stack([torch.tensor(n) for n in total_norms])
        total_norm = torch.norm(total_norm_tensor, 2).item()

        # Max gradient norm (to detect spikes in individual layers)
        max_norm = max(total_norms)

        return {
            "total_norm": total_norm,
            "nonzero": nonzero,
            "total": total,
            "max_norm": max_norm,
            "layer_stats": layer_stats,
        }
