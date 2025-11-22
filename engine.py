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
        self.log_steps = 0
        self.running_loss = 0
        self.start_time = time()
        self.grad_norm_sum = 0.0
        self.grad_norm_count = 0
        self.grad_nonzero_sum = 0
        self.grad_total_sum = 0
        # Enhanced gradient health tracking
        self.grad_max_norm_sum = 0.0
        self.grad_mean_norm_sum = 0.0
        self.grad_std_norm_sum = 0.0
        self.grad_p50_sum = 0.0
        self.grad_p95_sum = 0.0
        self.grad_p99_sum = 0.0
        self.grad_clip_count = 0  # Track how often gradients are clipped
        self.last_grad_stats = None  # Store last grad stats for per-layer logging
        # Lower decay for faster EMA updates with smaller dataset
        self.ema_decay = 0.999  # Changed from 0.9999 to 0.999 (10x faster)
        self.sigreg_loss_sum = 0.0
        self.sigreg_var_sum = 0.0
        self.sigreg_var_count = 0
        self.sigreg_enabled = getattr(self.args, "sigreg_lambda", 0) > 0

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

                with torch.no_grad():
                    x = encode_latents(self.vae, x, self.device)

                # Mixed precision training context
                amp_context = (
                    torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if self.use_amp
                    else torch.cuda.amp.autocast(enabled=False)
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

                # Compute gradient statistics before clipping
                grad_stats = self.compute_grad_stats()
                self.last_grad_stats = grad_stats  # Store for per-layer logging

                # Gradient clipping (common for diffusion models: 0.1-1.0)
                clip_value = getattr(self.args, "grad_clip", 1.0)
                if clip_value > 0:
                    unclipped_norm = grad_stats["total_norm"]
                    if self.scaler is not None:
                        self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                    # Track if clipping occurred
                    if unclipped_norm > clip_value:
                        self.grad_clip_count += 1

                # Accumulate gradient statistics
                self.grad_norm_sum += grad_stats["total_norm"]
                self.grad_norm_count += 1
                self.grad_nonzero_sum += grad_stats["nonzero"]
                self.grad_total_sum += grad_stats["total"]
                self.grad_max_norm_sum += grad_stats["max_norm"]
                self.grad_mean_norm_sum += grad_stats["mean_norm"]
                self.grad_std_norm_sum += grad_stats["std_norm"]
                self.grad_p50_sum += grad_stats["percentile_50"]
                self.grad_p95_sum += grad_stats["percentile_95"]
                self.grad_p99_sum += grad_stats["percentile_99"]

                # Optimizer step with mixed precision support
                if self.scaler is not None:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                update_ema(self.ema, self.model, decay=self.ema_decay)

                self.running_loss += loss.item()
                self.log_steps += 1
                self.train_steps += 1

                if self.train_steps % self.args.log_every == 0:
                    self.log_metrics()

                if (
                    self.train_steps % self.args.ckpt_every == 0
                    and self.train_steps > 0
                ):
                    self.save_checkpoint()
                    self.generate_samples()

    def log_metrics(self):
        end_time = time()
        steps_per_sec = self.log_steps / (end_time - self.start_time)
        avg_loss = self.running_loss / self.log_steps
        logger.info(
            f"(step={self.train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}"
        )

        if self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.get_last_lr()[0]
        else:
            # Schedule-free path: log the (constant) optimizer LR
            current_lr = self.opt.param_groups[0].get("lr", 0.0)
        # Compute gradient statistics
        avg_grad_norm = (
            self.grad_norm_sum / self.grad_norm_count if self.grad_norm_count else 0.0
        )
        frac_nonzero = (
            (self.grad_nonzero_sum / self.grad_total_sum)
            if self.grad_total_sum
            else 0.0
        )

        # Enhanced gradient health metrics
        avg_max_norm = (
            self.grad_max_norm_sum / self.grad_norm_count
            if self.grad_norm_count
            else 0.0
        )
        avg_mean_norm = (
            self.grad_mean_norm_sum / self.grad_norm_count
            if self.grad_norm_count
            else 0.0
        )
        avg_std_norm = (
            self.grad_std_norm_sum / self.grad_norm_count
            if self.grad_norm_count
            else 0.0
        )
        avg_p50 = (
            self.grad_p50_sum / self.grad_norm_count if self.grad_norm_count else 0.0
        )
        avg_p95 = (
            self.grad_p95_sum / self.grad_norm_count if self.grad_norm_count else 0.0
        )
        avg_p99 = (
            self.grad_p99_sum / self.grad_norm_count if self.grad_norm_count else 0.0
        )
        clip_frequency = (
            self.grad_clip_count / self.grad_norm_count if self.grad_norm_count else 0.0
        )

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
        stats = {
            "train/loss": avg_loss,
            "train/steps_per_sec": steps_per_sec,
            "train/learning_rate": current_lr,
            "train/ema_decay": self.ema_decay,
            # Gradient health metrics
            "grad/norm": avg_grad_norm,
            "grad/max_norm": avg_max_norm,
            "grad/mean_norm": avg_mean_norm,
            "grad/std_norm": avg_std_norm,
            "grad/percentile_50": avg_p50,
            "grad/percentile_95": avg_p95,
            "grad/percentile_99": avg_p99,
            "grad/fraction_nonzero": frac_nonzero,
            "grad/clip_frequency": clip_frequency,
            # System metrics
            "system/cuda_mem_alloc_bytes": float(cuda_mem_alloc),
            "system/cuda_mem_reserved_bytes": float(cuda_mem_reserved),
        }
        if self.sigreg_var_count:
            stats.update(
                {
                    "sigreg_loss": self.sigreg_loss_sum / self.sigreg_var_count,
                    "sigreg_proj_var_mean": self.sigreg_var_sum / self.sigreg_var_count,
                }
            )
        wandb_utils.log(stats, step=self.train_steps)

        self.running_loss = 0
        self.log_steps = 0
        self.grad_norm_sum = 0.0
        self.grad_norm_count = 0
        self.grad_nonzero_sum = 0
        self.grad_total_sum = 0
        # Reset enhanced gradient stats
        self.grad_max_norm_sum = 0.0
        self.grad_mean_norm_sum = 0.0
        self.grad_std_norm_sum = 0.0
        self.grad_p50_sum = 0.0
        self.grad_p95_sum = 0.0
        self.grad_p99_sum = 0.0
        self.grad_clip_count = 0
        self.start_time = time()
        self.sigreg_loss_sum = 0.0
        self.sigreg_var_sum = 0.0
        self.sigreg_var_count = 0

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
        output_dir = os.path.join(self.sample_dir, f"step_{self.train_steps:07d}")
        result = sample_pokemon(
            self.ema,
            self.vae,
            sample_args,
            self.device,
            num_samples=self.args.sample_batch_size,
            output_dir=output_dir,
            initial_noise=self.fixed_noise,
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
                    "seed": getattr(self.args, "seed", None),
                    "label": 0,
                    "file_path": file_path,
                    "sampler": getattr(self.args, "sampler", "unknown"),
                    "num_steps": sample_stats.get("mean_steps", 0)
                    if sample_stats
                    else 0,
                    "cfg_scale": getattr(self.args, "cfg_scale", 1.0),
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
                wandb.Image(samples[i], caption=f"Sample {i}")
                for i in range(preview_count)
            ]

        wandb_utils.log_samples_table(
            rows, step=self.train_steps, preview_images=preview_images
        )

        # Sampling diagnostics
        if sample_stats:
            diag = {
                "sampling/mean_steps": sample_stats.get("mean_steps", 0.0),
                "sampling/median_steps": sample_stats.get("median_steps", 0.0),
                "sampling/max_steps": sample_stats.get("max_steps", 0),
            }
            trace = sample_stats.get("grad_norm_trace", [])
            if trace:
                diag["sampling/grad_norm_first"] = trace[0]
                diag["sampling/grad_norm_last"] = trace[-1]
            energy = sample_stats.get("energy")
            if energy is not None:
                diag["energy/mean"] = energy.mean().item()
            wandb_utils.log(diag, step=self.train_steps)

        # Lightweight FID/IS hooks on fixed reference
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
                    {f"metric/{k}": v for k, v in metric_vals.items()},
                    step=self.train_steps,
                )

            # Compute LPIPS diversity (perceptual distance between samples)
            lpips_vals = self.metrics.compute_lpips_diversity(sample_tensor)
            if lpips_vals:
                wandb_utils.log(
                    {f"metric/{k}": v for k, v in lpips_vals.items()},
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
        """Enhanced gradient statistics for training health monitoring."""
        total_norms = []
        grad_values = []  # Collect all gradient values for percentile computation
        nonzero = 0
        total = 0
        layer_stats = {}  # Per-layer tracking

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.data
                param_norm = g.norm(2).item()
                total_norms.append(param_norm)

                # Collect gradient values for percentile computation
                grad_values.append(g.abs().flatten())

                nonzero += torch.count_nonzero(g).item()
                total += g.numel()

                # Per-layer stats for detailed monitoring
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
                "mean_norm": 0.0,
                "std_norm": 0.0,
                "percentile_50": 0.0,
                "percentile_95": 0.0,
                "percentile_99": 0.0,
                "layer_stats": {},
            }

        # Total gradient norm
        total_norm_tensor = torch.stack([torch.tensor(n) for n in total_norms])
        total_norm = torch.norm(total_norm_tensor, 2).item()

        # Max gradient norm (to detect spikes)
        max_norm = max(total_norms)

        # Statistics on per-layer norms
        mean_norm = total_norm_tensor.mean().item()
        std_norm = total_norm_tensor.std().item()

        # Percentiles for distribution analysis
        all_grad_values = torch.cat(grad_values)
        percentile_50 = torch.quantile(all_grad_values, 0.50).item()
        percentile_95 = torch.quantile(all_grad_values, 0.95).item()
        percentile_99 = torch.quantile(all_grad_values, 0.99).item()

        return {
            "total_norm": total_norm,
            "nonzero": nonzero,
            "total": total,
            "max_norm": max_norm,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "percentile_50": percentile_50,
            "percentile_95": percentile_95,
            "percentile_99": percentile_99,
            "layer_stats": layer_stats,
        }
