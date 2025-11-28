"""Visualization utilities for attention analysis and GIF generation."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
from PIL import Image
from tqdm import tqdm


class AttentionVisualizer:
    """Collects attention data during training for GIF generation at checkpoints."""

    def __init__(self, save_dir: Optional[str] = None, max_frames: int = 500):
        """
        Args:
            save_dir: Directory to save visualizations
            max_frames: Maximum frames to store (older frames discarded)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.max_frames = max_frames

        # Storage for raw data (converted to frames at checkpoint time)
        self.attention_data: List[Tuple[int, np.ndarray]] = []  # (step, attn_weights)
        self.diff_attn_data: List[Tuple[int, Dict[str, np.ndarray]]] = []  # (step, {attn1, attn2})
        self.entropy_data: List[Tuple[int, np.ndarray]] = []  # (step, per_head_entropy)
        self.similarity_data: List[Tuple[int, np.ndarray]] = []  # (step, similarity_matrix)
        self.sample_data: List[Tuple[int, np.ndarray]] = []  # (step, samples)
        self.metrics_history: Dict[str, List[Tuple[int, float]]] = {}

        # Cached frames for GIF generation (generated at checkpoint)
        self._attention_frames: List[Tuple[int, np.ndarray]] = []
        self._diff_attn_frames: List[Tuple[int, np.ndarray]] = []
        self._entropy_frames: List[Tuple[int, np.ndarray]] = []
        self._similarity_frames: List[Tuple[int, np.ndarray]] = []
        self._sample_frames: List[Tuple[int, np.ndarray]] = []
        self._num_registers: int = 0

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _fig_to_array(self, fig) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        arr = np.array(img)
        buf.close()
        return arr

    def _add_frame(self, storage: list, step: int, frame: np.ndarray):
        """Add frame to storage with max limit."""
        storage.append((step, frame))
        if len(storage) > self.max_frames:
            storage.pop(0)

    def add_metrics(self, step: int, metrics: Dict[str, float]):
        """Store metrics for plotting."""
        for k, v in metrics.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = []
            self.metrics_history[k].append((step, v))
            # Limit history
            if len(self.metrics_history[k]) > self.max_frames * 2:
                self.metrics_history[k] = self.metrics_history[k][-self.max_frames:]

    def create_attention_grid(
        self,
        attn: torch.Tensor,
        step: int,
        num_registers: int = 0,
        title_prefix: str = ""
    ) -> np.ndarray:
        """
        Create a grid visualization of attention from center patch for all heads.

        Args:
            attn: [num_heads, N, N] attention weights
            step: Current training step
            num_registers: Number of register tokens to skip
            title_prefix: Prefix for title

        Returns:
            Image as numpy array
        """
        if num_registers > 0:
            attn = attn[:, num_registers:, num_registers:]

        num_heads = attn.shape[0]
        num_patches = attn.shape[1]
        H = W = int(num_patches ** 0.5)
        center_idx = num_patches // 2

        # Create grid layout
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'{title_prefix}Attention from Center (Step {step})', fontsize=14)

        for head_idx in range(num_heads):
            row, col = head_idx // cols, head_idx % cols
            ax = axes[row, col]

            attn_from_center = attn[head_idx, center_idx].view(H, W).cpu().numpy()
            # Normalize
            attn_vis = attn_from_center - attn_from_center.min()
            attn_vis = attn_vis / (attn_vis.max() + 1e-8)

            im = ax.imshow(attn_vis, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(num_heads, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_entropy_chart(
        self,
        per_head_entropy: torch.Tensor,
        step: int
    ) -> np.ndarray:
        """Create per-head entropy bar chart."""
        per_head_ent = per_head_entropy.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(per_head_ent)), per_head_ent, color='steelblue', edgecolor='navy')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Entropy')
        ax.set_title(f'Per-Head Attention Entropy (Step {step})')
        mean_ent = per_head_ent.mean()
        ax.axhline(y=mean_ent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ent:.3f}')
        ax.legend()
        ax.set_xticks(range(len(per_head_ent)))
        plt.tight_layout()

        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_similarity_matrix(
        self,
        sim_matrix: torch.Tensor,
        step: int
    ) -> np.ndarray:
        """Create head similarity matrix heatmap."""
        sim = sim_matrix.cpu().numpy()
        num_heads = sim.shape[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Head Similarity Matrix (Step {step})')
        ax.set_xlabel('Head')
        ax.set_ylabel('Head')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_heads))
        plt.tight_layout()

        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_metrics_plot(self, step: int) -> np.ndarray:
        """Create a plot of metrics history."""
        if not self.metrics_history:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Select key metrics to plot
        key_metrics = [
            'entropy/normalized_mean',
            'sparsity/gini',
            'diversity/score',
            'spatial/local_ratio'
        ]
        available = [k for k in key_metrics if k in self.metrics_history]

        if not available:
            available = list(self.metrics_history.keys())[:4]

        n_plots = len(available)
        if n_plots == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, metric_name in enumerate(available[:4]):
            ax = axes[idx]
            history = self.metrics_history[metric_name]
            steps = [h[0] for h in history]
            values = [h[1] for h in history]
            ax.plot(steps, values, 'b-', linewidth=1)
            ax.set_title(metric_name.replace('/', ' / '))
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(available), 4):
            axes[idx].axis('off')

        fig.suptitle(f'Attention Metrics History (Step {step})', fontsize=14)
        plt.tight_layout()

        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def store_attention_data(self, attn: torch.Tensor, step: int, num_registers: int = 0):
        """Store raw attention data for later visualization."""
        self._num_registers = num_registers
        attn_np = attn.detach().cpu().numpy()
        self._add_frame(self.attention_data, step, attn_np)

    def store_diff_attention_data(self, attn1: torch.Tensor, attn2: torch.Tensor, step: int, num_registers: int = 0):
        """Store raw differential attention data for later visualization."""
        self._num_registers = num_registers
        data = {
            'attn1': attn1.detach().cpu().numpy(),
            'attn2': attn2.detach().cpu().numpy()
        }
        self._add_frame(self.diff_attn_data, step, data)

    def store_entropy_data(self, per_head_entropy: torch.Tensor, step: int):
        """Store raw entropy data for later visualization."""
        entropy_np = per_head_entropy.detach().cpu().numpy()
        self._add_frame(self.entropy_data, step, entropy_np)

    def store_similarity_data(self, sim_matrix: torch.Tensor, step: int):
        """Store raw similarity matrix data for later visualization."""
        sim_np = sim_matrix.detach().cpu().numpy()
        self._add_frame(self.similarity_data, step, sim_np)

    def store_sample_data(self, samples: Union[torch.Tensor, np.ndarray], step: int):
        """Store raw sample data for later visualization."""
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        self._add_frame(self.sample_data, step, samples)

    # Legacy methods for backward compatibility
    def add_attention_frame(self, attn: torch.Tensor, step: int, num_registers: int = 0):
        """Add attention visualization frame (legacy - stores data now)."""
        self.store_attention_data(attn, step, num_registers)

    def add_entropy_frame(self, per_head_entropy: torch.Tensor, step: int):
        """Add entropy chart frame (legacy - stores data now)."""
        self.store_entropy_data(per_head_entropy, step)

    def add_similarity_frame(self, sim_matrix: torch.Tensor, step: int):
        """Add similarity matrix frame (legacy - stores data now)."""
        self.store_similarity_data(sim_matrix, step)

    def add_sample_frame(self, samples: Union[torch.Tensor, np.ndarray], step: int):
        """Add generated samples frame (legacy - stores data now)."""
        self.store_sample_data(samples, step)

    def create_diff_attention_grid(
        self,
        attn1: np.ndarray,
        attn2: np.ndarray,
        step: int,
        num_registers: int = 0,
    ) -> np.ndarray:
        """
        Create a grid visualization for differential attention (attn1 and attn2 side by side).

        Args:
            attn1: [num_heads, N, N] primary attention weights
            attn2: [num_heads, N, N] subtracted attention weights
            step: Current training step
            num_registers: Number of register tokens to skip

        Returns:
            Image as numpy array
        """
        if num_registers > 0:
            attn1 = attn1[:, num_registers:, num_registers:]
            attn2 = attn2[:, num_registers:, num_registers:]

        num_heads = attn1.shape[0]
        num_patches = attn1.shape[1]
        H = W = int(num_patches ** 0.5)
        center_idx = num_patches // 2

        # Create grid layout: 2 columns per head (attn1, attn2)
        cols = min(4, num_heads)  # Max 4 heads per row
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows * 2, cols, figsize=(3 * cols, 3 * rows * 2))
        if rows * 2 == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows * 2 == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Differential Attention (Step {step})\nTop: Primary (A1), Bottom: Subtracted (A2)', fontsize=12)

        for head_idx in range(num_heads):
            row_base = (head_idx // cols) * 2
            col = head_idx % cols

            # Primary attention (attn1)
            ax1 = axes[row_base, col]
            attn1_from_center = attn1[head_idx, center_idx].reshape(H, W)
            attn1_vis = attn1_from_center - attn1_from_center.min()
            attn1_vis = attn1_vis / (attn1_vis.max() + 1e-8)
            ax1.imshow(attn1_vis, cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f'H{head_idx} A1', fontsize=9)
            ax1.axis('off')

            # Subtracted attention (attn2)
            ax2 = axes[row_base + 1, col]
            attn2_from_center = attn2[head_idx, center_idx].reshape(H, W)
            attn2_vis = attn2_from_center - attn2_from_center.min()
            attn2_vis = attn2_vis / (attn2_vis.max() + 1e-8)
            ax2.imshow(attn2_vis, cmap='magma', vmin=0, vmax=1)
            ax2.set_title(f'H{head_idx} A2', fontsize=9)
            ax2.axis('off')

        # Hide empty subplots
        for idx in range(num_heads, (rows) * cols):
            row_base = (idx // cols) * 2
            col = idx % cols
            if row_base < axes.shape[0] and col < axes.shape[1]:
                axes[row_base, col].axis('off')
                axes[row_base + 1, col].axis('off')

        plt.tight_layout()
        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_sample_grid(self, samples: np.ndarray, step: int) -> np.ndarray:
        """Create sample grid image from numpy array."""
        # Handle different formats
        if samples.ndim == 4:
            if samples.shape[1] in [1, 3, 4]:  # [N, C, H, W]
                samples = samples.transpose(0, 2, 3, 1)

        n_samples = min(samples.shape[0], 16)
        cols = min(4, n_samples)
        rows = (n_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Generated Samples (Step {step})', fontsize=14)

        for idx in range(n_samples):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]

            img = samples[idx]
            # Normalize to 0-1 if needed
            if img.min() < 0 or img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(n_samples, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_attention_grid_from_numpy(
        self,
        attn: np.ndarray,
        step: int,
        num_registers: int = 0,
        title_prefix: str = ""
    ) -> np.ndarray:
        """Create attention grid from numpy array (internal use)."""
        if num_registers > 0:
            attn = attn[:, num_registers:, num_registers:]

        num_heads = attn.shape[0]
        num_patches = attn.shape[1]
        H = W = int(num_patches ** 0.5)
        center_idx = num_patches // 2

        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'{title_prefix}Attention from Center (Step {step})', fontsize=14)

        for head_idx in range(num_heads):
            row, col = head_idx // cols, head_idx % cols
            ax = axes[row, col]

            attn_from_center = attn[head_idx, center_idx].reshape(H, W)
            attn_vis = attn_from_center - attn_from_center.min()
            attn_vis = attn_vis / (attn_vis.max() + 1e-8)

            ax.imshow(attn_vis, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.axis('off')

        for idx in range(num_heads, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_entropy_chart_from_numpy(self, per_head_entropy: np.ndarray, step: int) -> np.ndarray:
        """Create entropy chart from numpy array (internal use)."""
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(per_head_entropy)), per_head_entropy, color='steelblue', edgecolor='navy')
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Entropy')
        ax.set_title(f'Per-Head Attention Entropy (Step {step})')
        mean_ent = per_head_entropy.mean()
        ax.axhline(y=mean_ent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ent:.3f}')
        ax.legend()
        ax.set_xticks(range(len(per_head_entropy)))
        plt.tight_layout()

        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def create_similarity_matrix_from_numpy(self, sim_matrix: np.ndarray, step: int) -> np.ndarray:
        """Create similarity matrix from numpy array (internal use)."""
        num_heads = sim_matrix.shape[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Head Similarity Matrix (Step {step})')
        ax.set_xlabel('Head')
        ax.set_ylabel('Head')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_heads))
        plt.tight_layout()

        arr = self._fig_to_array(fig)
        plt.close(fig)
        return arr

    def generate_frames_from_data(self):
        """Convert stored raw data into visualization frames for GIF generation."""
        total_frames = (len(self.attention_data) + len(self.diff_attn_data) +
                       len(self.entropy_data) + len(self.similarity_data) + len(self.sample_data))

        if total_frames == 0:
            return

        pbar = tqdm(total=total_frames, desc="Generating GIF frames", leave=False)

        # Generate attention frames
        self._attention_frames = []
        for step, attn_np in self.attention_data:
            frame = self.create_attention_grid_from_numpy(attn_np, step, self._num_registers)
            self._attention_frames.append((step, frame))
            pbar.update(1)

        # Generate differential attention frames
        self._diff_attn_frames = []
        for step, data in self.diff_attn_data:
            frame = self.create_diff_attention_grid(
                data['attn1'], data['attn2'], step, self._num_registers
            )
            self._diff_attn_frames.append((step, frame))
            pbar.update(1)

        # Generate entropy frames
        self._entropy_frames = []
        for step, entropy_np in self.entropy_data:
            frame = self.create_entropy_chart_from_numpy(entropy_np, step)
            self._entropy_frames.append((step, frame))
            pbar.update(1)

        # Generate similarity frames
        self._similarity_frames = []
        for step, sim_np in self.similarity_data:
            frame = self.create_similarity_matrix_from_numpy(sim_np, step)
            self._similarity_frames.append((step, frame))
            pbar.update(1)

        # Generate sample frames
        self._sample_frames = []
        for step, samples_np in self.sample_data:
            frame = self.create_sample_grid(samples_np, step)
            self._sample_frames.append((step, frame))
            pbar.update(1)

        pbar.close()

    def create_gif(
        self,
        frames: List[Tuple[int, np.ndarray]],
        output_path: str,
        fps: int = 10,
        loop: int = 0
    ) -> str:
        """
        Create GIF from frames.

        Args:
            frames: List of (step, image_array) tuples
            output_path: Path to save GIF
            fps: Frames per second
            loop: Number of loops (0 = infinite)

        Returns:
            Path to saved GIF
        """
        if not frames:
            return None

        images = [Image.fromarray(frame[1]) for frame in frames]

        # Resize to consistent size (use first frame as reference)
        target_size = images[0].size
        images = [img.resize(target_size, Image.Resampling.LANCZOS) if img.size != target_size else img
                  for img in images]

        duration = int(1000 / fps)  # ms per frame

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )

        return output_path

    def generate_all_gifs(self, output_dir: str, step_suffix: str = "") -> Dict[str, str]:
        """
        Generate all GIFs from stored data.

        Args:
            output_dir: Directory to save GIFs
            step_suffix: Optional suffix for filenames (e.g., "_step_1000")

        Returns:
            Dict mapping GIF name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # First convert stored data to frames
        self.generate_frames_from_data()

        gifs = {}
        suffix = step_suffix if step_suffix else ""

        if self._attention_frames:
            path = str(output_dir / f'attention_grid{suffix}.gif')
            self.create_gif(self._attention_frames, path, fps=8)
            gifs['attention_grid'] = path

        if self._diff_attn_frames:
            path = str(output_dir / f'diff_attention_grid{suffix}.gif')
            self.create_gif(self._diff_attn_frames, path, fps=8)
            gifs['diff_attention_grid'] = path

        if self._entropy_frames:
            path = str(output_dir / f'entropy_evolution{suffix}.gif')
            self.create_gif(self._entropy_frames, path, fps=8)
            gifs['entropy_evolution'] = path

        if self._similarity_frames:
            path = str(output_dir / f'similarity_evolution{suffix}.gif')
            self.create_gif(self._similarity_frames, path, fps=8)
            gifs['similarity_evolution'] = path

        if self._sample_frames:
            path = str(output_dir / f'samples_evolution{suffix}.gif')
            self.create_gif(self._sample_frames, path, fps=5)
            gifs['samples_evolution'] = path

        # Create metrics history plot as final frame
        if self.metrics_history:
            last_step = max(h[-1][0] for h in self.metrics_history.values() if h)
            metrics_plot = self.create_metrics_plot(last_step)
            img = Image.fromarray(metrics_plot)
            path = str(output_dir / f'metrics_history{suffix}.png')
            img.save(path)
            gifs['metrics_history'] = path

        return gifs

    def clear_data(self):
        """Clear stored data after GIF generation to free memory."""
        self.attention_data.clear()
        self.diff_attn_data.clear()
        self.entropy_data.clear()
        self.similarity_data.clear()
        # Keep sample_data and metrics_history for full training visualization
        self._attention_frames.clear()
        self._diff_attn_frames.clear()
        self._entropy_frames.clear()
        self._similarity_frames.clear()

    def upload_gifs_to_wandb(self, wandb_utils, gifs: Dict[str, str], step: int):
        """Upload generated GIFs to wandb."""
        import wandb

        log_dict = {}
        for name, path in gifs.items():
            if path and Path(path).exists():
                if path.endswith('.gif'):
                    log_dict[f'summary/{name}'] = wandb.Video(path, fps=8, format='gif')
                else:
                    log_dict[f'summary/{name}'] = wandb.Image(path)

        if log_dict:
            wandb_utils.log(log_dict, step=step)


def create_combined_dashboard(
    attn: torch.Tensor,
    metrics: Dict[str, torch.Tensor],
    step: int,
    num_registers: int = 0,
    samples: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Create a combined dashboard with attention, metrics, and samples.

    Args:
        attn: [num_heads, N, N] attention weights
        metrics: Dict of computed metrics
        step: Current step
        num_registers: Number of register tokens
        samples: Optional generated samples [N, C, H, W]

    Returns:
        Dashboard image as numpy array
    """
    # Skip registers
    if num_registers > 0:
        attn = attn[:, num_registers:, num_registers:]

    num_heads = attn.shape[0]
    num_patches = attn.shape[1]
    H = W = int(num_patches ** 0.5)
    center_idx = num_patches // 2

    # Create figure with subplots
    has_samples = samples is not None and samples.numel() > 0

    if has_samples:
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1])

    # Row 1: Attention heads (first 4)
    for i in range(min(4, num_heads)):
        ax = fig.add_subplot(gs[0, i])
        attn_from_center = attn[i, center_idx].view(H, W).cpu().numpy()
        attn_vis = (attn_from_center - attn_from_center.min()) / (attn_from_center.max() - attn_from_center.min() + 1e-8)
        ax.imshow(attn_vis, cmap='viridis')
        ax.set_title(f'Head {i}')
        ax.axis('off')

    # Row 2: Entropy chart, Similarity matrix, Metrics
    # Entropy bar chart
    ax_entropy = fig.add_subplot(gs[1, :2])
    if 'entropy/per_head' in metrics:
        per_head_ent = metrics['entropy/per_head'].cpu().numpy()
        ax_entropy.bar(range(len(per_head_ent)), per_head_ent, color='steelblue')
        ax_entropy.axhline(y=per_head_ent.mean(), color='red', linestyle='--', label=f'Mean: {per_head_ent.mean():.3f}')
        ax_entropy.set_xlabel('Head')
        ax_entropy.set_ylabel('Entropy')
        ax_entropy.set_title('Per-Head Entropy')
        ax_entropy.legend()

    # Similarity matrix
    ax_sim = fig.add_subplot(gs[1, 2])
    if 'diversity/similarity_matrix' in metrics:
        sim = metrics['diversity/similarity_matrix'].cpu().numpy()
        im = ax_sim.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_sim.set_title('Head Similarity')
        plt.colorbar(im, ax=ax_sim, fraction=0.046)

    # Key metrics text
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis('off')
    text_lines = [f'Step: {step}', '']
    scalar_keys = ['entropy/normalized_mean', 'sparsity/gini', 'diversity/score', 'spatial/local_ratio']
    for key in scalar_keys:
        if key in metrics and metrics[key].dim() == 0:
            text_lines.append(f'{key.split("/")[1]}: {metrics[key].item():.4f}')
    ax_text.text(0.1, 0.9, '\n'.join(text_lines), transform=ax_text.transAxes,
                 fontsize=12, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 3: Samples (if available)
    if has_samples:
        samples_np = samples.cpu().numpy()
        if samples_np.shape[1] in [1, 3, 4]:  # [N, C, H, W]
            samples_np = samples_np.transpose(0, 2, 3, 1)

        for i in range(min(4, samples_np.shape[0])):
            ax = fig.add_subplot(gs[2, i])
            img = samples_np[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f'Sample {i}')
            ax.axis('off')

    fig.suptitle(f'Training Dashboard - Step {step}', fontsize=16)
    plt.tight_layout()

    # Convert to array
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    arr = np.array(img)
    buf.close()
    plt.close(fig)

    return arr
