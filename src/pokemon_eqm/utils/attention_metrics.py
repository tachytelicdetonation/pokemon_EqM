"""Attention quality metrics for monitoring transformer health.

This module provides metrics to evaluate attention quality during training:
- Entropy: Measures focus vs diffuse attention (detect collapse or uniform)
- Sparsity: Measures attention concentration and distribution
- Head Diversity: Measures whether heads learn different patterns
- Spatial: Measures local vs global attention patterns (for vision)

References:
- Stabilizing Transformer Training by Preventing Attention Entropy Collapse (ICML 2023)
- Crisp Attention: Regularizing Transformers via Structured Sparsity
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def compute_entropy(attn: torch.Tensor, eps: float = 1e-10) -> Dict[str, torch.Tensor]:
    """
    Compute entropy metrics for attention weights.

    Entropy measures how "spread out" the attention is:
    - Low entropy: sharp, focused on few tokens (potential collapse)
    - High entropy: diffuse, spread across many tokens (uniform)

    Args:
        attn: [B, num_heads, N, N] attention weights (post-softmax)
        eps: Small value for numerical stability

    Returns:
        Dict with:
            - entropy/per_head: Per-head entropy values [H]
            - entropy/mean: Mean entropy across heads
            - entropy/std: Std of entropy across heads
            - entropy/normalized_mean: Entropy normalized by max possible (0-1 range)
            - entropy/min: Minimum head entropy
            - entropy/max: Maximum head entropy
    """
    # Per-row entropy: H = -sum(p * log(p))
    log_attn = torch.log(attn + eps)
    entropy = -torch.sum(attn * log_attn, dim=-1)  # [B, H, N]

    # Average over queries and batch
    per_head_entropy = entropy.mean(dim=(0, 2))  # [H]

    # Normalize by max possible entropy (uniform distribution)
    seq_len = attn.shape[-1]
    max_entropy = torch.log(torch.tensor(seq_len, dtype=attn.dtype, device=attn.device))
    normalized = per_head_entropy / (max_entropy + eps)

    return {
        'entropy/per_head': per_head_entropy,
        'entropy/mean': per_head_entropy.mean(),
        'entropy/std': per_head_entropy.std(),
        'entropy/normalized_mean': normalized.mean(),
        'entropy/min': per_head_entropy.min(),
        'entropy/max': per_head_entropy.max(),
    }


def compute_sparsity(
    attn: torch.Tensor,
    threshold: float = 0.01,
    top_k: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Compute sparsity metrics for attention weights.

    Args:
        attn: [B, num_heads, N, N] attention weights
        threshold: Threshold below which attention is considered "sparse"
        top_k: Number of top positions to consider for concentration

    Returns:
        Dict with:
            - sparsity/threshold: Fraction of weights below threshold
            - sparsity/top_k_concentration: Mass in top-k positions
            - sparsity/gini: Gini coefficient (0=equal, 1=concentrated)
    """
    # Threshold sparsity: what fraction of attention weights are below threshold
    sparse_mask = (attn < threshold).float()
    sparsity = sparse_mask.mean()

    # Top-k concentration: how much attention mass is in top-k positions
    k = min(top_k, attn.shape[-1])
    top_k_vals, _ = torch.topk(attn, k=k, dim=-1)
    top_k_mass = top_k_vals.sum(dim=-1).mean()

    # Gini coefficient (measures inequality of attention distribution)
    B, H, N, _ = attn.shape
    attn_flat = attn.view(B, H, -1)  # [B, H, N*N]
    sorted_attn, _ = torch.sort(attn_flat, dim=-1)
    n = sorted_attn.shape[-1]
    index = torch.arange(1, n + 1, device=attn.device, dtype=attn.dtype)
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    gini = (2 * (index * sorted_attn).sum(dim=-1) / (n * sorted_attn.sum(dim=-1) + eps)) - (n + 1) / n
    gini = gini.mean()

    return {
        'sparsity/threshold': sparsity,
        'sparsity/top_k_concentration': top_k_mass,
        'sparsity/gini': gini,
    }


# Small constant for numerical stability
eps = 1e-10


def compute_head_diversity(attn: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute head diversity metrics.

    Measures whether different attention heads learn different patterns.
    Low diversity = heads are redundant and could be pruned.

    Args:
        attn: [B, num_heads, N, N] attention weights

    Returns:
        Dict with:
            - diversity/avg_similarity: Average pairwise cosine similarity
            - diversity/score: 1 - avg_similarity (higher = more diverse)
            - diversity/similarity_matrix: [H, H] similarity matrix for visualization
    """
    B, H, N, _ = attn.shape

    # Flatten attention patterns per head (average over batch first)
    attn_flat = attn.mean(dim=0).view(H, -1)  # [H, N*N]

    # Normalize for cosine similarity
    attn_norm = F.normalize(attn_flat, p=2, dim=-1)

    # Pairwise cosine similarity matrix
    sim_matrix = torch.mm(attn_norm, attn_norm.t())  # [H, H]

    # Average pairwise similarity (excluding diagonal)
    mask = ~torch.eye(H, device=attn.device, dtype=torch.bool)
    if mask.sum() > 0:
        avg_similarity = sim_matrix[mask].mean()
    else:
        avg_similarity = torch.tensor(0.0, device=attn.device)

    # Diversity score: higher = more diverse heads
    diversity_score = 1 - avg_similarity

    return {
        'diversity/avg_similarity': avg_similarity,
        'diversity/score': diversity_score,
        'diversity/similarity_matrix': sim_matrix,
    }


def compute_spatial_metrics(
    attn: torch.Tensor,
    spatial_size: int,
    num_registers: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Compute spatial attention metrics for vision models.

    Measures local vs global attention patterns:
    - Local attention: focuses on nearby patches
    - Global attention: attends to distant patches

    Args:
        attn: [B, num_heads, N, N] attention weights
        spatial_size: Height/width of spatial grid (assumes square)
        num_registers: Number of register tokens to skip

    Returns:
        Dict with:
            - spatial/avg_distance: Average distance attended to
            - spatial/local_ratio: Fraction of attention to nearby patches
    """
    B, H, N, _ = attn.shape

    # Skip register tokens
    if num_registers > 0:
        attn = attn[:, :, num_registers:, num_registers:]

    num_patches = attn.shape[-1]
    expected_patches = spatial_size * spatial_size

    # Verify dimensions match
    if num_patches != expected_patches:
        # Return placeholder values if dimensions don't match
        return {
            'spatial/avg_distance': torch.tensor(0.0, device=attn.device),
            'spatial/local_ratio': torch.tensor(0.0, device=attn.device),
        }

    # Create coordinate grids
    coords = torch.stack(torch.meshgrid(
        torch.arange(spatial_size, device=attn.device, dtype=attn.dtype),
        torch.arange(spatial_size, device=attn.device, dtype=attn.dtype),
        indexing='ij'
    ), dim=-1)  # [H, W, 2]
    coords_flat = coords.view(-1, 2)  # [N, 2]

    # Compute pairwise L2 distances
    diff = coords_flat.unsqueeze(0) - coords_flat.unsqueeze(1)  # [N, N, 2]
    distances = torch.norm(diff, dim=-1)  # [N, N]

    # Weighted average distance attended to
    avg_distance = (attn * distances.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()

    # Local attention ratio (within 2 patch distance)
    local_threshold = 2.0
    local_mask = (distances < local_threshold).float()
    local_attn = (attn * local_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()

    return {
        'spatial/avg_distance': avg_distance,
        'spatial/local_ratio': local_attn,
    }


def compute_all_metrics(
    attn: torch.Tensor,
    spatial_size: Optional[int] = None,
    num_registers: int = 0,
    sparsity_threshold: float = 0.01,
    top_k: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    Compute all attention quality metrics.

    Args:
        attn: [B, num_heads, N, N] attention weights (post-softmax)
        spatial_size: For spatial metrics (vision models). If None, skip spatial metrics.
        num_registers: Number of register tokens to skip for spatial metrics
        sparsity_threshold: Threshold for sparsity calculation
        top_k: Top-k for concentration metric

    Returns:
        Combined dict of all metrics. Scalar metrics can be logged directly,
        matrix metrics (like similarity_matrix) are for visualization.
    """
    metrics = {}

    # Core metrics (always computed)
    metrics.update(compute_entropy(attn))
    metrics.update(compute_sparsity(attn, threshold=sparsity_threshold, top_k=top_k))
    metrics.update(compute_head_diversity(attn))

    # Spatial metrics (only for vision models with known spatial layout)
    if spatial_size is not None and spatial_size > 0:
        metrics.update(compute_spatial_metrics(attn, spatial_size, num_registers))

    return metrics


def get_scalar_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Extract only scalar metrics (for logging).

    Args:
        metrics: Dict from compute_all_metrics

    Returns:
        Dict with only scalar values (matrices filtered out)
    """
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in metrics.items()
        if isinstance(v, torch.Tensor) and v.dim() == 0
    }
