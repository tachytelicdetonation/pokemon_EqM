# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def compute_spatial_decay_bias(dimensions, device, dtype, decay_rate=0.1, decay_radius=0.25, num_registers=0):
    """
    Compute uniform spatial decay bias in normalized [-1,+1] coordinate space.

    Patches within `decay_radius` attend freely (bias=0). Beyond that radius,
    a soft linear decay penalizes distant attention. Uses normalized coordinates
    for resolution-invariant behavior.

    Args:
        dimensions: Tuple of spatial dimensions (H, W)
        device: Target device
        dtype: Target dtype
        decay_rate: Decay factor outside radius (higher = stronger locality)
        decay_radius: Radius in normalized space [0, ~2.83]. Default 0.25 covers ~8 patches at 32x32.
        num_registers: Number of register tokens (they attend freely to everything)

    Returns:
        Spatial bias tensor of shape [1, 1, N, N] for broadcasting
    """
    H, W = dimensions
    num_patches = H * W

    # Normalized coordinates [-1, +1] matching LieRE DINOv3 mode
    y_norm = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2 - 1
    x_norm = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2 - 1
    yy, xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [H*W, 2]

    # Compute pairwise Euclidean distance in normalized space
    dist = torch.cdist(coords, coords, p=2)  # [H*W, H*W]

    # Free attention within radius, soft decay outside
    effective_dist = torch.clamp(dist - decay_radius, min=0)

    # Convert to negative bias (farther beyond radius = more negative = less attention)
    spatial_bias = -decay_rate * effective_dist  # [H*W, H*W]

    # Handle register tokens: they attend freely to everything
    if num_registers > 0:
        total_len = num_registers + num_patches
        full_bias = torch.zeros(total_len, total_len, device=device, dtype=torch.float32)
        full_bias[num_registers:, num_registers:] = spatial_bias
        spatial_bias = full_bias

    return spatial_bias.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]


def compute_patch_complexity(
    x: torch.Tensor,
    method: str = 'variance',
    num_registers: int = 0,
    normalize: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-patch complexity scores to guide attention toward hard regions.

    Based on 2024-2025 research on anti-curriculum learning and hardness-aware training:
    - HARDY-MER (arXiv:2508.06800): Multi-view hardness evaluation
    - TIACBM (ACL 2025): Task-informed anti-curriculum

    Higher complexity = harder to model = should receive more attention.

    Args:
        x: Input tensor [B, N, C] (patch embeddings)
        method: Complexity estimation method:
            - 'variance': Per-patch feature variance (simple, effective)
            - 'gradient': Gradient magnitude between adjacent patches
            - 'entropy': Shannon entropy of feature distribution
            - 'combined': Weighted combination of all methods
        num_registers: Number of register tokens to skip (always complexity=1)
        normalize: Normalize to [0, 1] range per sample
        temperature: Temperature for softmax-style normalization (higher = sharper)

    Returns:
        Complexity scores [B, N] in [0, 1] range (higher = more complex/harder)
    """
    B, N, C = x.shape

    # Handle register tokens (they always get complexity=1 to not be penalized)
    if num_registers > 0:
        x_patches = x[:, num_registers:]  # [B, N-num_reg, C]
    else:
        x_patches = x

    N_patches = x_patches.shape[1]

    if method == 'variance':
        # Feature variance per patch - higher variance = more information
        complexity = x_patches.var(dim=-1)  # [B, N_patches]

    elif method == 'gradient':
        # Gradient magnitude - high gradient = edge/detail region
        # Reshape to 2D grid for spatial gradients
        side = int(N_patches ** 0.5)
        if side * side == N_patches:
            x_2d = x_patches.view(B, side, side, C)
            # Sobel-like gradient approximation
            grad_x = (x_2d[:, :, 1:, :] - x_2d[:, :, :-1, :]).abs().mean(dim=-1)  # [B, H, W-1]
            grad_y = (x_2d[:, 1:, :, :] - x_2d[:, :-1, :, :]).abs().mean(dim=-1)  # [B, H-1, W]
            # Pad to original size - grad_x needs padding on dim 2, grad_y on dim 1
            grad_x = F.pad(grad_x, (0, 1), mode='replicate')  # [B, H, W]
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')  # [B, H, W]
            complexity = (grad_x + grad_y).view(B, N_patches)
        else:
            # Fallback to 1D gradient for non-square
            grad = (x_patches[:, 1:, :] - x_patches[:, :-1, :]).abs().mean(dim=-1)
            complexity = F.pad(grad, (0, 1), mode='replicate')

    elif method == 'entropy':
        # Shannon entropy of softmax-normalized features
        # Higher entropy = more distributed features = more complex
        eps = 1e-7
        probs = F.softmax(x_patches / temperature, dim=-1)  # [B, N_patches, C]
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # [B, N_patches]
        complexity = entropy

    elif method == 'combined':
        # Weighted combination of all methods
        var_score = x_patches.var(dim=-1)
        # Simple gradient approximation
        grad = torch.zeros(B, N_patches, device=x.device, dtype=x.dtype)
        grad[:, 1:] = (x_patches[:, 1:, :] - x_patches[:, :-1, :]).abs().mean(dim=-1)
        grad[:, 0] = grad[:, 1]
        # Entropy
        eps = 1e-7
        probs = F.softmax(x_patches / temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        # Normalize each and combine
        var_norm = (var_score - var_score.min(dim=-1, keepdim=True)[0]) / (var_score.max(dim=-1, keepdim=True)[0] - var_score.min(dim=-1, keepdim=True)[0] + eps)
        grad_norm = (grad - grad.min(dim=-1, keepdim=True)[0]) / (grad.max(dim=-1, keepdim=True)[0] - grad.min(dim=-1, keepdim=True)[0] + eps)
        entropy_norm = (entropy - entropy.min(dim=-1, keepdim=True)[0]) / (entropy.max(dim=-1, keepdim=True)[0] - entropy.min(dim=-1, keepdim=True)[0] + eps)
        complexity = 0.4 * var_norm + 0.4 * grad_norm + 0.2 * entropy_norm

    else:
        raise ValueError(f"Unknown complexity method: {method}")

    # Normalize to [0, 1]
    if normalize:
        min_c = complexity.min(dim=-1, keepdim=True)[0]
        max_c = complexity.max(dim=-1, keepdim=True)[0]
        complexity = (complexity - min_c) / (max_c - min_c + 1e-7)

    # Handle register tokens - they always get max complexity (1.0)
    if num_registers > 0:
        reg_complexity = torch.ones(B, num_registers, device=x.device, dtype=x.dtype)
        complexity = torch.cat([reg_complexity, complexity], dim=1)

    return complexity


def compute_complexity_attention_bias(
    complexity: torch.Tensor,
    scale: float = 1.0,
    mode: str = 'additive',
) -> torch.Tensor:
    """
    Convert patch complexity scores to attention bias.

    Biases attention toward high-complexity (hard) patches and away from
    low-complexity (easy) patches. Based on anti-curriculum learning research.

    Args:
        complexity: Per-patch complexity [B, N] in [0, 1]
        scale: Scaling factor for bias strength
        mode: Bias mode:
            - 'additive': Add bias to attention logits (softer)
            - 'multiplicative': Scale attention logits (sharper)

    Returns:
        Attention bias [B, 1, 1, N] for broadcasting to [B, H, N_q, N_k]
    """
    # Higher complexity = higher bias = more attention
    if mode == 'additive':
        # Shift complexity to [-0.5, 0.5] range, then scale
        # This penalizes low-complexity keys and boosts high-complexity ones
        bias = (complexity - 0.5) * scale
    elif mode == 'multiplicative':
        # Use as a multiplicative factor (complexity as soft mask)
        bias = complexity * scale
    else:
        raise ValueError(f"Unknown bias mode: {mode}")

    # Reshape for attention broadcasting: [B, 1, 1, N]
    return bias.unsqueeze(1).unsqueeze(2)


def compute_per_head_spatial_decay_bias(
    dimensions, device, dtype, num_heads,
    base_decay_rate=0.1, base_decay_radius=0.25, num_registers=0,
    decay_type='exponential', distance_type='l2', content_gate=None
):
    """
    ALiBi-style per-head spatial decay in normalized [-1,+1] coordinate space.

    Different attention heads have different locality preferences:
    - Head 0: Most local (high decay rate, small radius)
    - Head num_heads-1: Most global (low decay rate, large radius)

    Uses normalized coordinates [-1, +1] consistent with LieRE DINOv3 mode,
    making the bias resolution-invariant.

    References:
    - ALiBi (Attention with Linear Biases) extended to 2D
    - Radial Attention: exponential decay like physical signal decay
    - SDT (Spatial Decay Transformer): content-aware gating

    Args:
        dimensions: Tuple of spatial dimensions (H, W)
        device: Target device
        dtype: Target dtype
        num_heads: Number of attention heads
        base_decay_rate: Base decay rate for head 0 (most local)
        base_decay_radius: Base free-attention radius in normalized space for head 0
        num_registers: Number of register tokens (attend freely to everything)
        decay_type: 'exponential' (natural signal decay) or 'linear' (original)
        distance_type: 'l2' (Euclidean) or 'l1' (Manhattan)
        content_gate: Optional [B, num_heads, 1, 1] tensor from ContentAwareGate

    Returns:
        Spatial bias tensor of shape [1, num_heads, N, N] for per-head broadcasting
    """
    H, W = dimensions
    num_patches = H * W

    # Normalized coordinates [-1, +1] matching LieRE DINOv3 mode
    y_norm = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2 - 1
    x_norm = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2 - 1
    yy, xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [H*W, 2]

    # Compute pairwise distance in normalized space
    if distance_type == 'l1':
        # Manhattan distance: |y1-y2| + |x1-x2|
        dist = torch.cdist(coords, coords, p=1)  # Max ~4.0 (corner to corner)
    else:
        # Euclidean distance: sqrt((y1-y2)^2 + (x1-x2)^2)
        dist = torch.cdist(coords, coords, p=2)  # Max ~2.83 (corner to corner)

    # ALiBi geometric series: head 0 = most local, head num_heads-1 = most global
    head_idx = torch.arange(num_heads, device=device, dtype=torch.float32)
    decay_rates = base_decay_rate / (2 ** (8 * head_idx / num_heads))
    decay_radii = base_decay_radius * (2 ** (head_idx / num_heads))

    # Broadcast to compute per-head biases: [num_heads, H*W, H*W]
    dist_exp = dist.unsqueeze(0)  # [1, H*W, H*W]
    radii_exp = decay_radii.view(-1, 1, 1)  # [num_heads, 1, 1]
    rates_exp = decay_rates.view(-1, 1, 1)  # [num_heads, 1, 1]

    # effective_dist[h, i, j] = max(0, dist[i, j] - radius[h])
    effective_dist = torch.clamp(dist_exp - radii_exp, min=0)

    # Apply decay type
    if decay_type == 'exponential':
        # Exponential decay: bias approaches -rate as dist -> inf (like physical signal decay)
        # Formula: -rate * (1 - exp(-scale * dist))
        if content_gate is not None:
            # Content-modulated: gate [B, num_heads, 1, 1] controls decay strength
            # Note: content_gate will be applied in the attention forward pass
            # Here we just compute the base exponential decay
            spatial_bias = -rates_exp * (1 - torch.exp(-effective_dist))
        else:
            spatial_bias = -rates_exp * (1 - torch.exp(-effective_dist))
    else:
        # Linear decay (original): -rate * dist
        spatial_bias = -rates_exp * effective_dist

    # Handle register tokens: they attend freely to everything
    if num_registers > 0:
        total_len = num_registers + num_patches
        full_bias = torch.zeros(num_heads, total_len, total_len, device=device, dtype=torch.float32)
        full_bias[:, num_registers:, num_registers:] = spatial_bias
        spatial_bias = full_bias

    return spatial_bias.to(dtype=dtype).unsqueeze(0)  # [1, num_heads, N, N]


class LearnedSpatialDecay(nn.Module):
    """
    SDT-style learned spatial decay for attention.

    Instead of fixed decay formulas, learns per-token decay from content.
    Uses log-sigmoid to ensure decay is always non-positive (reduces attention).

    Formula:
        G[i] = log(sigmoid(x[i] @ W))  # per-token decay in (-∞, 0]
        bias[i,j] = 0.5 * (G[i] + G[j]) * distance(i,j) * alpha

    Reference: "Learning Spatial Decay for Vision Transformers" (SDT, 2025)
    """
    def __init__(self, head_dim, num_heads, distance_type='l2'):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.distance_type = distance_type

        # Per-token decay projection (shared across heads for efficiency)
        self.decay_proj = nn.Linear(head_dim, 1)

        # Learnable alpha per head (controls decay strength)
        self.alpha = nn.Parameter(torch.ones(num_heads) * 0.1)

    def forward(self, x, dimensions, num_registers=0):
        """
        Compute learned spatial decay bias.

        Args:
            x: Query or Key tensor [B, num_heads, N, head_dim]
            dimensions: (H, W) spatial dimensions
            num_registers: Number of register tokens (get zero decay)

        Returns:
            bias: [B, num_heads, N, N] with values ≤ 0
        """
        B, num_heads, N, head_dim = x.shape
        H, W = dimensions
        num_patches = H * W

        # Compute per-token decay using log-sigmoid (always ≤ 0)
        decay_logits = self.decay_proj(x)  # [B, num_heads, N, 1]
        G = F.logsigmoid(decay_logits)  # [B, num_heads, N, 1], in (-∞, 0]

        # Compute distance matrix (normalized coordinates)
        y_norm = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
        x_norm = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)

        if self.distance_type == 'l1':
            dist = torch.cdist(coords, coords, p=1)
        else:
            dist = torch.cdist(coords, coords, p=2)

        # Handle registers
        if num_registers > 0:
            G_patch = G[:, :, num_registers:, :]  # [B, num_heads, num_patches, 1]

            # Symmetric combination: M[i,j] = 0.5 * (G[i] + G[j])
            G_i = G_patch
            G_j = G_patch.transpose(-2, -1)
            decay_sym = 0.5 * (G_i + G_j)  # [B, num_heads, num_patches, num_patches]

            # Scale by distance and learnable alpha
            alpha = self.alpha.view(1, num_heads, 1, 1)
            bias_patches = decay_sym * dist.unsqueeze(0).unsqueeze(0) * alpha

            # Full bias with registers (registers have zero bias)
            full_bias = torch.zeros(B, num_heads, N, N, device=x.device, dtype=x.dtype)
            full_bias[:, :, num_registers:, num_registers:] = bias_patches
            return full_bias
        else:
            G_i = G
            G_j = G.transpose(-2, -1)
            decay_sym = 0.5 * (G_i + G_j)

            alpha = self.alpha.view(1, num_heads, 1, 1)
            return decay_sym * dist.unsqueeze(0).unsqueeze(0) * alpha


class SigmaReparam(nn.Module):
    """
    Spectral reparametrization for QKV projections (σReparam from Apple/ICLR 2025).

    Bounds the spectral norm of weight matrices to prevent attention entropy collapse.
    Uses power iteration for efficient spectral norm estimation.

    The key insight is that bounding ||W||_2 bounds ||QK^T||_F which bounds the maximum
    attention logit, preventing entropy collapse even without explicit entropy losses.

    Reference: "Stabilizing Transformer Training by Preventing Attention Entropy Collapse"
               Apple ML Research / ICLR 2025
    """
    def __init__(self, linear: nn.Linear, target_sigma: float = 1.0, n_power_iterations: int = 1):
        """
        Args:
            linear: The linear layer to wrap (typically QKV projection)
            target_sigma: Target spectral norm (default 1.0)
            n_power_iterations: Number of power iterations for spectral norm estimation
        """
        super().__init__()
        self.linear = linear
        self.target_sigma = target_sigma
        self.n_power_iterations = n_power_iterations

        # Initialize singular vectors for power iteration
        h, w = linear.weight.shape
        self.register_buffer('u', F.normalize(torch.randn(h), dim=0))
        self.register_buffer('v', F.normalize(torch.randn(w), dim=0))

    def _compute_spectral_norm(self):
        """Compute spectral norm via power iteration."""
        weight = self.linear.weight
        u, v = self.u, self.v

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight.t(), u), dim=0)
                u = F.normalize(torch.mv(weight, v), dim=0)
            self.u.copy_(u)
            self.v.copy_(v)

        # Compute spectral norm: sigma = u^T @ W @ v
        sigma = torch.dot(u, torch.mv(weight, v))
        return sigma

    def forward(self, x):
        """Apply spectral-normalized linear transformation."""
        sigma = self._compute_spectral_norm()
        scale = self.target_sigma / (sigma + 1e-8)
        return F.linear(x, self.linear.weight * scale, self.linear.bias)


class OutputGate(nn.Module):
    """
    G1 Output Gating for Attention Sink Elimination.

    Applies a head-specific, query-dependent sigmoid gate AFTER the attention computation,
    allowing the model to output "nothing" (sparse outputs) instead of being forced to
    attend somewhere due to softmax normalization.

    This eliminates attention sinks by breaking the sum-to-1 constraint at the output level.

    Formula:
        gate[b,h,n] = sigmoid((q_mean[b,h,n] @ W_gate + bias) / temperature)
        output = gate * attention_output

    When gate ≈ 0, the position outputs near-zero regardless of attention weights.

    Reference: "Gated Attention" (arXiv:2505.06708) - May 2025
    """
    def __init__(self, head_dim, num_heads, gate_init_bias=-2.0, temperature=1.0, gate_type='per_token'):
        """
        Args:
            head_dim: Dimension per attention head
            num_heads: Number of attention heads
            gate_init_bias: Initial bias (negative = gates start more closed, conservative)
            temperature: Temperature for sigmoid (higher = softer gates)
            gate_type: 'per_token' (N gates), 'per_head' (num_heads gates), or 'global' (1 gate)
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.gate_type = gate_type

        # Query projection to gate logits
        if gate_type == 'per_token':
            # Per-token, per-head gating: most expressive
            self.gate_proj = nn.Linear(head_dim, 1, bias=False)
            self.gate_bias = nn.Parameter(torch.full((num_heads,), gate_init_bias))
        elif gate_type == 'per_head':
            # Per-head gating only: simpler, less parameters
            self.gate_proj = nn.Linear(head_dim, 1, bias=False)
            self.gate_bias = nn.Parameter(torch.full((num_heads,), gate_init_bias))
        else:  # global
            # Single global gate: simplest
            self.gate_bias = nn.Parameter(torch.tensor(gate_init_bias))

        # Learnable temperature per head for adaptive sharpness
        self.learned_temp = nn.Parameter(torch.ones(num_heads) * temperature)

        self._init_weights()

    def _init_weights(self):
        """Initialize gate projection small so gates start near sigmoid(bias)."""
        if hasattr(self, 'gate_proj'):
            nn.init.normal_(self.gate_proj.weight, std=0.01)

    def forward(self, q, out, num_registers=0, return_gate=False):
        """
        Compute and apply output gate.

        Args:
            q: Query tensor [B, num_heads, N, head_dim]
            out: Attention output [B, num_heads, N, out_dim]
            num_registers: Number of register tokens (always get gate=1, no suppression)
            return_gate: If True, also return gate values for auxiliary losses

        Returns:
            Gated output [B, num_heads, N, out_dim]
            If return_gate=True: tuple of (gated_output, gate_values [B, H, N])
        """
        B, num_heads, N, _ = q.shape

        # Compute gate logits
        if self.gate_type == 'per_token':
            # Project query to scalar per token
            gate_logits = self.gate_proj(q).squeeze(-1)  # [B, H, N]
            gate_logits = gate_logits + self.gate_bias.view(1, num_heads, 1)
        elif self.gate_type == 'per_head':
            # Average query across tokens, then project
            q_mean = q.mean(dim=2)  # [B, H, head_dim]
            gate_logits = self.gate_proj(q_mean).squeeze(-1)  # [B, H]
            gate_logits = gate_logits + self.gate_bias
            gate_logits = gate_logits.unsqueeze(-1).expand(-1, -1, N)  # [B, H, N]
        else:  # global
            gate_logits = self.gate_bias.expand(B, num_heads, N)

        # Apply temperature-scaled sigmoid
        temp = self.learned_temp.view(1, num_heads, 1).clamp(min=0.1)
        gate = torch.sigmoid(gate_logits / temp)  # [B, H, N]

        # Register tokens should never be gated (they're meant to absorb attention)
        if num_registers > 0:
            gate = gate.clone()
            gate[:, :, :num_registers] = 1.0

        # Store gate values before adding dimension (for aux loss computation)
        gate_values = gate  # [B, H, N]

        # Apply gate to output
        gate = gate.unsqueeze(-1)  # [B, H, N, 1]
        gated_out = out * gate

        if return_gate:
            return gated_out, gate_values
        return gated_out


class MatrixGatedLambda(nn.Module):
    """
    M-DGSA: Content-dependent N×N lambda matrix from Q and K.

    Instead of a single scalar lambda applied uniformly, computes a per-position-pair
    lambda value based on query-key content, enabling content-dependent noise cancellation.

    Formula:
        lambda[i,j] = lambda_init + scale * (sigmoid(logits[i,j] / temp) - 0.5)
        where logits[i,j] = q_i @ W_q + k_j @ W_k + sum_d(W_qk[d] * q[i,d] * k[j,d])

    Reference: M-DGSA extends Microsoft Diff-Transformer (arXiv:2410.05258)
    """
    def __init__(self, head_dim, num_heads, lambda_init=0.5, scale=1.0, use_qk_interaction=True):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.lambda_init = lambda_init
        self.scale = scale
        self.use_qk_interaction = use_qk_interaction

        # Query and key contributions to gating logits
        self.W_q = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.W_k = nn.Parameter(torch.zeros(num_heads, head_dim))

        # QK multiplicative interaction for richer content modeling
        if use_qk_interaction:
            self.W_qk = nn.Parameter(torch.zeros(num_heads, head_dim))

        # Per-head learnable bias and temperature
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.temperature = nn.Parameter(torch.ones(num_heads))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights small so lambda starts near lambda_init."""
        nn.init.normal_(self.W_q, std=0.01)
        nn.init.normal_(self.W_k, std=0.01)
        if self.use_qk_interaction:
            nn.init.normal_(self.W_qk, std=0.01)

    def forward(self, q1, k1):
        """
        Compute N×N lambda gating matrix.

        Args:
            q1: Query tensor [B, num_heads, N, head_dim]
            k1: Key tensor [B, num_heads, N, head_dim]

        Returns:
            lambda_matrix: [B, num_heads, N, N] content-dependent lambda values
        """
        # Compute per-query and per-key contributions
        q_logits = torch.einsum('bhnd,hd->bhn', q1, self.W_q)  # [B, H, N]
        k_logits = torch.einsum('bhnd,hd->bhn', k1, self.W_k)  # [B, H, N]

        # Broadcast to [B, H, N, N]: outer sum
        logits = q_logits.unsqueeze(-1) + k_logits.unsqueeze(-2)

        # Add QK multiplicative interaction if enabled
        if self.use_qk_interaction:
            # Bilinear interaction: sum_d W_qk[h,d] * q[b,h,i,d] * k[b,h,j,d]
            qk_interaction = torch.einsum('bhid,bhjd,hd->bhij', q1, k1, self.W_qk)
            logits = logits + qk_interaction

        # Add per-head bias
        logits = logits + self.bias.view(1, -1, 1, 1)

        # Apply temperature-scaled sigmoid
        temp = self.temperature.view(1, -1, 1, 1).clamp(min=0.1)
        gate = torch.sigmoid(logits / temp)

        # Scale to [lambda_init - scale/2, lambda_init + scale/2]
        return self.lambda_init + self.scale * (gate - 0.5)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#################################################################################
#                                 Core EqM Model                                #
#################################################################################

class LieRE(nn.Module):
    """
    Lie Rotational Positional Encodings (LieRE) - learnable generalization of RoPE.
    Uses Lie group theory to create rotation matrices via matrix exponentials of
    skew-symmetric matrices (Lie algebra elements).

    Reference: https://arxiv.org/abs/2406.10322
    """
    def __init__(self, num_dim, dim, jitter_std=0.0, jitter_mode='gaussian',
                 pos_embed_shift=None, pos_embed_jitter=None, pos_embed_rescale=2.0):
        """
        Args:
            num_dim: Number of spatial dimensions (2 for images: H, W)
            dim: Head dimension
            jitter_std: Standard deviation for Gaussian jittering (when mode='gaussian').
                        Default 0.0 (disabled).
            jitter_mode: Jittering mode - 'gaussian' (simple) or 'dinov3' (log-uniform).
                        Default 'gaussian'.
            pos_embed_shift: DINOv3-style uniform shift in [-shift, shift]. None = disabled.
            pos_embed_jitter: DINOv3-style log-uniform jitter in [1/jitter, jitter]. None = disabled.
            pos_embed_rescale: DINOv3-style global rescale in [1/rescale, rescale]. Default 2.0.
        """
        super().__init__()
        self.num_dim = num_dim
        self.dim = dim
        self.jitter_std = jitter_std
        self.jitter_mode = jitter_mode
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale

        # Learnable generator parameters (Lie algebra)
        # Initialize with small random values
        self.generator_params = nn.Parameter(
            torch.randn(num_dim, dim, dim) * 0.02
        )

    def _make_skew_symmetric(self, matrices):
        """
        Convert arbitrary matrices to skew-symmetric matrices (A^T = -A).
        This ensures the matrix exponential produces valid rotation matrices.

        Args:
            matrices: Tensor of shape [..., dim, dim]

        Returns:
            Skew-symmetric matrices of same shape
        """
        # Extract upper triangular part (excluding diagonal)
        upper_tri = torch.triu(matrices, diagonal=1)
        # Create skew-symmetric: A - A^T
        skew = upper_tri - upper_tri.transpose(-2, -1)
        return skew

    def _get_jittered_positions(self, dimensions, device, training):
        """
        Generate position grid with optional jittering (Gaussian or DINOv3-style).

        Supports two jittering modes:
        - 'gaussian': Simple additive Gaussian noise
        - 'dinov3': Normalized coords + shift/jitter/rescale (log-uniform)

        Args:
            dimensions: Tuple of spatial dimensions (H, W) for 2D
            device: Target device
            training: Whether in training mode

        Returns:
            Position tensor of shape [H*W, num_dim]
        """
        if self.jitter_mode == 'dinov3':
            # DINOv3-style: normalized coordinates in [-1, +1]
            coords_list = []
            for dim_size in dimensions:
                # Patch centers: [0.5, 1.5, 2.5, ..., dim_size-0.5]
                coords = torch.arange(0.5, dim_size, dtype=torch.float32, device=device)
                # Normalize to [-1, +1]
                coords = coords / dim_size
                coords = 2.0 * coords - 1.0
                coords_list.append(coords)

            # Create meshgrid
            grids = torch.meshgrid(*coords_list, indexing='ij')
            positions = torch.stack([grid.flatten() for grid in grids], dim=1)

            # Apply DINOv3 augmentations during training
            if training:
                positions = self._augment_positions_dinov3(positions, device)

        elif self.jitter_mode == 'gaussian':
            # Original Gaussian mode
            if training and self.jitter_std > 0:
                # Generate base coordinate ranges for each dimension
                ranges = []
                for dim_size in dimensions:
                    # Base positions: [0, 1, 2, ..., dim_size-1]
                    base_pos = torch.arange(dim_size, device=device, dtype=torch.float32)
                    # Add Gaussian jitter
                    jitter = torch.randn(dim_size, device=device) * self.jitter_std
                    jittered_pos = base_pos + jitter
                    ranges.append(jittered_pos)

                # Create meshgrid and flatten
                grids = torch.meshgrid(*ranges, indexing='ij')
                positions = torch.stack([grid.flatten() for grid in grids], dim=1)
            else:
                # No jittering
                positions = torch.cartesian_prod(
                    *(torch.arange(dim_size, device=device, dtype=torch.float32)
                      for dim_size in dimensions)
                )
        else:
            raise ValueError(f"Unknown jitter_mode: {self.jitter_mode}")

        return positions  # Shape: [H*W, num_dim]

    def _augment_positions_dinov3(self, coords, device):
        """
        Apply DINOv3-style coordinate augmentations: shift, jitter, rescale.

        Args:
            coords: Position coordinates [H*W, num_dim]
            device: Target device

        Returns:
            Augmented coordinates [H*W, num_dim]
        """
        import numpy as np

        # 1. Shift: Uniform addition in [-shift, shift]
        if self.pos_embed_shift is not None:
            shift_hw = torch.empty((1, self.num_dim), device=device, dtype=coords.dtype)
            shift_hw = shift_hw.uniform_(-self.pos_embed_shift, self.pos_embed_shift)
            coords = coords + shift_hw

        # 2. Jitter: Log-uniform multiplication per dimension [1/jitter, jitter]
        if self.pos_embed_jitter is not None:
            jitter_range = np.log(self.pos_embed_jitter)
            jitter_hw = torch.empty((1, self.num_dim), device=device, dtype=coords.dtype)
            jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
            coords = coords * jitter_hw

        # 3. Rescale: Global log-uniform scaling [1/rescale, rescale]
        if self.pos_embed_rescale is not None:
            rescale_range = np.log(self.pos_embed_rescale)
            rescale_hw = torch.empty(1, device=device, dtype=coords.dtype)
            rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
            coords = coords * rescale_hw

        return coords

    def _get_rotations(self, dimensions, device, dtype, training=False):
        """
        Generate rotation matrices for given spatial dimensions using Cayley transform.

        The Cayley transform maps a skew-symmetric matrix A to a rotation matrix R:
        R = (I - A)^{-1} (I + A)

        This is computationally cheaper than matrix exponential and compatible with
        CUDA graphs / torch.compile.

        Args:
            dimensions: Tuple of spatial dimensions (H, W) for 2D
            device: Target device
            dtype: Target dtype
            training: Whether in training mode (enables jittering if configured)

        Returns:
            Rotation matrices of shape [num_positions, dim, dim]
        """
        # Create skew-symmetric matrices using Stanford's efficient approach
        # Extract upper triangular part and make skew-symmetric
        upper_triangle = torch.triu(self.generator_params, diagonal=1)
        skew_matrices = upper_triangle - upper_triangle.transpose(-1, -2)  # [num_dim, dim, dim]

        # Generate positions with optional jittering (DINOv3-style)
        # For 2D with dimensions=(32, 32): creates [1024, 2] tensor
        positions = self._get_jittered_positions(dimensions, device, training)  # Shape: [H*W, num_dim]

        # Vectorized computation using broadcasting (Stanford MIMI approach)
        # Reshape positions: [H*W, num_dim] -> [H*W, num_dim, 1, 1]
        # skew_matrices: [num_dim, dim, dim]
        # Broadcasting multiplication: [H*W, num_dim, dim, dim]
        in_basis_positions = positions.reshape(list(positions.shape) + [1, 1]) * skew_matrices

        # Sum over dimensions to get generator for each position: [H*W, dim, dim]
        A = torch.sum(in_basis_positions, dim=1)

        # Apply Cayley transform: R = (I - A)^-1 (I + A)
        # This is much faster than matrix_exp and supports CUDA graphs
        I = torch.eye(self.dim, device=device, dtype=dtype).unsqueeze(0) # Broadcastable Identity
        
        # Ensure A is in the correct dtype
        A = A.to(dtype=dtype)
        
        # Solve (I - A) * R = (I + A)
        # Using explicit inverse for better compatibility across backends (MPS/CUDA)
        # R = (I - A)^-1 @ (I + A)
        numerator = I + A
        denominator = I - A
        
        # Cast to float32 for inversion as linalg.inv doesn't support bfloat16
        rotation_matrices = torch.linalg.inv(denominator.float()) @ numerator.float()

        return rotation_matrices.to(dtype=dtype)  # [num_positions, dim, dim]

    def apply_rotations(self, x, dimensions):
        """
        Apply LieRE to query or key tensor.

        During training with jitter_std > 0, applies coordinate jittering to
        position grid for learning continuous positional representations.

        Args:
            x: Input tensor of shape [B, num_heads, seq_len, head_dim]
            dimensions: Tuple of spatial dimensions (H, W) for 2D

        Returns:
            Tensor with LieRE applied, same shape as input
        """
        B, num_heads, seq_len, head_dim = x.shape

        # Get rotation matrices for these dimensions (with optional jittering during training)
        rotations = self._get_rotations(dimensions, x.device, x.dtype, training=self.training)  # [seq_len, head_dim, head_dim]

        # Apply rotation: x @ R^T for each position
        # x: [B, num_heads, seq_len, head_dim]
        # rotations: [seq_len, head_dim, head_dim]

        # Reshape for batched matrix multiplication
        x_flat = x.reshape(B * num_heads, seq_len, head_dim)  # [B*num_heads, seq_len, head_dim]

        # Vectorized rotation application using einsum (NO LOOPS!)
        # x_flat: [B*num_heads, seq_len, head_dim]
        # rotations: [seq_len, head_dim, head_dim]
        # For each position i: output[:, i, :] = x_flat[:, i, :] @ rotations[i].T
        # Einsum notation: 'bnd,ned->bne' where b=batch, n=seq_len, d=head_dim, e=head_dim
        output = torch.einsum('bnd,ned->bne', x_flat, rotations)

        # Reshape back
        output = output.reshape(B, num_heads, seq_len, head_dim)
        return output


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        use_liere=False,
        liere_jitter_std=0.0,
        liere_jitter_mode='gaussian',
        liere_pos_embed_shift=None,
        liere_pos_embed_jitter=None,
        liere_pos_embed_rescale=2.0,
        spatial_dims=None,  # (H, W) of patch grid for LieRE
        num_registers=0,    # Number of register tokens to skip for LieRE
        use_spatial_decay=True,  # Enable spatial attention decay
        spatial_decay_rate=0.1,   # Base decay rate (higher = stronger locality)
        spatial_decay_radius=0.25,  # Base radius in normalized space
        use_per_head_decay=True,  # ALiBi-style per-head decay (different heads = different locality)
        decay_type='exponential',  # 'exponential', 'linear', or 'learned' (SDT-style)
        distance_type='l2',  # 'l2' (Euclidean) or 'l1' (Manhattan)
        use_content_gate=True,  # Content-aware gating (only for non-learned decay)
        use_output_gate=False,  # G1 output gating for attention sink elimination
        output_gate_type='per_token',  # 'per_token', 'per_head', or 'global'
        output_gate_init_bias=-2.0,  # Initial bias (negative = conservative, gates start more open)
        # σReparam for entropy collapse prevention (Apple/ICLR 2025)
        use_sigma_reparam=False,
        sigma_target=1.0,
        # Complexity bias for anti-curriculum attention (HARDY-MER, TIACBM 2024-2025)
        use_complexity_bias=False,  # Bias attention toward high-complexity patches
        complexity_method='variance',  # 'variance', 'gradient', 'entropy', 'combined'
        complexity_bias_scale=1.0,  # Scale for complexity bias strength
        # Head routing for complexity specialization (MoH 2024, attention orthogonality)
        use_head_routing=False,  # Enable learnable per-head complexity routing
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_liere = use_liere
        self.use_head_routing = use_head_routing
        self.spatial_dims = spatial_dims
        self.num_registers = num_registers
        self.use_spatial_decay = use_spatial_decay
        self.spatial_decay_rate = spatial_decay_rate
        self.spatial_decay_radius = spatial_decay_radius
        self.use_per_head_decay = use_per_head_decay
        self.decay_type = decay_type
        self.distance_type = distance_type
        self.use_content_gate = use_content_gate
        self.use_output_gate = use_output_gate
        self.use_sigma_reparam = use_sigma_reparam
        self.use_complexity_bias = use_complexity_bias
        self.complexity_method = complexity_method
        self.complexity_bias_scale = complexity_bias_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # σReparam: Wrap QKV with spectral normalization (Apple/ICLR 2025)
        if use_sigma_reparam:
            self.qkv_reparam = SigmaReparam(self.qkv, target_sigma=sigma_target)

        # SDT-style learned spatial decay (replaces content gate when decay_type='learned')
        if decay_type == 'learned' and use_spatial_decay:
            self.learned_decay = LearnedSpatialDecay(head_dim, num_heads, distance_type)
        elif use_content_gate and use_spatial_decay:
            # Legacy content-aware gate for non-learned decay types
            self.content_gate = LearnedSpatialDecay(head_dim, num_heads, distance_type)

        # G1 Output Gating for attention sink elimination (arXiv:2505.06708)
        if use_output_gate:
            self.output_gate = OutputGate(
                head_dim=head_dim,
                num_heads=num_heads,
                gate_init_bias=output_gate_init_bias,
                gate_type=output_gate_type
            )

        if self.use_liere:
            self.liere = LieRE(
                num_dim=2,
                dim=head_dim,
                jitter_std=liere_jitter_std,
                jitter_mode=liere_jitter_mode,
                pos_embed_shift=liere_pos_embed_shift,
                pos_embed_jitter=liere_pos_embed_jitter,
                pos_embed_rescale=liere_pos_embed_rescale
            )

        # Learnable per-head complexity routing (MoH 2024)
        # Initialized to 0 (neutral); training optimizes:
        # +values -> focus on complex regions, -values -> focus on simple regions
        if use_head_routing:
            self.head_routing = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer('head_routing', None)

    def forward(self, x, return_attention=False, return_aux_info=False):
        B, N, C = x.shape

        # Use σReparam QKV if enabled (for entropy collapse prevention)
        if self.use_sigma_reparam:
            qkv = self.qkv_reparam(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.use_liere:
            # Apply LieRE (learnable rotations) only to patch tokens, not register tokens
            if self.spatial_dims is not None:
                dimensions = self.spatial_dims
            else:
                # Fallback: compute from sequence length (for backwards compatibility)
                num_patches = N - self.num_registers
                side = int(num_patches ** 0.5)
                dimensions = (side, side)

            if self.num_registers > 0:
                # Split: register tokens don't get positional rotations
                q_reg, q_patch = q[:, :, :self.num_registers], q[:, :, self.num_registers:]
                k_reg, k_patch = k[:, :, :self.num_registers], k[:, :, self.num_registers:]
                # Apply rotations to patches only
                q_patch = self.liere.apply_rotations(q_patch, dimensions)
                k_patch = self.liere.apply_rotations(k_patch, dimensions)
                # Recombine
                q = torch.cat([q_reg, q_patch], dim=2)
                k = torch.cat([k_reg, k_patch], dim=2)
            else:
                q = self.liere.apply_rotations(q, dimensions)
                k = self.liere.apply_rotations(k, dimensions)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply spatial decay bias before softmax
        if self.use_spatial_decay:
            if self.spatial_dims is not None:
                dimensions = self.spatial_dims
            else:
                num_patches = N - self.num_registers
                side = int(num_patches ** 0.5)
                dimensions = (side, side)

            if self.decay_type == 'learned':
                # SDT-style: fully learned decay from content
                spatial_bias = self.learned_decay(q, dimensions, self.num_registers)
            elif self.use_per_head_decay:
                # ALiBi-style per-head decay with exponential/linear
                spatial_bias = compute_per_head_spatial_decay_bias(
                    dimensions, x.device, attn.dtype,
                    num_heads=self.num_heads,
                    base_decay_rate=self.spatial_decay_rate,
                    base_decay_radius=self.spatial_decay_radius,
                    num_registers=self.num_registers,
                    decay_type=self.decay_type,
                    distance_type=self.distance_type
                )
                # Apply content-aware gating if enabled
                if self.use_content_gate and hasattr(self, 'content_gate'):
                    content_gate = self.content_gate(q, dimensions, self.num_registers)
                    spatial_bias = spatial_bias + content_gate  # Add learned component
            else:
                # Uniform decay across all heads
                spatial_bias = compute_spatial_decay_bias(
                    dimensions, x.device, attn.dtype,
                    decay_rate=self.spatial_decay_rate,
                    decay_radius=self.spatial_decay_radius,
                    num_registers=self.num_registers
                )
            attn = attn + spatial_bias

        # Apply complexity bias to focus on hard patches (anti-curriculum attention)
        patch_complexity = None
        if self.use_complexity_bias:
            patch_complexity = compute_patch_complexity(
                x, method=self.complexity_method,
                num_registers=self.num_registers, normalize=True
            )
            complexity_bias = compute_complexity_attention_bias(
                patch_complexity, scale=self.complexity_bias_scale, mode='additive'
            )
            attn = attn + complexity_bias

        attn = attn.softmax(dim=-1)
        attn_weights = attn  # Save for visualization before dropout
        attn = self.attn_drop(attn)

        # Compute attention output
        out = attn @ v  # [B, num_heads, N, head_dim]

        # Store head outputs before gating (for HSIC diversity loss)
        head_outputs = out  # [B, num_heads, N, head_dim]

        # Apply G1 output gating if enabled (attention sink elimination)
        gate_values = None
        if self.use_output_gate:
            if return_aux_info:
                out, gate_values = self.output_gate(q, out, num_registers=self.num_registers, return_gate=True)
            else:
                out = self.output_gate(q, out, num_registers=self.num_registers)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention or return_aux_info:
            result = attn_weights
            if return_aux_info:
                result = {
                    'attn': attn_weights,
                    'head_outputs': head_outputs,
                    'gate_values': gate_values,
                    'patch_complexity': patch_complexity,
                    'head_routing': self.head_routing,  # For head specialization loss
                }
            return x, result
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (used in Differential Attention).
    """
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.weight is not None:
            return x_normed * self.weight
        return x_normed


def lambda_init_fn(depth):
    """
    Initialize lambda based on layer depth (from Microsoft Diff-Transformer paper).
    Deeper layers get larger lambda values for more aggressive noise cancellation.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialAttention(nn.Module):
    """
    Differential Attention mechanism from Microsoft's Diff-Transformer paper.
    Combines two attention computations: attn = softmax(Q1K1) - λ·softmax(Q2K2)

    This helps cancel attention noise and improves head diversity.
    Integrates with LieRE for learnable rotary positional embeddings.

    Reference: https://arxiv.org/abs/2410.05258
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        use_liere=False,
        liere_jitter_std=0.0,
        liere_jitter_mode='gaussian',
        liere_pos_embed_shift=None,
        liere_pos_embed_jitter=None,
        liere_pos_embed_rescale=2.0,
        spatial_dims=None,
        num_registers=0,
        layer_idx=0,  # Layer index for lambda initialization
        use_spatial_decay=True,  # Enable spatial attention decay
        spatial_decay_rate=0.1,   # Base decay rate (higher = stronger locality)
        spatial_decay_radius=0.25,  # Base radius in normalized space
        use_per_head_decay=True,  # ALiBi-style per-head decay
        decay_type='exponential',  # 'exponential', 'linear', or 'learned' (SDT-style)
        distance_type='l2',  # 'l2' (Euclidean) or 'l1' (Manhattan)
        use_content_gate=True,  # Content-aware gating (only for non-learned decay)
        use_matrix_lambda=False,  # M-DGSA: use content-dependent N×N lambda matrix
        matrix_lambda_scale=1.0,  # Scale for lambda range around lambda_init
        matrix_lambda_use_qk=True,  # Include QK multiplicative interaction in lambda
        use_output_gate=False,  # G1 output gating for attention sink elimination
        output_gate_type='per_token',  # 'per_token', 'per_head', or 'global'
        output_gate_init_bias=-2.0,  # Initial bias (negative = conservative, gates start more open)
        # σReparam for entropy collapse prevention (Apple/ICLR 2025)
        use_sigma_reparam=False,
        sigma_target=1.0,
        # Complexity bias for anti-curriculum attention (HARDY-MER, TIACBM 2024-2025)
        use_complexity_bias=False,  # Bias attention toward high-complexity patches
        complexity_method='variance',  # 'variance', 'gradient', 'entropy', 'combined'
        complexity_bias_scale=1.0,  # Scale for complexity bias strength
        # Head routing for complexity specialization (MoH 2024, attention orthogonality)
        use_head_routing=False,  # Enable learnable per-head complexity routing
    ):
        super().__init__()
        self.num_heads = num_heads
        # For differential attention, we split heads into pairs
        # So effective head_dim is halved compared to standard attention
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim ** -0.5
        self.use_liere = use_liere
        self.use_head_routing = use_head_routing
        self.spatial_dims = spatial_dims
        self.num_registers = num_registers
        self.layer_idx = layer_idx
        self.use_spatial_decay = use_spatial_decay
        self.spatial_decay_rate = spatial_decay_rate
        self.spatial_decay_radius = spatial_decay_radius
        self.use_per_head_decay = use_per_head_decay
        self.decay_type = decay_type
        self.distance_type = distance_type
        self.use_content_gate = use_content_gate
        self.use_matrix_lambda = use_matrix_lambda
        self.matrix_lambda_scale = matrix_lambda_scale
        self.matrix_lambda_use_qk = matrix_lambda_use_qk
        self.use_output_gate = use_output_gate
        self.use_sigma_reparam = use_sigma_reparam
        self.use_complexity_bias = use_complexity_bias
        self.complexity_method = complexity_method
        self.complexity_bias_scale = complexity_bias_scale

        # Q, K, V projections (same total dimension as standard attention)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # σReparam: Wrap QKV with spectral normalization (Apple/ICLR 2025)
        if use_sigma_reparam:
            self.qkv_reparam = SigmaReparam(self.qkv, target_sigma=sigma_target)

        # SDT-style learned spatial decay or content-aware gate
        if decay_type == 'learned' and use_spatial_decay:
            self.learned_decay = LearnedSpatialDecay(self.head_dim, num_heads, distance_type)
        elif use_content_gate and use_spatial_decay:
            self.content_gate = LearnedSpatialDecay(self.head_dim, num_heads, distance_type)

        # Lambda parameters for differential attention
        self.lambda_init = lambda_init_fn(layer_idx)

        if use_matrix_lambda:
            # M-DGSA: Content-dependent N×N lambda matrix
            self.matrix_lambda = MatrixGatedLambda(
                head_dim=self.head_dim,
                num_heads=num_heads,
                lambda_init=self.lambda_init,
                scale=matrix_lambda_scale,
                use_qk_interaction=matrix_lambda_use_qk
            )
        else:
            # Original scalar lambda parameters
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        # SubLayer normalization (RMSNorm as per the paper)
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        if self.use_liere:
            # LieRE operates on the halved head_dim for differential attention
            self.liere = LieRE(
                num_dim=2,
                dim=self.head_dim,
                jitter_std=liere_jitter_std,
                jitter_mode=liere_jitter_mode,
                pos_embed_shift=liere_pos_embed_shift,
                pos_embed_jitter=liere_pos_embed_jitter,
                pos_embed_rescale=liere_pos_embed_rescale
            )

        # G1 Output Gating for attention sink elimination (arXiv:2505.06708)
        # Note: For differential attention, output dim is 2*head_dim
        if use_output_gate:
            self.output_gate = OutputGate(
                head_dim=self.head_dim,
                num_heads=num_heads,
                gate_init_bias=output_gate_init_bias,
                gate_type=output_gate_type
            )

        # Learnable per-head complexity routing (MoH 2024)
        # Initialized to 0 (neutral); training optimizes:
        # +values -> focus on complex regions, -values -> focus on simple regions
        if use_head_routing:
            self.head_routing = nn.Parameter(torch.zeros(num_heads))
        else:
            self.register_buffer('head_routing', None)

    def forward(self, x, return_attention=False, return_aux_info=False):
        B, N, C = x.shape

        # Project to Q, K, V (use σReparam if enabled)
        if self.use_sigma_reparam:
            qkv = self.qkv_reparam(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, N, C]

        # Reshape for differential attention: split into 2*num_heads with head_dim each
        # q, k: [B, N, 2*num_heads, head_dim]
        # v: [B, N, num_heads, 2*head_dim]
        q = q.view(B, N, 2 * self.num_heads, self.head_dim)
        k = k.view(B, N, 2 * self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, 2 * self.head_dim)

        # Split Q and K into pairs for differential computation
        q = q.view(B, N, self.num_heads, 2, self.head_dim)
        k = k.view(B, N, self.num_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]  # [B, N, num_heads, head_dim]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        # Transpose for attention: [B, num_heads, N, head_dim]
        q1 = q1.permute(0, 2, 1, 3)
        q2 = q2.permute(0, 2, 1, 3)
        k1 = k1.permute(0, 2, 1, 3)
        k2 = k2.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, N, 2*head_dim]

        # Apply LieRE to both Q/K pairs
        if self.use_liere:
            if self.spatial_dims is not None:
                dimensions = self.spatial_dims
            else:
                num_patches = N - self.num_registers
                side = int(num_patches ** 0.5)
                dimensions = (side, side)

            if self.num_registers > 0:
                # Split register tokens (don't apply LieRE to them)
                q1_reg, q1_patch = q1[:, :, :self.num_registers], q1[:, :, self.num_registers:]
                q2_reg, q2_patch = q2[:, :, :self.num_registers], q2[:, :, self.num_registers:]
                k1_reg, k1_patch = k1[:, :, :self.num_registers], k1[:, :, self.num_registers:]
                k2_reg, k2_patch = k2[:, :, :self.num_registers], k2[:, :, self.num_registers:]

                # Apply LieRE to patch tokens
                q1_patch = self.liere.apply_rotations(q1_patch, dimensions)
                q2_patch = self.liere.apply_rotations(q2_patch, dimensions)
                k1_patch = self.liere.apply_rotations(k1_patch, dimensions)
                k2_patch = self.liere.apply_rotations(k2_patch, dimensions)

                # Recombine
                q1 = torch.cat([q1_reg, q1_patch], dim=2)
                q2 = torch.cat([q2_reg, q2_patch], dim=2)
                k1 = torch.cat([k1_reg, k1_patch], dim=2)
                k2 = torch.cat([k2_reg, k2_patch], dim=2)
            else:
                q1 = self.liere.apply_rotations(q1, dimensions)
                q2 = self.liere.apply_rotations(q2, dimensions)
                k1 = self.liere.apply_rotations(k1, dimensions)
                k2 = self.liere.apply_rotations(k2, dimensions)

        # Compute two attention matrices
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        # Apply spatial decay bias before softmax
        if self.use_spatial_decay:
            if self.spatial_dims is not None:
                dimensions = self.spatial_dims
            else:
                num_patches = N - self.num_registers
                side = int(num_patches ** 0.5)
                dimensions = (side, side)

            if self.decay_type == 'learned':
                # SDT-style: fully learned decay from content
                spatial_bias = self.learned_decay(q1, dimensions, self.num_registers)
            elif self.use_per_head_decay:
                # ALiBi-style per-head decay with exponential/linear
                spatial_bias = compute_per_head_spatial_decay_bias(
                    dimensions, x.device, attn1.dtype,
                    num_heads=self.num_heads,
                    base_decay_rate=self.spatial_decay_rate,
                    base_decay_radius=self.spatial_decay_radius,
                    num_registers=self.num_registers,
                    decay_type=self.decay_type,
                    distance_type=self.distance_type
                )
                # Apply content-aware gating if enabled
                if self.use_content_gate and hasattr(self, 'content_gate'):
                    content_gate = self.content_gate(q1, dimensions, self.num_registers)
                    spatial_bias = spatial_bias + content_gate
            else:
                # Uniform decay across all heads
                spatial_bias = compute_spatial_decay_bias(
                    dimensions, x.device, attn1.dtype,
                    decay_rate=self.spatial_decay_rate,
                    decay_radius=self.spatial_decay_radius,
                    num_registers=self.num_registers
                )
            attn1 = attn1 + spatial_bias
            attn2 = attn2 + spatial_bias

        # Apply complexity bias to focus on hard patches (anti-curriculum attention)
        patch_complexity = None
        if self.use_complexity_bias:
            patch_complexity = compute_patch_complexity(
                x, method=self.complexity_method,
                num_registers=self.num_registers, normalize=True
            )
            complexity_bias = compute_complexity_attention_bias(
                patch_complexity, scale=self.complexity_bias_scale, mode='additive'
            )
            attn1 = attn1 + complexity_bias
            attn2 = attn2 + complexity_bias

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)

        # Save attention weights for visualization before dropout
        attn1_weights = attn1
        attn2_weights = attn2

        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        # Compute attention outputs
        out1 = attn1 @ v  # [B, num_heads, N, 2*head_dim]
        out2 = attn2 @ v

        # Compute lambda and apply differential attention
        if self.use_matrix_lambda:
            # M-DGSA: Content-dependent N×N lambda matrix
            lambda_matrix = self.matrix_lambda(q1, k1)  # [B, num_heads, N, N]
            # Apply matrix lambda element-wise to second attention before value aggregation
            weighted_attn2 = lambda_matrix * attn2  # [B, num_heads, N, N]
            out2_weighted = weighted_attn2 @ v  # [B, num_heads, N, 2*head_dim]
            out = out1 - out2_weighted
            lambda_for_return = lambda_matrix
        else:
            # Original scalar lambda computation
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            out = out1 - lambda_full * out2  # [B, num_heads, N, 2*head_dim]
            lambda_for_return = lambda_full

        # Store head outputs before gating (for HSIC diversity loss)
        head_outputs = out  # [B, num_heads, N, 2*head_dim]

        # Apply G1 output gating if enabled (attention sink elimination)
        # This allows the model to output "nothing" for positions where attention is uninformative
        gate_values = None
        if self.use_output_gate:
            if return_aux_info:
                out, gate_values = self.output_gate(q1, out, num_registers=self.num_registers, return_gate=True)
            else:
                out = self.output_gate(q1, out, num_registers=self.num_registers)

        # Apply sublayer normalization and scaling
        out = out.permute(0, 2, 1, 3)  # [B, N, num_heads, 2*head_dim]
        out = self.subln(out)
        out = out * (1 - self.lambda_init)

        # Reshape and project
        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention or return_aux_info:
            result = {'attn1': attn1_weights, 'attn2': attn2_weights, 'lambda': lambda_for_return}
            if return_aux_info:
                result['head_outputs'] = head_outputs
                result['gate_values'] = gate_values
                result['lambda_matrix'] = lambda_for_return if self.use_matrix_lambda else None
                result['patch_complexity'] = patch_complexity
                result['head_routing'] = self.head_routing  # For head specialization loss
            return out, result
        return out


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Supports both standard and differential attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_diff_attn=False, layer_idx=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Choose attention type
        if use_diff_attn:
            self.attn = DifferentialAttention(
                hidden_size, num_heads=num_heads, qkv_bias=True,
                layer_idx=layer_idx, **block_kwargs
            )
        else:
            # Filter out differential attention-specific kwargs for standard Attention
            diff_attn_keys = {'use_spatial_decay', 'spatial_decay_rate', 'spatial_decay_radius',
                            'use_per_head_decay', 'decay_type', 'distance_type', 'use_content_gate',
                            'use_matrix_lambda', 'matrix_lambda_scale', 'matrix_lambda_use_qk'}
            attn_kwargs = {k: v for k, v in block_kwargs.items() if k not in diff_attn_keys}
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **attn_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, return_attention=False, return_aux_info=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        if return_attention or return_aux_info:
            attn_out, attn_info = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                return_attention=return_attention,
                return_aux_info=return_aux_info
            )
            x = x + gate_msa.unsqueeze(1) * attn_out
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x, attn_info
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class EqM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        uncond=True,
        ebm='none',
        use_liere=False,
        liere_jitter_std=0.0,
        liere_jitter_mode='gaussian',
        liere_pos_embed_shift=None,
        liere_pos_embed_jitter=None,
        liere_pos_embed_rescale=2.0,
        num_registers=0,  # Number of register tokens (attention sinks)
        use_diff_attn=False,  # Use differential attention (from Microsoft Diff-Transformer)
        use_spatial_decay=True,  # Enable spatial attention decay
        spatial_decay_rate=0.1,   # Base decay rate (higher = stronger locality)
        spatial_decay_radius=0.25,  # Base radius in normalized [-1,+1] space
        use_per_head_decay=True,  # ALiBi-style per-head decay (different heads = different locality)
        decay_type='exponential',  # 'exponential' (natural signal decay) or 'linear' (original)
        distance_type='l2',  # 'l2' (Euclidean) or 'l1' (Manhattan)
        use_content_gate=True,  # Content-aware gating (SDT-style)
        use_matrix_lambda=False,  # M-DGSA: use content-dependent N×N lambda matrix
        matrix_lambda_scale=1.0,  # Scale for lambda range around lambda_init
        matrix_lambda_use_qk=True,  # Include QK multiplicative interaction in lambda
        use_output_gate=False,  # G1 output gating for attention sink elimination (arXiv:2505.06708)
        output_gate_type='per_token',  # 'per_token', 'per_head', or 'global'
        output_gate_init_bias=-2.0,  # Initial bias (negative = conservative, gates start more open)
        # σReparam for entropy collapse prevention (Apple/ICLR 2025)
        use_sigma_reparam=False,
        sigma_target=1.0,
        # Complexity bias for anti-curriculum attention (HARDY-MER, TIACBM 2024-2025)
        use_complexity_bias=False,
        complexity_method='variance',
        complexity_bias_scale=1.0,
        # Head routing for complexity specialization (MoH 2024, attention orthogonality)
        use_head_routing=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_diff_attn = use_diff_attn

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Register tokens (attention sinks) - from "Vision Transformers Need Registers"
        self.num_registers = num_registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, hidden_size))

        # Compute spatial dimensions for LieRE (patch grid size)
        spatial_side = input_size // patch_size
        self.spatial_dims = (spatial_side, spatial_side)

        block_kwargs = dict(
            use_liere=use_liere,
            liere_jitter_std=liere_jitter_std,
            liere_jitter_mode=liere_jitter_mode,
            liere_pos_embed_shift=liere_pos_embed_shift,
            liere_pos_embed_jitter=liere_pos_embed_jitter,
            liere_pos_embed_rescale=liere_pos_embed_rescale,
            spatial_dims=self.spatial_dims,
            num_registers=num_registers,
            use_spatial_decay=use_spatial_decay,
            spatial_decay_rate=spatial_decay_rate,
            spatial_decay_radius=spatial_decay_radius,
            use_per_head_decay=use_per_head_decay,
            decay_type=decay_type,
            distance_type=distance_type,
            use_content_gate=use_content_gate,
            use_matrix_lambda=use_matrix_lambda,
            matrix_lambda_scale=matrix_lambda_scale,
            matrix_lambda_use_qk=matrix_lambda_use_qk,
            use_output_gate=use_output_gate,
            output_gate_type=output_gate_type,
            output_gate_init_bias=output_gate_init_bias,
            use_sigma_reparam=use_sigma_reparam,
            sigma_target=sigma_target,
            use_complexity_bias=use_complexity_bias,
            complexity_method=complexity_method,
            complexity_bias_scale=complexity_bias_scale,
            use_head_routing=use_head_routing,
        )

        # Create blocks with layer indices for differential attention lambda initialization
        self.blocks = nn.ModuleList([
            SiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                use_diff_attn=use_diff_attn, layer_idx=i, **block_kwargs
            ) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.uncond = uncond
        self.ebm = ebm

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize register tokens:
        if self.num_registers > 0:
            nn.init.normal_(self.register_tokens, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x0, t, y, return_act=False, return_registers=False, get_energy=False, train=False,
                return_attention=False, attention_layer_idx=-1, return_aux_info=False, return_embeddings=False):
        """
        Forward pass of EqM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        return_registers: if True, also return the register token outputs for SIGReg
        return_attention: if True, also return attention weights from specified layer
        attention_layer_idx: which layer to extract attention from (-1 = last layer)
        return_aux_info: if True, also return auxiliary info (head_outputs, gate_values, lambda_matrix) for aux losses
        return_embeddings: if True, also return patch embeddings before final layer (for LejEPA)
        """
        x0.requires_grad_(True)
        # if self.uncond: # removes noise/time conditioning by setting to 0
        #     t = torch.zeros_like(t)
        act = []
        attention_weights = None
        x = self.x_embedder(x0) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # Prepend register tokens (attention sinks)
        if self.num_registers > 0:
            reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)  # (N, num_reg, D)
            x = torch.cat([reg_tokens, x], dim=1)  # (N, num_reg + T, D)

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        # Determine which layer to extract attention from
        num_blocks = len(self.blocks)
        target_layer = attention_layer_idx if attention_layer_idx >= 0 else num_blocks + attention_layer_idx

        aux_info = None
        for i, block in enumerate(self.blocks):
            if (return_attention or return_aux_info) and i == target_layer:
                x, block_info = block(x, c, return_attention=return_attention, return_aux_info=return_aux_info)
                attention_weights = block_info  # For backward compatibility
                aux_info = block_info  # Full auxiliary info
            else:
                x = block(x, c)
            act.append(x)

        # Split registers from patches before final layer
        registers = None
        embeddings = None
        if self.num_registers > 0:
            registers = x[:, :self.num_registers]  # (N, num_reg, D)
            x = x[:, self.num_registers:]          # (N, T, D)

        # Store embeddings before final layer (for LejEPA)
        if return_embeddings:
            embeddings = x.clone()  # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)

        # explicit energy
        E=0
        if self.ebm == 'l2':
            E = -torch.sum(x**2, dim=(1,2,3))/2
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0]
        if self.ebm == 'dot':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0]
        if self.ebm == 'mean':
            E = torch.sum(x*x0, dim=(1,2,3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()],[x0],create_graph=train)[0]
        if get_energy:
            return x, -E
        # Build return tuple based on requested outputs
        # Order: (x, act, registers, embeddings, aux_info)
        result = [x]

        if return_act:
            result.append(act)

        if return_registers:
            result.append(registers)

        if return_embeddings:
            result.append(embeddings)

        if return_attention or return_aux_info:
            result.append(aux_info)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def forward_with_cfg(self, x, t, y, cfg_scale, return_act=False, return_registers=False, get_energy=False, train=False):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, return_act=return_act, return_registers=return_registers, get_energy=get_energy, train=train)

        registers = None
        if get_energy:
            x, E = model_out
            model_out = x
        if return_act:
            if return_registers:
                model_out, act, registers = model_out
            else:
                model_out, act = model_out
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            if return_registers:
                return torch.cat([eps, rest], dim=1), act, registers
            return torch.cat([eps, rest], dim=1), act
        elif return_registers:
            model_out, registers = model_out

        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
        if return_registers:
            return torch.cat([eps, rest], dim=1), registers
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   EqM Configs                                  #
#################################################################################

def EqM_XL_2(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def EqM_XL_4(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def EqM_XL_8(**kwargs):
    return EqM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def EqM_L_2(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def EqM_L_4(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def EqM_L_8(**kwargs):
    return EqM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def EqM_B_2(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def EqM_B_4(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def EqM_B_8(**kwargs):
    return EqM(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def EqM_S_2(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def EqM_S_4(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def EqM_S_8(**kwargs):
    return EqM(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


EqM_models = {
    'EqM-XL/2': EqM_XL_2,  'EqM-XL/4': EqM_XL_4,  'EqM-XL/8': EqM_XL_8,
    'EqM-L/2':  EqM_L_2,   'EqM-L/4':  EqM_L_4,   'EqM-L/8':  EqM_L_8,
    'EqM-B/2':  EqM_B_2,   'EqM-B/4':  EqM_B_4,   'EqM-B/8':  EqM_B_8,
    'EqM-S/2':  EqM_S_2,   'EqM-S/4':  EqM_S_4,   'EqM-S/8':  EqM_S_8,
}
