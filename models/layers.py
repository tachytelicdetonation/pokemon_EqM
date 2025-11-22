import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
        """
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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_g = self.fc1_g(x)
        x_x = self.fc1_x(x)
        x = self.act(x_g) * x_x
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LieRE(nn.Module):
    """
    Lie Rotational Positional Encodings (LieRE) - learnable generalization of RoPE.
    Uses Lie group theory to create rotation matrices via matrix exponentials of
    skew-symmetric matrices (Lie algebra elements).

    Reference: https://arxiv.org/abs/2406.10322
    """
    def __init__(self, num_dim, dim):
        """
        Args:
            num_dim: Number of spatial dimensions (2 for images: H, W)
            dim: Head dimension
        """
        super().__init__()
        self.num_dim = num_dim
        self.dim = dim

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

    def _get_rotations(self, dimensions, device, dtype):
        """
        Generate rotation matrices for given spatial dimensions.

        Note: Caching is disabled for LieRE since the generator_params are learnable
        and change during training. Gradients must flow through this computation.

        Args:
            dimensions: Tuple of spatial dimensions (H, W) for 2D
            device: Target device
            dtype: Target dtype

        Returns:
            Rotation matrices of shape [num_positions, dim, dim]
        """
        # Create skew-symmetric matrices from generator parameters
        skew_matrices = self._make_skew_symmetric(self.generator_params)  # [num_dim, dim, dim]

        # Compute rotation matrices via matrix exponential (Lie group elements)
        # matrix_exp maps from Lie algebra (skew-symmetric) to Lie group (rotations)
        rotations = torch.matrix_exp(skew_matrices.float())  # [num_dim, dim, dim]
        rotations = rotations.to(dtype)

        # Generate position encodings for each dimension
        # For 2D: dimensions = (H, W)
        all_positions = []
        for dim_idx, dim_size in enumerate(dimensions):
            positions = torch.arange(dim_size, device=device, dtype=torch.float32)
            all_positions.append(positions)

        # Create meshgrid for 2D positions
        if self.num_dim == 2:
            pos_h, pos_w = torch.meshgrid(all_positions[0], all_positions[1], indexing='ij')
            pos_h = pos_h.reshape(-1)  # [H*W]
            pos_w = pos_w.reshape(-1)  # [H*W]
            positions_list = [pos_h, pos_w]
        else:
            # Fallback for other dimensions
            positions_list = all_positions

        # Compute cumulative rotations for each position
        num_positions = len(positions_list[0])

        # Build list of rotation matrices (avoid inplace operations for autograd)
        rotation_list = []
        for pos_idx in range(num_positions):
            # Start with identity
            R_cumulative = torch.eye(self.dim, device=device, dtype=dtype)

            # Apply rotation for each dimension
            for dim_idx in range(self.num_dim):
                p = positions_list[dim_idx][pos_idx].item()
                if p != 0:
                    # R^p = exp(p * log(R)) = exp(p * skew_matrix)
                    scaled_skew = skew_matrices[dim_idx] * p
                    R_pow = torch.matrix_exp(scaled_skew.float()).to(dtype)
                    R_cumulative = R_cumulative @ R_pow

            rotation_list.append(R_cumulative)

        # Stack into a single tensor
        rotation_matrices = torch.stack(rotation_list, dim=0)  # [num_positions, dim, dim]

        return rotation_matrices

    def apply_rotations(self, x, dimensions):
        """
        Apply LieRE to query or key tensor.

        Args:
            x: Input tensor of shape [B, num_heads, seq_len, head_dim]
            dimensions: Tuple of spatial dimensions (H, W) for 2D

        Returns:
            Tensor with LieRE applied, same shape as input
        """
        B, num_heads, seq_len, head_dim = x.shape

        # Get rotation matrices for these dimensions
        rotations = self._get_rotations(dimensions, x.device, x.dtype)  # [seq_len, head_dim, head_dim]

        # Apply rotation: x @ R^T for each position
        # x: [B, num_heads, seq_len, head_dim]
        # rotations: [seq_len, head_dim, head_dim]

        # Reshape for batched matrix multiplication
        x_flat = x.reshape(B * num_heads, seq_len, head_dim)  # [B*num_heads, seq_len, head_dim]

        # Apply rotations: [B*num_heads, seq_len, head_dim] @ [seq_len, head_dim, head_dim]
        # We need to apply different rotation to each position
        output = torch.zeros_like(x_flat)
        for i in range(seq_len):
            output[:, i, :] = x_flat[:, i, :] @ rotations[i].T

        # Reshape back
        output = output.reshape(B, num_heads, seq_len, head_dim)
        return output


class DifferentialAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        kv_heads=None,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        use_qk_norm=True,
        head_drop=0.0,
        window_size=None,
        use_rope=True,
        rope_base=10000,
        use_liere=False,
    ):
        """
        Differential (two-branch) attention with QK normalization,
        grouped/shared KV heads, head drop regularization, safer lambda init,
        and positional encodings (2D Axial RoPE or LieRE).
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.kv_heads = kv_heads or num_heads  # multi-/grouped-query support
        assert dim % self.kv_heads == 0, 'dim should be divisible by kv_heads'
        assert self.num_heads % self.kv_heads == 0, 'num_heads must be divisible by kv_heads'
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm
        self.head_drop = head_drop
        self.window_size = window_size
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.use_liere = use_liere

        # Can't use both RoPE and LieRE simultaneously
        if self.use_rope and self.use_liere:
            raise ValueError("Cannot use both RoPE and LieRE. Set use_liere=True and use_rope=False for LieRE.")

        # RoPE requires even head dimension (split in half for H and W)
        if self.use_rope:
            assert self.head_dim % 2 == 0, f'head_dim must be even for RoPE, got {self.head_dim}'

        # Initialize LieRE if requested
        if self.use_liere:
            self.liere = LieRE(num_dim=2, dim=self.head_dim)  # 2D for images (H, W)

        # Separate V projections to avoid capacity loss after subtraction.
        self.q_proj = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, 2 * dim * self.kv_heads // self.num_heads, bias=qkv_bias)
        self.v_proj1 = nn.Linear(dim, dim * self.kv_heads // self.num_heads, bias=qkv_bias)
        self.v_proj2 = nn.Linear(dim, dim * self.kv_heads // self.num_heads, bias=qkv_bias)

        # Start lambda small via softplus so out1 dominates early training.
        self.lambda_p = nn.Parameter(torch.full((num_heads, 1, 1), -2.0))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._mask_cache = {}
        self._rope_cache = {}  # Cache for RoPE embeddings

    def _get_window_mask(self, seq_len, device, dtype):
        if self.window_size is None:
            return None
        key = (seq_len, device, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]
        side = int(seq_len ** 0.5)
        if side * side != seq_len:
            # 1D fallback
            idx = torch.arange(seq_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
        else:
            h = torch.arange(side, device=device)
            w = torch.arange(side, device=device)
            gh, gw = torch.meshgrid(h, w, indexing='ij')
            coords = torch.stack([gh.reshape(-1), gw.reshape(-1)], dim=1)
            dist = (coords[:, None, :] - coords[None, :, :]).abs()
            dist = dist.max(dim=-1).values  # Chebyshev distance
        mask = dist > self.window_size
        # Convert to additive mask for SDPA (masked positions = -inf)
        mask = mask.to(dtype)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _rotate_half(x):
        """
        Rotates half the hidden dims of the input for RoPE application.
        Used to implement complex multiplication in the rotary embedding.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _get_rope(self, seq_len, device, dtype):
        """
        Generate 2D Axial RoPE (Rotary Position Embeddings) for a square grid.
        Splits head_dim in half: first half for height, second half for width.

        Args:
            seq_len: Number of patches (assumes square grid: seq_len = H * W)
            device: Target device for the embeddings
            dtype: Target dtype for the embeddings

        Returns:
            cos, sin: Cosine and sine embeddings of shape [1, 1, seq_len, head_dim]
        """
        # Check cache first
        key = (seq_len, device, dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]

        # Compute grid size (assume square)
        side = int(seq_len ** 0.5)
        if side * side != seq_len:
            # Fallback to 1D for non-square sequences
            dim_half = self.head_dim // 2
            inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, dim_half, 2, device=device, dtype=torch.float32) / dim_half))
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            theta = torch.outer(pos, inv_freq)
            theta = theta.repeat(1, 2)  # [seq_len, dim_half]
            # Pad to full head_dim
            theta = torch.cat([theta, theta], dim=1)  # [seq_len, head_dim]
        else:
            # 2D axial RoPE
            dim_half = self.head_dim // 2
            dim_quarter = dim_half // 2

            # Inverse frequencies for RoPE
            inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, dim_quarter, 1, device=device, dtype=torch.float32) / dim_quarter))

            # Create 2D position grid
            pos_h = torch.arange(side, device=device, dtype=torch.float32)
            pos_w = torch.arange(side, device=device, dtype=torch.float32)
            grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
            pos_h = grid_h.reshape(-1)  # [seq_len]
            pos_w = grid_w.reshape(-1)  # [seq_len]

            # Compute theta for height and width separately
            theta_h = torch.outer(pos_h, inv_freq)  # [seq_len, dim_quarter]
            theta_w = torch.outer(pos_w, inv_freq)  # [seq_len, dim_quarter]

            # Concatenate: first half for height, second half for width
            theta = torch.cat([theta_h, theta_w], dim=1)  # [seq_len, dim_half]
            # Duplicate to cover full head_dim (apply same rotation to both halves)
            theta = torch.cat([theta, theta], dim=1)  # [seq_len, head_dim]

        # Compute cos and sin
        cos = theta.cos().to(dtype)  # [seq_len, head_dim]
        sin = theta.sin().to(dtype)  # [seq_len, head_dim]

        # Reshape to [1, 1, seq_len, head_dim] for broadcasting
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Cache the result - clone to avoid reference issues with torch.compile
        self._rope_cache[key] = (cos.clone(), sin.clone())
        return cos, sin

    def _apply_rope(self, x):
        """
        Apply 2D Axial RoPE to query or key tensor.

        Args:
            x: Input tensor of shape [B, num_heads, seq_len, head_dim]

        Returns:
            Tensor with RoPE applied, same shape as input
        """
        _, _, seq_len, _ = x.shape
        cos, sin = self._get_rope(seq_len, x.device, x.dtype)

        # Apply rotary embedding: x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x):
        B, N, C = x.shape

        # Prefer flash/memory-efficient SDPA when available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

        q = self.q_proj(x)
        k = self.k_proj(x)
        v1 = self.v_proj1(x)
        v2 = self.v_proj2(x)

        q = q.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k = k.reshape(B, N, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        v1 = v1.reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v2 = v2.reshape(B, N, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q1, q2 = q[0], q[1]
        k1, k2 = k[0], k[1]

        # Expand grouped KV to per-head tensors if kv_heads < num_heads
        if self.kv_heads != self.num_heads:
            repeat = self.num_heads // self.kv_heads
            k1 = k1.repeat_interleave(repeat, dim=1)
            k2 = k2.repeat_interleave(repeat, dim=1)
            v1 = v1.repeat_interleave(repeat, dim=1)
            v2 = v2.repeat_interleave(repeat, dim=1)

        # Apply positional encoding to Q and K (before QK normalization)
        if self.use_rope:
            # Apply 2D Axial RoPE (fixed rotations)
            q1 = self._apply_rope(q1)
            q2 = self._apply_rope(q2)
            k1 = self._apply_rope(k1)
            k2 = self._apply_rope(k2)
        elif self.use_liere:
            # Apply LieRE (learnable rotations)
            # Compute spatial dimensions from sequence length
            side = int(N ** 0.5)
            dimensions = (side, side)  # (H, W)
            q1 = self.liere.apply_rotations(q1, dimensions)
            q2 = self.liere.apply_rotations(q2, dimensions)
            k1 = self.liere.apply_rotations(k1, dimensions)
            k2 = self.liere.apply_rotations(k2, dimensions)

        if self.use_qk_norm:
            # Cast to float32 for norm operations (FP8 not supported by linalg.vector_norm)
            # torch.compile will fuse these casts for performance
            q1 = q1 / (q1.float().norm(dim=-1, keepdim=True) + 1e-6)
            q2 = q2 / (q2.float().norm(dim=-1, keepdim=True) + 1e-6)
            k1 = k1 / (k1.float().norm(dim=-1, keepdim=True) + 1e-6)
            k2 = k2 / (k2.float().norm(dim=-1, keepdim=True) + 1e-6)

        attn_mask = self._get_window_mask(N, x.device, x.dtype)

        out1 = F.scaled_dot_product_attention(
            q1, k1, v1,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out2 = F.scaled_dot_product_attention(
            q2, k2, v2,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        lam = F.softplus(self.lambda_p)
        out = out1 - lam * out2

        if self.head_drop > 0 and self.training:
            keep = 1.0 - self.head_drop
            mask = torch.empty((1, self.num_heads, 1, 1), device=x.device, dtype=out.dtype).bernoulli_(keep) / keep
            out = out * mask

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = DifferentialAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
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

# Positional Embedding Functions
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
