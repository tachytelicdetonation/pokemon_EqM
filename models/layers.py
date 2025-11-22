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
        use_rope=True,
        rope_base=10000,
        head_drop=0.0,
        window_size=None,
    ):
        """
        Differential (two-branch) attention with optional RoPE, QK normalization, 
        grouped/shared KV heads, head drop regularization, and safer lambda init.
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.kv_heads = kv_heads or num_heads  # multi-/grouped-query support
        assert dim % self.kv_heads == 0, 'dim should be divisible by kv_heads'
        assert self.num_heads % self.kv_heads == 0, 'num_heads must be divisible by kv_heads'
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, 'head_dim must be even for RoPE'
        self.rope_base = rope_base
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.head_drop = head_drop
        self.window_size = window_size

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

        self._rope_cache = {}
        self._mask_cache = {}

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _get_rope(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key in self._rope_cache:
            return self._rope_cache[key]

        head_dim = self.head_dim
        # assume square grid; fall back to 1D if not square
        side = int(seq_len ** 0.5)
        if side * side != seq_len:
            side = seq_len
            pos_h = torch.arange(seq_len, device=device, dtype=torch.float32)
            pos_w = torch.zeros_like(pos_h)
        else:
            h = torch.arange(side, device=device, dtype=torch.float32)
            w = torch.arange(side, device=device, dtype=torch.float32)
            grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')
            pos_h = grid_h.reshape(-1)
            pos_w = grid_w.reshape(-1)

        dim_half = head_dim // 2
        freq_seq = torch.arange(dim_half, device=device, dtype=torch.float32)
        inv_freq = torch.pow(torch.tensor(self.rope_base, device=device, dtype=torch.float32), -freq_seq / dim_half)

        theta_h = pos_h[:, None] * inv_freq[None, :]
        theta_w = pos_w[:, None] * inv_freq[None, :]
        theta = torch.cat([theta_h, theta_w], dim=1)  # [seq_len, head_dim]

        cos = torch.cos(theta)[None, None, :, :].to(dtype=dtype)  # [1,1,N,D]
        sin = torch.sin(theta)[None, None, :, :].to(dtype=dtype)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def _apply_rope(self, q):
        # q: (B, heads, N, head_dim)
        cos, sin = self._get_rope(q.size(2), q.device, q.dtype)
        return (q * cos) + (self._rotate_half(q) * sin)

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

        if self.use_rope:
            q1 = self._apply_rope(q1)
            q2 = self._apply_rope(q2)
            k1 = self._apply_rope(k1)
            k2 = self._apply_rope(k2)

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
