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
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_liere = use_liere

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.use_liere:
            # Apply LieRE (learnable rotations)
            side = int(N ** 0.5)
            dimensions = (side, side)
            q = self.liere.apply_rotations(q, dimensions)
            k = self.liere.apply_rotations(k, dimensions)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
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
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

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

        block_kwargs = dict(
            use_liere=use_liere,
            liere_jitter_std=liere_jitter_std,
            liere_jitter_mode=liere_jitter_mode,
            liere_pos_embed_shift=liere_pos_embed_shift,
            liere_pos_embed_jitter=liere_pos_embed_jitter,
            liere_pos_embed_rescale=liere_pos_embed_rescale,
        )

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
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

    def forward(self, x0, t, y, return_act=False, return_registers=False, get_energy=False, train=False):
        """
        Forward pass of EqM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        return_registers: if True, also return the register token outputs for SIGReg
        """
        x0.requires_grad_(True)
        if self.uncond: # removes noise/time conditioning by setting to 0
            t = torch.zeros_like(t)
        act = []
        x = self.x_embedder(x0) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # Prepend register tokens (attention sinks)
        if self.num_registers > 0:
            reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)  # (N, num_reg, D)
            x = torch.cat([reg_tokens, x], dim=1)  # (N, num_reg + T, D)

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, num_reg + T, D)
            act.append(x)

        # Split registers from patches before final layer
        registers = None
        if self.num_registers > 0:
            registers = x[:, :self.num_registers]  # (N, num_reg, D)
            x = x[:, self.num_registers:]          # (N, T, D)

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
        if return_act:
            if return_registers:
                return x, act, registers
            return x, act
        if return_registers:
            return x, registers
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, return_act=False, get_energy=False, train=False):
        """
        Forward pass of EqM, but also batches the uncondional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, return_act=return_act, get_energy=get_energy, train=train)
        if get_energy:
            x, E = model_out
            model_out=x
        if return_act:
            act = model_out[1]
            model_out = model_out[0]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1), act
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        if get_energy:
            return torch.cat([eps, rest], dim=1), E
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
