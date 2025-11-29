"""
JEPA-EqM: A Unified Framework for Understanding and Generation.

Theoretical Foundation
======================

This module unifies JEPA (representation learning) and EqM (generation)
based on a key insight: both methods learn to recover information from
corrupted inputs.

The Mathematical Connection
---------------------------

JEPA corruption:   Masking (discrete information removal)
EqM corruption:    Noising (continuous information destruction)

Both learn: p(z_clean | z_corrupted)

The crucial insight from LejEPA:
- SIGReg forces embeddings to be z ~ N(0, I) (isotropic Gaussian)
- EqM's prior is also ε ~ N(0, I)
- THEY ARE THE SAME DISTRIBUTION!

Therefore:
- A point in "noise space" is also a valid "embedding"
- A point in "embedding space" is also valid "noise"
- Denoising, Predicting, and Generating become THE SAME operation

The Unified Corruption Model
----------------------------

Define unified corruption C(z, t, m):
    - z: clean embedding
    - t ∈ [0,1]: noise level (t=1 clean, t=0 pure noise)
    - m ∈ {0,1}^N: mask (1 = masked/hidden)

For each patch i:
    If m_i = 1 (masked):
        z_corrupted[i] = [MASK] token  (no information)
    Else (visible):
        z_corrupted[i] = t * z[i] + (1-t) * ε[i]  (noisy)

The model learns to predict z_clean from z_corrupted.

The Unified Objective
---------------------

L = L_prediction + λ₁·L_SIGReg + λ₂·L_velocity

Where:
- L_prediction: Predict all patch embeddings (JEPA-style)
  ||f_θ(z_corrupted, t, m) - z_clean||²

- L_SIGReg: Force embeddings to be Gaussian
  This is what UNIFIES the two methods!

- L_velocity: Predict transport direction (EqM-style)
  ||v_θ(z_corrupted, t) - (z_clean - ε)||²

Why This Works (Information-Theoretic View)
--------------------------------------------

1. Masking removes information discretely (some patches unknown)
2. Noising removes information continuously (all patches uncertain)
3. The model learns a SINGLE function: "recover information"
4. SIGReg ensures the "fully corrupted" state = Gaussian = valid sample
5. Generation = start from Gaussian, iteratively recover information

The Architecture Flow
---------------------

Input: Clean image x

Step 1 (JEPA-style): Patchify and Mask
    patches = Patchify(x)           # [B, N, patch_dim]
    z_clean = Embed(patches)        # [B, N, D]
    mask = RandomMask(N)            # [N] boolean

Step 2 (EqM-style): Add Noise
    ε ~ N(0, I)                     # [B, N, D]
    t ~ Uniform(0, 1)               # [B]
    z_noisy = t * z_clean + (1-t) * ε

Step 3 (Unified): Apply both corruptions
    z_input = z_noisy
    z_input[mask] = MASK_TOKEN      # Masked positions get [MASK]

Step 4: Transformer processes corrupted input
    z_encoded = Transformer(z_input, t, mask)

Step 5: Dual prediction heads
    z_predicted = PredictionHead(z_encoded)     # Predict all embeddings
    velocity = VelocityHead(z_encoded, t)       # Predict flow direction

Step 6: Losses
    L_pred = MSE(z_predicted, z_clean)          # Reconstruction
    L_vel = MSE(velocity, z_clean - ε)          # Velocity matching
    L_sig = SIGReg(z_encoded)                   # Gaussian constraint

References:
- LejEPA: arXiv:2511.08544
- EqM: arXiv:2510.02300
- I-JEPA: arXiv:2301.08243
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class JEPAEqMConfig:
    """Configuration for JEPA-EqM model."""
    # Image
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 4  # VAE latent channels

    # Architecture
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # JEPA
    mask_ratio: float = 0.6
    mask_type: str = 'random'  # 'random' or 'block'
    predictor_depth: int = 6

    # Loss weights
    lambda_sigreg: float = 0.05
    lambda_velocity: float = 1.0
    lambda_prediction: float = 0.1

    # SIGReg
    sigreg_num_slices: int = 1024


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN for time conditioning."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # AdaLN modulation (6 = 2 for norm1 + 2 for norm2 + 2 for mlp gate)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B, D] conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # MLP with AdaLN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class JEPAEqMEncoder(nn.Module):
    """
    Unified encoder that processes corrupted (masked + noised) input.

    The encoder sees:
    - Visible patches with noise added (EqM corruption)
    - Masked patches replaced with [MASK] token (JEPA corruption)
    - Timestep conditioning (how much noise)
    - Mask information (which patches are hidden)
    """
    def __init__(self, config: JEPAEqMConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

        # Mask embedding (learned embedding indicating mask status)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.mask_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        z_corrupted: torch.Tensor,  # [B, N, D] corrupted embeddings
        t: torch.Tensor,             # [B] timestep
        mask: torch.Tensor,          # [B, N] boolean mask
    ) -> torch.Tensor:
        """
        Encode corrupted input.

        Args:
            z_corrupted: Already corrupted embeddings (noised + masked)
            t: Timestep (noise level)
            mask: Boolean mask (True = masked/hidden)

        Returns:
            z_encoded: [B, N, D] encoded representations
        """
        B, N, D = z_corrupted.shape

        # Add position embedding
        x = z_corrupted + self.pos_embed

        # Add mask indicator embedding (helps model know which are masked)
        mask_indicator = mask.float().unsqueeze(-1) * self.mask_embed  # [B, N, D]
        x = x + mask_indicator

        # Time conditioning
        t_emb = self.time_embed(t)  # [B, D]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        x = self.norm(x)
        return x


class PredictionHead(nn.Module):
    """
    JEPA-style prediction head.

    Predicts embeddings for ALL positions (both masked and visible).
    This learns to "complete" the representation.
    """
    def __init__(self, config: JEPAEqMConfig):
        super().__init__()

        # Lightweight transformer predictor
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads // 2,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.predictor_depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Simple conditioning (no time, just predicts clean)
        self.cond = nn.Parameter(torch.zeros(1, config.embed_dim))

    def forward(self, z_encoded: torch.Tensor) -> torch.Tensor:
        """Predict clean embeddings from encoded corrupted input."""
        B = z_encoded.shape[0]
        c = self.cond.expand(B, -1)

        x = z_encoded
        for block in self.blocks:
            x = block(x, c)

        x = self.norm(x)
        x = self.proj(x)
        return x


class VelocityHead(nn.Module):
    """
    EqM-style velocity prediction head.

    Predicts the velocity field v = z_clean - z_noise.
    This is the "direction" to move from noise toward data.
    """
    def __init__(self, config: JEPAEqMConfig):
        super().__init__()

        self.norm = nn.LayerNorm(config.embed_dim, elementwise_affine=False)

        # AdaLN for time conditioning
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embed_dim, 2 * config.embed_dim),
        )

        # Output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        # Initialize to zero (start predicting nothing, learn the velocity)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z_encoded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity from encoded input."""
        t_emb = self.time_embed(t)  # [B, D]
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)

        x = self.norm(z_encoded) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.proj(x)

        return velocity


class JEPAEqM(nn.Module):
    """
    Unified JEPA-EqM Model.

    Combines:
    - JEPA's masked prediction (learning to "see"/understand)
    - EqM's velocity prediction (learning to generate)
    - SIGReg's Gaussian constraint (unifying the two)

    The key insight: Both methods learn p(z_clean | z_corrupted).
    SIGReg ensures embedding space = noise space, unifying them.
    """
    def __init__(self, config: JEPAEqMConfig):
        super().__init__()
        self.config = config

        # Core components
        self.patch_embed = PatchEmbedding(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Encoder
        self.encoder = JEPAEqMEncoder(config)

        # Prediction heads
        self.prediction_head = PredictionHead(config)  # JEPA
        self.velocity_head = VelocityHead(config)      # EqM

        # Decoder: embedding -> patches -> image
        self.decoder_proj = nn.Linear(
            config.embed_dim,
            config.patch_size * config.patch_size * config.in_channels
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> patches."""
        return self.patch_embed(x)

    def unpatchify(self, z: torch.Tensor) -> torch.Tensor:
        """Patches -> image."""
        # z: [B, N, D] -> [B, C, H, W]
        x = self.decoder_proj(z)  # [B, N, p*p*C]

        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        p = self.config.patch_size
        c = self.config.in_channels

        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)

        return x

    def random_mask(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Generate random mask."""
        num_mask = int(N * self.config.mask_ratio)
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def corrupt(
        self,
        z_clean: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply unified corruption: noise + masking.

        Args:
            z_clean: [B, N, D] clean embeddings
            t: [B] noise level (1 = clean, 0 = pure noise)
            mask: [B, N] boolean (True = masked)

        Returns:
            z_corrupted: [B, N, D] corrupted embeddings
            epsilon: [B, N, D] noise used (for velocity target)
        """
        B, N, D = z_clean.shape
        device = z_clean.device

        # Sample noise
        epsilon = torch.randn_like(z_clean)

        # EqM-style noise interpolation: z_t = t * z_clean + (1-t) * epsilon
        t_expanded = t.view(B, 1, 1)  # [B, 1, 1]
        z_noisy = t_expanded * z_clean + (1 - t_expanded) * epsilon

        # JEPA-style masking: replace masked positions with [MASK] token
        z_corrupted = z_noisy.clone()
        mask_tokens = self.mask_token.expand(B, N, -1)
        z_corrupted = torch.where(mask.unsqueeze(-1), mask_tokens, z_corrupted)

        return z_corrupted, epsilon

    def forward(
        self,
        x: torch.Tensor,                    # [B, C, H, W] clean image
        t: Optional[torch.Tensor] = None,   # [B] timestep (random if None)
        mask: Optional[torch.Tensor] = None, # [B, N] mask (random if None)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with unified corruption.

        Returns dict with all components needed for loss computation.
        """
        B = x.shape[0]
        device = x.device

        # 1. Get clean embeddings
        z_clean = self.patchify(x)  # [B, N, D]
        N = z_clean.shape[1]

        # 2. Sample timestep if not provided
        if t is None:
            t = torch.rand(B, device=device)

        # 3. Generate mask if not provided
        if mask is None:
            mask = self.random_mask(B, N, device)

        # 4. Apply unified corruption
        z_corrupted, epsilon = self.corrupt(z_clean, t, mask)

        # 5. Encode
        z_encoded = self.encoder(z_corrupted, t, mask)

        # 6. Predictions
        z_predicted = self.prediction_head(z_encoded)  # JEPA
        velocity = self.velocity_head(z_encoded, t)     # EqM

        # 7. Targets
        velocity_target = z_clean - epsilon  # v = z_clean - noise

        return {
            'z_clean': z_clean,
            'z_corrupted': z_corrupted,
            'z_encoded': z_encoded,
            'z_predicted': z_predicted,
            'velocity': velocity,
            'velocity_target': velocity_target,
            'epsilon': epsilon,
            't': t,
            'mask': mask,
        }

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_steps: int = 50,
        device: torch.device = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using the unified model.

        Starts from Gaussian noise (which IS the embedding space thanks to SIGReg)
        and iteratively denoises using the velocity prediction.
        """
        if device is None:
            device = next(self.parameters()).device

        N = self.num_patches
        D = self.config.embed_dim

        # Start from pure noise (t=0)
        z = torch.randn(batch_size, N, D, device=device)

        # Iteratively denoise
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step / num_steps  # t goes from 0 to 1
            t = torch.full((batch_size,), t_val, device=device)

            # No masking during generation
            mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)

            # Encode current state
            z_encoded = self.encoder(z, t, mask)

            # Predict velocity
            v = self.velocity_head(z_encoded, t)

            # Euler step: z_{t+dt} = z_t + dt * v
            z = z + dt * v * guidance_scale

        # Decode to image
        x = self.unpatchify(z)

        return x


class JEPAEqMLoss(nn.Module):
    """
    Unified loss for JEPA-EqM.

    L_total = L_velocity + λ_pred * L_prediction + λ_sig * L_SIGReg

    Where:
    - L_velocity: EqM loss - predict direction from noise to clean
    - L_prediction: JEPA loss - predict masked embeddings
    - L_SIGReg: Gaussian constraint - unifies the two methods
    """
    def __init__(self, config: JEPAEqMConfig):
        super().__init__()
        self.config = config

        # Import SIGReg
        from pokemon_eqm.losses.lejepa_loss import SlicingUnivariateTest, EppsPulley
        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(),
            num_slices=config.sigreg_num_slices,
        )

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs: Dict from model forward pass

        Returns:
            Dict with loss components and 'total'
        """
        z_clean = outputs['z_clean']
        z_encoded = outputs['z_encoded']
        z_predicted = outputs['z_predicted']
        velocity = outputs['velocity']
        velocity_target = outputs['velocity_target']
        mask = outputs['mask']

        losses = {}

        # 1. Velocity Loss (EqM) - on ALL positions
        # This is the main generative loss
        L_velocity = F.mse_loss(velocity, velocity_target)
        losses['velocity'] = L_velocity

        # 2. Prediction Loss (JEPA) - primarily on MASKED positions
        # But also lightly on visible positions for consistency

        # Loss on masked positions (main JEPA objective)
        if mask.any():
            pred_masked = z_predicted[mask]
            clean_masked = z_clean[mask]
            # Normalize before loss (I-JEPA style)
            pred_norm = F.layer_norm(pred_masked, (pred_masked.shape[-1],))
            clean_norm = F.layer_norm(clean_masked, (clean_masked.shape[-1],))
            L_pred_masked = F.mse_loss(pred_norm, clean_norm)
        else:
            L_pred_masked = torch.tensor(0.0, device=z_clean.device)

        # Light loss on visible positions (consistency)
        if (~mask).any():
            pred_visible = z_predicted[~mask]
            clean_visible = z_clean[~mask]
            L_pred_visible = F.mse_loss(pred_visible, clean_visible)
        else:
            L_pred_visible = torch.tensor(0.0, device=z_clean.device)

        L_prediction = L_pred_masked + 0.1 * L_pred_visible
        losses['prediction'] = L_prediction
        losses['prediction_masked'] = L_pred_masked
        losses['prediction_visible'] = L_pred_visible

        # 3. SIGReg Loss - THE UNIFYING CONSTRAINT
        # This forces embeddings to be Gaussian, which means:
        # - "noise" in EqM is a valid embedding
        # - "embedding" in JEPA is valid noise
        # - The two methods operate in the SAME space!
        L_sigreg = self.sigreg(z_encoded)
        losses['sigreg'] = L_sigreg

        # Total loss
        total = (
            self.config.lambda_velocity * L_velocity +
            self.config.lambda_prediction * L_prediction +
            self.config.lambda_sigreg * L_sigreg
        )
        losses['total'] = total

        return losses


def create_jepa_eqm(
    image_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 4,
    model_size: str = 'base',
    **kwargs,
) -> Tuple[JEPAEqM, JEPAEqMLoss]:
    """
    Create JEPA-EqM model and loss.

    Args:
        image_size: Input image size
        patch_size: Patch size
        in_channels: Input channels
        model_size: 'small', 'base', 'large'

    Returns:
        (model, loss_fn)
    """
    size_configs = {
        'small': dict(embed_dim=384, depth=12, num_heads=6, predictor_depth=4),
        'base': dict(embed_dim=768, depth=12, num_heads=12, predictor_depth=6),
        'large': dict(embed_dim=1024, depth=24, num_heads=16, predictor_depth=8),
    }

    config_dict = size_configs.get(model_size, size_configs['base'])
    config_dict.update(kwargs)
    config_dict['image_size'] = image_size
    config_dict['patch_size'] = patch_size
    config_dict['in_channels'] = in_channels

    config = JEPAEqMConfig(**config_dict)

    model = JEPAEqM(config)
    loss_fn = JEPAEqMLoss(config)

    return model, loss_fn
