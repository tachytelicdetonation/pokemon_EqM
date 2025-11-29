"""
Unified LejEPA + EqM Architecture.

This module implements a novel unified architecture that combines:
- LejEPA's masked prediction and SIGReg regularization
- EqM's equilibrium matching for generative modeling

The key insight: both methods benefit from isotropic Gaussian embeddings.
- LejEPA uses SIGReg to enforce Gaussian embeddings
- EqM uses Gaussian noise as prior

By training them together, the representation space naturally aligns
with the generative prior, potentially improving both understanding
and generation.

Architecture:
    Input (noisy + masked) → Shared Encoder → Embeddings
                                                 ↓
                              ┌──────────────────┴──────────────────┐
                              ↓                                    ↓
                        JEPA Predictor                         EqM Head
                              ↓                                    ↓
                        Predict masked                      Predict velocity
                              ↓                                    ↓
                        JEPA Loss                              EqM Loss
                                        +
                                   SIGReg(embeddings)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from einops import rearrange


class PatchMasker(nn.Module):
    """
    Generates random masks for patches (LejEPA-style).

    Supports:
    - Random masking: randomly select patches to mask
    - Block masking: mask contiguous spatial regions
    """
    def __init__(
        self,
        mask_ratio: float = 0.6,
        mask_type: str = 'random',
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def forward(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks for context and target patches.

        Args:
            batch_size: Number of samples
            num_patches: Total number of patches
            device: Target device

        Returns:
            context_mask: [B, N] bool, True = visible (context)
            target_mask: [B, N] bool, True = masked (to predict)
        """
        num_mask = int(num_patches * self.mask_ratio)

        if self.mask_type == 'block':
            target_mask = self._block_mask(batch_size, num_patches, num_mask, device)
        else:
            target_mask = self._random_mask(batch_size, num_patches, num_mask, device)

        context_mask = ~target_mask
        return context_mask, target_mask

    def _random_mask(self, B, N, num_mask, device):
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def _block_mask(self, B, N, num_mask, device):
        # Assume square grid
        H = W = int(N ** 0.5)
        if H * W != N:
            return self._random_mask(B, N, num_mask, device)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        block_size = max(2, int(H * (num_mask / N) ** 0.5))

        for b in range(B):
            top = torch.randint(0, H - block_size + 1, (1,)).item()
            left = torch.randint(0, W - block_size + 1, (1,)).item()
            for h in range(top, min(top + block_size, H)):
                for w in range(left, min(left + block_size, W)):
                    mask[b, h * W + w] = True
        return mask


class JEPAPredictorHead(nn.Module):
    """
    Predictor network for JEPA-style masked prediction.

    Takes context embeddings and predicts target embeddings.
    Lightweight MLP following I-JEPA design.
    """
    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Predictor transformer (lightweight)
        self.predictor_embed = nn.Linear(embed_dim, predictor_dim)

        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=num_heads,
                dim_feedforward=predictor_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target embeddings from context.

        Args:
            context_embeddings: [B, N, D] embeddings (masked positions have context info)
            target_mask: [B, N] bool, True = positions to predict

        Returns:
            predictions: [B, N, D] predicted embeddings for ALL positions
                        (only target positions matter for loss)
        """
        B, N, D = context_embeddings.shape

        # Replace target positions with mask token
        x = context_embeddings.clone()
        mask_tokens = self.mask_token.expand(B, N, -1)
        x = torch.where(target_mask.unsqueeze(-1), mask_tokens, x)

        # Project to predictor dimension
        x = self.predictor_embed(x)

        # Transformer blocks
        for block in self.predictor_blocks:
            x = block(x)

        x = self.predictor_norm(x)
        predictions = self.predictor_proj(x)

        return predictions


class EqMHead(nn.Module):
    """
    EqM velocity prediction head.

    Takes embeddings and predicts velocity field for equilibrium matching.
    """
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        # AdaLN for time conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # Initialize
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity from embeddings.

        Args:
            x: [B, N, D] patch embeddings
            c: [B, D] conditioning (timestep + optional class)

        Returns:
            velocity: [B, C, H, W] predicted velocity field
        """
        # AdaLN modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Project to patches
        x = self.proj(x)  # [B, N, p*p*C]

        # Unpatchify
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        p = self.patch_size
        c = self.out_channels

        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)

        return x


class UnifiedLejEPAEqM(nn.Module):
    """
    Unified LejEPA + EqM Model.

    Combines masked prediction (LejEPA) with equilibrium matching (EqM)
    in a single architecture with shared encoder.

    Training:
        1. Input image is noised (EqM interpolation)
        2. Some patches are masked (LejEPA masking)
        3. Encoder produces embeddings from visible noisy patches
        4. JEPA predictor predicts masked patch embeddings
        5. EqM head predicts velocity field
        6. Loss = EqM_loss + λ₁·SIGReg + λ₂·JEPA_loss
    """
    def __init__(
        self,
        # Image parameters
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 4,  # VAE latent channels
        # Model parameters
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        # LejEPA parameters
        mask_ratio: float = 0.6,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        # EqM parameters
        out_channels: int = 4,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Learnable mask token (for context positions during JEPA)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Time embedding (for EqM)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Shared transformer encoder
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # JEPA predictor head
        self.jepa_predictor = JEPAPredictorHead(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=num_heads // 2,
        )

        # EqM velocity head
        out_ch = out_channels * 2 if learn_sigma else out_channels
        self.eqm_head = EqMHead(
            embed_dim=embed_dim,
            out_channels=out_ch,
            patch_size=patch_size,
        )

        # Masker
        self.masker = PatchMasker(mask_ratio=mask_ratio)

        # SIGReg will be applied externally

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)

    def encode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input patches to embeddings.

        Args:
            x: [B, C, H, W] input (possibly noisy)
            t: [B] timesteps
            target_mask: [B, N] bool, positions to mask (optional)

        Returns:
            embeddings: [B, N, D]
        """
        # Patch embed
        x = self.patch_embed(x)  # [B, D, H/p, W/p]
        x = rearrange(x, 'b d h w -> b (h w) d')  # [B, N, D]

        # Add position embedding
        x = x + self.pos_embed

        # Replace masked positions with mask token (if masking)
        if target_mask is not None:
            mask_tokens = self.mask_token.expand(x.shape[0], x.shape[1], -1)
            x = torch.where(target_mask.unsqueeze(-1), mask_tokens, x)

        # Add time embedding (broadcast to all patches)
        t_emb = self.timestep_embedding(t)  # [B, D]
        x = x + t_emb.unsqueeze(1)

        # Transformer encoder
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        enable_jepa: bool = True,
        target_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.

        Args:
            x: [B, C, H, W] noisy input (EqM interpolation)
            t: [B] timesteps
            enable_jepa: Whether to apply masking and JEPA prediction
            target_mask: Pre-computed mask (optional, will generate if None)
            return_embeddings: Whether to return embeddings for SIGReg

        Returns:
            Dict with:
                - 'velocity': [B, C, H, W] EqM velocity prediction
                - 'jepa_predictions': [B, N, D] predicted target embeddings (if enable_jepa)
                - 'embeddings': [B, N, D] encoder embeddings (if return_embeddings)
                - 'target_mask': [B, N] mask used (if enable_jepa)
        """
        B = x.shape[0]
        device = x.device
        outputs = {}

        # Generate mask if needed
        if enable_jepa and target_mask is None:
            _, target_mask = self.masker(B, self.num_patches, device)

        # Encode (with masking if enabled)
        embeddings = self.encode(x, t, target_mask if enable_jepa else None)

        if return_embeddings:
            outputs['embeddings'] = embeddings

        # JEPA prediction
        if enable_jepa and target_mask is not None:
            jepa_predictions = self.jepa_predictor(embeddings, target_mask)
            outputs['jepa_predictions'] = jepa_predictions
            outputs['target_mask'] = target_mask

        # EqM velocity prediction
        t_emb = self.timestep_embedding(t)
        velocity = self.eqm_head(embeddings, t_emb)
        outputs['velocity'] = velocity

        return outputs

    def get_target_embeddings(
        self,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get target embeddings from clean image (for JEPA loss).

        Uses t=1 (fully clean) and no masking.
        Can use EMA encoder in practice.
        """
        t = torch.ones(x_clean.shape[0], device=x_clean.device)
        with torch.no_grad():
            embeddings = self.encode(x_clean, t, target_mask=None)
        return embeddings


class UnifiedTrainingLoss(nn.Module):
    """
    Combined loss for unified LejEPA + EqM training.

    Total Loss = EqM_loss + λ_sigreg * SIGReg + λ_jepa * JEPA_loss
    """
    def __init__(
        self,
        lambda_sigreg: float = 0.05,
        lambda_jepa: float = 0.1,
        sigreg_num_slices: int = 1024,
    ):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.lambda_jepa = lambda_jepa

        # Import SIGReg
        from pokemon_eqm.losses.lejepa_loss import SlicingUnivariateTest, EppsPulley
        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(),
            num_slices=sigreg_num_slices,
        )

    def forward(
        self,
        # Model outputs
        velocity_pred: torch.Tensor,
        embeddings: torch.Tensor,
        jepa_predictions: Optional[torch.Tensor],
        target_mask: Optional[torch.Tensor],
        # Targets
        velocity_target: torch.Tensor,
        target_embeddings: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            velocity_pred: [B, C, H, W] predicted velocity
            embeddings: [B, N, D] encoder embeddings
            jepa_predictions: [B, N, D] JEPA predictions (optional)
            target_mask: [B, N] mask for JEPA (optional)
            velocity_target: [B, C, H, W] target velocity (x1 - x0)
            target_embeddings: [B, N, D] target for JEPA (from clean image)

        Returns:
            Dict with all loss components and 'total'
        """
        losses = {}

        # 1. EqM Loss (velocity prediction)
        eqm_loss = F.mse_loss(velocity_pred, velocity_target)
        losses['eqm'] = eqm_loss

        # 2. SIGReg Loss (Gaussian constraint on embeddings)
        sigreg_loss = self.sigreg(embeddings)
        losses['sigreg'] = sigreg_loss

        # 3. JEPA Loss (masked prediction)
        jepa_loss = torch.tensor(0.0, device=embeddings.device)
        if jepa_predictions is not None and target_embeddings is not None and target_mask is not None:
            # Only compute loss on masked positions
            pred_masked = jepa_predictions[target_mask]  # [num_masked, D]
            target_masked = target_embeddings[target_mask]  # [num_masked, D]

            # Normalize before computing loss (following I-JEPA)
            pred_norm = F.layer_norm(pred_masked, (pred_masked.shape[-1],))
            target_norm = F.layer_norm(target_masked, (target_masked.shape[-1],))

            jepa_loss = F.mse_loss(pred_norm, target_norm)
        losses['jepa'] = jepa_loss

        # Total
        total = eqm_loss + self.lambda_sigreg * sigreg_loss + self.lambda_jepa * jepa_loss
        losses['total'] = total

        return losses


def create_unified_model(
    image_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 4,
    model_size: str = 'base',
    **kwargs,
) -> UnifiedLejEPAEqM:
    """
    Factory function for unified model.

    Args:
        image_size: Input image size
        patch_size: Patch size
        in_channels: Input channels (4 for VAE latent)
        model_size: 'small', 'base', 'large', 'huge'
    """
    configs = {
        'small': dict(embed_dim=384, depth=12, num_heads=6),
        'base': dict(embed_dim=768, depth=12, num_heads=12),
        'large': dict(embed_dim=1024, depth=24, num_heads=16),
        'huge': dict(embed_dim=1280, depth=32, num_heads=16),
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)

    return UnifiedLejEPAEqM(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        **config,
    )
