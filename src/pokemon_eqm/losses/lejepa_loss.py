"""
LejEPA Loss Module: Combining JEPA with Equilibrium Matching.

This module implements the LejEPA (Lean Joint-Embedding Predictive Architecture)
losses for integration with EqM generative modeling. The key innovation from LejEPA
is heuristics-free self-supervised learning using SIGReg (Sketched Isotropic
Gaussian Regularization).

Key components:
1. SIGReg Loss - Forces embeddings to follow isotropic Gaussian distribution
2. Multi-View Invariance Loss - Aligns representations across augmented views
3. JEPA Prediction Loss - Predicts masked patches from context

Reference: arXiv:2511.08544 - "LeJEPA: Provable and Scalable Self-Supervised Learning"
GitHub: https://github.com/rbalestr-lab/lejepa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List

# Try to import the lejepa package for proper SIGReg implementation
try:
    import lejepa
    LEJEPA_AVAILABLE = True
except ImportError:
    LEJEPA_AVAILABLE = False


class EppsPulley(nn.Module):
    """
    Epps-Pulley Test for Univariate Normality.

    PyTorch implementation matching the reference lejepa repo:
    https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/univariate/epps_pulley.py

    Compares empirical characteristic function to theoretical normal CF
    using numerical integration with trapezoidal rule.
    """
    def __init__(
        self,
        t_range: Tuple[float, float] = (-3, 3),
        n_points: int = 10,
        weight_type: str = 'gaussian',
    ):
        """
        Args:
            t_range: Range for characteristic function evaluation
            n_points: Number of integration points
            weight_type: 'gaussian' or 'uniform' weighting
        """
        super().__init__()
        self.t_range = t_range
        self.n_points = n_points
        self.weight_type = weight_type

    def empirical_cf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical characteristic function.

        φ̂(t) = (1/n) Σ_j exp(i·t·X_j)
        """
        x_expanded = x.unsqueeze(1)  # [N, 1]
        t_expanded = t.unsqueeze(0)  # [1, K]

        real_part = torch.cos(t_expanded * x_expanded)
        imag_part = torch.sin(t_expanded * x_expanded)

        empirical_real = torch.mean(real_part, dim=0)
        empirical_imag = torch.mean(imag_part, dim=0)

        return torch.complex(empirical_real.float(), empirical_imag.float())

    def normal_cf(self, t: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
        """
        Theoretical CF for normal distribution: exp(iμt - σ²t²/2)
        """
        magnitude = torch.exp(-0.5 * sigma**2 * t**2)
        phase = mu * t

        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        return torch.complex(real_part.float(), imag_part.float())

    def weight_function(self, t: torch.Tensor) -> torch.Tensor:
        """Weight function for integration."""
        if self.weight_type == "gaussian":
            return torch.exp(-(t**2) / 2)
        elif self.weight_type == "uniform":
            return torch.ones_like(t)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Epps-Pulley test statistic.

        Args:
            x: Standardized samples [N] (should be zero-mean, unit-variance)

        Returns:
            Scalar test statistic (higher = more deviation from normality)
        """
        device = x.device

        with torch.no_grad():
            t = torch.linspace(*self.t_range, self.n_points, device=device)
            phi_normal = self.normal_cf(t, mu=0.0, sigma=1.0)
            weights = self.weight_function(t)

        phi_emp = self.empirical_cf(x, t)
        diff = phi_emp - phi_normal
        squared_diff = torch.real(diff * torch.conj(diff))

        integrand = squared_diff * weights
        integral = torch.trapezoid(integrand, t)

        return integral


# Alias for backward compatibility
EppsPulleyTest = EppsPulley


class SlicingUnivariateTest(nn.Module):
    """
    Slicing-based Multivariate Test (SIGReg).

    PyTorch implementation matching the reference lejepa repo:
    https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/multivariate/slicing.py

    Projects high-dimensional embeddings onto random 1D directions and applies
    univariate normality tests to each slice. This is the core of LejEPA's
    heuristics-free approach.

    The loss encourages embeddings to follow an isotropic (spherical) Gaussian
    distribution, which:
    1. Prevents representation collapse
    2. Enables optimal information geometry
    3. Works without momentum encoders or stop-gradients
    """
    def __init__(
        self,
        univariate_test: Optional[nn.Module] = None,
        num_slices: int = 1024,
        sampler: str = 'gaussian',
        clip_value: Optional[float] = None,
        reduction: str = 'mean',
    ):
        """
        Args:
            univariate_test: Univariate test module (default: EppsPulley)
            num_slices: Number of random projections
            sampler: 'gaussian' for standard normal projections
            clip_value: Optional clipping for noise filtering
            reduction: 'mean', 'sum', or None
        """
        super().__init__()
        self.univariate_test = univariate_test or EppsPulley()
        self.num_slices = num_slices
        self.sampler = sampler
        self.clip_value = clip_value
        self.reduction = reduction

        # Global step counter for varied projections
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def _sample_projections(self, D: int, device: torch.device) -> torch.Tensor:
        """Sample random projection directions."""
        # Use step as part of seed for reproducibility across distributed training
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self._step.item()) + 42)

        if self.sampler == 'gaussian':
            projections = torch.randn(D, self.num_slices, generator=generator, device=device)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

        # L2 normalize to unit vectors
        projections = F.normalize(projections, p=2, dim=0)

        return projections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for embeddings.

        Args:
            x: Embeddings [N, D] where N=samples, D=dimension

        Returns:
            Scalar loss (or [num_slices] if reduction=None)
        """
        # Handle batched input: flatten to [N, D]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        N, D = x.shape
        device = x.device

        # Increment step counter
        self._step += 1

        # Sample random projections
        projections = self._sample_projections(D, device)

        # Project: [N, D] @ [D, K] -> [N, K]
        projected = x @ projections

        # Optional clipping for noise filtering
        if self.clip_value is not None:
            projected = torch.clamp(projected, -self.clip_value, self.clip_value)

        # Standardize each slice (zero mean, unit variance)
        projected = (projected - projected.mean(dim=0)) / (projected.std(dim=0) + 1e-7)

        # Apply univariate test to each slice
        losses = []
        for k in range(self.num_slices):
            loss_k = self.univariate_test(projected[:, k])
            losses.append(loss_k)

        losses = torch.stack(losses)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


# Alias for backward compatibility
SlicedSIGReg = SlicingUnivariateTest


class CFSIGReg(nn.Module):
    """
    Characteristic Function SIGReg (CF-SIGReg).

    An efficient alternative to sliced SIGReg that directly computes the
    characteristic function matching loss without explicit slicing.

    This is the implementation used in your existing sigreg.py, enhanced
    with better numerical stability.
    """
    def __init__(
        self,
        num_projections: int = 64,
        num_freqs: int = 17,
        freq_max: float = 5.0,
        grad_clip: float = 5.0,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.num_freqs = num_freqs
        self.freq_max = freq_max
        self.grad_clip = grad_clip

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CF-SIGReg loss.

        Args:
            z: Embeddings [B, *] (will be flattened to [B, D])

        Returns:
            (loss, variance_per_projection) tuple
        """
        eps = 1e-6

        # Flatten
        if z.dim() > 2:
            z_flat = z.reshape(z.shape[0], -1)
        else:
            z_flat = z

        device = z.device

        # Random projections
        proj = torch.randn(self.num_projections, z_flat.shape[1], device=device)
        proj = proj / (proj.norm(dim=1, keepdim=True).clamp_min(eps))
        projected = z_flat @ proj.t()  # [B, P]

        # Gradient clipping for stability
        if self.grad_clip is not None:
            projected = projected.clamp(-self.grad_clip, self.grad_clip)

        # Frequency grid
        freqs = torch.linspace(-self.freq_max, self.freq_max, self.num_freqs, device=device)
        phase = projected.unsqueeze(-1) * freqs  # [B, P, F]
        window = torch.exp(-0.5 * (freqs / self.freq_max) ** 2)

        # Empirical CF
        empirical_cf = torch.exp(1j * phase).mean(dim=0)  # [P, F]

        # Target: standard normal CF
        gaussian_cf = torch.exp(-0.5 * freqs**2)

        # MSE with windowing
        diff_real = empirical_cf.real - gaussian_cf
        diff_imag = empirical_cf.imag
        cf_mse = (diff_real.pow(2) + diff_imag.pow(2)) * window

        loss = cf_mse.mean()
        variance = projected.var(dim=0).detach()

        return loss, variance


class MultiViewInvarianceLoss(nn.Module):
    """
    Multi-View Invariance Loss.

    Encourages similar representations for different augmented views of the
    same image, which is key to JEPA-style self-supervised learning.

    Loss = -cos_sim(z_v1, z_v2) or ||z_v1 - z_v2||²
    """
    def __init__(
        self,
        loss_type: str = 'cosine',
        temperature: float = 0.1,
    ):
        """
        Args:
            loss_type: 'cosine' for cosine similarity, 'mse' for L2
            temperature: Temperature for cosine similarity loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariance loss between two views.

        Args:
            z1: Embeddings from view 1 [B, D]
            z2: Embeddings from view 2 [B, D]

        Returns:
            Scalar loss
        """
        if self.loss_type == 'cosine':
            # Normalize
            z1_norm = F.normalize(z1, p=2, dim=-1)
            z2_norm = F.normalize(z2, p=2, dim=-1)

            # Negative cosine similarity
            cos_sim = (z1_norm * z2_norm).sum(dim=-1)
            loss = -cos_sim.mean()

        elif self.loss_type == 'mse':
            loss = F.mse_loss(z1, z2)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class JEPAPredictor(nn.Module):
    """
    JEPA Predictor Module.

    A lightweight predictor network that predicts target patch representations
    from context patch representations. This enables self-supervised learning
    by predicting masked regions from visible context.

    Architecture: MLP with hidden layer
    """
    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Input/output embedding dimension
            predictor_dim: Hidden dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        in_dim = embed_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, predictor_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = predictor_dim

        layers.append(nn.Linear(in_dim, embed_dim))

        self.predictor = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict target representations from context.

        Args:
            x: Context embeddings [B, N, D]

        Returns:
            Predicted target embeddings [B, N, D]
        """
        return self.predictor(x)


class JEPAPredictionLoss(nn.Module):
    """
    JEPA Prediction Loss.

    Computes the loss for predicting masked target patches from context patches.
    Can use either MSE or smooth L1 loss.
    """
    def __init__(
        self,
        loss_type: str = 'mse',
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize_targets = normalize_targets

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction loss.

        Args:
            predictions: Predicted representations [B, N, D]
            targets: Target representations [B, N, D]
            mask: Binary mask [B, N] indicating which positions to compute loss

        Returns:
            Scalar loss
        """
        if self.normalize_targets:
            targets = F.layer_norm(targets, targets.shape[-1:])
            predictions = F.layer_norm(predictions, predictions.shape[-1:])

        if self.loss_type == 'mse':
            diff = (predictions - targets).pow(2).mean(dim=-1)  # [B, N]
        elif self.loss_type == 'smooth_l1':
            diff = F.smooth_l1_loss(predictions, targets, reduction='none').mean(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if mask is not None:
            # Only compute loss on masked positions
            diff = diff * mask.float()
            loss = diff.sum() / (mask.sum() + 1e-7)
        else:
            loss = diff.mean()

        return loss


class MaskGenerator(nn.Module):
    """
    Mask Generator for JEPA-style training.

    Generates random masks for patch tokens to define context (visible)
    and target (predicted) regions.
    """
    def __init__(
        self,
        mask_ratio: float = 0.6,
        mask_type: str = 'random',
        min_mask_ratio: float = 0.4,
        max_mask_ratio: float = 0.8,
    ):
        """
        Args:
            mask_ratio: Default fraction of patches to mask
            mask_type: 'random', 'block', or 'scheduled' (varies during training)
            min_mask_ratio: Minimum mask ratio (for scheduled)
            max_mask_ratio: Maximum mask ratio (for scheduled)
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def forward(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        training_progress: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context and target masks.

        Args:
            batch_size: Batch size
            num_patches: Number of patch tokens
            device: Target device
            training_progress: Progress through training [0, 1] for scheduled masking

        Returns:
            (context_mask, target_mask) boolean tensors [B, N]
            context_mask: True for visible patches
            target_mask: True for patches to predict
        """
        if self.mask_type == 'scheduled':
            # Curriculum: start with more masking, reduce over time
            mask_ratio = self.max_mask_ratio - training_progress * (
                self.max_mask_ratio - self.min_mask_ratio
            )
        else:
            mask_ratio = self.mask_ratio

        num_mask = int(num_patches * mask_ratio)

        if self.mask_type == 'block':
            # Block masking - mask contiguous regions
            target_mask = self._generate_block_mask(
                batch_size, num_patches, num_mask, device
            )
        else:
            # Random masking
            target_mask = self._generate_random_mask(
                batch_size, num_patches, num_mask, device
            )

        context_mask = ~target_mask

        return context_mask, target_mask

    def _generate_random_mask(
        self,
        batch_size: int,
        num_patches: int,
        num_mask: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate random masks."""
        # Random permutation per sample
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        # Create mask: 1 for masked, 0 for visible
        mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)

        return mask

    def _generate_block_mask(
        self,
        batch_size: int,
        num_patches: int,
        num_mask: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate block masks (for 2D patch grids)."""
        # Assume square grid
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            # Fall back to random if not square
            return self._generate_random_mask(batch_size, num_patches, num_mask, device)

        mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)

        # Block size (aim for ~mask_ratio coverage)
        block_size = max(2, int(grid_size * (num_mask / num_patches) ** 0.5))

        for b in range(batch_size):
            # Random top-left corner for each sample
            max_start = grid_size - block_size
            start_h = torch.randint(0, max_start + 1, (1,), device=device).item()
            start_w = torch.randint(0, max_start + 1, (1,), device=device).item()

            # Create block
            for h in range(start_h, min(start_h + block_size, grid_size)):
                for w in range(start_w, min(start_w + block_size, grid_size)):
                    idx = h * grid_size + w
                    mask[b, idx] = True

        return mask


class LejEPALoss(nn.Module):
    """
    Combined LejEPA Loss for EqM Integration.

    This module combines all LejEPA components:
    1. SIGReg loss on embeddings (isotropic Gaussian constraint)
    2. Multi-view invariance loss (representation alignment)
    3. JEPA prediction loss (masked patch prediction)

    The losses are weighted and combined with the main EqM loss.
    """
    def __init__(
        self,
        # SIGReg parameters
        use_sigreg: bool = True,
        sigreg_weight: float = 0.05,
        sigreg_num_slices: int = 1024,
        sigreg_type: str = 'cf',  # 'cf' (characteristic function) or 'sliced'

        # Invariance parameters
        use_invariance: bool = True,
        invariance_weight: float = 0.02,
        invariance_type: str = 'cosine',

        # JEPA prediction parameters
        use_prediction: bool = True,
        prediction_weight: float = 0.1,
        embed_dim: int = 768,
        predictor_dim: int = 384,
        mask_ratio: float = 0.6,

        # Warmup
        warmup_steps: int = 1000,
    ):
        super().__init__()

        self.use_sigreg = use_sigreg
        self.sigreg_weight = sigreg_weight
        self.use_invariance = use_invariance
        self.invariance_weight = invariance_weight
        self.use_prediction = use_prediction
        self.prediction_weight = prediction_weight
        self.warmup_steps = warmup_steps

        # Initialize components
        if use_sigreg:
            if LEJEPA_AVAILABLE and sigreg_type == 'sliced':
                # Use official lejepa implementation
                univariate_test = lejepa.univariate.EppsPulley()
                self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
                    univariate_test=univariate_test,
                    num_slices=sigreg_num_slices,
                )
            elif sigreg_type == 'sliced':
                self.sigreg_loss = SlicedSIGReg(num_slices=sigreg_num_slices)
            else:
                self.sigreg_loss = CFSIGReg(num_projections=sigreg_num_slices // 16)

        if use_invariance:
            self.invariance_loss = MultiViewInvarianceLoss(loss_type=invariance_type)

        if use_prediction:
            self.predictor = JEPAPredictor(
                embed_dim=embed_dim,
                predictor_dim=predictor_dim,
            )
            self.prediction_loss = JEPAPredictionLoss()
            self.mask_generator = MaskGenerator(mask_ratio=mask_ratio)

    def compute_sigreg(
        self,
        embeddings: torch.Tensor,
        registers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SIGReg loss on embeddings.

        Args:
            embeddings: Patch embeddings [B, N, D]
            registers: Register token embeddings [B, R, D] (optional)

        Returns:
            Dict with 'sigreg_patches' and optionally 'sigreg_registers'
        """
        losses = {}

        # Flatten batch and sequence for SIGReg
        B, N, D = embeddings.shape
        emb_flat = embeddings.reshape(B * N, D)

        if isinstance(self.sigreg_loss, CFSIGReg):
            loss, _ = self.sigreg_loss(emb_flat)
        else:
            loss = self.sigreg_loss(emb_flat)

        losses['sigreg_patches'] = loss

        if registers is not None:
            B, R, D = registers.shape
            reg_flat = registers.reshape(B * R, D)

            if isinstance(self.sigreg_loss, CFSIGReg):
                reg_loss, _ = self.sigreg_loss(reg_flat)
            else:
                reg_loss = self.sigreg_loss(reg_flat)

            losses['sigreg_registers'] = reg_loss

        return losses

    def compute_invariance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariance loss between two views.

        Args:
            z1: Embeddings from view 1 [B, D] or [B, N, D]
            z2: Embeddings from view 2 [B, D] or [B, N, D]

        Returns:
            Scalar loss
        """
        # If sequence dim exists, pool
        if z1.dim() == 3:
            z1 = z1.mean(dim=1)
        if z2.dim() == 3:
            z2 = z2.mean(dim=1)

        return self.invariance_loss(z1, z2)

    def compute_prediction(
        self,
        embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        training_progress: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute JEPA prediction loss.

        Args:
            embeddings: Context embeddings [B, N, D]
            target_embeddings: Target embeddings [B, N, D]
            training_progress: Training progress [0, 1]

        Returns:
            (loss, context_mask, target_mask)
        """
        B, N, D = embeddings.shape

        # Ensure predictor is on the same device as input
        device = embeddings.device
        if next(self.predictor.parameters()).device != device:
            self.predictor = self.predictor.to(device)

        # Generate masks
        context_mask, target_mask = self.mask_generator(
            B, N, embeddings.device, training_progress
        )

        # Get context tokens (expand mask for gathering)
        context_indices = context_mask.nonzero(as_tuple=True)

        # Simple approach: predict all target positions from averaged context
        context_pooled = (embeddings * context_mask.unsqueeze(-1).float()).sum(dim=1)
        context_pooled = context_pooled / (context_mask.sum(dim=1, keepdim=True).float() + 1e-7)

        # Expand to predict all positions
        context_expanded = context_pooled.unsqueeze(1).expand(-1, N, -1)

        # Predict
        predictions = self.predictor(context_expanded)

        # Compute loss only on target positions
        loss = self.prediction_loss(predictions, target_embeddings, target_mask)

        return loss, context_mask, target_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        registers: Optional[torch.Tensor] = None,
        embeddings_view2: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        train_step: int = 0,
        total_steps: int = 100000,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all LejEPA losses.

        Args:
            embeddings: Main embeddings [B, N, D]
            registers: Register embeddings [B, R, D] (optional)
            embeddings_view2: Second view embeddings [B, N, D] (for invariance)
            target_embeddings: Target for prediction [B, N, D] (defaults to embeddings)
            train_step: Current training step
            total_steps: Total training steps

        Returns:
            Dict with all loss components and 'total'
        """
        losses = {}
        total = torch.tensor(0.0, device=embeddings.device)

        # Warmup scaling
        if self.warmup_steps > 0 and train_step < self.warmup_steps:
            warmup_scale = train_step / self.warmup_steps
        else:
            warmup_scale = 1.0

        # 1. SIGReg loss
        if self.use_sigreg:
            sigreg_losses = self.compute_sigreg(embeddings, registers)
            losses.update(sigreg_losses)

            sigreg_total = sum(sigreg_losses.values())
            total = total + warmup_scale * self.sigreg_weight * sigreg_total
            losses['sigreg_total'] = sigreg_total

        # 2. Invariance loss
        if self.use_invariance and embeddings_view2 is not None:
            inv_loss = self.compute_invariance(embeddings, embeddings_view2)
            losses['invariance'] = inv_loss
            total = total + warmup_scale * self.invariance_weight * inv_loss

        # 3. Prediction loss
        if self.use_prediction:
            if target_embeddings is None:
                target_embeddings = embeddings.detach()  # Self-prediction

            training_progress = train_step / total_steps
            pred_loss, ctx_mask, tgt_mask = self.compute_prediction(
                embeddings, target_embeddings, training_progress
            )
            losses['prediction'] = pred_loss
            total = total + warmup_scale * self.prediction_weight * pred_loss

        losses['total'] = total
        return losses
