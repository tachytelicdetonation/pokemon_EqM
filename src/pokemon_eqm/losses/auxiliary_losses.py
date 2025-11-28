"""
Auxiliary losses for attention mechanism improvement.
Based on 2025 research: σReparam, GateRA, HSIC, D-Gating.

References:
- σReparam: Apple/ICLR 2025 - Stabilizing Transformer Training
- GateRA: arXiv:2511.17582 (Nov 2025) - Entropy regularization for gating
- HSIC: Kornblith et al. - Head decorrelation via independence criterion
- D-Gating: arXiv:2509.23898 - Structured sparsity-inducing penalty
- Disagreement: arXiv:1810.10183 - Multi-head attention disagreement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AuxiliaryLossComputer(nn.Module):
    """
    Computes all auxiliary losses for attention regularization.

    Loss Categories:
    1. Entropy Regularization: Floor/ceiling to prevent collapse/uniformity
    2. Gate Regularization: Encourage binary gating decisions (GateRA)
    3. Head Diversity: HSIC-based decorrelation between heads
    4. Position Disagreement: Minimize attention pattern overlap
    5. Lambda Regularization: Smoothness and entropy for M-DGSA
    """

    def __init__(
        self,
        # Entropy floor/ceiling
        entropy_floor_threshold: float = 0.3,
        entropy_floor_weight: float = 0.02,
        entropy_ceiling_threshold: float = 0.85,
        entropy_ceiling_weight: float = 0.02,
        # Gate regularization (GateRA + D-Gating)
        gate_entropy_weight: float = 0.01,
        gate_sparsity_weight: float = 0.005,
        # Head diversity (HSIC)
        hsic_weight: float = 0.01,
        hsic_sigma: float = 1.0,
        hsic_sample_fraction: float = 0.25,
        # Position disagreement
        position_disagreement_weight: float = 0.005,
        # Lambda regularization (for M-DGSA)
        lambda_smoothness_weight: float = 0.001,
        lambda_entropy_weight: float = 0.01,
        # Hard focus (anti-curriculum attention)
        hard_focus_weight: float = 0.02,
        complexity_diversity_weight: float = 0.01,
        # Head specialization (2024-2025 research: MoH, attention orthogonality)
        complexity_ortho_weight: float = 0.01,
        load_balance_weight: float = 0.005,
        # Warmup
        warmup_steps: int = 1000,
    ):
        super().__init__()
        # Entropy parameters
        self.entropy_floor_threshold = entropy_floor_threshold
        self.entropy_floor_weight = entropy_floor_weight
        self.entropy_ceiling_threshold = entropy_ceiling_threshold
        self.entropy_ceiling_weight = entropy_ceiling_weight

        # Gate parameters
        self.gate_entropy_weight = gate_entropy_weight
        self.gate_sparsity_weight = gate_sparsity_weight

        # HSIC parameters
        self.hsic_weight = hsic_weight
        self.hsic_sigma = hsic_sigma
        self.hsic_sample_fraction = hsic_sample_fraction

        # Position disagreement
        self.position_disagreement_weight = position_disagreement_weight

        # Lambda parameters
        self.lambda_smoothness_weight = lambda_smoothness_weight
        self.lambda_entropy_weight = lambda_entropy_weight

        # Hard focus parameters
        self.hard_focus_weight = hard_focus_weight
        self.complexity_diversity_weight = complexity_diversity_weight

        # Head specialization parameters
        self.complexity_ortho_weight = complexity_ortho_weight
        self.load_balance_weight = load_balance_weight

        # Warmup
        self.warmup_steps = warmup_steps

    def entropy_floor_loss(self, entropy_normalized: torch.Tensor) -> torch.Tensor:
        """
        Prevent attention entropy collapse (too sharp/peaked).

        When entropy falls below threshold, loss becomes positive,
        pushing entropy back up to prevent one-hot attention.

        Args:
            entropy_normalized: Normalized entropy in [0, 1]

        Returns:
            Scalar loss (0 if entropy >= threshold)
        """
        deficit = self.entropy_floor_threshold - entropy_normalized
        return self.entropy_floor_weight * F.relu(deficit)

    def entropy_ceiling_loss(self, entropy_normalized: torch.Tensor) -> torch.Tensor:
        """
        Prevent uniform attention (too diffuse).

        When entropy exceeds threshold, loss becomes positive,
        encouraging sharper, more focused attention patterns.

        Args:
            entropy_normalized: Normalized entropy in [0, 1]

        Returns:
            Scalar loss (0 if entropy <= threshold)
        """
        excess = entropy_normalized - self.entropy_ceiling_threshold
        return self.entropy_ceiling_weight * F.relu(excess)

    def gate_entropy_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """
        GateRA: Encourage near-binary gating decisions.

        Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p) is maximized at p=0.5
        and minimized at p=0 or p=1. We minimize this to push gates binary.

        Args:
            gate: Gate values in [0, 1], shape [B, H, N] or [B, H]

        Returns:
            Scalar loss (lower when gates are binary)
        """
        eps = 1e-7
        gate = gate.clamp(eps, 1 - eps)
        binary_entropy = -gate * torch.log(gate) - (1 - gate) * torch.log(1 - gate)
        return self.gate_entropy_weight * binary_entropy.mean()

    def gate_sparsity_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """
        D-Gating: Soft sparsity penalty.

        Penalizes g*(1-g) which is maximized at g=0.5 and zero at g=0 or g=1.
        This provides a smooth gradient toward binary values.

        Args:
            gate: Gate values in [0, 1]

        Returns:
            Scalar loss (lower when gates are binary)
        """
        return self.gate_sparsity_weight * torch.abs(gate * (1 - gate)).mean()

    def hsic_loss(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        HSIC-based head decorrelation.

        Hilbert-Schmidt Independence Criterion measures statistical dependence
        between random variables. HSIC = 0 iff variables are independent.
        We minimize HSIC between all head pairs to encourage diverse representations.

        Uses RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))

        Args:
            head_outputs: [B, H, N, d] per-head attention outputs

        Returns:
            Scalar HSIC loss (minimize for more independent heads)
        """
        B, H, N, d = head_outputs.shape

        # Flatten batch and sequence: [B*N, H, d]
        outputs_flat = head_outputs.permute(0, 2, 1, 3).reshape(B * N, H, d)

        # Sample for efficiency (O(n^2) kernel computation)
        num_samples = max(64, int(outputs_flat.shape[0] * self.hsic_sample_fraction))
        if num_samples < outputs_flat.shape[0]:
            indices = torch.randperm(outputs_flat.shape[0], device=head_outputs.device)[:num_samples]
            outputs_flat = outputs_flat[indices]

        n = outputs_flat.shape[0]

        # Compute RBF kernels per head: K_h[i,j] = exp(-||z_h[i] - z_h[j]||^2 / (2*sigma^2))
        kernels = []
        for h in range(H):
            z = outputs_flat[:, h]  # [n, d]
            dist_sq = torch.cdist(z, z, p=2).pow(2)
            K = torch.exp(-dist_sq / (2 * self.hsic_sigma ** 2))
            kernels.append(K)

        # Centering matrix: H_c = I - (1/n) * 1*1^T
        H_center = torch.eye(n, device=head_outputs.device) - 1.0 / n

        # HSIC between all head pairs: HSIC(X,Y) = (1/n^2) * tr(K_X @ H @ K_Y @ H)
        hsic_sum = 0.0
        num_pairs = 0
        for i in range(H):
            for j in range(i + 1, H):
                KH_i = kernels[i] @ H_center
                KH_j = kernels[j] @ H_center
                # HSIC = trace(KH_i @ KH_j) / n^2 = sum(KH_i * KH_j.T) / n^2
                hsic = (KH_i * KH_j.T).sum() / (n ** 2)
                hsic_sum += hsic
                num_pairs += 1

        return self.hsic_weight * hsic_sum / max(num_pairs, 1)

    def position_disagreement_loss(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Position disagreement: minimize overlap of attended positions across heads.

        Based on Li et al. disagreement regularization. Computes cosine similarity
        between flattened attention patterns of each head pair and minimizes it.

        Args:
            attn: [B, H, N, N] attention weights (post-softmax)

        Returns:
            Scalar loss (lower = more diverse attention patterns)
        """
        B, H, N, _ = attn.shape

        # Average over batch for stability
        attn_avg = attn.mean(dim=0)  # [H, N, N]

        # L2 normalize per head (flatten spatial dims)
        attn_norm = F.normalize(attn_avg.view(H, -1), p=2, dim=-1)  # [H, N*N]

        # Pairwise cosine similarity
        sim_matrix = attn_norm @ attn_norm.T  # [H, H]

        # Sum upper triangular (excluding diagonal - self-similarity is always 1)
        mask = torch.triu(torch.ones(H, H, device=attn.device), diagonal=1).bool()
        return self.position_disagreement_weight * sim_matrix[mask].mean()

    def lambda_smoothness_loss(self, lambda_matrix: torch.Tensor) -> torch.Tensor:
        """
        Encourage spatial smoothness in M-DGSA lambda matrix.

        Penalizes large gradients in lambda values between neighboring positions,
        encouraging smooth transitions in noise cancellation strength.

        Args:
            lambda_matrix: [B, H, N, N] lambda values from MatrixGatedLambda

        Returns:
            Scalar loss (lower = smoother lambda patterns)
        """
        # Compute gradients in both spatial directions (last two dims are N x N)
        grad_x = lambda_matrix[:, :, :, 1:] - lambda_matrix[:, :, :, :-1]
        grad_y = lambda_matrix[:, :, 1:, :] - lambda_matrix[:, :, :-1, :]
        return self.lambda_smoothness_weight * (grad_x.abs().mean() + grad_y.abs().mean())

    def lambda_entropy_loss(self, lambda_matrix: torch.Tensor) -> torch.Tensor:
        """
        Encourage peaked lambda distributions (sharp noise cancellation).

        For each query position, we want lambda to be concentrated on a few
        key positions rather than uniformly spread. Low entropy = more peaked.

        Args:
            lambda_matrix: [B, H, N, N] lambda values

        Returns:
            Scalar loss (lower = more peaked lambda per query)
        """
        eps = 1e-7
        # Normalize lambda to probability distribution per query (softmax over keys)
        lambda_normalized = F.softmax(lambda_matrix, dim=-1)
        # Entropy per query position: H = -sum(p * log(p))
        entropy = -torch.sum(lambda_normalized * torch.log(lambda_normalized + eps), dim=-1)
        return self.lambda_entropy_weight * entropy.mean()

    def hard_focus_loss(
        self,
        attn_weights: torch.Tensor,
        patch_complexity: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        """
        Hard Focus Loss: Penalize attention to low-complexity (easy) patches.

        Based on anti-curriculum learning research (HARDY-MER, TIACBM 2024-2025):
        Forces attention to focus on high-complexity regions first, implementing
        a "hard-to-easy" attention strategy.

        The loss computes how much attention goes to easy patches (low complexity)
        and penalizes it. This encourages the model to attend to harder, more
        informative regions of the image.

        Args:
            attn_weights: Post-softmax attention [B, H, N_q, N_k]
            patch_complexity: Per-patch complexity scores [B, N] in [0, 1]
                Higher = more complex/harder
            num_registers: Number of register tokens (excluded from penalty)

        Returns:
            Scalar loss (lower when attention focuses on high-complexity patches)
        """
        B, H, N_q, N_k = attn_weights.shape

        # Ensure complexity has correct shape
        if patch_complexity.shape[1] != N_k:
            # Complexity might be for patches only, need to handle registers
            if num_registers > 0 and patch_complexity.shape[1] == N_k - num_registers:
                # Prepend 1.0 for register tokens (max complexity = no penalty)
                reg_complexity = torch.ones(B, num_registers, device=patch_complexity.device, dtype=patch_complexity.dtype)
                patch_complexity = torch.cat([reg_complexity, patch_complexity], dim=1)

        # Invert complexity to get "easiness" score: easy = 1 - complexity
        # Higher easiness = lower complexity = we want to penalize attention here
        easiness = 1.0 - patch_complexity  # [B, N_k]

        # Reshape for broadcasting: [B, 1, 1, N_k]
        easiness = easiness.unsqueeze(1).unsqueeze(2)

        # Compute weighted sum: how much attention goes to easy patches
        # attn_weights: [B, H, N_q, N_k], easiness: [B, 1, 1, N_k]
        easy_attention = (attn_weights * easiness).sum(dim=-1)  # [B, H, N_q]

        # Skip register tokens as queries (they should attend freely)
        if num_registers > 0:
            easy_attention = easy_attention[:, :, num_registers:]

        # Mean across all dimensions
        return self.hard_focus_weight * easy_attention.mean()

    def complexity_diversity_loss(
        self,
        attn_weights: torch.Tensor,
        patch_complexity: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        """
        Complexity Diversity Loss: Encourage heads to attend to different complexity levels.

        Some heads should focus on fine details (high complexity), others on
        global context (low complexity). This loss encourages diversity by
        penalizing heads that all attend to the same complexity level.

        Args:
            attn_weights: Post-softmax attention [B, H, N_q, N_k]
            patch_complexity: Per-patch complexity scores [B, N] in [0, 1]
            num_registers: Number of register tokens

        Returns:
            Scalar loss (lower when heads attend to diverse complexity levels)
        """
        B, H, N_q, N_k = attn_weights.shape

        # Handle register tokens in complexity
        if patch_complexity.shape[1] != N_k:
            if num_registers > 0 and patch_complexity.shape[1] == N_k - num_registers:
                reg_complexity = torch.ones(B, num_registers, device=patch_complexity.device, dtype=patch_complexity.dtype)
                patch_complexity = torch.cat([reg_complexity, patch_complexity], dim=1)

        # Compute mean complexity attended by each head
        # attn_weights: [B, H, N_q, N_k], complexity: [B, N_k] -> [B, 1, 1, N_k]
        complexity_expanded = patch_complexity.unsqueeze(1).unsqueeze(2)
        mean_complexity_per_head = (attn_weights * complexity_expanded).sum(dim=-1).mean(dim=-1)  # [B, H]

        # Penalize variance being too low (all heads similar complexity preference)
        head_variance = mean_complexity_per_head.var(dim=-1)  # [B]

        # We want variance to be high, so penalize low variance
        # Use negative variance as loss (or 1/variance clamped)
        target_variance = 0.1  # Target minimum variance
        variance_deficit = F.relu(target_variance - head_variance)

        return self.complexity_diversity_weight * variance_deficit.mean()

    def head_aware_hard_focus_loss(
        self,
        attn_weights: torch.Tensor,
        patch_complexity: torch.Tensor,
        head_routing: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        """
        Per-head hard focus loss respecting head specialization.

        Based on MoH (Mixture-of-Heads, 2024) and attention specialization research.
        Only penalizes heads with positive routing (complex-focused) for attending
        to easy patches. Context heads (negative routing) are not penalized.

        Uses soft weighting via sigmoid so gradient flows smoothly through routing.

        Args:
            attn_weights: Post-softmax attention [B, H, N_q, N_k]
            patch_complexity: Per-patch complexity scores [B, N] in [0, 1]
            head_routing: Per-head routing values [H] (learnable)
                +values -> focus on complex, -values -> focus on simple
            num_registers: Number of register tokens

        Returns:
            Scalar loss (lower when complex-focused heads focus on hard patches)
        """
        B, H, N_q, N_k = attn_weights.shape

        # Handle register tokens in complexity
        if patch_complexity.shape[1] != N_k:
            if num_registers > 0 and patch_complexity.shape[1] == N_k - num_registers:
                reg_complexity = torch.ones(B, num_registers, device=patch_complexity.device, dtype=patch_complexity.dtype)
                patch_complexity = torch.cat([reg_complexity, patch_complexity], dim=1)

        # Soft weighting: heads with higher routing get penalized more
        # sigmoid(routing * 2) maps: -inf->0, 0->0.5, +inf->1
        routing_weight = torch.sigmoid(head_routing * 2)  # [H] in (0, 1)

        # Easiness per key position
        easiness = 1.0 - patch_complexity  # [B, N_k]

        # Attention to easy patches per head: [B, H, N_q]
        easy_attn = (attn_weights * easiness.view(B, 1, 1, N_k)).sum(dim=-1)

        # Weight by routing (complex-focused heads penalized more)
        weighted = easy_attn * routing_weight.view(1, H, 1)

        # Skip register queries
        if num_registers > 0:
            weighted = weighted[:, :, num_registers:]

        return self.hard_focus_weight * weighted.mean()

    def complexity_ortho_loss(
        self,
        attn_weights: torch.Tensor,
        patch_complexity: torch.Tensor,
        num_registers: int = 0,
    ) -> torch.Tensor:
        """
        Complexity-targeted orthogonal loss: force heads to have diverse complexity preferences.

        Based on orthogonal head regularization research (2024-2025). Instead of making
        heads orthogonal in general attention patterns, we specifically force them to
        attend to different complexity levels.

        This naturally spreads heads across all complexity levels, with some focusing
        on high-complexity regions and others on low-complexity context.

        Args:
            attn_weights: Post-softmax attention [B, H, N_q, N_k]
            patch_complexity: Per-patch complexity scores [B, N_k] in [0, 1]
            num_registers: Number of register tokens

        Returns:
            Scalar loss (lower when heads have orthogonal complexity preferences)
        """
        B, H, N_q, N_k = attn_weights.shape

        # Handle register tokens
        if patch_complexity.shape[1] != N_k:
            if num_registers > 0 and patch_complexity.shape[1] == N_k - num_registers:
                reg_complexity = torch.ones(B, num_registers, device=patch_complexity.device, dtype=patch_complexity.dtype)
                patch_complexity = torch.cat([reg_complexity, patch_complexity], dim=1)

        # Per-head complexity preference: what complexity each head attends to
        # attn_weights: [B, H, N_q, N_k], complexity: [B, N_k]
        complexity_exp = patch_complexity.view(B, 1, 1, N_k)
        # Weighted average complexity per head per query: [B, H, N_q]
        complexity_per_query = (attn_weights * complexity_exp).sum(dim=-1)
        # Average across queries and batch: [H]
        complexity_pref = complexity_per_query.mean(dim=(0, 2))

        # Normalize to unit vector for orthogonality computation
        C = F.normalize(complexity_pref.unsqueeze(1), dim=0)  # [H, 1]

        # Gram matrix: measures pairwise similarity
        gram = C @ C.T  # [H, H]

        # Target: identity (orthogonal preferences means off-diagonal should be 0)
        # But for a 1D preference, perfect orthogonality isn't achievable
        # Instead, we minimize off-diagonal similarity
        identity = torch.eye(H, device=gram.device)
        off_diag = gram - identity

        return self.complexity_ortho_weight * (off_diag ** 2).mean()

    def load_balance_loss(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        Load balancing loss: ensure all heads contribute to output.

        Inspired by Mixture-of-Experts load balancing (MoE, 2024). Prevents some
        heads from dominating while others are underutilized. This is especially
        important with head routing, as we want all specialized heads to contribute.

        Args:
            head_outputs: Per-head outputs [B, H, N, d]

        Returns:
            Scalar loss (lower when all heads contribute equally)
        """
        B, H, N, d = head_outputs.shape

        # Compute per-head contribution (Frobenius norm)
        head_norms = head_outputs.norm(dim=(-2, -1))  # [B, H]

        # Normalize to get fraction of total output per head
        total_norm = head_norms.sum(dim=-1, keepdim=True) + 1e-7
        head_fractions = head_norms / total_norm  # [B, H]

        # Target: uniform distribution 1/H
        target = 1.0 / H

        # Penalize deviation from uniform
        return self.load_balance_weight * ((head_fractions - target) ** 2).mean()

    def forward(
        self,
        train_step: int,
        attn_weights: Optional[torch.Tensor] = None,  # [B, H, N, N]
        head_outputs: Optional[torch.Tensor] = None,  # [B, H, N, d]
        gate_values: Optional[torch.Tensor] = None,   # [B, H, N] or [B, H]
        lambda_matrix: Optional[torch.Tensor] = None, # [B, H, N, N]
        entropy_normalized: Optional[torch.Tensor] = None,  # scalar or [H]
        patch_complexity: Optional[torch.Tensor] = None,  # [B, N] complexity scores
        num_registers: int = 0,  # Number of register tokens
        head_routing: Optional[torch.Tensor] = None,  # [H] per-head routing for specialization
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses.

        Args:
            train_step: Current training step (for warmup scaling)
            attn_weights: Post-softmax attention weights
            head_outputs: Per-head outputs before final projection
            gate_values: Output gate values from G1
            lambda_matrix: Lambda values from M-DGSA
            entropy_normalized: Pre-computed normalized entropy
            patch_complexity: Per-patch complexity scores for hard focus loss
            num_registers: Number of register tokens
            head_routing: Per-head routing values for head specialization [H]
                +values -> focus on complex, -values -> focus on simple

        Returns:
            Dict with individual loss components and 'total' sum.
            Keys: entropy_floor, entropy_ceiling, gate_entropy, gate_sparsity,
                  hsic, position_disagreement, lambda_smoothness, lambda_entropy,
                  hard_focus, complexity_diversity, complexity_ortho, load_balance, total
        """
        losses = {}

        # Determine device from any available tensor
        device = None
        for tensor in [attn_weights, head_outputs, gate_values, lambda_matrix, patch_complexity]:
            if tensor is not None:
                device = tensor.device
                break

        if device is None:
            # No tensors provided, return zero loss
            return {'total': torch.tensor(0.0)}

        total = torch.tensor(0.0, device=device)

        # Warmup scaling: gradually increase aux loss influence
        if self.warmup_steps > 0 and train_step < self.warmup_steps:
            warmup_scale = train_step / self.warmup_steps
        else:
            warmup_scale = 1.0

        # 1. Entropy regularization
        if entropy_normalized is not None:
            if isinstance(entropy_normalized, (int, float)):
                entropy_normalized = torch.tensor(entropy_normalized, device=device)
            losses['entropy_floor'] = self.entropy_floor_loss(entropy_normalized)
            losses['entropy_ceiling'] = self.entropy_ceiling_loss(entropy_normalized)
            total = total + warmup_scale * (losses['entropy_floor'] + losses['entropy_ceiling'])

        # 2. Gate regularization (GateRA + D-Gating)
        if gate_values is not None:
            losses['gate_entropy'] = self.gate_entropy_loss(gate_values)
            losses['gate_sparsity'] = self.gate_sparsity_loss(gate_values)
            total = total + warmup_scale * (losses['gate_entropy'] + losses['gate_sparsity'])

        # 3. Head diversity (HSIC)
        if head_outputs is not None:
            losses['hsic'] = self.hsic_loss(head_outputs)
            total = total + warmup_scale * losses['hsic']

        # 4. Position disagreement
        if attn_weights is not None:
            losses['position_disagreement'] = self.position_disagreement_loss(attn_weights)
            total = total + warmup_scale * losses['position_disagreement']

        # 5. Lambda regularization (for M-DGSA)
        if lambda_matrix is not None:
            losses['lambda_smoothness'] = self.lambda_smoothness_loss(lambda_matrix)
            losses['lambda_entropy'] = self.lambda_entropy_loss(lambda_matrix)
            total = total + warmup_scale * (losses['lambda_smoothness'] + losses['lambda_entropy'])

        # 6. Hard focus losses (anti-curriculum attention)
        if patch_complexity is not None and attn_weights is not None:
            # Use head-aware loss if head_routing available, otherwise fall back to global
            if head_routing is not None:
                losses['hard_focus'] = self.head_aware_hard_focus_loss(
                    attn_weights, patch_complexity, head_routing, num_registers
                )
            else:
                losses['hard_focus'] = self.hard_focus_loss(attn_weights, patch_complexity, num_registers)
            losses['complexity_diversity'] = self.complexity_diversity_loss(attn_weights, patch_complexity, num_registers)
            total = total + warmup_scale * (losses['hard_focus'] + losses['complexity_diversity'])

            # 7. Complexity-targeted orthogonal loss (head specialization)
            losses['complexity_ortho'] = self.complexity_ortho_loss(attn_weights, patch_complexity, num_registers)
            total = total + warmup_scale * losses['complexity_ortho']

        # 8. Load balancing loss (ensure all heads contribute)
        if head_outputs is not None:
            losses['load_balance'] = self.load_balance_loss(head_outputs)
            total = total + warmup_scale * losses['load_balance']

        losses['total'] = total
        return losses
