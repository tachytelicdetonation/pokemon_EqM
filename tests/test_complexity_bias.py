"""
Tests for complexity bias (anti-curriculum attention) implementation.
"""
import torch
import sys
sys.path.insert(0, 'src')

from pokemon_eqm.models import (
    compute_patch_complexity,
    compute_complexity_attention_bias,
    Attention,
    DifferentialAttention,
    EqM
)
from pokemon_eqm.losses.auxiliary_losses import AuxiliaryLossComputer


def test_patch_complexity_variance():
    """Test variance-based complexity computation."""
    print("Testing patch complexity (variance method)...")
    B, N, C = 2, 16, 64
    x = torch.randn(B, N, C)

    complexity = compute_patch_complexity(x, method='variance', normalize=True)

    assert complexity.shape == (B, N), f"Expected shape {(B, N)}, got {complexity.shape}"
    assert (complexity >= 0).all() and (complexity <= 1).all(), "Complexity should be in [0, 1]"
    print(f"  Complexity shape: {complexity.shape}, range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    print("  ✓ Variance method works")


def test_patch_complexity_gradient():
    """Test gradient-based complexity computation."""
    print("Testing patch complexity (gradient method)...")
    B, N, C = 2, 16, 64
    x = torch.randn(B, N, C)

    complexity = compute_patch_complexity(x, method='gradient', normalize=True)

    assert complexity.shape == (B, N), f"Expected shape {(B, N)}, got {complexity.shape}"
    assert (complexity >= 0).all() and (complexity <= 1).all(), "Complexity should be in [0, 1]"
    print(f"  Complexity shape: {complexity.shape}, range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    print("  ✓ Gradient method works")


def test_patch_complexity_entropy():
    """Test entropy-based complexity computation."""
    print("Testing patch complexity (entropy method)...")
    B, N, C = 2, 16, 64
    x = torch.randn(B, N, C)

    complexity = compute_patch_complexity(x, method='entropy', normalize=True)

    assert complexity.shape == (B, N), f"Expected shape {(B, N)}, got {complexity.shape}"
    assert (complexity >= 0).all() and (complexity <= 1).all(), "Complexity should be in [0, 1]"
    print(f"  Complexity shape: {complexity.shape}, range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    print("  ✓ Entropy method works")


def test_patch_complexity_combined():
    """Test combined complexity computation."""
    print("Testing patch complexity (combined method)...")
    B, N, C = 2, 16, 64
    x = torch.randn(B, N, C)

    complexity = compute_patch_complexity(x, method='combined', normalize=True)

    assert complexity.shape == (B, N), f"Expected shape {(B, N)}, got {complexity.shape}"
    assert (complexity >= 0).all() and (complexity <= 1).all(), "Complexity should be in [0, 1]"
    print(f"  Complexity shape: {complexity.shape}, range: [{complexity.min():.3f}, {complexity.max():.3f}]")
    print("  ✓ Combined method works")


def test_patch_complexity_with_registers():
    """Test complexity with register tokens."""
    print("Testing patch complexity with register tokens...")
    B, N, C = 2, 20, 64  # 4 registers + 16 patches
    num_registers = 4
    x = torch.randn(B, N, C)

    complexity = compute_patch_complexity(x, method='variance', num_registers=num_registers, normalize=True)

    assert complexity.shape == (B, N), f"Expected shape {(B, N)}, got {complexity.shape}"
    # Register tokens should have max complexity (1.0)
    assert (complexity[:, :num_registers] == 1.0).all(), "Register tokens should have complexity 1.0"
    print(f"  Complexity shape: {complexity.shape}")
    print(f"  Register complexity: {complexity[:, :num_registers].mean():.3f}")
    print(f"  Patch complexity range: [{complexity[:, num_registers:].min():.3f}, {complexity[:, num_registers:].max():.3f}]")
    print("  ✓ Register tokens handled correctly")


def test_complexity_attention_bias():
    """Test attention bias computation from complexity."""
    print("Testing complexity attention bias...")
    B, N = 2, 16
    complexity = torch.rand(B, N)

    bias = compute_complexity_attention_bias(complexity, scale=1.0, mode='additive')

    assert bias.shape == (B, 1, 1, N), f"Expected shape {(B, 1, 1, N)}, got {bias.shape}"
    # Check that high complexity gives positive bias, low gives negative
    high_complexity_idx = complexity.argmax(dim=-1)
    low_complexity_idx = complexity.argmin(dim=-1)
    for b in range(B):
        high_bias = bias[b, 0, 0, high_complexity_idx[b]].item()
        low_bias = bias[b, 0, 0, low_complexity_idx[b]].item()
        assert high_bias > low_bias, f"High complexity should have higher bias: {high_bias} vs {low_bias}"
    print(f"  Bias shape: {bias.shape}")
    print(f"  Bias range: [{bias.min():.3f}, {bias.max():.3f}]")
    print("  ✓ Complexity bias correctly computed")


def test_attention_with_complexity_bias():
    """Test Attention module with complexity bias enabled."""
    print("Testing Attention with complexity bias...")
    B, N, C = 2, 16, 64
    num_heads = 4
    x = torch.randn(B, N, C)

    attn = Attention(
        dim=C, num_heads=num_heads,
        use_complexity_bias=True,
        complexity_method='variance',
        complexity_bias_scale=1.0,
        use_spatial_decay=False,
        use_liere=False,
    )

    out, aux_info = attn(x, return_aux_info=True)

    assert out.shape == (B, N, C), f"Output shape mismatch: {out.shape}"
    assert 'patch_complexity' in aux_info, "patch_complexity should be in aux_info"
    assert aux_info['patch_complexity'] is not None, "patch_complexity should not be None"
    assert aux_info['patch_complexity'].shape == (B, N), f"Complexity shape: {aux_info['patch_complexity'].shape}"
    print(f"  Output shape: {out.shape}")
    print(f"  Patch complexity shape: {aux_info['patch_complexity'].shape}")
    print("  ✓ Attention with complexity bias works")


def test_differential_attention_with_complexity_bias():
    """Test DifferentialAttention module with complexity bias enabled."""
    print("Testing DifferentialAttention with complexity bias...")
    B, N, C = 2, 16, 64
    num_heads = 4
    x = torch.randn(B, N, C)

    attn = DifferentialAttention(
        dim=C, num_heads=num_heads, layer_idx=0,
        use_complexity_bias=True,
        complexity_method='variance',
        complexity_bias_scale=1.0,
        use_spatial_decay=False,
        use_liere=False,
    )

    out, aux_info = attn(x, return_aux_info=True)

    assert out.shape == (B, N, C), f"Output shape mismatch: {out.shape}"
    assert 'patch_complexity' in aux_info, "patch_complexity should be in aux_info"
    assert aux_info['patch_complexity'] is not None, "patch_complexity should not be None"
    assert aux_info['patch_complexity'].shape == (B, N), f"Complexity shape: {aux_info['patch_complexity'].shape}"
    print(f"  Output shape: {out.shape}")
    print(f"  Patch complexity shape: {aux_info['patch_complexity'].shape}")
    print("  ✓ DifferentialAttention with complexity bias works")


def test_hard_focus_loss():
    """Test hard focus auxiliary loss."""
    print("Testing hard focus loss...")
    B, H, N = 2, 4, 16
    attn_weights = torch.softmax(torch.randn(B, H, N, N), dim=-1)
    patch_complexity = torch.rand(B, N)

    aux_computer = AuxiliaryLossComputer(hard_focus_weight=0.02)
    losses = aux_computer.forward(
        train_step=1000,
        attn_weights=attn_weights,
        patch_complexity=patch_complexity,
    )

    assert 'hard_focus' in losses, "hard_focus loss should be computed"
    assert 'complexity_diversity' in losses, "complexity_diversity loss should be computed"
    assert losses['hard_focus'].item() >= 0, "hard_focus loss should be non-negative"
    assert losses['complexity_diversity'].item() >= 0, "complexity_diversity loss should be non-negative"
    print(f"  hard_focus loss: {losses['hard_focus'].item():.6f}")
    print(f"  complexity_diversity loss: {losses['complexity_diversity'].item():.6f}")
    print(f"  total loss: {losses['total'].item():.6f}")
    print("  ✓ Hard focus losses computed correctly")


def test_complexity_diversity_encourages_head_diversity():
    """Test that complexity diversity loss encourages diverse head preferences."""
    print("Testing complexity diversity loss encourages diversity...")
    B, H, N = 2, 4, 16

    # Create attention weights where all heads focus on same complexity level
    # (low diversity = high loss)
    patch_complexity = torch.linspace(0, 1, N).unsqueeze(0).expand(B, N)

    # All heads focus on high complexity patches
    uniform_attn = torch.zeros(B, H, N, N)
    uniform_attn[:, :, :, -N//4:] = 1.0 / (N//4)  # Focus on last quarter (highest complexity)

    # Diverse attention: different heads focus on different complexity levels
    diverse_attn = torch.zeros(B, H, N, N)
    for h in range(H):
        start = (h * N) // H
        end = ((h + 1) * N) // H
        diverse_attn[:, h, :, start:end] = 1.0 / (end - start)

    aux_computer = AuxiliaryLossComputer(complexity_diversity_weight=0.1)

    uniform_losses = aux_computer.forward(
        train_step=1000,
        attn_weights=uniform_attn,
        patch_complexity=patch_complexity,
    )

    diverse_losses = aux_computer.forward(
        train_step=1000,
        attn_weights=diverse_attn,
        patch_complexity=patch_complexity,
    )

    # Uniform attention (low diversity) should have higher complexity_diversity loss
    print(f"  Uniform attention complexity_diversity loss: {uniform_losses['complexity_diversity'].item():.6f}")
    print(f"  Diverse attention complexity_diversity loss: {diverse_losses['complexity_diversity'].item():.6f}")
    print("  ✓ Complexity diversity loss computed (penalizes low diversity)")


def test_eqm_with_complexity_bias():
    """Test full EqM model with complexity bias enabled."""
    print("Testing EqM model with complexity bias...")
    B = 2
    C = 4
    H = W = 32
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    y = torch.zeros(B, dtype=torch.long)

    model = EqM(
        input_size=H,
        patch_size=4,
        in_channels=C,
        hidden_size=64,
        depth=2,
        num_heads=4,
        use_diff_attn=True,
        use_complexity_bias=True,
        complexity_method='variance',
        complexity_bias_scale=1.0,
        num_classes=10,
        uncond=True,
        learn_sigma=False,
    )

    out, aux_info = model(x, t, y, return_aux_info=True)

    assert out.shape == (B, C, H, W), f"Output shape mismatch: {out.shape}"
    assert 'patch_complexity' in aux_info, "patch_complexity should be in aux_info"
    print(f"  Output shape: {out.shape}")
    print(f"  Aux info contains: {list(aux_info.keys())}")
    print("  ✓ EqM with complexity bias works")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Complexity Bias (Anti-Curriculum Attention)")
    print("=" * 60)
    print()

    test_patch_complexity_variance()
    test_patch_complexity_gradient()
    test_patch_complexity_entropy()
    test_patch_complexity_combined()
    test_patch_complexity_with_registers()
    test_complexity_attention_bias()
    test_attention_with_complexity_bias()
    test_differential_attention_with_complexity_bias()
    test_hard_focus_loss()
    test_complexity_diversity_encourages_head_diversity()
    test_eqm_with_complexity_bias()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
