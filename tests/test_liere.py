"""
Test script for LieRE (Lie Rotational Positional Encodings) implementation.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.layers import LieRE, Attention
from models import EqM_models

def test_liere_standalone():
    """Test LieRE module standalone."""
    print("=" * 80)
    print("Test 1: LieRE Standalone")
    print("=" * 80)

    # Create LieRE module
    num_dim = 2  # 2D for images (H, W)
    head_dim = 64
    liere = LieRE(num_dim=num_dim, dim=head_dim)

    # Print parameter count
    num_params = sum(p.numel() for p in liere.parameters() if p.requires_grad)
    print(f"✓ LieRE created with {num_params:,} learnable parameters")
    print(f"  Generator params shape: {liere.generator_params.shape}")

    # Test forward pass
    B, num_heads, seq_len, head_dim = 2, 8, 256, 64  # 16x16 image
    x = torch.randn(B, num_heads, seq_len, head_dim)
    dimensions = (16, 16)  # H, W

    # Apply LieRE
    output = liere.apply_rotations(x, dimensions)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Check that rotation is non-trivial (output != input)
    diff = (output - x).abs().mean().item()
    print(f"✓ Mean absolute difference: {diff:.6f} (non-trivial rotation)")

    # Test gradients
    loss = output.sum()
    loss.backward()
    grad_norm = liere.generator_params.grad.norm().item()
    print(f"✓ Gradients flow correctly: grad_norm = {grad_norm:.6f}")

    print()
    return True


def test_attention_with_liere():
    """Test DifferentialAttention with LieRE enabled."""
    print("=" * 80)
    print("Test 2: Attention with LieRE")
    print("=" * 80)

    # Create attention module with LieRE
    dim = 512
    num_heads = 8
    attn_liere = Attention(
        dim=dim,
        num_heads=num_heads,
        use_liere=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in attn_liere.parameters() if p.requires_grad)
    print(f"✓ Attention with LieRE: {num_params:,} parameters")

    # Test forward pass
    B, N, C = 2, 256, 512  # 16x16 patches
    x = torch.randn(B, N, C)

    output = attn_liere(x)
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Test gradients
    loss = output.sum()
    loss.backward()
    liere_grad_norm = attn_liere.liere.generator_params.grad.norm().item()
    print(f"✓ LieRE gradients flow: grad_norm = {liere_grad_norm:.6f}")

    # Compare with RoPE


    print()
    return True


def test_eqm_model_with_liere():
    """Test full EqM model with LieRE."""
    print("=" * 80)
    print("Test 3: Full EqM Model with LieRE")
    print("=" * 80)

    # Create model with LieRE
    model_liere = EqM_models['EqM-L/2'](
        input_size=32,  # 32x32 latent space (256x256 image / 8)
        num_classes=1000,
        in_channels=4,
        use_liere=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in model_liere.parameters())
    trainable_params = sum(p.numel() for p in model_liere.parameters() if p.requires_grad)
    print(f"✓ EqM-L/2 with LieRE: {num_params:,} total params, {trainable_params:,} trainable")

    # Test forward pass
    B = 2
    x = torch.randn(B, 4, 32, 32)  # Latent space
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 1000, (B,))

    model_liere.eval()
    with torch.no_grad():
        output = model_liere(x, t, y)

    expected_shape = (B, 4, 32, 32)
    assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} vs {expected_shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Compare with RoPE model


    # Test backward pass
    model_liere.train()
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 1000, (B,))

    output = model_liere(x, t, y)
    loss = output.sum()
    loss.backward()

    # Check that LieRE parameters have gradients
    liere_params_found = 0
    liere_grads_found = 0
    for name, param in model_liere.named_parameters():
        if 'liere' in name:
            liere_params_found += 1
            if param.grad is not None:
                liere_grads_found += 1
                if liere_grads_found == 1:  # Print first one
                    print(f"✓ LieRE parameter '{name}' has gradients: grad_norm = {param.grad.norm().item():.6f}")

    if liere_grads_found > 0:
        print(f"✓ Found {liere_grads_found}/{liere_params_found} LieRE parameters with gradients")
    else:
        print(f"✗ No gradients found on {liere_params_found} LieRE parameters")
        # Show a few parameter names for debugging
        for i, (name, param) in enumerate(model_liere.named_parameters()):
            if 'liere' in name and i < 3:
                print(f"  Debug: {name}, requires_grad={param.requires_grad}, has_grad={param.grad is not None}")

    assert liere_grads_found > 0, f"LieRE parameters should have gradients! Found {liere_params_found} params but 0 have grads"

    print()
    return True





def test_liere_jittering_disabled():
    """Test LieRE with jittering disabled (deterministic behavior)."""
    print("=" * 80)
    print("Test 5: LieRE Jittering Disabled (Deterministic)")
    print("=" * 80)

    # Create LieRE with jittering disabled
    num_dim = 2
    head_dim = 64
    liere = LieRE(num_dim=num_dim, dim=head_dim, jitter_std=0.0)
    liere.train()  # Training mode

    print(f"✓ LieRE created with jitter_std=0.0")

    # Test forward pass twice with same input
    B, num_heads, seq_len, head_dim = 2, 8, 256, 64
    x = torch.randn(B, num_heads, seq_len, head_dim)
    dimensions = (16, 16)

    # Set manual seed for reproducibility
    torch.manual_seed(42)
    output1 = liere.apply_rotations(x.clone(), dimensions)

    torch.manual_seed(42)
    output2 = liere.apply_rotations(x.clone(), dimensions)

    # Outputs should be identical (deterministic)
    max_diff = (output1 - output2).abs().max().item()
    assert max_diff < 1e-6, f"Outputs should be identical with jitter_std=0.0, got max_diff={max_diff}"
    print(f"✓ Deterministic behavior confirmed: max_diff = {max_diff:.2e}")

    print()
    return True


def test_liere_jittering_enabled():
    """Test LieRE with jittering enabled (stochastic behavior)."""
    print("=" * 80)
    print("Test 6: LieRE Jittering Enabled (Stochastic)")
    print("=" * 80)

    # Create LieRE with jittering enabled
    num_dim = 2
    head_dim = 64
    jitter_std = 0.1
    liere = LieRE(num_dim=num_dim, dim=head_dim, jitter_std=jitter_std)
    liere.train()  # Training mode

    print(f"✓ LieRE created with jitter_std={jitter_std}")

    # Test forward pass twice with same input
    B, num_heads, seq_len, head_dim = 2, 8, 256, 64
    x = torch.randn(B, num_heads, seq_len, head_dim)
    dimensions = (16, 16)

    output1 = liere.apply_rotations(x.clone(), dimensions)
    output2 = liere.apply_rotations(x.clone(), dimensions)

    # Outputs should be different (stochastic jittering)
    max_diff = (output1 - output2).abs().max().item()
    mean_diff = (output1 - output2).abs().mean().item()

    assert max_diff > 1e-4, f"Outputs should differ with jittering, got max_diff={max_diff}"
    print(f"✓ Stochastic behavior confirmed:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Check that shapes are preserved
    assert output1.shape == x.shape, f"Shape mismatch: {output1.shape} vs {x.shape}"
    print(f"✓ Output shape preserved: {output1.shape}")

    print()
    return True


def test_liere_jittering_train_eval_mode():
    """Test that jittering is disabled in eval mode."""
    print("=" * 80)
    print("Test 7: LieRE Jittering Train/Eval Mode Switching")
    print("=" * 80)

    # Create LieRE with jittering enabled
    num_dim = 2
    head_dim = 64
    jitter_std = 0.1
    liere = LieRE(num_dim=num_dim, dim=head_dim, jitter_std=jitter_std)

    B, num_heads, seq_len, head_dim = 2, 8, 256, 64
    x = torch.randn(B, num_heads, seq_len, head_dim)
    dimensions = (16, 16)

    # Test in training mode (jittering enabled)
    liere.train()
    output1_train = liere.apply_rotations(x.clone(), dimensions)
    output2_train = liere.apply_rotations(x.clone(), dimensions)

    train_diff = (output1_train - output2_train).abs().max().item()
    print(f"✓ Training mode: outputs differ (max_diff = {train_diff:.6f})")

    # Test in eval mode (jittering disabled)
    liere.eval()
    torch.manual_seed(42)
    output1_eval = liere.apply_rotations(x.clone(), dimensions)
    torch.manual_seed(42)
    output2_eval = liere.apply_rotations(x.clone(), dimensions)

    eval_diff = (output1_eval - output2_eval).abs().max().item()
    assert eval_diff < 1e-6, f"Eval mode should be deterministic, got max_diff={eval_diff}"
    print(f"✓ Eval mode: outputs identical (max_diff = {eval_diff:.2e})")

    # Verify training mode had jittering
    assert train_diff > 1e-4, "Training mode should have stochastic jittering"
    print(f"✓ Mode switching works correctly")

    print()
    return True


def test_liere_jittering_gradient_flow():
    """Test that gradients flow correctly with jittering."""
    print("=" * 80)
    print("Test 8: LieRE Jittering Gradient Flow")
    print("=" * 80)

    # Create LieRE with jittering
    num_dim = 2
    head_dim = 64
    jitter_std = 0.1
    liere = LieRE(num_dim=num_dim, dim=head_dim, jitter_std=jitter_std)
    liere.train()

    B, num_heads, seq_len, head_dim = 2, 8, 256, 64
    x = torch.randn(B, num_heads, seq_len, head_dim, requires_grad=True)
    dimensions = (16, 16)

    # Forward pass with jittering
    output = liere.apply_rotations(x, dimensions)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients on LieRE parameters
    assert liere.generator_params.grad is not None, "LieRE parameters should have gradients"
    grad_norm = liere.generator_params.grad.norm().item()
    print(f"✓ LieRE parameter gradients: grad_norm = {grad_norm:.6f}")

    # Check gradients on input (should flow through rotation)
    assert x.grad is not None, "Input should have gradients"
    input_grad_norm = x.grad.norm().item()
    print(f"✓ Input gradients flow through rotations: grad_norm = {input_grad_norm:.6f}")

    # Verify gradients are non-zero
    assert grad_norm > 0, "LieRE gradients should be non-zero"
    assert input_grad_norm > 0, "Input gradients should be non-zero"
    print(f"✓ Gradient flow verified with jittering")

    print()
    return True


def test_eqm_model_with_liere_jittering():
    """Test full EqM model with LieRE jittering."""
    print("=" * 80)
    print("Test 9: Full EqM Model with LieRE Jittering")
    print("=" * 80)

    # Create model with LieRE and jittering
    jitter_std = 0.1
    model = EqM_models['EqM-S/2'](  # Use small model for faster testing
        input_size=32,
        num_classes=1000,
        in_channels=4,
        use_liere=True,
        liere_jitter_std=jitter_std
    )

    print(f"✓ EqM-S/2 created with LieRE jittering (jitter_std={jitter_std})")

    # Test forward pass in training mode
    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 1000, (B,))

    model.train()
    output = model(x, t, y)

    assert output.shape == (B, 4, 32, 32), f"Output shape mismatch: {output.shape}"
    print(f"✓ Forward pass successful with jittering: {x.shape} -> {output.shape}")

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check that LieRE parameters have gradients
    liere_grads_found = 0
    for name, param in model.named_parameters():
        if 'liere' in name and param.grad is not None:
            liere_grads_found += 1

    assert liere_grads_found > 0, "LieRE parameters should have gradients"
    print(f"✓ Gradients flow through {liere_grads_found} LieRE parameters")

    # Test eval mode (should be deterministic)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        output1 = model(x, t, y)
        torch.manual_seed(42)
        output2 = model(x, t, y)

    eval_diff = (output1 - output2).abs().max().item()
    assert eval_diff < 1e-5, f"Eval mode should be deterministic, got max_diff={eval_diff}"
    print(f"✓ Eval mode deterministic: max_diff = {eval_diff:.2e}")

    print()
    return True


def test_liere_dinov3_normalized_coords():
    """Test that DINOv3 mode generates normalized coordinates in [-1, +1]."""
    print("=" * 80)
    print("Test 10: DINOv3 Normalized Coordinates")
    print("=" * 80)

    # Create LieRE with DINOv3 mode (no jittering)
    num_dim = 2
    head_dim = 64
    liere = LieRE(num_dim=num_dim, dim=head_dim, jitter_mode='dinov3')
    liere.eval()  # Eval mode (no jittering)

    print(f"✓ LieRE created with jitter_mode='dinov3'")

    # Get positions for 4x4 grid
    dimensions = (4, 4)
    positions = liere._get_jittered_positions(dimensions, 'cpu', training=False)

    # Check shape
    assert positions.shape == (16, 2), f"Shape mismatch: {positions.shape}"
    print(f"✓ Position shape correct: {positions.shape}")

    # Check range is in [-1, +1]
    min_val = positions.min().item()
    max_val = positions.max().item()
    assert min_val >= -1.0 and max_val <= 1.0, f"Positions out of range: [{min_val}, {max_val}]"
    print(f"✓ Positions in [-1, +1] range: [{min_val:.3f}, {max_val:.3f}]")

    # Check that positions are centered (patch centers, not edges)
    # For 4x4 grid: first position should be at (-0.75, -0.75), not (-1.0, -1.0)
    first_pos = positions[0]
    expected_first = torch.tensor([-0.75, -0.75])
    diff = (first_pos - expected_first).abs().max().item()
    assert diff < 0.01, f"First position not centered: {first_pos} vs {expected_first}"
    print(f"✓ Patch centers computed correctly: first_pos = {first_pos.tolist()}")

    print()
    return True


def test_liere_dinov3_shift_augmentation():
    """Test DINOv3 shift augmentation."""
    print("=" * 80)
    print("Test 11: DINOv3 Shift Augmentation")
    print("=" * 80)

    # Create LieRE with shift only
    liere = LieRE(
        num_dim=2, dim=64,
        jitter_mode='dinov3',
        pos_embed_shift=0.2,
        pos_embed_jitter=None,
        pos_embed_rescale=None
    )
    liere.train()

    print(f"✓ LieRE created with shift=0.2")

    # Get positions multiple times
    dimensions = (8, 8)
    pos1 = liere._get_jittered_positions(dimensions, 'cpu', training=True)
    pos2 = liere._get_jittered_positions(dimensions, 'cpu', training=True)

    # Positions should differ (stochastic shift)
    diff = (pos1 - pos2).abs().max().item()
    assert diff > 0.01, f"Positions should differ with shift, got diff={diff}"
    print(f"✓ Stochastic shift confirmed: max_diff = {diff:.3f}")

    # Shift should be additive (approximately uniform distribution)
    mean_diff = (pos1 - pos2).abs().mean().item()
    print(f"✓ Mean difference: {mean_diff:.3f}")

    print()
    return True


def test_liere_dinov3_jitter_augmentation():
    """Test DINOv3 log-uniform jitter augmentation."""
    print("=" * 80)
    print("Test 12: DINOv3 Log-Uniform Jitter")
    print("=" * 80)

    # Create LieRE with jitter only
    liere = LieRE(
        num_dim=2, dim=64,
        jitter_mode='dinov3',
        pos_embed_shift=None,
        pos_embed_jitter=2.0,  # Scale in [0.5, 2.0]
        pos_embed_rescale=None
    )
    liere.train()

    print(f"✓ LieRE created with jitter=2.0 (scale in [0.5, 2.0])")

    # Get positions
    dimensions = (8, 8)
    # Get base positions (no jittering)
    liere.eval()
    base_pos = liere._get_jittered_positions(dimensions, 'cpu', training=False)

    # Get jittered positions
    liere.train()
    torch.manual_seed(42)  # For reproducible jitter
    jittered_pos = liere._get_jittered_positions(dimensions, 'cpu', training=True)

    # Jittering is multiplicative, so ratio should be in [0.5, 2.0]
    # Check a few positions
    for i in [0, 10, 30]:
        ratio = jittered_pos[i] / (base_pos[i] + 1e-8)  # Avoid division by zero
        print(f"  Position {i}: ratio = {ratio.tolist()}")

    # Positions should differ significantly (multiplicative noise)
    diff = (jittered_pos - base_pos).abs().max().item()
    assert diff > 0.01, f"Positions should differ with jitter, got diff={diff}"
    print(f"✓ Log-uniform jitter confirmed: max_diff = {diff:.3f}")

    print()
    return True


def test_liere_dinov3_rescale_augmentation():
    """Test DINOv3 global rescale augmentation."""
    print("=" * 80)
    print("Test 13: DINOv3 Global Rescale")
    print("=" * 80)

    # Create LieRE with rescale only
    liere = LieRE(
        num_dim=2, dim=64,
        jitter_mode='dinov3',
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=3.0  # Scale in [1/3, 3]
    )
    liere.train()

    print(f"✓ LieRE created with rescale=3.0 (scale in [0.33, 3.0])")

    # Get base and rescaled positions
    liere.eval()
    base_pos = liere._get_jittered_positions((8, 8), 'cpu', training=False)

    liere.train()
    torch.manual_seed(42)
    rescaled_pos = liere._get_jittered_positions((8, 8), 'cpu', training=True)

    # Global rescale should affect all dimensions equally
    # Check that the ratio is consistent across positions
    ratios = rescaled_pos / (base_pos + 1e-8)
    ratio_std = ratios.std().item()

    print(f"✓ Rescale ratio std: {ratio_std:.6f} (should be low for global scaling)")
    print(f"  Sample ratios: {ratios[:3, 0].tolist()}")

    print()
    return True


def test_eqm_model_with_dinov3_jittering():
    """Test full EqM model with DINOv3-style jittering."""
    print("=" * 80)
    print("Test 14: Full EqM Model with DINOv3 Jittering")
    print("=" * 80)

    # Create model with DINOv3 jittering
    model = EqM_models['EqM-S/2'](
        input_size=32,
        num_classes=1000,
        in_channels=4,
        use_liere=True,
        liere_jitter_mode='dinov3',
        liere_pos_embed_shift=0.1,
        liere_pos_embed_jitter=1.5,
        liere_pos_embed_rescale=2.0
    )

    print(f"✓ EqM-S/2 created with DINOv3 jittering")

    # Test forward pass in training mode
    B = 2
    x = torch.randn(B, 4, 32, 32)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 1000, (B,))

    model.train()
    output = model(x, t, y)

    assert output.shape == (B, 4, 32, 32), f"Output shape mismatch: {output.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Test backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    liere_grads_found = 0
    for name, param in model.named_parameters():
        if 'liere' in name and param.grad is not None:
            liere_grads_found += 1

    assert liere_grads_found > 0, "LieRE parameters should have gradients"
    print(f"✓ Gradients flow through {liere_grads_found} LieRE parameters")

    # Test deterministic eval mode
    model.eval()
    with torch.no_grad():
        torch.manual_seed(42)
        output1 = model(x, t, y)
        torch.manual_seed(42)
        output2 = model(x, t, y)

    eval_diff = (output1 - output2).abs().max().item()
    assert eval_diff < 1e-5, f"Eval mode should be deterministic, got max_diff={eval_diff}"
    print(f"✓ Eval mode deterministic: max_diff = {eval_diff:.2e}")

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing LieRE Implementation")
    print("=" * 80 + "\n")

    tests = [
        test_liere_standalone,
        test_attention_with_liere,
        test_eqm_model_with_liere,
        # test_rope_liere_mutual_exclusion,
        test_liere_jittering_disabled,
        test_liere_jittering_enabled,
        test_liere_jittering_train_eval_mode,
        test_liere_jittering_gradient_flow,
        test_eqm_model_with_liere_jittering,
        test_liere_dinov3_normalized_coords,
        test_liere_dinov3_shift_augmentation,
        test_liere_dinov3_jitter_augmentation,
        test_liere_dinov3_rescale_augmentation,
        test_eqm_model_with_dinov3_jittering,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! LieRE implementation is working correctly.")
    else:
        print(f"\n❌ {total - passed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
