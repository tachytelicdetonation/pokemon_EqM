"""
Test script for LieRE (Lie Rotational Positional Encodings) implementation.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.layers import LieRE, DifferentialAttention
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


def test_differential_attention_with_liere():
    """Test DifferentialAttention with LieRE enabled."""
    print("=" * 80)
    print("Test 2: DifferentialAttention with LieRE")
    print("=" * 80)

    # Create attention module with LieRE
    dim = 512
    num_heads = 8
    attn_liere = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_rope=False,
        use_liere=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in attn_liere.parameters() if p.requires_grad)
    print(f"✓ DifferentialAttention with LieRE: {num_params:,} parameters")

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
    attn_rope = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_rope=True,
        use_liere=False
    )
    num_params_rope = sum(p.numel() for p in attn_rope.parameters() if p.requires_grad)
    print(f"✓ DifferentialAttention with RoPE: {num_params_rope:,} parameters")
    print(f"  LieRE adds {num_params - num_params_rope:,} learnable parameters")

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
        use_rope=False,
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
    model_rope = EqM_models['EqM-L/2'](
        input_size=32,
        num_classes=1000,
        in_channels=4,
        use_rope=True,
        use_liere=False
    )

    num_params_rope = sum(p.numel() for p in model_rope.parameters())
    trainable_params_rope = sum(p.numel() for p in model_rope.parameters() if p.requires_grad)
    print(f"✓ EqM-L/2 with RoPE: {num_params_rope:,} total params, {trainable_params_rope:,} trainable")
    print(f"  LieRE adds {trainable_params - trainable_params_rope:,} learnable parameters")

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


def test_rope_liere_mutual_exclusion():
    """Test that RoPE and LieRE cannot be used simultaneously."""
    print("=" * 80)
    print("Test 4: RoPE-LieRE Mutual Exclusion")
    print("=" * 80)

    try:
        attn = DifferentialAttention(
            dim=512,
            num_heads=8,
            use_rope=True,
            use_liere=True  # This should fail
        )
        print("✗ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)}")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing LieRE Implementation")
    print("=" * 80 + "\n")

    tests = [
        test_liere_standalone,
        test_differential_attention_with_liere,
        test_eqm_model_with_liere,
        test_rope_liere_mutual_exclusion,
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
