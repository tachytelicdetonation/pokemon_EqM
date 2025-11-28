"""
Test suite for M-DGSA (Matrix Differential Gated Self-Attention).
"""
import torch
import sys
import os

# Add src directory to path for proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pokemon_eqm.models import EqM_models, MatrixGatedLambda, DifferentialAttention


def test_matrix_gated_lambda_shape():
    """Test MatrixGatedLambda output shape."""
    print("=" * 60)
    print("Test 1: MatrixGatedLambda Shape")
    print("=" * 60)

    B, num_heads, N, head_dim = 2, 8, 64, 32
    module = MatrixGatedLambda(head_dim, num_heads, lambda_init=0.5)

    q1 = torch.randn(B, num_heads, N, head_dim)
    k1 = torch.randn(B, num_heads, N, head_dim)

    lambda_matrix = module(q1, k1)

    assert lambda_matrix.shape == (B, num_heads, N, N), \
        f"Expected {(B, num_heads, N, N)}, got {lambda_matrix.shape}"
    print(f"Output shape: {lambda_matrix.shape}")
    print("PASSED\n")
    return True


def test_matrix_gated_lambda_range():
    """Test lambda values are in expected range."""
    print("=" * 60)
    print("Test 2: MatrixGatedLambda Value Range")
    print("=" * 60)

    lambda_init = 0.6
    scale = 1.0
    module = MatrixGatedLambda(32, 8, lambda_init=lambda_init, scale=scale)

    q1 = torch.randn(2, 8, 64, 32)
    k1 = torch.randn(2, 8, 64, 32)

    lambda_matrix = module(q1, k1)

    min_val = lambda_matrix.min().item()
    max_val = lambda_matrix.max().item()
    mean_val = lambda_matrix.mean().item()

    expected_min = lambda_init - scale / 2
    expected_max = lambda_init + scale / 2

    print(f"Lambda range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"Expected range: [{expected_min:.4f}, {expected_max:.4f}]")
    print(f"Mean lambda: {mean_val:.4f} (expected ~{lambda_init})")

    # Allow small margin for numerical precision
    assert min_val >= expected_min - 0.01, f"Lambda {min_val} below expected minimum {expected_min}"
    assert max_val <= expected_max + 0.01, f"Lambda {max_val} above expected maximum {expected_max}"
    print("PASSED\n")
    return True


def test_differential_attention_with_mdgsa():
    """Test DifferentialAttention with M-DGSA enabled."""
    print("=" * 60)
    print("Test 3: DifferentialAttention with M-DGSA")
    print("=" * 60)

    dim, num_heads = 512, 8
    attn = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_matrix_lambda=True,
        layer_idx=0,
        use_spatial_decay=False,  # Disable for simpler test
        use_liere=False
    )

    B, N = 2, 64  # 8x8 patches
    x = torch.randn(B, N, dim)

    out = attn(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("PASSED\n")
    return True


def test_mdgsa_gradient_flow():
    """Verify gradients flow through matrix lambda."""
    print("=" * 60)
    print("Test 4: M-DGSA Gradient Flow")
    print("=" * 60)

    module = MatrixGatedLambda(32, 8, lambda_init=0.5, use_qk_interaction=True)

    q1 = torch.randn(2, 8, 64, 32, requires_grad=True)
    k1 = torch.randn(2, 8, 64, 32, requires_grad=True)

    lambda_matrix = module(q1, k1)
    loss = lambda_matrix.sum()
    loss.backward()

    # Check gradients on module parameters
    assert module.W_q.grad is not None, "W_q should have gradients"
    assert module.W_k.grad is not None, "W_k should have gradients"
    assert module.W_qk.grad is not None, "W_qk should have gradients"

    # Check gradients on inputs
    assert q1.grad is not None, "q1 should have gradients"
    assert k1.grad is not None, "k1 should have gradients"

    print(f"W_q grad norm: {module.W_q.grad.norm().item():.6f}")
    print(f"W_k grad norm: {module.W_k.grad.norm().item():.6f}")
    print(f"W_qk grad norm: {module.W_qk.grad.norm().item():.6f}")
    print(f"q1 grad norm: {q1.grad.norm().item():.6f}")
    print(f"k1 grad norm: {k1.grad.norm().item():.6f}")
    print("PASSED\n")
    return True


def test_differential_attention_backward():
    """Test backward pass through DifferentialAttention with M-DGSA."""
    print("=" * 60)
    print("Test 5: DifferentialAttention Backward with M-DGSA")
    print("=" * 60)

    dim, num_heads = 256, 4
    attn = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_matrix_lambda=True,
        layer_idx=0,
        use_spatial_decay=False,
        use_liere=False
    )

    B, N = 2, 16
    x = torch.randn(B, N, dim, requires_grad=True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    # Check that matrix_lambda parameters have gradients
    assert attn.matrix_lambda.W_q.grad is not None, "W_q should have gradients"
    assert attn.matrix_lambda.W_k.grad is not None, "W_k should have gradients"
    assert x.grad is not None, "Input should have gradients"

    print(f"matrix_lambda.W_q grad norm: {attn.matrix_lambda.W_q.grad.norm().item():.6f}")
    print(f"matrix_lambda.W_k grad norm: {attn.matrix_lambda.W_k.grad.norm().item():.6f}")
    print(f"Input grad norm: {x.grad.norm().item():.6f}")
    print("PASSED\n")
    return True


def test_eqm_model_with_mdgsa():
    """Test full EqM model with M-DGSA."""
    print("=" * 60)
    print("Test 6: Full EqM Model with M-DGSA")
    print("=" * 60)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    model = EqM_models['EqM-S/2'](
        input_size=8,
        num_classes=4,
        in_channels=4,
        use_diff_attn=True,
        use_matrix_lambda=True,
        use_liere=False,
        use_spatial_decay=False
    ).to(device)

    B = 2
    x = torch.randn(B, 4, 8, 8).to(device)
    t = torch.rand(B).to(device)
    y = torch.randint(0, 4, (B,)).to(device)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, t, y)

    assert out.shape[0] == B, f"Expected batch size {B}, got {out.shape[0]}"
    assert out.shape[2] == 8 and out.shape[3] == 8, f"Expected spatial size 8x8, got {out.shape[2]}x{out.shape[3]}"
    print(f"Forward pass successful: {x.shape} -> {out.shape}")

    # Test backward pass
    model.train()
    out = model(x, t, y)
    loss = out.sum()
    loss.backward()

    # Check that M-DGSA parameters have gradients
    mdgsa_grads = 0
    for name, param in model.named_parameters():
        if 'matrix_lambda' in name and param.grad is not None:
            mdgsa_grads += 1

    print(f"M-DGSA parameters with gradients: {mdgsa_grads}")
    assert mdgsa_grads > 0, "M-DGSA parameters should have gradients"
    print("PASSED\n")
    return True


def test_mdgsa_vs_scalar_lambda_comparison():
    """Compare M-DGSA output shape with scalar lambda."""
    print("=" * 60)
    print("Test 7: M-DGSA vs Scalar Lambda Comparison")
    print("=" * 60)

    dim, num_heads = 256, 4
    B, N = 2, 16

    # M-DGSA attention
    attn_mdgsa = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_matrix_lambda=True,
        layer_idx=0,
        use_spatial_decay=False,
        use_liere=False
    )

    # Scalar lambda attention
    attn_scalar = DifferentialAttention(
        dim=dim,
        num_heads=num_heads,
        use_matrix_lambda=False,
        layer_idx=0,
        use_spatial_decay=False,
        use_liere=False
    )

    x = torch.randn(B, N, dim)

    out_mdgsa = attn_mdgsa(x)
    out_scalar = attn_scalar(x)

    assert out_mdgsa.shape == out_scalar.shape, \
        f"Output shapes should match: M-DGSA {out_mdgsa.shape} vs scalar {out_scalar.shape}"
    print(f"M-DGSA output shape: {out_mdgsa.shape}")
    print(f"Scalar output shape: {out_scalar.shape}")
    print("PASSED\n")
    return True


def main():
    """Run all M-DGSA tests."""
    print("\n" + "=" * 60)
    print("Testing M-DGSA Implementation")
    print("=" * 60 + "\n")

    tests = [
        test_matrix_gated_lambda_shape,
        test_matrix_gated_lambda_range,
        test_differential_attention_with_mdgsa,
        test_mdgsa_gradient_flow,
        test_differential_attention_backward,
        test_eqm_model_with_mdgsa,
        test_mdgsa_vs_scalar_lambda_comparison,
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
