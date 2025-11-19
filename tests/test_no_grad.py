import torch

def test_no_grad_behavior():
    x = torch.randn(5, 5)
    
    with torch.no_grad():
        x.requires_grad_(True)
        y = x * 2
        print(f"Inside no_grad: x.requires_grad={x.requires_grad}")
        print(f"Inside no_grad: y.requires_grad={y.requires_grad}")
        print(f"Inside no_grad: y.grad_fn={y.grad_fn}")
        
        try:
            g = torch.autograd.grad(y.sum(), x)
            print("Gradient computation succeeded (unexpected)")
        except Exception as e:
            print(f"Gradient computation failed as expected: {e}")

if __name__ == "__main__":
    test_no_grad_behavior()
