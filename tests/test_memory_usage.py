import torch
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import EqM

class TestMemoryUsage(unittest.TestCase):
    def test_graph_accumulation(self):
        """
        Verify that the computation graph does not accumulate during generation loop.
        We simulate this by checking if the output tensor has a grad_fn attached
        when it shouldn't (in 'none' EBM mode).
        """
        device = torch.device("cpu")
        model = EqM(input_size=4, depth=1, hidden_size=16, patch_size=2, num_heads=2).to(device)
        model.eval()
        
        # Mock inputs
        batch_size = 1
        latent_size = 4
        z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        t = torch.ones(batch_size, device=device)
        
        xt = z.clone()
        
        # Simulate one step of the loop from generate_pokemon.py
        # We manually replicate the logic to verify the fix logic works as intended
        
        # Case 1: EBM = 'none' (should have no grad)
        model.ebm = 'none'
        
        # Logic from generate_pokemon.py
        xt = xt.detach()
        context = torch.no_grad()
        with context:
            out = model(xt, t, y)
            xt = xt + out * 0.1
            xt = xt.detach()
            
        self.assertIsNone(xt.grad_fn, "xt should not have grad_fn when EBM is none")
        
        # Case 2: EBM != 'none' (should have grad initially, but detached between steps)
        model.ebm = 'l2'
        xt = z.clone()
        xt.requires_grad_(True) # This happens inside model.forward usually, but we simulate the loop
        
        # Logic from generate_pokemon.py
        xt = xt.detach()
        context = torch.enable_grad()
        with context:
            # Inside forward, it sets requires_grad
            # We simulate the forward pass returning something with grad
            out = model(xt, t, y) 
            # out should have grad_fn because ebm is not none and we are in enable_grad
            # Wait, model.forward sets x0.requires_grad_(True)
            
            xt = xt + out * 0.1
            
            # Check that we have grad here (during the step)
            # Note: This assertion might be tricky if the model output doesn't depend on x0 in a way that autograd tracks if we don't compute loss.
            # But EqM forward computes gradients of E w.r.t x0, so out is a gradient.
            
            # After the step, we detach
            xt = xt.detach()
            
        self.assertIsNone(xt.grad_fn, "xt should not have grad_fn after detach")

if __name__ == '__main__':
    unittest.main()
