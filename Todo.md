# Project Todo

## High Priority
- [ ] Align `configs/production.json` hyperparameters with `reference_repo/train.py`
- [ ] Verify `scripts/train.py` parity with reference implementation
- [ ] Complete repository refactoring (move scripts, organize modules)

## Medium Priority
- [ ] Implement proper DDP support in `scripts/train.py` if needed
- [ ] Add unit tests for `EqM` model components
- [ ] Visualize attention maps in wandb (per-head, to verify spatial decay)

## Low Priority
- [ ] Documentation updates

## Research Ideas: Auxiliary Losses for Spatial Attention

### Head Diversity Loss
Force different heads to learn different attention patterns:
```python
attn_flat = attn.view(B, num_heads, -1)  # [B, H, N*N]
similarity = torch.corrcoef(attn_flat)   # correlation between heads
loss_diversity = lambda_div * similarity.triu(diagonal=1).pow(2).mean()
```

### Attention Entropy Loss
Prevent attention collapse (too peaked) or uniform (too diffuse):
```python
attn_entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
loss_entropy = lambda_ent * (target_entropy - attn_entropy).pow(2).mean()
```

### Spatial Smoothness Loss
Encourage spatially coherent attention maps:
```python
attn_2d = attn.view(B, num_heads, H, W, H, W)
grad_y = attn_2d[:,:,1:,:,:,:] - attn_2d[:,:,:-1,:,:,:]
grad_x = attn_2d[:,:,:,1:,:,:] - attn_2d[:,:,:,:-1,:,:]
loss_smooth = lambda_smooth * (grad_y.pow(2).mean() + grad_x.pow(2).mean())
```

### Scale-Aware Reconstruction Loss
Weight loss by head locality (local heads = detail, global heads = structure):
```python
head_weights = [0.5, 0.3, 0.2, ...]  # tunable per-head weights
loss = sum(w * MSE(output_from_head_h) for h, w in enumerate(head_weights))
```

### When to Use
- Head diversity: if attention maps collapse (all heads look same)
- Entropy: if training unstable or attention too sparse/diffuse
- Smoothness: if attention maps look noisy/fragmented
- Scale-aware: if detail reconstruction is poor
