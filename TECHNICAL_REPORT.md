# Pokemon EqM — Technical Report
Date: 2025-11-20  
Repo: `pokemon_eqm`

## 1. Goal and Scope
Standalone reproduction of the Equilibrium Matching (EqM) generative framework, adapted to synthesize Pokemon artwork. The code base trains a transformer-based EqM backbone on VAE latents of Pokemon images, evaluates with EMA weights, and supports multiple sampling strategies (gradient descent or ODE/SDE-based transport).

## 2. Data Pipeline
- **Source**: Official artwork sprites pulled from PokeAPI via `scripts/download_data.py`, which expands `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{id}.png` and saves to `data/raw`.
- **Dataset class**: `datasets/pokemon.py` gathers all `.png/.jpg/.jpeg` files, converts to RGB, applies torchvision transforms (resize to `image_size`, random horizontal flip, normalization to [-1,1]), and returns label `0` (currently single-class/unconditional).
- **VAE latents**: Images are encoded to latents with `utils/vae.py::encode_latents`, using Wan/Qwen VAE if `qwen_image_vae.safetensors` is present, otherwise Stable Diffusion VAE. Scaling differs per VAE (Wan: ×0.661, SD: ×0.18215) to normalize latent variance.

## 3. Model Architecture (EqM Transformer)
- **Registry**: `models/__init__.py` defines size variants (S/B/L/XL with patch sizes 2/4/8). Training uses `EqM-XL/2` in production config and `EqM-S/2` in test config.
- **Backbone** (`models/core.py`):
  - Patch embedding (`timm.PatchEmbed`) over VAE latents of spatial size `image_size/8`; fixed 2D sinusoidal position embeddings.
  - Conditioning: timestep embedding (`TimestepEmbedder` with sinusoidal basis + MLP) and label embedding (`LabelEmbedder` with classifier-free dropout); summed to form conditioning vector `c`.
  - Stacked `SiTBlock` layers (`models/layers.py`), each using differential attention, RMSNorm, SwiGLU MLP, and adaptive LayerNorm modulation that gates both attention and MLP residuals.
  - Final adaptive LayerNorm + linear projection back to patch tokens, followed by unpatchify to image grid. When `learn_sigma=True` (default), the network predicts mean and log-variance; trainer currently slices off sigma and trains on the mean.
- **Energy-based options**: `ebm` flag (`none|l2|dot|mean`) enables gradient-based energy computation inside `forward`; used for NGD sampling or energy outputs.
- **Classifier-free guidance**: `forward_with_cfg` splits batch into conditioned/unconditioned halves and fuses with `cfg_scale`.

## 4. Transport Formulation and Objectives
- **Path choices** (`transport/path.py`): Linear (IC), generalized VP (GVP), or VP diffusion-style paths. Each defines time-dependent coefficients `alpha_t`, `sigma_t`, drifts, and helper transforms between velocity/score/noise parameterizations.
- **Transport wrapper** (`transport/__init__.py`, `transport/transport.py`): Selects model type (velocity/score/noise), loss weighting (none/velocity/likelihood), and epsilon guards (`train_eps`, `sample_eps`) based on path/model combo.
- **Training sample construction**: `Transport.sample` draws latent noise `x0`, picks `t∼U[t0,t1]`, and couples `(x0,x1)` through the chosen path to get `(xt, ut)`.
- **Losses** (`Transport.training_losses`):
  - Velocity mode (default): MSE between model output and target vector field `ut`, scaled by a time-dependent factor `ct(t)`; optional dispersive loss on final transformer activations when `return_act` is requested.
  - Score/noise modes: same weighting but converts between parameterizations using path-specific drift/variance terms.
- **Auxiliary pieces**: `get_drift` and `get_score` expose probability-flow ODE components; `Sampler` wraps SDE (Euler/Heun) and ODE (Euler/Heun/Dopri5) integrators from `transport/integrators.py`, plus likelihood ODE for log-prob estimation.

## 5. Training Pipeline (`train_pokemon.py`, `engine.py`)
- **Argument surface**: `args.py` loads CLI/`@configs/*.config` files. Key knobs: model variant, image size, path type, prediction head, loss weight, VAE path/type, CFG scale (for sampling), logging and checkpoint cadence. Optimizer is fixed to schedule-free AdamW.
- **Optimization**: ScheduleFree AdamW (`schedulefree.AdamWScheduleFree`) with optional built-in warmup; no external LR schedulers.
- **EMA**: Shadow model updated every step with decay 0.9999 (`engine.update_ema`).
- **Loop**:
  1) Load/transform images, encode to latents (no gradients through VAE).  
  2) Build model kwargs (`y`, optional activations), compute transport loss, backprop, optimizer step, optional scheduler step (none when schedule-free), EMA update.  
  3) Periodic logging to stdout and Weights & Biases (`utils/wandb.py`), checkpointing to `results/<run>/checkpoints`, and EMA-based sample generation.
- **Resume**: If `--run_id` is provided, the trainer loads the latest checkpoint in the matching run directory and resumes step counter.

## 6. Generation Pipeline (`generate_pokemon.py`)
- **Inputs**: Random Gaussian latents shaped to VAE channels (`vae_channels`) and spatial size `image_size/8`; class labels default to zero. If `cfg_scale>1`, batches are doubled with null labels for CFG.
- **Samplers**:
  - Gradient Descent (`gd`): Iterative `xt += model(xt,t,y)*stepsize` with `torch.no_grad` unless EBM is active.
  - Natural Gradient Descent (`ngd`): Uses momentum-like accumulator `m` with `mu` coefficient, still driven by model outputs.
  - ODE-based (`ode_dopri5|ode_euler|ode_heun`): Uses `Sampler.sample_ode` with drift derived from transport equations; steps over `[t0,t1]` with `num_sampling_steps`.
- **Decoding**: Final latents are sliced back to conditioned half if CFG was used, decoded via VAE (`decode_latents`), scaled to uint8, and written to `output_dir` (default `generated_samples/` or experiment `samples/` during training).

## 7. Configurations and Orchestration
- **Production config** (`configs/prod.config`): EqM-XL/2, 256×256 images, batch 32, 1000 epochs.
- **Test config** (`configs/test.config`): EqM-S/2, 64×64 images, batch 8, 10 epochs, frequent logging/checkpoints, explicit VAE/data paths.
- **Pipeline script** (`run_pipeline.sh`): Installs `uv`, creates venv, installs deps, downloads data if missing, trains with prod config, then finds latest checkpoint and generates samples with `generate_pokemon.py`.
- **Dependencies** (`requirements.txt`): torch/torchvision, timm, diffusers, accelerate, torchdiffeq, wandb, schedulefree.

## 8. Experiment Tracking and Outputs
- **Logging**: WandB run names stamped with timestamp; when WANDB_KEY is absent or init fails, a local directory under `results/` is still created.
- **Checkpoints & samples**: Stored under `results/<run_id>/checkpoints` and `results/<run_id>/samples/step_xxxxxxx/`. Checkpoints include model, EMA, optimizer state, and args for reproducibility.

## 9. Testing and Validation
- `tests/test_model_forward.py`: Smoke test for EqM forward shapes on CPU/CUDA/MPS.
- `tests/test_memory_usage.py`: Ensures generation loop detaches correctly to avoid graph growth, both in plain and EBM modes.
- `tests/test_no_grad.py`: Demonstrates autograd behavior inside `torch.no_grad` (regression guard for generation loop detaching).

## 10. Notable Design Choices
- **Transformer over latents**: Operating in VAE latent space drastically reduces spatial resolution (×1/8), making large EqM variants tractable.
- **Differential attention + AdaLN gates**: Now uses two value projections (contrastive subtraction), softplus-gated lambda init≈0.14, QK normalization, optional RoPE, grouped KV heads, head-drop, and windowed masking; improves stability and efficiency while keeping the contrastive residual design.
- **Transport flexibility**: Swappable paths (Linear/GVP/VP) and objectives (velocity/score/noise) make the code ready for ablations; default favors velocity with time-dependent scaling `ct(t)` to align with energy-compatible targets.
- **EMA sampling**: All published samples use EMA weights to reduce noise and improve perceptual quality.
- **Safety switches**: `train_eps`/`sample_eps` guard against numerical issues near t=0/1; CFG implemented without duplicating full forward passes beyond the concatenation trick.

## 11. How to Reproduce
1) `bash run_pipeline.sh` (creates venv, installs deps, downloads data, trains with production settings, generates samples).  
2) For quick sanity: `python train_pokemon.py @configs/test.config` then `python generate_pokemon.py --ckpt <latest_test_ckpt>`.  
3) To switch samplers: set `--sampler ode_dopri5` (or `gd`, `ngd`) and adjust `--num-sampling-steps` / `--stepsize`.  
4) To tweak architecture: choose any key from `EqM_models` (e.g., `EqM-B/4`) and adjust `--image-size` accordingly (must be divisible by 8).

## 12. Current Limitations and Next Steps
- Single-class setup; label embedding is ready for multi-class but dataset assigns all labels to 0.
- Dispersive loss is disabled in default training (since `return_act` is False); enabling it requires setting `return_act=True` where the trainer builds `model_kwargs`.
- SDE sampling is implemented but not currently wired in the CLI (only ODE and GD/NGD exposed); exposing it would enable stochastic samplers.
- Data quality tied to PokeAPI official artwork; no filtering or augmentations beyond flip/resize/normalize.
- No automated eval metrics (FID/KID); could be added using saved checkpoints and decoded samples.

## 13. File/Directory Map
- Core model: `models/core.py`, blocks and layers: `models/layers.py`, registry: `models/__init__.py`
- Transport math: `transport/transport.py`, `transport/path.py`, `transport/integrators.py`, helpers: `transport/utils.py`
- Training loop: `train_pokemon.py`, `engine.py`
- Sampling CLI: `generate_pokemon.py`
- Data utils: `datasets/pokemon.py`, downloader: `scripts/download_data.py`
- VAE + logging: `utils/vae.py`, `utils/wandb.py`
- Configs: `configs/prod.config`, `configs/test.config`
- Tests: `tests/`

## 14. Summary
The project integrates a high-capacity EqM transformer operating on VAE latents with a flexible transport formulation, enabling velocity-, score-, or noise-based training. The pipeline is end-to-end: data acquisition from PokeAPI, latent-space modeling with classifier-free conditioning, robust training loop with EMA and schedule-free optimization, and versatile deterministic or ODE-based sampling. The modular structure (paths, samplers, model registry, VAE abstraction) supports quick ablations while preserving a reproducible default path for Pokemon image synthesis.
