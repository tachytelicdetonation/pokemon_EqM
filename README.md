# Pokémon EqM

Experiments in Pokémon image generation using Equilibrium Matching (EqM), with
training, sampling, latent precomputation, and representation-learning variants.
This is a research repository; it does not currently publish a consolidated
benchmark or a downloadable trained checkpoint.

## What is here

| Area | Entry point |
| --- | --- |
| Training | [scripts/train.py](scripts/train.py) |
| Sampling | [scripts/generate.py](scripts/generate.py) |
| Latent preparation | [scripts/precompute_latents.py](scripts/precompute_latents.py) |
| Model and transport | [src/pokemon_eqm](src/pokemon_eqm) |
| Experiment configurations | [configs](configs) |
| Model checks | [tests](tests) |

## Setup and training

Clone the repository and install its dependencies from the repository root:

```bash
git clone https://github.com/tachytelicdetonation/pokemon_EqM.git
cd pokemon_EqM
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Review [configs/production.json](configs/production.json) before starting a run.
The pipeline below downloads data when needed, creates the training-data symlink,
starts training, and samples from the latest checkpoint. It is a training job,
not a lightweight installation check.

```bash
bash run_pipeline.sh
```

With the dataset already prepared, invoke training directly:

```bash
python scripts/train.py --config configs/production.json
```

Sampling lives in `scripts/generate.py`; use the checkpoint and matching model
settings from your run. [run_pipeline.sh](run_pipeline.sh) contains the full
sampling invocation used by the production pipeline. Generated data, checkpoints,
and sample outputs are run artifacts, not evidence of a published benchmark.

## Attribution

The core model architecture, transport logic, and utility code are adapted from
the official implementation of **Equilibrium Matching: Unraveling the Equilibrium
of Diffusion Models** (arXiv:2510.02300). This repository integrates those
components into a Pokémon training pipeline and explores additional model and
representation-learning variants. The original method belongs to its authors.
