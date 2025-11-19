# Pokemon EqM

This project implements Equilibrium Matching (EqM) for generating Pokemon images.

## Attribution

The core model architecture (`models.py`), transport logic (`transport/`), and utility scripts (`wandb_utils.py`) are adapted from the official implementation of **Equilibrium Matching: Unraveling the Equilibrium of Diffusion Models** (arXiv:2510.02300).

We have integrated these components into a standalone pipeline for training on the Pokemon dataset.

## Setup

Run the full pipeline (setup + training):
```bash
bash run_pipeline.sh
```

Or install manually:
```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python download_data.py
python train_pokemon.py
```
