#!/bin/bash
set -e

# 1. Install uv if not exists
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 2. Create and activate venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi
source .venv/bin/activate

# 3. Install requirements
echo "Installing dependencies..."
uv pip install -r requirements.txt

# 3.5 Download Wan/Qwen VAE (used by default via args.py -> vae_path)
VAE_FILE="qwen_image_vae.safetensors"
VAE_URL="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/${VAE_FILE}"
if [ ! -f "$VAE_FILE" ]; then
    echo "Downloading VAE (${VAE_FILE}) from Comfy-Org/Qwen-Image_ComfyUI..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$VAE_URL" -o "$VAE_FILE"
    else
        python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
repo_id = "Comfy-Org/Qwen-Image_ComfyUI"
filename = "split_files/vae/qwen_image_vae.safetensors"
local_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
dest = "qwen_image_vae.safetensors"
if local_path != dest:
    shutil.copyfile(local_path, dest)
print(f"Downloaded {dest}")
PY
    fi
else
    echo "VAE already present at ${VAE_FILE}, skipping download."
fi

# 4. Download data
if [ ! -d "data/raw" ]; then
    echo "Downloading data..."
    python scripts/download_data.py
fi

# 5. Run training
echo "Starting training..."
python train_pokemon.py --config production

# 6. Generate samples
echo "Generating samples..."
# Find the latest checkpoint (assuming standard structure results/pokemon-eqm-*/checkpoints/*.pt)
LATEST_CKPT=$(find results -name "*.pt" | sort -V | tail -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found!"
else
    echo "Using checkpoint: $LATEST_CKPT"
    python generate_pokemon.py --ckpt "$LATEST_CKPT" --config production
fi
