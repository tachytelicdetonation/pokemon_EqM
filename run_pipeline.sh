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

# 4. Download data
if [ ! -d "data/raw" ]; then
    echo "Downloading data..."
    python download_data.py
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

