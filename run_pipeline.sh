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
    python scripts/download_data.py
fi

# 5. Run training
echo "Starting training..."
python train_pokemon.py --config base_128

# 6. Generate samples from the newest run's latest checkpoint
echo "Generating samples..."
LATEST_RUN=$(ls -td results/pokemon-eqm-* 2>/dev/null | head -n 1 || true)
if [ -z "$LATEST_RUN" ]; then
    echo "No run directory found under results/. Skipping sample generation."
    exit 0
fi

LATEST_CKPT=$(find "$LATEST_RUN"/checkpoints -maxdepth 1 -name "*.pt" -print | sort -V | tail -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found in $LATEST_RUN/checkpoints. Skipping sample generation."
    exit 0
fi

echo "Using checkpoint: $LATEST_CKPT"
python generate_pokemon.py --ckpt "$LATEST_CKPT" --config base_128 --output_dir "$LATEST_RUN/generated_samples"
