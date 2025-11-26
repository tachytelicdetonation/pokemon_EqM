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
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw)" ]; then
    echo "Downloading data..."
    python src/pokemon_eqm/utils/download_dataset.py
fi

# 5. Run training
echo "Starting training..."
python scripts/train.py --config configs/production.json

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
# Pass arguments matching production.json where relevant
python scripts/generate.py \
    --ckpt "$LATEST_CKPT" \
    --model "EqM-B/2" \
    --image-size 256 \
    --num-samples 16 \
    --batch-size 16 \
    --output-dir "$LATEST_RUN/generated_samples" \
    --use-liere \
    --uncond
