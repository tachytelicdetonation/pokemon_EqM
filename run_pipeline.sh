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
python train_pokemon.py
