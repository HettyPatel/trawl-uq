#!/bin/bash
# Setup script for trawl-uq environment
# Run: bash setup_env.sh

set -e

echo "=============================================="
echo "Setting up trawl-uq environment"
echo "=============================================="

# Create virtual environment
if [ ! -d "trawl-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv trawl-env
else
    echo "Virtual environment already exists."
fi

# Activate
source trawl-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 - adjust if needed)
echo "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing trawl-uq package..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data/eval_sets
mkdir -p results

# Create MCQ eval set if it doesn't exist
if [ ! -f "data/eval_sets/eval_set_mcq_nq_open_200.json" ]; then
    echo "Creating MCQ evaluation set..."
    python scripts/create_mcq_eval_set.py --dataset nq_open --samples 200
else
    echo "MCQ evaluation set already exists."
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source trawl-env/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python experiments/07_mcq_entropy_noise_removal.py --model gpt2 --model-type gpt2 --layer 11 --rank 20"
echo ""
