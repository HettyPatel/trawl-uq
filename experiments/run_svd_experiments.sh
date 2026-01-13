#!/bin/bash
# Run SVD truncation experiments - Layer 30 and 31
# 8 experiments total:
# Layer 31: Llama-2 mlp_in/out, Llama-3 mlp_in/out
# Layer 30: Llama-2 mlp_in/out, Llama-3 mlp_in/out

set -e  # Exit on error

echo "========================================"
echo "SVD Truncation Experiments - Layer 30 & 31"
echo "========================================"

# Activate environment
source /home/hpate061/trawl-uq/trawl-env/bin/activate

cd /home/hpate061/trawl-uq

# ========== LAYER 31 ==========

# Experiment 1: Llama-2-7b, layer 31, mlp_in
echo ""
echo "========================================"
echo "Experiment 1/8: Llama-2-7b layer 31 mlp_in"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 31 \
    --matrix mlp_in \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama

# Experiment 2: Llama-2-7b, layer 31, mlp_out
echo ""
echo "========================================"
echo "Experiment 2/8: Llama-2-7b layer 31 mlp_out"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 31 \
    --matrix mlp_out \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama

# Experiment 3: Llama-3-8b, layer 31, mlp_in
echo ""
echo "========================================"
echo "Experiment 3/8: Llama-3-8b layer 31 mlp_in"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 31 \
    --matrix mlp_in \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --model-type llama

# Experiment 4: Llama-3-8b, layer 31, mlp_out
echo ""
echo "========================================"
echo "Experiment 4/8: Llama-3-8b layer 31 mlp_out"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 31 \
    --matrix mlp_out \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --model-type llama

# ========== LAYER 30 ==========

# Experiment 5: Llama-2-7b, layer 30, mlp_in
echo ""
echo "========================================"
echo "Experiment 5/8: Llama-2-7b layer 30 mlp_in"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 30 \
    --matrix mlp_in \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama

# Experiment 6: Llama-2-7b, layer 30, mlp_out
echo ""
echo "========================================"
echo "Experiment 6/8: Llama-2-7b layer 30 mlp_out"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 30 \
    --matrix mlp_out \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama

# Experiment 7: Llama-3-8b, layer 30, mlp_in
echo ""
echo "========================================"
echo "Experiment 7/8: Llama-3-8b layer 30 mlp_in"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 30 \
    --matrix mlp_in \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --model-type llama

# Experiment 8: Llama-3-8b, layer 30, mlp_out
echo ""
echo "========================================"
echo "Experiment 8/8: Llama-3-8b layer 30 mlp_out"
echo "========================================"
python experiments/04_svd_truncation.py \
    --layer 30 \
    --matrix mlp_out \
    --dataset nq_open \
    --samples 10 \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --model-type llama

echo ""
echo "========================================"
echo "All 8 experiments complete!"
echo "========================================"
echo "All experiments complete!"
echo "========================================"
