#!/bin/bash
# Run all entropy experiments with generation
# Models: Llama-2, Llama-3, GPT-2, GPT-J
# Experiments: CP decomposition (last 2 layers), SVD truncation (last layer, mlp_in and mlp_out)

set -e  # Exit on error

# Activate environment
source /home/hpate061/trawl-uq/trawl-env/bin/activate

echo "=============================================="
echo "ENTROPY EXPERIMENTS - FULL RUN"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# ============================================
# EXPERIMENT 1: CP DECOMPOSITION (NOISE REMOVAL)
# ============================================
echo "=============================================="
echo "EXPERIMENT 1: CP DECOMPOSITION"
echo "=============================================="

# Llama-2-7b-chat (32 layers: 30, 31)
echo ""
echo "--- Llama-2-7b-chat Layer 30 ---"
python experiments/05_entropy_noise_removal.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layer 30 \
    --rank 50 \
    --decomposition cp

echo ""
echo "--- Llama-2-7b-chat Layer 31 ---"
python experiments/05_entropy_noise_removal.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layer 31 \
    --rank 50 \
    --decomposition cp

# Llama-3-8B (32 layers: 30, 31)
echo ""
echo "--- Llama-3-8B Layer 30 ---"
python experiments/05_entropy_noise_removal.py \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type llama \
    --layer 30 \
    --rank 50 \
    --decomposition cp

echo ""
echo "--- Llama-3-8B Layer 31 ---"
python experiments/05_entropy_noise_removal.py \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type llama \
    --layer 31 \
    --rank 50 \
    --decomposition cp

# GPT-2 (12 layers: 10, 11)
echo ""
echo "--- GPT-2 Layer 10 ---"
python experiments/05_entropy_noise_removal.py \
    --model gpt2 \
    --model-type gpt2 \
    --layer 10 \
    --rank 50 \
    --decomposition cp

echo ""
echo "--- GPT-2 Layer 11 ---"
python experiments/05_entropy_noise_removal.py \
    --model gpt2 \
    --model-type gpt2 \
    --layer 11 \
    --rank 50 \
    --decomposition cp

# GPT-J-6B (28 layers: 26, 27)
echo ""
echo "--- GPT-J-6B Layer 26 ---"
python experiments/05_entropy_noise_removal.py \
    --model EleutherAI/gpt-j-6B \
    --model-type gptj \
    --layer 26 \
    --rank 50 \
    --decomposition cp

echo ""
echo "--- GPT-J-6B Layer 27 ---"
python experiments/05_entropy_noise_removal.py \
    --model EleutherAI/gpt-j-6B \
    --model-type gptj \
    --layer 27 \
    --rank 50 \
    --decomposition cp


# ============================================
# EXPERIMENT 2: SVD TRUNCATION
# ============================================
echo ""
echo "=============================================="
echo "EXPERIMENT 2: SVD TRUNCATION"
echo "=============================================="

# Llama-2-7b-chat (last layer: 31)
echo ""
echo "--- Llama-2-7b-chat Layer 31 MLP_IN ---"
python experiments/06_entropy_svd_truncation.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layer 31 \
    --matrix mlp_in

echo ""
echo "--- Llama-2-7b-chat Layer 31 MLP_OUT ---"
python experiments/06_entropy_svd_truncation.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layer 31 \
    --matrix mlp_out

# Llama-3-8B (last layer: 31)
echo ""
echo "--- Llama-3-8B Layer 31 MLP_IN ---"
python experiments/06_entropy_svd_truncation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type llama \
    --layer 31 \
    --matrix mlp_in

echo ""
echo "--- Llama-3-8B Layer 31 MLP_OUT ---"
python experiments/06_entropy_svd_truncation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --model-type llama \
    --layer 31 \
    --matrix mlp_out

# GPT-2 (last layer: 11)
echo ""
echo "--- GPT-2 Layer 11 MLP_IN ---"
python experiments/06_entropy_svd_truncation.py \
    --model gpt2 \
    --model-type gpt2 \
    --layer 11 \
    --matrix mlp_in

echo ""
echo "--- GPT-2 Layer 11 MLP_OUT ---"
python experiments/06_entropy_svd_truncation.py \
    --model gpt2 \
    --model-type gpt2 \
    --layer 11 \
    --matrix mlp_out

# GPT-J-6B (last layer: 27)
echo ""
echo "--- GPT-J-6B Layer 27 MLP_IN ---"
python experiments/06_entropy_svd_truncation.py \
    --model EleutherAI/gpt-j-6B \
    --model-type gptj \
    --layer 27 \
    --matrix mlp_in

echo ""
echo "--- GPT-J-6B Layer 27 MLP_OUT ---"
python experiments/06_entropy_svd_truncation.py \
    --model EleutherAI/gpt-j-6B \
    --model-type gptj \
    --layer 27 \
    --matrix mlp_out


echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "End time: $(date)"
