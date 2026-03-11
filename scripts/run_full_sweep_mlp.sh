#!/bin/bash
# Full MLP sweep: all 32 layers, mlp_in + mlp_out
# Estimated time: ~3-4 hours on 1 GPU (500 MCQ samples)
# Uses --resume to skip already-completed layers

set -e

echo "=========================================="
echo "Full MLP Sweep (Llama-2-7b, 32 layers)"
echo "=========================================="

source /data/home/hpate061/trawl-uq/trawl-env/bin/activate

python experiments/15_full_layer_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layers 0-31 \
    --matrices mlp \
    --chunk-size 100 \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
    --resume \
    "$@"

echo "MLP sweep complete!"
