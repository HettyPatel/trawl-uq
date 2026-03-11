#!/bin/bash
# Full sweep: all 32 layers, ALL matrices (MLP + gate + attention)
# Estimated time: ~12-16 hours on 1 GPU (500 MCQ samples)
# Uses --resume to skip already-completed layers
# Can be interrupted and restarted safely

set -e

echo "=========================================="
echo "Full Sweep - ALL matrices (Llama-2-7b)"
echo "=========================================="

source /data/home/hpate061/trawl-uq/trawl-env/bin/activate

python experiments/15_full_layer_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --model-type llama \
    --layers 0-31 \
    --matrices all \
    --chunk-size 100 \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
    --resume \
    "$@"

echo "Full sweep complete!"
