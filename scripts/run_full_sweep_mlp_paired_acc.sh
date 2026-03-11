#!/bin/bash
# Full MLP sweep with PAIRED removal + ACCURACY-ONLY classification
# Uses experiment 16 (accuracy-based) instead of experiment 15 (entropy-based)
# 32 layers × 1 paired job each = 32 jobs

source trawl-env/bin/activate

python experiments/16_full_layer_sweep_acc.py \
    --layers 0-31 \
    --matrices mlp \
    --paired \
    "$@"
