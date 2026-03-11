#!/bin/bash
# Full MLP sweep with PAIRED removal (mlp_in + mlp_out removed together)
# This gives stronger perturbation and cleaner noise/signal separation
# 32 layers × 1 paired job each = 32 jobs

source trawl-env/bin/activate

python experiments/15_full_layer_sweep.py \
    --layers 0-31 \
    --matrices mlp \
    --paired \
    "$@"
