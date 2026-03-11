#!/bin/bash
# CP Component Sweep — all 32 layers, rank 40
# Uses experiment 17 (CP decomposition component removal)

source trawl-env/bin/activate

python experiments/17_cp_component_sweep.py \
    --layers 0-31 \
    --rank 40 \
    "$@"
