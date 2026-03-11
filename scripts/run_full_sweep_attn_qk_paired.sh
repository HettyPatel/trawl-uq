#!/bin/bash
# Full Layer Sweep - Attention Q+K paired, Llama-2-7b, ARC 500q
source trawl-env/bin/activate

echo "=== Llama-2-7b-chat-hf — Attention Q+K Paired ==="
python experiments/15_full_layer_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --layers 0-31 \
    --matrices custom --matrix-list attn_q attn_k \
    --paired \
    --chunk-size 100 \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
    --resume \
    "$@"
