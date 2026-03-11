#!/bin/bash
# Full Layer Sweep - Paired MLP in+out, Llama-2-7b, MMLU 200q
source trawl-env/bin/activate

echo "=== Llama-2-7b-chat-hf — MMLU 200 questions ==="
python experiments/15_full_layer_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --layers 0-31 \
    --matrices mlp \
    --paired \
    --chunk-size 100 \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_200.json \
    --resume \
    "$@"
