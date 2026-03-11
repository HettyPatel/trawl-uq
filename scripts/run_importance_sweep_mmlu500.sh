#!/bin/bash
# Exp 20: Importance Sweep — Llama-2, MMLU-500, k=5 and k=10
source trawl-env/bin/activate

echo "=== Exp 20: Importance Sweep — Llama-2-7b-chat-hf — MMLU-500 — k=5 ==="
python experiments/20_importance_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --layers 0-31 \
    --matrices mlp \
    --paired \
    --chunk-size 100 \
    --flip-threshold 5 \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_500.json \
    --resume \
    "$@"

echo "=== Exp 20: Importance Sweep — Llama-2-7b-chat-hf — MMLU-500 — k=10 ==="
python experiments/20_importance_sweep.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --layers 0-31 \
    --matrices mlp \
    --paired \
    --chunk-size 100 \
    --flip-threshold 10 \
    --eval-set data/eval_sets/eval_set_mcq_mmlu_500.json \
    --resume \
    "$@"
