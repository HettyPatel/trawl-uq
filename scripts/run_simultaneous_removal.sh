#!/bin/bash
# Experiment 19: Simultaneous All-Layer Noise Removal
# Removes all noise chunks from all layers at once and evaluates
source trawl-env/bin/activate

echo "=== Llama-2-7b-chat-hf ==="
python experiments/19_simultaneous_noise_removal.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --sweep-dir results/full_sweep/Llama-2-7b-chat-hf_paired_mlp_in+mlp_out_chunk100 \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
    "$@"

echo ""
echo "=== Meta-Llama-3-8B-Instruct ==="
python experiments/19_simultaneous_noise_removal.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --sweep-dir results/full_sweep/Meta-Llama-3-8B-Instruct_paired_mlp_in+mlp_out_chunk100_eval_set_mcq_arc_challenge_500 \
    --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
    "$@"
