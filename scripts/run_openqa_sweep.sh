#!/bin/bash
# Open QA Full Layer Sweep
# Uses NQ-Open 200 questions with keyword match accuracy + mean neg logprob
source trawl-env/bin/activate

python experiments/18_openqa_full_layer_sweep.py \
    --layers 0-31 \
    --paired \
    --chunk-size 100 \
    --max-new-tokens 10 \
    --eval-set data/eval_sets/eval_set_nq_open_200.json \
    --resume \
    "$@"
