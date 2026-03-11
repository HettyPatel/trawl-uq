#!/bin/bash
# Run SVD noise removal with mlp_in + mlp_out on layers 0-4 (MCQ ARC Challenge)
# Usage: bash scripts/run_noise_removal_first5_layers.sh

source /home/hpate061/trawl-uq/trawl-env/bin/activate

EVAL_SET="data/eval_sets/eval_set_mcq_arc_challenge_200.json"
DATASET="mcq"

for LAYER in 0 1 2 3 4; do
    echo "=========================================="
    echo "Layer $LAYER, mlp_in + mlp_out"
    echo "=========================================="
    python experiments/14_svd_noise_removal.py \
        --layer $LAYER --matrix mlp_in mlp_out \
        --dataset $DATASET --eval-set $EVAL_SET
done

echo ""
echo "=========================================="
echo "All experiments complete! Generating plots..."
echo "=========================================="

for dir in results/svd_noise_removal/*mlp_in+mlp_out*mcq*; do
    if [ -f "$dir/results.pkl" ]; then
        echo "Plotting: $dir"
        python scripts/plot_svd_noise_removal.py --results "$dir/results.pkl"
    fi
done

echo "Done!"
