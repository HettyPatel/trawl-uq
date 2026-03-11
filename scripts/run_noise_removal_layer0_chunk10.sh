#!/bin/bash
# Run SVD noise removal on layer 0 with mlp_in + mlp_out, chunk size 10 (MCQ ARC Challenge)
# Usage: bash scripts/run_noise_removal_layer0_chunk10.sh

source /home/hpate061/trawl-uq/trawl-env/bin/activate

EVAL_SET="data/eval_sets/eval_set_mcq_arc_challenge_200.json"
DATASET="mcq"

echo "=========================================="
echo "Layer 0, mlp_in + mlp_out, chunk_size=10"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 0 --matrix mlp_in mlp_out \
    --chunk-size 10 \
    --dataset $DATASET --eval-set $EVAL_SET

echo ""
echo "=========================================="
echo "Generating plots..."
echo "=========================================="

for dir in results/svd_noise_removal/*layer0*chunk10*mcq*; do
    if [ -f "$dir/results.pkl" ]; then
        echo "Plotting: $dir"
        python scripts/plot_svd_noise_removal.py --results "$dir/results.pkl"
    fi
done

echo "Done!"
