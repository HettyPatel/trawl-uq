#!/bin/bash
# Run all SVD noise removal experiments with open-ended QA generation
# Same 6 configurations as MCQ: mlp_in, mlp_out, mlp_in+mlp_out for layers 0 and 31
# Usage: bash scripts/run_noise_removal_open_qa.sh

source /home/hpate061/trawl-uq/trawl-env/bin/activate

DATASET="open_qa"

echo "=========================================="
echo "Experiment 1/6: Layer 31, mlp_out"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 31 --matrix mlp_out \
    --dataset $DATASET

echo "=========================================="
echo "Experiment 2/6: Layer 0, mlp_out"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 0 --matrix mlp_out \
    --dataset $DATASET

echo "=========================================="
echo "Experiment 3/6: Layer 31, mlp_in"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 31 --matrix mlp_in \
    --dataset $DATASET

echo "=========================================="
echo "Experiment 4/6: Layer 0, mlp_in"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 0 --matrix mlp_in \
    --dataset $DATASET

echo "=========================================="
echo "Experiment 5/6: Layer 31, mlp_in + mlp_out"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 31 --matrix mlp_in mlp_out \
    --dataset $DATASET

echo "=========================================="
echo "Experiment 6/6: Layer 0, mlp_in + mlp_out"
echo "=========================================="
python experiments/14_svd_noise_removal.py \
    --layer 0 --matrix mlp_in mlp_out \
    --dataset $DATASET

echo ""
echo "=========================================="
echo "All experiments complete! Generating plots..."
echo "=========================================="

for dir in results/svd_noise_removal/*open_qa*; do
    if [ -f "$dir/results.pkl" ]; then
        echo "Plotting: $dir"
        python scripts/plot_svd_noise_removal.py --results "$dir/results.pkl"
    fi
done

echo "Done!"
