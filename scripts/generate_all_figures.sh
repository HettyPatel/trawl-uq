#!/bin/bash
# Generate figures for MCQ entropy experiment results
# Finds all results.pkl files and generates plots

set -e

# Activate environment
source /home/hpate061/trawl-uq/trawl-env/bin/activate

echo "=============================================="
echo "GENERATING FIGURES FOR MCQ EXPERIMENTS"
echo "=============================================="
echo ""

# MCQ CP Decomposition results
echo "--- MCQ CP Decomposition Results ---"
for results_file in results/mcq_entropy_noise_removal/*/results.pkl; do
    if [ -f "$results_file" ]; then
        echo "Processing: $results_file"
        python scripts/plot_mcq_results.py --results "$results_file"
        echo ""
    fi
done

# MCQ SVD Truncation results
echo "--- MCQ SVD Truncation Results ---"
for results_file in results/mcq_entropy_svd/*/results.pkl; do
    if [ -f "$results_file" ]; then
        echo "Processing: $results_file"
        python scripts/plot_mcq_results.py --results "$results_file"
        echo ""
    fi
done

echo "=============================================="
echo "ALL FIGURES GENERATED"
echo "=============================================="

# List all figure directories
echo ""
echo "Figure directories:"
find results -name "figures" -type d 2>/dev/null | sort
