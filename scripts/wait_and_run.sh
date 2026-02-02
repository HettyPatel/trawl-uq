#!/bin/bash
# Wait for any GPU to have enough memory, then run experiments
# Usage: bash scripts/wait_and_run.sh [MIN_FREE_MB]

MIN_FREE_MB=${1:-20000}  # 20GB default

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)

echo "=============================================="
echo "Waiting for any GPU (0-$((NUM_GPUS-1))) to have ${MIN_FREE_MB}MB free..."
echo "Checking every 60 seconds. Press Ctrl+C to cancel."
echo "=============================================="

while true; do
    echo ""
    echo "[$(date '+%H:%M:%S')] Checking GPUs..."

    # Check each GPU
    for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID)
        echo "  GPU $GPU_ID: ${FREE_MB}MB free"

        if [ "$FREE_MB" -ge "$MIN_FREE_MB" ]; then
            echo ""
            echo "=============================================="
            echo "GPU $GPU_ID has ${FREE_MB}MB free! Starting experiments..."
            echo "=============================================="

            # Activate environment and run
            source /home/hpate061/trawl-uq/trawl-env/bin/activate
            CUDA_VISIBLE_DEVICES=$GPU_ID bash scripts/run_all_experiments.sh

            echo ""
            echo "=============================================="
            echo "All experiments complete!"
            echo "=============================================="
            exit 0
        fi
    done

    echo "  No GPU has ${MIN_FREE_MB}MB free. Waiting 60s..."
    sleep 60
done
