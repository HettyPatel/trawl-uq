"""
Split an eval set JSON into two halves: discovery and validation.

Usage:
    python scripts/split_eval_set.py \
        --input data/eval_sets/eval_set_mcq_mmlu_test_14042.json \
        --output-dir data/eval_sets \
        --seed 42
"""

import json
import argparse
import numpy as np
from pathlib import Path


def split_eval_set(input_path: str, output_dir: str, seed: int = 42):
    with open(input_path) as f:
        data = json.load(f)

    samples = data['samples']
    metadata = data.get('metadata', {})
    n = len(samples)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    mid = n // 2

    discovery = [samples[i] for i in indices[:mid]]
    validation = [samples[i] for i in indices[mid:]]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem  # e.g. eval_set_mcq_mmlu_test_14042
    disc_path = output_dir / f"{stem}_discovery.json"
    val_path  = output_dir / f"{stem}_validation.json"

    for path, split_samples, split_name in [
        (disc_path, discovery, 'discovery'),
        (val_path, validation, 'validation'),
    ]:
        with open(path, 'w') as f:
            json.dump({
                'metadata': {**metadata, 'split_half': split_name,
                             'n_samples': len(split_samples), 'split_seed': seed},
                'samples': split_samples,
            }, f, indent=2)
        print(f"Saved {len(split_samples)} samples → {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split eval set JSON into discovery/validation halves")
    parser.add_argument('--input', type=str, required=True, help='Input eval set JSON')
    parser.add_argument('--output-dir', type=str, default='data/eval_sets')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split_eval_set(args.input, args.output_dir, args.seed)
