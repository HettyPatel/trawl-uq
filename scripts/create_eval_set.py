"""
Create and save a fixed evaluation set for reproducible experiments.

This script creates a JSON file with a fixed subset of samples that can be
loaded by all experiments to ensure consistent evaluation.

Usage:
    python scripts/create_eval_set.py --dataset nq_open --samples 200
    python scripts/create_eval_set.py --dataset hotpotqa --samples 200
"""

import sys
sys.path.append('.')

import json
import argparse
from pathlib import Path
from src.generation.datasets import get_dataset


def create_eval_set(
    dataset_name: str,
    num_samples: int,
    output_dir: str = "data/eval_sets"
):
    """
    Create and save a fixed evaluation set.

    Args:
        dataset_name: Name of dataset (nq_open, hotpotqa, coqa)
        num_samples: Number of samples to include
        output_dir: Directory to save the eval set
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name} validation set...")
    dataset = get_dataset(dataset_name, split="validation", num_samples=num_samples)
    dataset.load(None)

    samples = dataset.data
    print(f"Loaded {len(samples)} samples")

    # Show sample statistics
    answer_lengths = [len(s['answer'].split()) for s in samples]
    print(f"\nAnswer length statistics:")
    print(f"  Min: {min(answer_lengths)} words")
    print(f"  Max: {max(answer_lengths)} words")
    print(f"  Mean: {sum(answer_lengths)/len(answer_lengths):.1f} words")

    # Show a few examples
    print(f"\nExample samples:")
    for i, s in enumerate(samples[:3]):
        print(f"  {i+1}. Q: {s['question'][:60]}...")
        print(f"     A: {s['answer']}")

    # Save to JSON
    output_file = output_dir / f"eval_set_{dataset_name}_{num_samples}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": dataset_name,
                "split": "validation",
                "num_samples": num_samples,
                "description": f"Fixed evaluation set for {dataset_name}"
            },
            "samples": samples
        }, f, indent=2)

    print(f"\nSaved evaluation set to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def load_eval_set(filepath: str):
    """
    Load a saved evaluation set.

    Args:
        filepath: Path to the eval set JSON file

    Returns:
        List of sample dictionaries
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    print(f"Dataset: {data['metadata']['dataset']}")

    return data['samples']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fixed evaluation set")
    parser.add_argument("--dataset", type=str, default="nq_open",
                        choices=["nq_open", "hotpotqa", "coqa"],
                        help="Dataset to use (default: nq_open)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of samples (default: 200)")
    parser.add_argument("--output-dir", type=str, default="data/eval_sets",
                        help="Output directory (default: data/eval_sets)")

    args = parser.parse_args()

    create_eval_set(
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output_dir
    )
