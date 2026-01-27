"""
Create MCQ (Multiple Choice) evaluation set for cleaner entropy measurements.

Each question gets 4 options (A, B, C, D):
- 1 correct answer (from gold answers)
- 3 distractors (gold answers from other questions)
- Correct answer position is randomized

Usage:
    python scripts/create_mcq_eval_set.py --dataset nq_open --samples 200
"""

import sys
sys.path.append('.')

import json
import random
import argparse
from pathlib import Path
from src.generation.datasets import get_dataset


def create_mcq_eval_set(
    dataset_name: str,
    num_samples: int,
    output_dir: str = "data/eval_sets",
    seed: int = 42
):
    """
    Create MCQ evaluation set with randomized answer positions.

    Args:
        dataset_name: Name of dataset (nq_open, hotpotqa, coqa)
        num_samples: Number of samples to include
        output_dir: Directory to save the eval set
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name} validation set...")
    # Load extra samples to have enough distractors
    dataset = get_dataset(dataset_name, split="validation", num_samples=num_samples + 100)
    dataset.load(None)

    all_samples = dataset.data
    print(f"Loaded {len(all_samples)} samples")

    # Collect all unique answers for distractors
    all_answers = []
    for s in all_samples:
        if 'all_answers' in s:
            all_answers.extend(s['all_answers'])
        else:
            all_answers.append(s['answer'])

    # Remove duplicates while preserving some variety
    unique_answers = list(set(all_answers))
    print(f"Collected {len(unique_answers)} unique answers for distractors")

    # Create MCQ samples
    mcq_samples = []
    letters = ['A', 'B', 'C', 'D']

    for i, sample in enumerate(all_samples[:num_samples]):
        question = sample['question']
        correct_answer = sample['answer']

        # Get 3 distractors (different from correct answer)
        distractors = []
        attempts = 0
        while len(distractors) < 3 and attempts < 100:
            distractor = random.choice(unique_answers)
            # Avoid same answer or very similar
            if (distractor != correct_answer and
                distractor.lower() != correct_answer.lower() and
                distractor not in distractors):
                distractors.append(distractor)
            attempts += 1

        # If not enough distractors, pad with generic wrong answers
        while len(distractors) < 3:
            distractors.append(f"Unknown answer {len(distractors) + 1}")

        # Combine correct + distractors and shuffle
        all_options = [correct_answer] + distractors
        random.shuffle(all_options)

        # Find correct answer position
        correct_idx = all_options.index(correct_answer)
        correct_letter = letters[correct_idx]

        # Build MCQ prompt
        mcq_prompt = f"Question: {question}\n"
        for j, opt in enumerate(all_options):
            mcq_prompt += f"{letters[j]}. {opt}\n"
        mcq_prompt += "Answer:"

        mcq_samples.append({
            'id': f"{dataset_name}_mcq_{i}",
            'question': question,
            'mcq_prompt': mcq_prompt,
            'options': {letters[j]: opt for j, opt in enumerate(all_options)},
            'correct_answer': correct_answer,
            'correct_letter': correct_letter,
            'distractors': distractors
        })

    # Verify distribution of correct positions
    position_counts = {l: 0 for l in letters}
    for s in mcq_samples:
        position_counts[s['correct_letter']] += 1

    print(f"\nCorrect answer position distribution:")
    for l, count in position_counts.items():
        print(f"  {l}: {count} ({100*count/len(mcq_samples):.1f}%)")

    # Show examples
    print(f"\nExample MCQ samples:")
    for i, s in enumerate(mcq_samples[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(s['mcq_prompt'])
        print(f"Correct: {s['correct_letter']}")

    # Save to JSON
    output_file = output_dir / f"eval_set_mcq_{dataset_name}_{num_samples}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": dataset_name,
                "format": "mcq",
                "split": "validation",
                "num_samples": num_samples,
                "num_options": 4,
                "seed": seed,
                "description": f"MCQ evaluation set for {dataset_name}"
            },
            "samples": mcq_samples
        }, f, indent=2)

    print(f"\nSaved MCQ evaluation set to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def load_mcq_eval_set(filepath: str):
    """
    Load a saved MCQ evaluation set.

    Args:
        filepath: Path to the eval set JSON file

    Returns:
        List of MCQ sample dictionaries
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    print(f"Dataset: {data['metadata']['dataset']}")

    return data['samples']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MCQ evaluation set")
    parser.add_argument("--dataset", type=str, default="nq_open",
                        choices=["nq_open", "hotpotqa", "coqa"],
                        help="Dataset to use (default: nq_open)")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of samples (default: 200)")
    parser.add_argument("--output-dir", type=str, default="data/eval_sets",
                        help="Output directory (default: data/eval_sets)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    create_mcq_eval_set(
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output_dir,
        seed=args.seed
    )
