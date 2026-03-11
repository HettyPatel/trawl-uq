"""
Create a BoolQ MCQ evaluation set in the same format as ARC Challenge.

BoolQ has yes/no questions with a passage. We format them as 2-option MCQ:
  Question: [passage context] [question]
  A. Yes
  B. No
  Answer:

The existing compute_mcq_entropy_and_nll function defaults to ['A','B','C','D']
but C and D will get near-zero probability since they're not in the prompt.

Usage:
    python scripts/create_boolq_eval_set.py
    python scripts/create_boolq_eval_set.py --num-samples 500
    python scripts/create_boolq_eval_set.py --split validation --num-samples 300
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def create_boolq_eval_set(
    split: str = "validation",
    num_samples: int = 500,
    seed: int = 42,
    output_dir: str = "data/eval_sets",
):
    random.seed(seed)

    print(f"Loading BoolQ dataset (split={split})...")
    ds = load_dataset("google/boolq", split=split)
    print(f"Loaded {len(ds)} samples")

    # Sample if needed
    if num_samples < len(ds):
        indices = random.sample(range(len(ds)), num_samples)
        indices.sort()
    else:
        indices = list(range(len(ds)))
        num_samples = len(ds)

    samples = []
    for i, idx in enumerate(indices):
        item = ds[idx]
        question = item['question']
        passage = item['passage']
        answer = item['answer']  # True/False

        correct_letter = 'A' if answer else 'B'
        correct_answer = 'Yes' if answer else 'No'

        # Format MCQ prompt with passage context
        mcq_prompt = (
            f"Passage: {passage}\n\n"
            f"Question: {question}\n"
            f"A. Yes\n"
            f"B. No\n"
            f"Answer:"
        )

        samples.append({
            'id': f'boolq_mcq_{i}',
            'question': question,
            'passage': passage,
            'mcq_prompt': mcq_prompt,
            'options': {
                'A': 'Yes',
                'B': 'No',
            },
            'correct_answer': correct_answer,
            'correct_letter': correct_letter,
            'distractors': ['No'] if answer else ['Yes'],
        })

    # Save
    output_path = Path(output_dir) / f"eval_set_mcq_boolq_{num_samples}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'metadata': {
            'dataset': 'boolq',
            'format': 'mcq',
            'split': split,
            'num_samples': num_samples,
            'num_options': 2,
            'seed': seed,
            'description': 'MCQ evaluation set from BoolQ (yes/no reading comprehension)',
        },
        'samples': samples,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Stats
    num_yes = sum(1 for s in samples if s['correct_letter'] == 'A')
    num_no = num_samples - num_yes
    print(f"\nSaved {num_samples} samples to {output_path}")
    print(f"  Yes (A): {num_yes} ({num_yes/num_samples*100:.1f}%)")
    print(f"  No (B): {num_no} ({num_no/num_samples*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BoolQ MCQ eval set")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_boolq_eval_set(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )
