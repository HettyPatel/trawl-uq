"""
Create MCQ evaluation set from the ARC Challenge dataset.

ARC Challenge already comes in MCQ format with options and answer keys,
so we just need to download, filter to 4-option questions, and convert
to our standard eval set JSON format.

Source: https://huggingface.co/datasets/allenai/ai2_arc (ARC-Challenge split)

Usage:
    python scripts/create_arc_eval_set.py
    python scripts/create_arc_eval_set.py --samples 200 --seed 42
    python scripts/create_arc_eval_set.py --samples 500 --split test --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def create_arc_eval_set(
    num_samples: int = 200,
    output_dir: str = "data/eval_sets",
    seed: int = 42,
    split: str = "validation"
):
    """
    Create MCQ evaluation set from ARC Challenge dataset.

    Args:
        num_samples: Number of samples to include
        output_dir: Directory to save the eval set
        seed: Random seed for reproducibility
        split: Dataset split ('validation' or 'test')
    """
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ARC Challenge {split} set from HuggingFace...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    print(f"Loaded {len(dataset)} samples")

    # Filter to 4-option questions with A/B/C/D labels
    letters = ['A', 'B', 'C', 'D']
    valid_samples = []
    for sample in dataset:
        labels = sample['choices']['label']
        if labels == letters:
            valid_samples.append(sample)

    print(f"Samples with exactly 4 options (A/B/C/D): {len(valid_samples)} / {len(dataset)}")

    if len(valid_samples) < num_samples:
        print(f"WARNING: Only {len(valid_samples)} valid samples available, "
              f"using all of them instead of {num_samples}")
        num_samples = len(valid_samples)

    # Shuffle and select
    random.shuffle(valid_samples)
    selected = valid_samples[:num_samples]

    # Convert to our MCQ eval set format
    mcq_samples = []
    for i, sample in enumerate(selected):
        question = sample['question']
        options_text = sample['choices']['text']
        answer_key = sample['answerKey']

        # Build options dict
        options = {letters[j]: text for j, text in enumerate(options_text)}

        # Build MCQ prompt (same format as existing eval sets)
        mcq_prompt = f"Question: {question}\n"
        for j, text in enumerate(options_text):
            mcq_prompt += f"{letters[j]}. {text}\n"
        mcq_prompt += "Answer:"

        correct_answer = options[answer_key]
        distractors = [text for l, text in options.items() if l != answer_key]

        mcq_samples.append({
            'id': f"arc_challenge_mcq_{i}",
            'question': question,
            'mcq_prompt': mcq_prompt,
            'options': options,
            'correct_answer': correct_answer,
            'correct_letter': answer_key,
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
        print(f"Correct: {s['correct_letter']} ({s['correct_answer']})")

    # Save
    output_file = output_dir / f"eval_set_mcq_arc_challenge_{num_samples}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": "arc_challenge",
                "format": "mcq",
                "split": split,
                "num_samples": num_samples,
                "num_options": 4,
                "seed": seed,
                "description": "MCQ evaluation set from ARC Challenge (science reasoning)"
            },
            "samples": mcq_samples
        }, f, indent=2)

    print(f"\nSaved MCQ evaluation set to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ARC Challenge MCQ evaluation set")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of samples (default: 200)")
    parser.add_argument("--output-dir", type=str, default="data/eval_sets",
                        help="Output directory (default: data/eval_sets)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test", "train"],
                        help="Dataset split (default: validation). 'test'=1144, 'train'=1094.")

    args = parser.parse_args()
    create_arc_eval_set(
        num_samples=args.samples,
        output_dir=args.output_dir,
        seed=args.seed,
        split=args.split
    )
