"""
Create MCQ evaluation set from the MMLU dataset.

MMLU (Massive Multitask Language Understanding) covers 57 subjects across
STEM, humanities, social sciences, etc. Each question has 4 choices (A/B/C/D).

Usage:
    python scripts/create_mmlu_eval_set.py
    python scripts/create_mmlu_eval_set.py --samples 200 --seed 42
    python scripts/create_mmlu_eval_set.py --samples 500 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def create_mmlu_eval_set(
    num_samples: int = 200,
    output_dir: str = "data/eval_sets",
    seed: int = 42,
    split: str = "validation",
):
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMLU ({split}) from HuggingFace...")
    dataset = load_dataset("cais/mmlu", "all", split=split)
    print(f"Loaded {len(dataset)} samples across all subjects")

    letters = ['A', 'B', 'C', 'D']

    # All MMLU questions have exactly 4 choices
    valid_samples = list(dataset)
    print(f"Valid 4-choice samples: {len(valid_samples)}")

    if len(valid_samples) < num_samples:
        print(f"WARNING: Only {len(valid_samples)} samples available, using all.")
        num_samples = len(valid_samples)

    random.shuffle(valid_samples)
    selected = valid_samples[:num_samples]

    mcq_samples = []
    for i, sample in enumerate(selected):
        question = sample['question']
        choices = sample['choices']           # list of 4 strings
        answer_idx = sample['answer']         # int 0-3
        answer_key = letters[answer_idx]

        options = {letters[j]: text for j, text in enumerate(choices)}

        mcq_prompt = f"Question: {question}\n"
        for j, text in enumerate(choices):
            mcq_prompt += f"{letters[j]}. {text}\n"
        mcq_prompt += "Answer:"

        correct_answer = choices[answer_idx]
        distractors = [text for j, text in enumerate(choices) if j != answer_idx]

        mcq_samples.append({
            'id': f"mmlu_mcq_{i}",
            'question': question,
            'subject': sample.get('subject', 'unknown'),
            'mcq_prompt': mcq_prompt,
            'options': options,
            'correct_answer': correct_answer,
            'correct_letter': answer_key,
            'distractors': distractors,
        })

    # Distribution checks
    position_counts = {l: 0 for l in letters}
    subject_counts = {}
    for s in mcq_samples:
        position_counts[s['correct_letter']] += 1
        subject_counts[s['subject']] = subject_counts.get(s['subject'], 0) + 1

    print(f"\nCorrect answer position distribution:")
    for l, count in position_counts.items():
        print(f"  {l}: {count} ({100*count/len(mcq_samples):.1f}%)")

    print(f"\nTop 10 subjects represented:")
    for subj, count in sorted(subject_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {subj}: {count}")

    print(f"\nExample MCQ samples:")
    for i, s in enumerate(mcq_samples[:2]):
        print(f"\n--- Sample {i+1} ({s['subject']}) ---")
        print(s['mcq_prompt'])
        print(f"Correct: {s['correct_letter']} ({s['correct_answer']})")

    split_tag = split if split != "validation" else ""
    split_prefix = f"_{split_tag}" if split_tag else ""
    output_file = output_dir / f"eval_set_mcq_mmlu{split_prefix}_{num_samples}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": "mmlu",
                "format": "mcq",
                "split": split,
                "num_samples": num_samples,
                "num_options": 4,
                "seed": seed,
                "description": "MCQ evaluation set from MMLU (multi-subject academic)"
            },
            "samples": mcq_samples
        }, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MMLU MCQ evaluation set")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="data/eval_sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "test", "dev"])
    args = parser.parse_args()
    create_mmlu_eval_set(
        num_samples=args.samples,
        output_dir=args.output_dir,
        seed=args.seed,
        split=args.split,
    )
