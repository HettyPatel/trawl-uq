"""
MCQ Random Matrix Replacement Control Experiment

Control experiment: replaces a weight matrix with a random matrix
matching the original's statistics (mean, std), then measures
entropy/accuracy changes. Demonstrates that the original matrix
structure matters and that effects seen in SVD truncation are not
simply due to weight perturbation.

Experiment flow:
1. Load MCQ evaluation set
2. Measure baseline MCQ entropy/accuracy on all samples
3. For each trial: replace matrix with random, evaluate, restore
4. Report aggregated results (mean/std across trials)
5. Save results (plotting done separately)
"""

import sys
sys.path.append('.')

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.evaluation.metrics import compute_mcq_entropy_and_nll
from src.decomposition.svd import (
    get_svd_stats,
    update_layer_with_svd,
    restore_original_weight
)


def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def evaluate_mcq_on_samples(model, tokenizer, samples, device="cuda"):
    """
    Evaluate MCQ entropy and NLL on all samples.

    Returns:
        List of dicts with metrics for each sample
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating MCQ", leave=False):
        mcq_prompt = sample['mcq_prompt']
        correct_letter = sample['correct_letter']

        metrics = compute_mcq_entropy_and_nll(
            mcq_prompt=mcq_prompt,
            correct_letter=correct_letter,
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        results.append({
            'sample_id': sample['id'],
            'question': sample['question'],
            'correct_answer': sample['correct_answer'],
            'correct_letter': correct_letter,
            **metrics
        })

    return results


def generate_random_matrix(original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Generate a random matrix matching the original's shape and statistics (mean, std).

    Args:
        original_weight: The original weight matrix to match
        seed: Random seed for reproducibility

    Returns:
        Random matrix on same device/dtype as original
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    mean = original_weight.float().mean().item()
    std = original_weight.float().std().item()

    random_weight = torch.normal(
        mean=mean, std=std, size=original_weight.shape, generator=generator
    )

    return random_weight.to(original_weight.device).to(original_weight.dtype)


def run_random_matrix_control(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_mcq_nq_open_200.json",
    target_layer: int = 6,
    matrix_type: str = "mlp_out",
    num_trials: int = 10,
    random_seed_start: int = 200,
    device: str = "cuda"
):
    """
    Run random matrix replacement control experiment.

    Replaces the target weight matrix with random matrices (matching original
    statistics) and measures entropy/accuracy changes across multiple trials.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama', 'gpt2', 'gptj')
        eval_set_path: Path to MCQ evaluation set JSON
        target_layer: Which layer to modify
        matrix_type: 'mlp_in' or 'mlp_out'
        num_trials: Number of random replacement trials
        random_seed_start: Starting seed for random matrix generation
        device: Device for computation
    """
    seed_everything(42)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    output_dir = Path(f"results/mcq_random_control/{model_short}_layer{target_layer}_{matrix_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MCQ RANDOM MATRIX REPLACEMENT CONTROL EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Matrix: {matrix_type}")
    print(f"Trials: {num_trials}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load eval set
    samples = load_mcq_eval_set(eval_set_path)
    print(f"Using {len(samples)} MCQ samples")

    # Save config
    config = {
        'model_name': model_name,
        'model_type': model_type,
        'eval_set_path': eval_set_path,
        'target_layer': target_layer,
        'matrix_type': matrix_type,
        'num_trials': num_trials,
        'random_seed_start': random_seed_start,
        'num_samples': len(samples),
        'experiment_type': 'random_matrix_control',
        'timestamp': timestamp
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # Get original weight
    print(f"\nGetting weight from layer {target_layer} {matrix_type}...")
    if model_type == "llama":
        if matrix_type == "mlp_in":
            original_weight = model.model.layers[target_layer].mlp.up_proj.weight.data.clone()
        else:
            original_weight = model.model.layers[target_layer].mlp.down_proj.weight.data.clone()
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            original_weight = model.transformer.h[target_layer].mlp.c_fc.weight.data.clone()
        else:
            original_weight = model.transformer.h[target_layer].mlp.c_proj.weight.data.clone()
    elif model_type == "gptj":
        if matrix_type == "mlp_in":
            original_weight = model.transformer.h[target_layer].mlp.fc_in.weight.data.clone()
        else:
            original_weight = model.transformer.h[target_layer].mlp.fc_out.weight.data.clone()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Get matrix stats
    svd_stats = get_svd_stats(original_weight, device)
    print(f"Matrix shape: {svd_stats['shape']}")

    random_matrix_stats = {
        'original_mean': original_weight.float().mean().item(),
        'original_std': original_weight.float().std().item(),
        'original_norm': torch.norm(original_weight.float()).item()
    }
    print(f"Original weight mean: {random_matrix_stats['original_mean']:.6f}")
    print(f"Original weight std: {random_matrix_stats['original_std']:.6f}")

    # ========== Phase 1: Baseline ==========
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE (Original Model)")
    print("=" * 70)

    baseline_results = evaluate_mcq_on_samples(model, tokenizer, samples, device)

    baseline_avg_entropy = np.mean([r['entropy'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['nll'] for r in baseline_results])
    baseline_accuracy = np.mean([r['is_correct'] for r in baseline_results])

    print(f"Baseline avg entropy: {baseline_avg_entropy:.4f} (max: 1.386)")
    print(f"Baseline avg NLL: {baseline_avg_nll:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")

    # ========== Phase 2: Random Replacement Trials ==========
    print("\n" + "=" * 70)
    print(f"PHASE 2: RANDOM MATRIX REPLACEMENT ({num_trials} trials)")
    print("=" * 70)

    trial_results = []

    for trial_idx in range(num_trials):
        trial_seed = random_seed_start + trial_idx
        print(f"\nTrial {trial_idx+1}/{num_trials} (seed={trial_seed})")

        # Generate random matrix
        random_weight = generate_random_matrix(original_weight, trial_seed)

        # Update model with random matrix
        update_layer_with_svd(model, target_layer, random_weight, matrix_type, model_type)

        # Evaluate
        trial_eval = evaluate_mcq_on_samples(model, tokenizer, samples, device)

        avg_entropy = np.mean([r['entropy'] for r in trial_eval])
        avg_nll = np.mean([r['nll'] for r in trial_eval])
        accuracy = np.mean([r['is_correct'] for r in trial_eval])

        trial_results.append({
            'trial_idx': trial_idx,
            'seed': trial_seed,
            'avg_entropy': avg_entropy,
            'avg_nll': avg_nll,
            'accuracy': accuracy,
            'entropy_change': avg_entropy - baseline_avg_entropy,
            'nll_change': avg_nll - baseline_avg_nll,
            'accuracy_change': accuracy - baseline_accuracy,
            'random_weight_norm': torch.norm(random_weight.float()).item(),
            'per_sample': trial_eval
        })

        print(f"  Entropy: {avg_entropy:.4f} ({avg_entropy - baseline_avg_entropy:+.4f})")
        print(f"  Accuracy: {accuracy*100:.1f}% ({(accuracy - baseline_accuracy)*100:+.1f}pp)")

        # Restore original weight
        restore_original_weight(model, target_layer, original_weight, matrix_type, model_type)

    # ========== Aggregate and Save ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    aggregated = {
        'avg_entropy_mean': np.mean([t['avg_entropy'] for t in trial_results]),
        'avg_entropy_std': np.std([t['avg_entropy'] for t in trial_results]),
        'avg_nll_mean': np.mean([t['avg_nll'] for t in trial_results]),
        'avg_nll_std': np.std([t['avg_nll'] for t in trial_results]),
        'accuracy_mean': np.mean([t['accuracy'] for t in trial_results]),
        'accuracy_std': np.std([t['accuracy'] for t in trial_results]),
        'entropy_change_mean': np.mean([t['entropy_change'] for t in trial_results]),
        'entropy_change_std': np.std([t['entropy_change'] for t in trial_results]),
        'accuracy_change_mean': np.mean([t['accuracy_change'] for t in trial_results]),
        'accuracy_change_std': np.std([t['accuracy_change'] for t in trial_results])
    }

    results = {
        'config': config,
        'svd_stats': svd_stats,
        'random_matrix_stats': random_matrix_stats,
        'baseline': {
            'avg_entropy': baseline_avg_entropy,
            'avg_nll': baseline_avg_nll,
            'accuracy': baseline_accuracy,
            'per_sample': baseline_results
        },
        'trial_results': trial_results,
        'aggregated': aggregated
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Baseline: {baseline_accuracy*100:.1f}% accuracy, {baseline_avg_entropy:.4f} entropy")
    print(f"Random replacement ({num_trials} trials):")
    print(f"  Accuracy: {aggregated['accuracy_mean']*100:.1f}% (±{aggregated['accuracy_std']*100:.1f}pp)")
    print(f"  Entropy: {aggregated['avg_entropy_mean']:.4f} (±{aggregated['avg_entropy_std']:.4f})")
    print(f"  Accuracy change: {aggregated['accuracy_change_mean']*100:+.1f}pp (±{aggregated['accuracy_change_std']*100:.1f}pp)")
    print(f"  Entropy change: {aggregated['entropy_change_mean']:+.4f} (±{aggregated['entropy_change_std']:.4f})")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCQ random matrix replacement control experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str, default="data/eval_sets/eval_set_mcq_nq_open_200.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--matrix", type=str, default="mlp_out", choices=["mlp_in", "mlp_out"])
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--test", action="store_true", help="Quick test with 2 trials")

    args = parser.parse_args()

    num_trials = 2 if args.test else args.num_trials

    run_random_matrix_control(
        model_name=args.model,
        model_type=args.model_type,
        eval_set_path=args.eval_set,
        target_layer=args.layer,
        matrix_type=args.matrix,
        num_trials=num_trials,
        random_seed_start=args.seed_start
    )
