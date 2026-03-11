"""
MCQ Combined SVD Truncation Experiment

Truncates both mlp_in and mlp_out weight matrices simultaneously at the same
reduction level, then evaluates MCQ entropy/accuracy. This tests whether the
effects of SVD truncation compound, cancel, or remain independent when applied
to both MLP matrices in a single layer at once.

Experiment flow:
1. Load MCQ evaluation set
2. SVD decompose both mlp_in and mlp_out weight matrices
3. Measure baseline MCQ entropy/accuracy
4. For each reduction %: truncate both matrices, apply both, evaluate, restore both
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
    decompose_weight_svd,
    select_svd_components,
    reconstruct_from_svd,
    get_svd_stats,
    update_layer_with_svd,
    restore_original_weight,
    compute_energy_retention,
    LASER_REDUCTION_PERCENTAGES,
    reduction_to_keep_ratio
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


def get_original_weight(model, layer_idx, matrix_type, model_type):
    """Get a clone of the original weight matrix from the model."""
    if model_type == "llama":
        if matrix_type == "mlp_in":
            return model.model.layers[layer_idx].mlp.up_proj.weight.data.clone()
        else:
            return model.model.layers[layer_idx].mlp.down_proj.weight.data.clone()
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            return model.transformer.h[layer_idx].mlp.c_fc.weight.data.clone()
        else:
            return model.transformer.h[layer_idx].mlp.c_proj.weight.data.clone()
    elif model_type == "gptj":
        if matrix_type == "mlp_in":
            return model.transformer.h[layer_idx].mlp.fc_in.weight.data.clone()
        else:
            return model.transformer.h[layer_idx].mlp.fc_out.weight.data.clone()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_mcq_combined_svd_truncation(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_mcq_nq_open_200.json",
    target_layer: int = 6,
    reduction_percentages: list = None,
    component_counts: list = None,
    mode: str = "top",
    num_trials: int = 5,
    random_seed_start: int = 100,
    device: str = "cuda",
    checkpoint_every: int = 5
):
    """
    Run combined mlp_in + mlp_out SVD truncation experiment.

    Truncates both MLP matrices at the same reduction level simultaneously,
    then evaluates MCQ entropy/accuracy once with both modifications active.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama', 'gpt2', 'gptj')
        eval_set_path: Path to MCQ evaluation set JSON
        target_layer: Which layer to apply SVD
        reduction_percentages: List of reduction percentages to test
        component_counts: Exact component counts to test (overrides reduction_percentages)
        mode: Component selection mode ('top', 'bottom', 'random')
        num_trials: Number of trials for random mode
        random_seed_start: Starting seed for random trials
        device: Device for computation
        checkpoint_every: Save checkpoint every N reduction levels
    """
    seed_everything(42)

    if reduction_percentages is None:
        reduction_percentages = LASER_REDUCTION_PERCENTAGES

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    mode_suffix = f"_{mode}" if mode != "top" else ""
    output_dir = Path(f"results/mcq_combined_svd/{model_short}_layer{target_layer}_combined{mode_suffix}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MCQ COMBINED SVD TRUNCATION EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Matrices: mlp_in + mlp_out (combined)")
    print(f"Mode: {mode}")
    print(f"Reduction levels: {len(reduction_percentages)}")
    if mode == "random":
        print(f"Random trials: {num_trials}")
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
        'matrix_type': 'combined',
        'reduction_percentages': reduction_percentages,
        'num_samples': len(samples),
        'timestamp': timestamp,
        'experiment_type': 'combined_svd_truncation',
        'mode': mode,
        'num_trials': num_trials if mode == 'random' else 1,
        'random_seed_start': random_seed_start if mode == 'random' else None
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # Get original weights for both matrices
    print(f"\nGetting weights from layer {target_layer}...")
    original_weight_in = get_original_weight(model, target_layer, "mlp_in", model_type)
    original_weight_out = get_original_weight(model, target_layer, "mlp_out", model_type)
    print(f"  mlp_in shape: {original_weight_in.shape}")
    print(f"  mlp_out shape: {original_weight_out.shape}")

    # Get SVD stats for both
    svd_stats_in = get_svd_stats(original_weight_in, device)
    svd_stats_out = get_svd_stats(original_weight_out, device)
    print(f"  mlp_in max rank: {svd_stats_in['max_rank']}, effective rank: {svd_stats_in['effective_rank']:.1f}")
    print(f"  mlp_out max rank: {svd_stats_out['max_rank']}, effective rank: {svd_stats_out['effective_rank']:.1f}")

    # Decompose both once
    print("\nPerforming SVD decomposition on both matrices...")
    U_in, S_in, Vh_in = decompose_weight_svd(original_weight_in, device)
    U_out, S_out, Vh_out = decompose_weight_svd(original_weight_out, device)
    print(f"  mlp_in: {len(S_in)} singular values")
    print(f"  mlp_out: {len(S_out)} singular values")

    # If component_counts specified, convert to reduction percentages
    # Use the smaller total_components for the conversion (more conservative)
    if component_counts is not None:
        total_in = len(S_in)
        total_out = len(S_out)
        min_total = min(total_in, total_out)
        reduction_percentages = []
        for k in component_counts:
            if k > min_total:
                print(f"  Warning: requested {k} components but min matrix has {min_total}, skipping")
                continue
            pct = (1.0 - k / min_total) * 100.0
            reduction_percentages.append(pct)
            k_in = max(1, int(total_in * (k / min_total)))
            k_out = max(1, int(total_out * (k / min_total)))
            print(f"  {k} components → {pct:.4f}% reduction (mlp_in: {k_in}, mlp_out: {k_out})")
        config['component_counts'] = component_counts
        config['reduction_percentages'] = reduction_percentages

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

    # ========== Phase 2: Combined Truncation Sweep ==========
    print("\n" + "=" * 70)
    print(f"PHASE 2: COMBINED TRUNCATION SWEEP ({len(reduction_percentages)} levels, mode={mode})")
    print("=" * 70)

    truncation_results = []

    for i, reduction_pct in enumerate(tqdm(reduction_percentages, desc="Testing reductions")):
        keep_ratio = reduction_to_keep_ratio(reduction_pct)
        components_kept_in = max(1, int(len(S_in) * keep_ratio))
        components_kept_out = max(1, int(len(S_out) * keep_ratio))

        print(f"\nReduction {reduction_pct}% (mlp_in: keep {components_kept_in}/{len(S_in)}, "
              f"mlp_out: keep {components_kept_out}/{len(S_out)}, mode={mode})")

        if mode in ("top", "bottom"):
            # Single pass for top/bottom
            U_sel_in, S_sel_in, Vh_sel_in, idx_in = select_svd_components(
                U_in, S_in, Vh_in, keep_ratio, mode=mode
            )
            U_sel_out, S_sel_out, Vh_sel_out, idx_out = select_svd_components(
                U_out, S_out, Vh_out, keep_ratio, mode=mode
            )

            weight_lr_in = reconstruct_from_svd(U_sel_in, S_sel_in, Vh_sel_in)
            weight_lr_out = reconstruct_from_svd(U_sel_out, S_sel_out, Vh_sel_out)

            energy_in = compute_energy_retention(S_in, S_sel_in)
            energy_out = compute_energy_retention(S_out, S_sel_out)

            # Apply both truncated matrices simultaneously
            update_layer_with_svd(model, target_layer, weight_lr_in, "mlp_in", model_type)
            update_layer_with_svd(model, target_layer, weight_lr_out, "mlp_out", model_type)

            # Evaluate once with both modifications active
            trunc_eval = evaluate_mcq_on_samples(model, tokenizer, samples, device)

            avg_entropy = np.mean([r['entropy'] for r in trunc_eval])
            avg_nll = np.mean([r['nll'] for r in trunc_eval])
            accuracy = np.mean([r['is_correct'] for r in trunc_eval])
            entropy_change = avg_entropy - baseline_avg_entropy
            nll_change = avg_nll - baseline_avg_nll
            accuracy_change = accuracy - baseline_accuracy

            truncation_results.append({
                'reduction_percent': reduction_pct,
                'keep_ratio': keep_ratio,
                'components_kept_in': components_kept_in,
                'components_kept_out': components_kept_out,
                'total_components_in': len(S_in),
                'total_components_out': len(S_out),
                'energy_retention_in': energy_in,
                'energy_retention_out': energy_out,
                'energy_retention': (energy_in + energy_out) / 2,
                'avg_entropy': avg_entropy,
                'avg_nll': avg_nll,
                'accuracy': accuracy,
                'entropy_change': entropy_change,
                'nll_change': nll_change,
                'accuracy_change': accuracy_change,
                'per_sample': trunc_eval,
                'mode': mode,
                'selected_indices_in': idx_in.tolist(),
                'selected_indices_out': idx_out.tolist()
            })

            print(f"  Entropy: {avg_entropy:.4f} ({entropy_change:+.4f})")
            print(f"  Accuracy: {accuracy*100:.1f}% ({accuracy_change*100:+.1f}pp)")
            print(f"  Energy retention: mlp_in={energy_in*100:.2f}%, mlp_out={energy_out*100:.2f}%")

            # Restore both original weights
            restore_original_weight(model, target_layer, original_weight_in, "mlp_in", model_type)
            restore_original_weight(model, target_layer, original_weight_out, "mlp_out", model_type)

        elif mode == "random":
            # Multiple trials for random mode
            trial_results = []

            for trial_idx in range(num_trials):
                trial_seed = random_seed_start + trial_idx

                U_sel_in, S_sel_in, Vh_sel_in, idx_in = select_svd_components(
                    U_in, S_in, Vh_in, keep_ratio, mode="random", seed=trial_seed
                )
                U_sel_out, S_sel_out, Vh_sel_out, idx_out = select_svd_components(
                    U_out, S_out, Vh_out, keep_ratio, mode="random", seed=trial_seed + 1000
                )

                weight_lr_in = reconstruct_from_svd(U_sel_in, S_sel_in, Vh_sel_in)
                weight_lr_out = reconstruct_from_svd(U_sel_out, S_sel_out, Vh_sel_out)

                energy_in = compute_energy_retention(S_in, S_sel_in)
                energy_out = compute_energy_retention(S_out, S_sel_out)

                # Apply both
                update_layer_with_svd(model, target_layer, weight_lr_in, "mlp_in", model_type)
                update_layer_with_svd(model, target_layer, weight_lr_out, "mlp_out", model_type)

                trunc_eval = evaluate_mcq_on_samples(model, tokenizer, samples, device)

                avg_entropy = np.mean([r['entropy'] for r in trunc_eval])
                avg_nll = np.mean([r['nll'] for r in trunc_eval])
                accuracy = np.mean([r['is_correct'] for r in trunc_eval])

                trial_results.append({
                    'trial_idx': trial_idx,
                    'seed': trial_seed,
                    'energy_retention_in': energy_in,
                    'energy_retention_out': energy_out,
                    'energy_retention': (energy_in + energy_out) / 2,
                    'avg_entropy': avg_entropy,
                    'avg_nll': avg_nll,
                    'accuracy': accuracy,
                    'entropy_change': avg_entropy - baseline_avg_entropy,
                    'nll_change': avg_nll - baseline_avg_nll,
                    'accuracy_change': accuracy - baseline_accuracy,
                    'selected_indices_in': idx_in.tolist(),
                    'selected_indices_out': idx_out.tolist(),
                    'per_sample': trunc_eval
                })

                print(f"  Trial {trial_idx+1}/{num_trials}: acc={accuracy*100:.1f}% entropy={avg_entropy:.4f}")

                # Restore both
                restore_original_weight(model, target_layer, original_weight_in, "mlp_in", model_type)
                restore_original_weight(model, target_layer, original_weight_out, "mlp_out", model_type)

            # Aggregate across trials
            agg_entropy = np.mean([t['avg_entropy'] for t in trial_results])
            agg_nll = np.mean([t['avg_nll'] for t in trial_results])
            agg_accuracy = np.mean([t['accuracy'] for t in trial_results])
            agg_energy = np.mean([t['energy_retention'] for t in trial_results])

            truncation_results.append({
                'reduction_percent': reduction_pct,
                'keep_ratio': keep_ratio,
                'components_kept_in': components_kept_in,
                'components_kept_out': components_kept_out,
                'total_components_in': len(S_in),
                'total_components_out': len(S_out),
                'mode': 'random',
                'num_trials': num_trials,
                'energy_retention': agg_energy,
                'avg_entropy': agg_entropy,
                'avg_nll': agg_nll,
                'accuracy': agg_accuracy,
                'entropy_change': agg_entropy - baseline_avg_entropy,
                'nll_change': agg_nll - baseline_avg_nll,
                'accuracy_change': agg_accuracy - baseline_accuracy,
                'entropy_std': np.std([t['avg_entropy'] for t in trial_results]),
                'nll_std': np.std([t['avg_nll'] for t in trial_results]),
                'accuracy_std': np.std([t['accuracy'] for t in trial_results]),
                'energy_std': np.std([t['energy_retention'] for t in trial_results]),
                'trial_results': trial_results,
                'per_sample': None
            })

            print(f"  Aggregated: acc={agg_accuracy*100:.1f}% "
                  f"(±{np.std([t['accuracy'] for t in trial_results])*100:.1f}pp) "
                  f"entropy={agg_entropy:.4f}")

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': config,
                'svd_stats_in': svd_stats_in,
                'svd_stats_out': svd_stats_out,
                'baseline': {
                    'avg_entropy': baseline_avg_entropy,
                    'avg_nll': baseline_avg_nll,
                    'accuracy': baseline_accuracy,
                    'per_sample': baseline_results
                },
                'truncation_results': truncation_results,
                'completed_levels': i + 1
            }
            with open(output_dir / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

    # ========== Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'config': config,
        'svd_stats_in': svd_stats_in,
        'svd_stats_out': svd_stats_out,
        'baseline': {
            'avg_entropy': baseline_avg_entropy,
            'avg_nll': baseline_avg_nll,
            'accuracy': baseline_accuracy,
            'per_sample': baseline_results
        },
        'truncation_results': truncation_results
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Baseline entropy: {baseline_avg_entropy:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")

    # Find sweet spot (best accuracy with lowest components)
    best_acc = max(r['accuracy'] for r in truncation_results)
    best_results = [r for r in truncation_results if r['accuracy'] >= best_acc * 0.95]
    if best_results:
        most_compressed = max(best_results, key=lambda x: x['reduction_percent'])
        print(f"\nBest tradeoff: {most_compressed['reduction_percent']}% reduction")
        print(f"  Keeps {most_compressed['components_kept_in']} mlp_in + "
              f"{most_compressed['components_kept_out']} mlp_out components")
        print(f"  Accuracy: {most_compressed['accuracy']*100:.1f}%")
        print(f"  Energy retention: {most_compressed['energy_retention']*100:.2f}%")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCQ combined mlp_in + mlp_out SVD truncation experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str, default="data/eval_sets/eval_set_mcq_nq_open_200.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--test", action="store_true", help="Quick test with fewer reduction levels")
    parser.add_argument("--components", type=int, nargs="+",
                        help="Exact component counts to test (e.g. --components 10 5 3 2 1)")
    parser.add_argument("--mode", type=str, default="top", choices=["top", "bottom", "random"],
                        help="SVD component selection mode: top (largest SV), bottom (smallest SV), random")
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Number of random trials for mode=random (ignored for top/bottom)")
    parser.add_argument("--random-seed-start", type=int, default=100,
                        help="Starting seed for random trials")

    args = parser.parse_args()

    kwargs = dict(
        model_name=args.model,
        model_type=args.model_type,
        eval_set_path=args.eval_set,
        target_layer=args.layer,
        mode=args.mode,
        num_trials=args.num_trials,
        random_seed_start=args.random_seed_start,
        checkpoint_every=args.checkpoint_every
    )

    if args.test:
        print("Running quick test...")
        kwargs['reduction_percentages'] = [10, 50, 90]
        kwargs['checkpoint_every'] = 1
        if args.mode == 'random':
            kwargs['num_trials'] = 2
        run_mcq_combined_svd_truncation(**kwargs)
    elif args.components:
        kwargs['component_counts'] = sorted(args.components, reverse=True)
        run_mcq_combined_svd_truncation(**kwargs)
    else:
        run_mcq_combined_svd_truncation(**kwargs)
