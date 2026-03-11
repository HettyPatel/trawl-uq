"""
CP Decomposition Component Sweep — Spectral Anatomy via CP

Same approach as experiment 15 (SVD chunk removal) but using CP decomposition.
CP jointly decomposes mlp_in + mlp_out by transposing and stacking them into
a [2, hidden_size, intermediate_size] tensor, then factoring into rank-1
components. We remove CP components one by one, measure MCQ entropy + accuracy,
classify as noise/signal, then cumulatively stack noise components.

Key differences from exp 15 (SVD):
  - Joint decomposition: mlp_in + mlp_out together (inherently paired)
  - Component removal: 1 CP component at a time (not chunks of SVs)
  - Lossy: CP at rank < full is approximate, so we measure a decomposed baseline
  - Weight update: reconstruct both fc_in and fc_out from modified CP factors

Usage:
    # Full sweep, all 32 layers
    python experiments/17_cp_component_sweep.py --layers 0-31 --rank 40

    # Single layer test
    python experiments/17_cp_component_sweep.py --layers 31 --test

    # Resume interrupted run
    python experiments/17_cp_component_sweep.py --layers 0-31 --rank 40 --resume

    # Quick test
    python experiments/17_cp_component_sweep.py --layers 31 --rank 5 --test
"""

import sys
sys.path.append('.')

import json
import csv
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.evaluation.metrics import compute_mcq_entropy_and_nll
from src.decomposition.cp import (
    decompose_fc_layer_cp,
    remove_cp_component,
    reconstruct_weights_cp,
    get_cp_component_importance,
)
from src.decomposition.tucker import get_fc_layer_weights
from src.decomposition.model_utils import update_fc_layer_weights


# =============================================================================
# Helper functions (reused from exp 15)
# =============================================================================

def parse_layer_spec(spec: str, num_layers: int = 32) -> list:
    """
    Parse layer specification string into list of layer indices.

    Supports:
        "0-31"      -> [0, 1, ..., 31]
        "0,1,4,31"  -> [0, 1, 4, 31]
        "0-4,12,31" -> [0, 1, 2, 3, 4, 12, 31]
        "31"        -> [31]
    """
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def evaluate_mcq_on_samples(model, tokenizer, samples, device="cuda"):
    """Evaluate MCQ entropy and NLL on all samples."""
    results = []
    for sample in tqdm(samples, desc="Evaluating MCQ", leave=False):
        metrics = compute_mcq_entropy_and_nll(
            mcq_prompt=sample['mcq_prompt'],
            correct_letter=sample['correct_letter'],
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        results.append({
            'sample_id': sample['id'],
            'question': sample['question'],
            'correct_answer': sample['correct_answer'],
            'correct_letter': sample['correct_letter'],
            **metrics
        })
    return results


# =============================================================================
# Single layer CP experiment
# =============================================================================

def run_single_layer_cp(
    model, tokenizer, mcq_samples,
    baseline_mcq, baseline_mcq_entropy, baseline_mcq_nll, baseline_mcq_accuracy,
    layer_idx: int, rank: int,
    model_type: str,
    output_dir: Path, model_short: str,
    device: str = "cuda",
):
    """
    Run CP component removal for a single layer.

    For each of the `rank` CP components:
      1. Remove it (zero its lambda weight)
      2. Reconstruct both mlp_in and mlp_out
      3. Evaluate MCQ entropy + accuracy
      4. Classify as noise/signal/etc.

    Then cumulatively stack noise components.
    """
    result_path = output_dir / f"{model_short}_layer{layer_idx}_cp_rank{rank}.pkl"

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} | CP rank={rank}")
    print(f"{'='*60}")

    # Get original weights
    fc_in_weight, fc_out_weight = get_fc_layer_weights(model, layer_idx, model_type=model_type)
    original_fc_in = fc_in_weight.clone()
    original_fc_out = fc_out_weight.clone()

    # CP decomposition
    print(f"  Decomposing layer {layer_idx} with CP rank {rank}...")
    cp_result = decompose_fc_layer_cp(fc_in_weight, fc_out_weight, rank=rank, device=device)
    cp_weights = cp_result[0]   # [rank] — lambda values
    cp_factors = cp_result[1:]  # [factor_0, factor_1, factor_2]

    print(f"  CP weights shape: {cp_weights.shape}")
    for i, f in enumerate(cp_factors):
        print(f"  Factor {i} shape: {f.shape}")

    # Component importance scores
    importance = get_cp_component_importance(cp_weights, cp_factors)
    importance_order = torch.argsort(importance, descending=True).cpu().numpy()
    print(f"  Top-5 important components: {importance_order[:5]} "
          f"(scores: {importance[importance_order[:5]].cpu().numpy()})")

    # ==================== Decomposed Baseline ====================
    # Reconstruct with ALL components to measure CP approximation error
    print(f"\n  Evaluating decomposed baseline (all {rank} components)...")
    fc_in_recon, fc_out_recon = reconstruct_weights_cp(cp_weights, cp_factors)
    update_fc_layer_weights(model, layer_idx, fc_in_recon, fc_out_recon, model_type=model_type)

    decomp_baseline_mcq = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    decomp_baseline_entropy = np.nanmean([r['entropy'] for r in decomp_baseline_mcq])
    decomp_baseline_nll = np.nanmean([r['nll'] for r in decomp_baseline_mcq])
    decomp_baseline_accuracy = np.mean([r['is_correct'] for r in decomp_baseline_mcq])

    print(f"  Original baseline:    {baseline_mcq_accuracy*100:.1f}% acc, {baseline_mcq_entropy:.4f} entropy")
    print(f"  Decomposed baseline:  {decomp_baseline_accuracy*100:.1f}% acc, {decomp_baseline_entropy:.4f} entropy")
    print(f"  CP approximation loss: {(decomp_baseline_accuracy - baseline_mcq_accuracy)*100:+.1f}pp acc, "
          f"{decomp_baseline_entropy - baseline_mcq_entropy:+.4f} entropy")

    # Restore original weights
    update_fc_layer_weights(model, layer_idx, original_fc_in, original_fc_out, model_type=model_type)

    # ==================== Component Removal Loop ====================
    print(f"\n  Removing components one by one...")
    component_results = []

    for comp_idx in tqdm(range(rank), desc=f"L{layer_idx}/CP", leave=False):
        # Remove single component
        modified_weights, _ = remove_cp_component(cp_weights, cp_factors, comp_idx)

        # Reconstruct both matrices
        fc_in_recon, fc_out_recon = reconstruct_weights_cp(modified_weights, cp_factors)

        # Update model
        update_fc_layer_weights(model, layer_idx, fc_in_recon, fc_out_recon, model_type=model_type)

        # Evaluate
        mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        mcq_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
        mcq_nll = np.nanmean([r['nll'] for r in mcq_eval])
        mcq_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

        # Compute changes from DECOMPOSED baseline (not original)
        if np.isnan(mcq_entropy):
            mcq_entropy_change = float('nan')
            mcq_nll_change = float('nan')
        else:
            mcq_entropy_change = mcq_entropy - decomp_baseline_entropy
            mcq_nll_change = mcq_nll - decomp_baseline_nll

        acc_change = mcq_accuracy - decomp_baseline_accuracy

        # Also compute changes from original baseline (for reference)
        if np.isnan(mcq_entropy):
            orig_entropy_change = float('nan')
        else:
            orig_entropy_change = mcq_entropy - baseline_mcq_entropy
        orig_acc_change = mcq_accuracy - baseline_mcq_accuracy

        component_results.append({
            'component_idx': comp_idx,
            'importance_score': importance[comp_idx].item(),
            'lambda_weight': cp_weights[comp_idx].item(),
            'mcq_entropy': mcq_entropy,
            'mcq_nll': mcq_nll,
            'mcq_accuracy': mcq_accuracy,
            'mcq_entropy_change': mcq_entropy_change,
            'mcq_nll_change': mcq_nll_change,
            'mcq_accuracy_change': acc_change,
            'mcq_entropy_change_from_original': orig_entropy_change,
            'mcq_accuracy_change_from_original': orig_acc_change,
            'mcq_per_sample': mcq_eval,
        })

        # Classification label (based on decomposed baseline comparison)
        if np.isnan(mcq_entropy_change):
            label = "CRITICAL"
        elif mcq_entropy_change < 0 and acc_change > 0:
            label = "NOISE"
        elif mcq_entropy_change < 0 and acc_change <= 0:
            label = "conf_wrng"
        elif mcq_entropy_change >= 0 and acc_change <= 0:
            label = "SIGNAL"
        else:
            label = "unc_rght"

        # Print progress
        if np.isnan(mcq_entropy_change):
            print(f"    comp{comp_idx:>3} (lambda={cp_weights[comp_idx].item():+.3f}) "
                  f"ent=NaN  acc={acc_change*100:+.1f}pp  -> {label}")
        else:
            print(f"    comp{comp_idx:>3} (lambda={cp_weights[comp_idx].item():+.3f}) "
                  f"ent={mcq_entropy_change:+.4f}  acc={acc_change*100:+.1f}pp  -> {label}")

        # Restore original weights
        update_fc_layer_weights(model, layer_idx, original_fc_in, original_fc_out, model_type=model_type)

    # ==================== Classify Components ====================
    true_noise = []
    confident_wrong = []
    true_signal = []
    uncertain_right = []
    critical_signal = []

    for c in component_results:
        if np.isnan(c['mcq_entropy_change']):
            critical_signal.append(c['component_idx'])
            continue
        if c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] > 0:
            true_noise.append(c['component_idx'])
        elif c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] <= 0:
            confident_wrong.append(c['component_idx'])
        elif c['mcq_entropy_change'] >= 0 and c['mcq_accuracy_change'] <= 0:
            true_signal.append(c['component_idx'])
        else:
            uncertain_right.append(c['component_idx'])

    crit_str = f", {len(critical_signal)} critical" if critical_signal else ""
    print(f"\n  Classification: {len(true_noise)} noise, {len(true_signal)} signal, "
          f"{len(confident_wrong)} conf_wrong, {len(uncertain_right)} unc_right{crit_str}")

    # ==================== Cumulative Noise Removal ====================
    cumulative_results = []
    if true_noise:
        noise_sorted = sorted(true_noise)
        print(f"\n  Cumulative noise removal ({len(noise_sorted)} noise components)...")
        print(f"  Noise components: {noise_sorted}")

        for i, _ in enumerate(noise_sorted):
            components_to_remove = noise_sorted[:i+1]

            # Zero out all noise components up to this step
            modified_weights = cp_weights.clone()
            for comp_idx in components_to_remove:
                modified_weights[comp_idx] = 0.0

            # Reconstruct and update
            fc_in_recon, fc_out_recon = reconstruct_weights_cp(modified_weights, cp_factors)
            update_fc_layer_weights(model, layer_idx, fc_in_recon, fc_out_recon, model_type=model_type)

            # Evaluate
            mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
            cum_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
            cum_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

            if np.isnan(cum_entropy):
                cum_entropy_change = float('nan')
            else:
                cum_entropy_change = cum_entropy - decomp_baseline_entropy

            cum_acc_change = cum_accuracy - decomp_baseline_accuracy

            cumulative_results.append({
                'step': i,
                'num_components_removed': i + 1,
                'component_indices_removed': list(components_to_remove),
                'mcq_entropy': cum_entropy,
                'mcq_accuracy': cum_accuracy,
                'mcq_entropy_change': cum_entropy_change,
                'mcq_accuracy_change': cum_acc_change,
            })

            print(f"    Removed {i+1} noise components {components_to_remove}: "
                  f"acc={cum_accuracy*100:.1f}% ({cum_acc_change*100:+.1f}pp), "
                  f"ent={cum_entropy:.4f} ({cum_entropy_change:+.4f})")

            # Restore
            update_fc_layer_weights(model, layer_idx, original_fc_in, original_fc_out, model_type=model_type)
    else:
        print(f"\n  No noise components found — skipping cumulative removal.")

    # ==================== Summary Stats ====================
    valid_components = [c for c in component_results if not np.isnan(c['mcq_entropy_change'])]
    sum_entropy_change = sum(c['mcq_entropy_change'] for c in valid_components)
    sum_accuracy_change = sum(c['mcq_accuracy_change'] for c in valid_components)
    total_signal = len(true_signal) + len(critical_signal)
    noise_fraction = len(true_noise) / rank if rank > 0 else 0

    # ==================== Save ====================
    result = {
        'config': {
            'layer': layer_idx,
            'decomposition_type': 'cp',
            'rank': rank,
            'num_components': rank,
            'model_type': model_type,
            'matrix_type': 'mlp_in+mlp_out',
        },
        'original_baseline_mcq': {
            'avg_entropy': baseline_mcq_entropy,
            'avg_nll': baseline_mcq_nll,
            'accuracy': baseline_mcq_accuracy,
        },
        'decomposed_baseline_mcq': {
            'avg_entropy': decomp_baseline_entropy,
            'avg_nll': decomp_baseline_nll,
            'accuracy': decomp_baseline_accuracy,
            'per_sample': decomp_baseline_mcq,
        },
        'component_importance': {
            'scores': importance.cpu().numpy().tolist(),
            'order_by_importance': importance_order.tolist(),
        },
        'component_results': component_results,
        'cumulative_results': cumulative_results,
        'classification': {
            'true_noise': true_noise,
            'confident_wrong': confident_wrong,
            'true_signal': true_signal,
            'uncertain_right': uncertain_right,
            'critical_signal': critical_signal,
        },
        'summary': {
            'noise_fraction': noise_fraction,
            'signal_fraction': total_signal / rank if rank > 0 else 0,
            'sum_entropy_change': sum_entropy_change,
            'sum_accuracy_change': sum_accuracy_change,
            'num_true_noise': len(true_noise),
            'num_true_signal': len(true_signal),
            'num_confident_wrong': len(confident_wrong),
            'num_uncertain_right': len(uncertain_right),
            'num_critical_signal': len(critical_signal),
        },
    }

    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n  Saved: {result_path}")
    print(f"  Net entropy: {sum_entropy_change:+.4f}, Net accuracy: {sum_accuracy_change*100:+.1f}pp, "
          f"Noise fraction: {noise_fraction:.1%}")

    return result


# =============================================================================
# Summary CSV generation
# =============================================================================

def generate_summary_csv(output_dir: Path, model_short: str):
    """Generate a summary CSV from all completed result files."""
    csv_path = output_dir / f"{model_short}_summary.csv"
    rows = []

    for pkl_file in sorted(output_dir.glob(f"{model_short}_layer*.pkl")):
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)

        config = result['config']
        summary = result['summary']
        orig_baseline = result['original_baseline_mcq']
        decomp_baseline = result['decomposed_baseline_mcq']

        rows.append({
            'layer': config['layer'],
            'rank': config['rank'],
            'decomposition_type': config['decomposition_type'],
            'original_baseline_entropy': orig_baseline['avg_entropy'],
            'original_baseline_accuracy': orig_baseline['accuracy'],
            'decomposed_baseline_entropy': decomp_baseline['avg_entropy'],
            'decomposed_baseline_accuracy': decomp_baseline['accuracy'],
            'num_true_noise': summary['num_true_noise'],
            'num_true_signal': summary['num_true_signal'],
            'num_confident_wrong': summary['num_confident_wrong'],
            'num_uncertain_right': summary['num_uncertain_right'],
            'num_critical_signal': summary.get('num_critical_signal', 0),
            'noise_fraction': summary['noise_fraction'],
            'signal_fraction': summary['signal_fraction'],
            'sum_entropy_change': summary['sum_entropy_change'],
            'sum_accuracy_change': summary['sum_accuracy_change'],
        })

    if rows:
        rows.sort(key=lambda r: r['layer'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV saved to: {csv_path}")
        print(f"Total completed: {len(rows)} layers")
    else:
        print("No results found to summarize.")


# =============================================================================
# Main sweep
# =============================================================================

def run_cp_sweep(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    layers: list = None,
    rank: int = 40,
    mcq_eval_set_path: str = "data/eval_sets/eval_set_mcq_arc_challenge_500.json",
    device: str = "cuda",
    resume: bool = False,
):
    """
    Run CP component removal sweep across specified layers.
    One job per layer (CP inherently decomposes mlp_in + mlp_out jointly).
    """
    seed_everything(42)

    if layers is None:
        layers = list(range(32))

    model_short = model_name.split("/")[-1]
    output_dir = Path(f"results/cp_sweep/{model_short}_rank{rank}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    skipped = 0
    for layer in layers:
        result_path = output_dir / f"{model_short}_layer{layer}_cp_rank{rank}.pkl"
        if resume and result_path.exists():
            skipped += 1
            continue
        jobs.append(layer)

    total_jobs = len(jobs) + skipped
    print("=" * 70)
    print("CP COMPONENT SWEEP - Spectral Anatomy via CP Decomposition")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"CP rank: {rank}")
    print(f"Eval set: {mcq_eval_set_path}")
    print(f"Output: {output_dir}")
    print(f"Jobs: {len(jobs)} to run, {skipped} already completed, {total_jobs} total")
    print("=" * 70)

    if not jobs:
        print("\nAll jobs already completed! Generating summary...")
        generate_summary_csv(output_dir, model_short)
        return

    # Load eval set
    mcq_samples = load_mcq_eval_set(mcq_eval_set_path)

    # Save sweep config
    sweep_config = {
        'model_name': model_name,
        'model_type': model_type,
        'layers': layers,
        'rank': rank,
        'decomposition_type': 'cp',
        'mcq_eval_set_path': mcq_eval_set_path,
        'num_mcq_samples': len(mcq_samples),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(output_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # Baseline evaluation (original model, once for all jobs)
    print("\nEvaluating baseline MCQ (original model)...")
    baseline_mcq = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    baseline_mcq_entropy = np.nanmean([r['entropy'] for r in baseline_mcq])
    baseline_mcq_nll = np.nanmean([r['nll'] for r in baseline_mcq])
    baseline_mcq_accuracy = np.mean([r['is_correct'] for r in baseline_mcq])
    print(f"  Baseline: {baseline_mcq_accuracy*100:.1f}% accuracy, "
          f"{baseline_mcq_entropy:.4f} entropy, {baseline_mcq_nll:.4f} NLL")

    # Save baseline
    with open(output_dir / 'baseline.pkl', 'wb') as f:
        pickle.dump({
            'mcq_entropy': baseline_mcq_entropy,
            'mcq_nll': baseline_mcq_nll,
            'mcq_accuracy': baseline_mcq_accuracy,
            'mcq_per_sample': baseline_mcq,
        }, f)

    # Run jobs
    for job_idx, layer in enumerate(jobs):
        print(f"\n[Job {job_idx+1}/{len(jobs)}] Layer {layer}, CP rank={rank}")

        run_single_layer_cp(
            model=model,
            tokenizer=tokenizer,
            mcq_samples=mcq_samples,
            baseline_mcq=baseline_mcq,
            baseline_mcq_entropy=baseline_mcq_entropy,
            baseline_mcq_nll=baseline_mcq_nll,
            baseline_mcq_accuracy=baseline_mcq_accuracy,
            layer_idx=layer,
            rank=rank,
            model_type=model_type,
            output_dir=output_dir,
            model_short=model_short,
            device=device,
        )

        torch.cuda.empty_cache()

    # Generate summary
    print("\n" + "=" * 70)
    print("CP SWEEP COMPLETE")
    print("=" * 70)
    generate_summary_csv(output_dir, model_short)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CP Component Sweep - Spectral Anatomy via CP Decomposition"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama",
                        choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--layers", type=str, default="0-31",
                        help="Layer specification: '0-31', '0,1,4,31', etc.")
    parser.add_argument("--rank", type=int, default=40,
                        help="CP decomposition rank (number of components, default: 40)")
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_arc_challenge_500.json",
                        help="Path to MCQ evaluation set JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed layers")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: rank=5, single layer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    args = parser.parse_args()

    # Determine number of layers from model
    num_layers = 32
    if "gpt2" in args.model.lower() and "xl" not in args.model.lower():
        num_layers = 12
    elif "gpt-j" in args.model.lower() or "gptj" in args.model.lower():
        num_layers = 28

    layers = parse_layer_spec(args.layers, num_layers)

    rank = 5 if args.test else args.rank
    if args.test:
        print("TEST MODE: rank=5")

    run_cp_sweep(
        model_name=args.model,
        model_type=args.model_type,
        layers=layers,
        rank=rank,
        mcq_eval_set_path=args.eval_set,
        device=args.device,
        resume=args.resume,
    )
