"""
Experiment 19: Simultaneous All-Layer Noise Removal

Loads existing sweep results (from experiment 15) to get per-layer noise
classifications, then removes ALL noise chunks from ALL layers at once
and evaluates the model. Tests whether noise removal is additive across layers.

Usage:
    python experiments/19_simultaneous_noise_removal.py \
        --sweep-dir results/full_sweep/Llama-2-7b-chat-hf_paired_mlp_in+mlp_out_chunk100 \
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json
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
import argparse

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.evaluation.metrics import compute_mcq_entropy_and_nll
from src.decomposition.svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
)


# =============================================================================
# Helpers (from experiment 15)
# =============================================================================

def load_mcq_eval_set(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def evaluate_mcq_on_samples(model, tokenizer, samples, device="cuda"):
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
            'correct_letter': sample['correct_letter'],
            **metrics
        })
    return results


def get_original_weight(model, layer_idx, matrix_type, model_type):
    if model_type == "llama":
        weight_map = {
            'mlp_in': lambda: model.model.layers[layer_idx].mlp.up_proj.weight.data,
            'mlp_out': lambda: model.model.layers[layer_idx].mlp.down_proj.weight.data,
            'gate_proj': lambda: model.model.layers[layer_idx].mlp.gate_proj.weight.data,
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return weight_map[matrix_type]().clone()


def summarize_mcq(results):
    """Compute accuracy and avg entropy from MCQ results."""
    accuracy = np.mean([r['is_correct'] for r in results])
    avg_entropy = np.mean([r['entropy'] for r in results])
    avg_nll = np.mean([r['nll'] for r in results])
    return {'accuracy': accuracy, 'avg_entropy': avg_entropy, 'avg_nll': avg_nll}


# =============================================================================
# Load sweep results
# =============================================================================

def load_sweep_noise(sweep_dir: Path):
    """Load noise chunk classifications from existing sweep results.

    Returns:
        dict: {layer_idx: {
            'noise_indices': [chunk_idx, ...],
            'chunk_ranges': [(start, end), ...],  # for noise chunks only
            'total_chunks': int,
            'noise_fraction': float,
        }}
    """
    sweep_dir = Path(sweep_dir)

    # Auto-detect model_short from pickle filenames
    layer_pkls = sorted(sweep_dir.glob("*_layer*.pkl"))
    if not layer_pkls:
        raise FileNotFoundError(f"No layer pickle files found in {sweep_dir}")
    model_short = layer_pkls[0].stem.split('_layer')[0]

    noise_info = {}
    for pkl_file in sorted(sweep_dir.glob(f"{model_short}_layer*.pkl")):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        layer_idx = data['config']['layer']
        classification = data['classification']
        noise_indices = classification.get('true_noise', [])

        # Build chunk_idx → (start, end) map
        chunk_range_map = {}
        for c in data['chunk_results']:
            chunk_range_map[c['chunk_idx']] = (c['chunk_start'], c['chunk_end'])

        # Get ranges for noise chunks only
        noise_ranges = [chunk_range_map[ci] for ci in noise_indices if ci in chunk_range_map]

        noise_info[layer_idx] = {
            'noise_indices': noise_indices,
            'chunk_ranges': noise_ranges,
            'total_chunks': data['config']['num_chunks'],
            'noise_fraction': data['summary']['noise_fraction'],
        }

    print(f"Loaded noise info for {len(noise_info)} layers from {sweep_dir}")
    return noise_info


# =============================================================================
# Main experiment
# =============================================================================

def run_simultaneous_removal(args):
    device = args.device
    model_type = args.model_type

    # Load model
    print(f"\n{'='*60}")
    print(f"Experiment 19: Simultaneous All-Layer Noise Removal")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Sweep dir: {args.sweep_dir}")
    print(f"Eval set: {args.eval_set}")
    print(f"Device: {device}")

    seed_everything(42)
    model, tokenizer = load_model_and_tokenizer(args.model, device=device)
    mcq_samples = load_mcq_eval_set(args.eval_set)

    # Load noise classifications from sweep
    noise_info = load_sweep_noise(args.sweep_dir)

    # Print per-layer noise summary
    total_noise = 0
    total_chunks = 0
    print(f"\n{'Layer':>5} {'Noise':>6} {'Total':>6} {'Fraction':>9}")
    print("-" * 30)
    for layer_idx in sorted(noise_info.keys()):
        info = noise_info[layer_idx]
        n = len(info['noise_indices'])
        t = info['total_chunks']
        total_noise += n
        total_chunks += t
        marker = " ***" if n >= 10 else ""
        print(f"{layer_idx:>5} {n:>6} {t:>6} {info['noise_fraction']:>8.1%}{marker}")
    print("-" * 30)
    print(f"{'TOTAL':>5} {total_noise:>6} {total_chunks:>6} {total_noise/total_chunks:>8.1%}")

    # Baseline evaluation
    print(f"\n--- Baseline evaluation ---")
    baseline_results = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    baseline = summarize_mcq(baseline_results)
    print(f"Baseline: accuracy={baseline['accuracy']:.1%}, "
          f"entropy={baseline['avg_entropy']:.4f}, nll={baseline['avg_nll']:.4f}")

    # Store original weights and apply noise removal for all layers
    print(f"\n--- Removing noise from all {len(noise_info)} layers simultaneously ---")
    original_weights = {}  # {(layer, mt): weight_tensor}
    matrix_types = ['mlp_in', 'mlp_out']  # paired

    for layer_idx in sorted(noise_info.keys()):
        info = noise_info[layer_idx]
        noise_ranges = info['chunk_ranges']

        if len(noise_ranges) == 0:
            print(f"  Layer {layer_idx}: 0 noise chunks — skip")
            continue

        print(f"  Layer {layer_idx}: removing {len(info['noise_indices'])} noise chunks "
              f"({info['noise_fraction']:.1%})...", end=" ")

        for mt in matrix_types:
            # Clone original
            orig = get_original_weight(model, layer_idx, mt, model_type)
            original_weights[(layer_idx, mt)] = orig

            # SVD decompose
            U, S, Vh = decompose_weight_svd(orig, device=device)

            # Remove noise chunks: compute keep indices
            remove_indices = set()
            for start, end in noise_ranges:
                remove_indices.update(range(start, end))
            keep_indices = sorted(set(range(len(S))) - remove_indices)
            keep_indices_t = torch.tensor(keep_indices, device=S.device, dtype=torch.long)

            # Reconstruct with noise removed
            W_modified = reconstruct_from_svd(
                U[:, keep_indices_t], S[keep_indices_t], Vh[keep_indices_t, :]
            )

            # Update model — DO NOT restore
            update_layer_with_svd(model, layer_idx, W_modified, mt, model_type)

        print("done")

    # Evaluate with all noise removed
    print(f"\n--- Evaluation with all noise removed ---")
    modified_results = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    modified = summarize_mcq(modified_results)

    acc_change = modified['accuracy'] - baseline['accuracy']
    ent_change = modified['avg_entropy'] - baseline['avg_entropy']

    print(f"Modified: accuracy={modified['accuracy']:.1%}, "
          f"entropy={modified['avg_entropy']:.4f}, nll={modified['avg_nll']:.4f}")
    print(f"\n{'='*60}")
    print(f"RESULT: accuracy change = {acc_change*100:+.1f}pp, "
          f"entropy change = {ent_change:+.4f}")
    print(f"Total noise chunks removed: {total_noise}/{total_chunks} "
          f"({total_noise/total_chunks:.1%})")
    print(f"{'='*60}")

    # Restore original weights
    print("\nRestoring original weights...")
    for (layer_idx, mt), orig in original_weights.items():
        restore_original_weight(model, layer_idx, orig, mt, model_type)
    print("Restored.")

    # Save results
    output_dir = Path("results/simultaneous_removal")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = args.model.split('/')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{model_short}_simultaneous_{timestamp}.pkl"

    result = {
        'config': {
            'model': args.model,
            'model_short': model_short,
            'sweep_dir': str(args.sweep_dir),
            'eval_set': args.eval_set,
            'matrix_types': matrix_types,
            'timestamp': timestamp,
        },
        'baseline': baseline,
        'modified': modified,
        'changes': {
            'accuracy_change': acc_change,
            'entropy_change': ent_change,
            'nll_change': modified['avg_nll'] - baseline['avg_nll'],
        },
        'per_layer': [
            {
                'layer': layer_idx,
                'num_noise_chunks': len(noise_info[layer_idx]['noise_indices']),
                'total_chunks': noise_info[layer_idx]['total_chunks'],
                'noise_indices': noise_info[layer_idx]['noise_indices'],
                'noise_fraction': noise_info[layer_idx]['noise_fraction'],
            }
            for layer_idx in sorted(noise_info.keys())
        ],
        'totals': {
            'total_noise_removed': total_noise,
            'total_chunks': total_chunks,
            'overall_noise_fraction': total_noise / total_chunks if total_chunks > 0 else 0,
        },
        'baseline_per_sample': baseline_results,
        'modified_per_sample': modified_results,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nSaved results to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 19: Remove all noise chunks from all layers simultaneously"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama")
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Path to existing sweep results directory")
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_arc_challenge_500.json")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_simultaneous_removal(args)


if __name__ == "__main__":
    main()
