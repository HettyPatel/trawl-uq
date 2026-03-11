"""
Experiment 21: Structured Pruning via Importance Maps

Loads importance classification from experiment 20, removes all UNIMPORTANT
SVD chunks, and measures retained model accuracy. Tests whether the ~66-83%
of chunks classified as unimportant can be safely removed.

Two modes:
  1. ALL-AT-ONCE: Remove unimportant chunks from ALL layers simultaneously.
     This gives the actual compression result.
  2. PER-LAYER: Remove unimportant chunks one layer at a time.
     This shows which layers tolerate pruning best.

No new sweep needed — reads exp 20 results for the importance map, loads
the model once, applies structured pruning, evaluates.

Usage:
    # All-at-once pruning using exp 20 results (k=5)
    python experiments/21_structured_pruning.py \\
        --importance-dir results/importance_sweep/Llama-2-7b-chat-hf_paired_mlp_in+mlp_out_chunk100_k5_eval_set_mcq_arc_challenge_500 \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json

    # Per-layer mode
    python experiments/21_structured_pruning.py \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --per-layer

    # Both modes
    python experiments/21_structured_pruning.py \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --both
"""

import sys
sys.path.append('.')

import json
import torch
import pickle
import numpy as np
import glob
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
# Helpers
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
            'question': sample['question'],
            'correct_answer': sample['correct_answer'],
            'correct_letter': sample['correct_letter'],
            **metrics
        })
    return results


def get_original_weight(model, layer_idx, matrix_type, model_type="llama"):
    if model_type == "llama":
        weight_map = {
            'mlp_in':   lambda: model.model.layers[layer_idx].mlp.up_proj.weight.data,
            'mlp_out':  lambda: model.model.layers[layer_idx].mlp.down_proj.weight.data,
            'gate_proj':lambda: model.model.layers[layer_idx].mlp.gate_proj.weight.data,
            'attn_q':   lambda: model.model.layers[layer_idx].self_attn.q_proj.weight.data,
            'attn_k':   lambda: model.model.layers[layer_idx].self_attn.k_proj.weight.data,
            'attn_v':   lambda: model.model.layers[layer_idx].self_attn.v_proj.weight.data,
            'attn_o':   lambda: model.model.layers[layer_idx].self_attn.o_proj.weight.data,
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return weight_map[matrix_type]().clone()


def reconstruct_keeping_chunks(U, S, Vh, keep_chunk_indices, chunk_size, total_components):
    """Reconstruct weight keeping only the specified chunks."""
    keep_sv_indices = []
    for ci in keep_chunk_indices:
        start = ci * chunk_size
        end = min(start + chunk_size, total_components)
        keep_sv_indices.extend(range(start, end))

    keep_sv_indices = sorted(keep_sv_indices)
    if not keep_sv_indices:
        # Everything removed — return zeros
        return torch.zeros(U.shape[0], Vh.shape[1], device=U.device), 1.0

    keep_t = torch.tensor(keep_sv_indices, device=S.device, dtype=torch.long)
    U_k = U[:, keep_t]
    S_k = S[keep_t]
    Vh_k = Vh[keep_t, :]
    W = reconstruct_from_svd(U_k, S_k, Vh_k)

    total_energy = torch.sum(S ** 2).item()
    kept_energy = torch.sum(S_k ** 2).item()
    energy_removed = 1.0 - (kept_energy / total_energy) if total_energy > 0 else 0.0

    return W, energy_removed


# =============================================================================
# Load importance map from exp 20
# =============================================================================

def load_importance_map(importance_dir: Path, flip_threshold_override=None):
    """
    Load importance classifications from exp 20 results.
    If flip_threshold_override is set, re-classify using raw flip counts
    instead of the pre-computed classification.
    Returns: dict of layer -> {important_chunks, unimportant_chunks, config}
    """
    pkls = sorted(glob.glob(str(importance_dir / "*.pkl")))
    pkls = [p for p in pkls if 'baseline' not in Path(p).name]

    importance_map = {}
    config_info = None

    for p in pkls:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        layer = data['config']['layer']

        if flip_threshold_override is not None:
            # Re-classify from raw flip counts
            important = []
            unimportant = []
            critical = []
            for chunk in data['chunk_results']:
                ci = chunk['chunk_idx']
                if chunk['importance'] == 'critical':
                    critical.append(ci)
                elif chunk['flip_count'] >= flip_threshold_override:
                    important.append(ci)
                else:
                    unimportant.append(ci)
        else:
            important = data['classification']['important']
            unimportant = data['classification']['unimportant']
            critical = data['classification']['critical']

        importance_map[layer] = {
            'important': important,
            'unimportant': unimportant,
            'critical': critical,
            'matrix_types_paired': data['config'].get('matrix_types_paired', []),
            'matrix_type': data['config']['matrix_type'],
            'chunk_size': data['config']['chunk_size'],
            'total_components': data['config']['total_components'],
            'num_chunks': data['config']['num_chunks'],
            'flip_threshold': flip_threshold_override if flip_threshold_override is not None else data['config'].get('flip_threshold', None),
        }
        if config_info is None:
            config_info = data['config']

    layers = sorted(importance_map.keys())
    total_imp = sum(len(importance_map[l]['important']) for l in layers)
    total_unimp = sum(len(importance_map[l]['unimportant']) for l in layers)
    total_crit = sum(len(importance_map[l]['critical']) for l in layers)
    total = total_imp + total_unimp + total_crit

    k_used = flip_threshold_override if flip_threshold_override is not None else config_info.get('flip_threshold', '?')
    print(f"Loaded importance map: {len(layers)} layers (k={k_used})")
    print(f"  Important:   {total_imp}/{total} ({100*total_imp/total:.1f}%)")
    print(f"  Unimportant: {total_unimp}/{total} ({100*total_unimp/total:.1f}%)")
    print(f"  Critical:    {total_crit}/{total} ({100*total_crit/total:.1f}%)")

    return importance_map, config_info


# =============================================================================
# All-at-once pruning
# =============================================================================

def run_all_at_once(
    model, tokenizer, mcq_samples,
    baseline_accuracy, baseline_entropy,
    importance_map, model_type,
    device="cuda",
):
    """
    Remove ALL unimportant chunks from ALL layers simultaneously.
    Keep important + critical chunks only.
    """
    print("\n" + "=" * 60)
    print("ALL-AT-ONCE PRUNING")
    print("=" * 60)

    layers = sorted(importance_map.keys())
    original_weights = {}
    total_svs = 0
    kept_svs_total = 0
    total_chunks_all = 0
    kept_chunks_all = 0

    for layer_idx in tqdm(layers, desc="Pruning layers"):
        info = importance_map[layer_idx]
        matrix_types = info['matrix_types_paired']
        chunk_size = info['chunk_size']
        total_components = info['total_components']

        # Keep important + critical chunks
        keep_chunks = sorted(set(info['important'] + info['critical']))
        remove_chunks = info['unimportant']

        for mt in matrix_types:
            original_weights[(layer_idx, mt)] = get_original_weight(model, layer_idx, mt, model_type)
            W = original_weights[(layer_idx, mt)]
            U, S, Vh = decompose_weight_svd(W, device)

            W_pruned, energy_removed = reconstruct_keeping_chunks(
                U, S, Vh, keep_chunks, chunk_size, total_components
            )
            update_layer_with_svd(model, layer_idx, W_pruned, mt, model_type)

            # Count singular values kept vs total
            kept_svs = sum(min((ci + 1) * chunk_size, total_components) - ci * chunk_size for ci in keep_chunks)
            total_svs += total_components
            kept_svs_total += kept_svs

        total_chunks_all += info['num_chunks']
        kept_chunks_all += len(keep_chunks)

        print(f"  Layer {layer_idx}: keep {len(keep_chunks)}/{info['num_chunks']} chunks, "
              f"remove {len(remove_chunks)}")

    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    pruned_accuracy = float(np.mean([r['is_correct'] for r in mcq_eval]))
    pruned_entropy = float(np.nanmean([r['entropy'] for r in mcq_eval]))

    acc_retained = pruned_accuracy / baseline_accuracy * 100 if baseline_accuracy > 0 else 0
    sv_removed_pct = (1.0 - kept_svs_total / total_svs) * 100 if total_svs > 0 else 0
    chunks_removed_pct = (1.0 - kept_chunks_all / total_chunks_all) * 100 if total_chunks_all > 0 else 0

    print(f"\n  Baseline accuracy:  {baseline_accuracy*100:.1f}%")
    print(f"  Pruned accuracy:    {pruned_accuracy*100:.1f}%")
    print(f"  Accuracy retained:  {acc_retained:.1f}%")
    print(f"  Accuracy change:    {(pruned_accuracy - baseline_accuracy)*100:+.1f}pp")
    print(f"  Baseline entropy:   {baseline_entropy:.4f}")
    print(f"  Pruned entropy:     {pruned_entropy:.4f}")
    print(f"  Chunks kept:        {kept_chunks_all}/{total_chunks_all} ({100*kept_chunks_all/total_chunks_all:.1f}%)")
    print(f"  SVs kept:           {kept_svs_total}/{total_svs} ({100*kept_svs_total/total_svs:.1f}%)")
    print(f"  SVs removed:        {sv_removed_pct:.1f}%")

    # Restore all weights
    for (layer_idx, mt), W_orig in original_weights.items():
        restore_original_weight(model, layer_idx, W_orig, mt, model_type)

    return {
        'mode': 'all_at_once',
        'baseline_accuracy': baseline_accuracy,
        'baseline_entropy': baseline_entropy,
        'pruned_accuracy': pruned_accuracy,
        'pruned_entropy': pruned_entropy,
        'accuracy_change': pruned_accuracy - baseline_accuracy,
        'accuracy_retained_pct': acc_retained,
        'total_svs': total_svs,
        'kept_svs': kept_svs_total,
        'sv_removed_pct': sv_removed_pct,
        'total_chunks': total_chunks_all,
        'kept_chunks': kept_chunks_all,
        'chunks_removed_pct': chunks_removed_pct,
        'per_sample': mcq_eval,
    }


# =============================================================================
# Per-layer pruning
# =============================================================================

def run_per_layer(
    model, tokenizer, mcq_samples,
    baseline_accuracy, baseline_entropy,
    importance_map, model_type,
    device="cuda",
):
    """
    Remove unimportant chunks from ONE layer at a time.
    Shows which layers tolerate pruning best.
    """
    print("\n" + "=" * 60)
    print("PER-LAYER PRUNING")
    print("=" * 60)

    layers = sorted(importance_map.keys())
    per_layer_results = []

    for layer_idx in layers:
        info = importance_map[layer_idx]
        matrix_types = info['matrix_types_paired']
        chunk_size = info['chunk_size']
        total_components = info['total_components']

        keep_chunks = sorted(set(info['important'] + info['critical']))
        remove_chunks = info['unimportant']

        if not remove_chunks:
            print(f"  Layer {layer_idx}: nothing to remove (0 unimportant)")
            per_layer_results.append({
                'layer': layer_idx,
                'n_removed': 0,
                'n_kept': info['num_chunks'],
                'pruned_accuracy': baseline_accuracy,
                'accuracy_change': 0.0,
            })
            continue

        # Save originals and prune
        originals = {}
        for mt in matrix_types:
            originals[mt] = get_original_weight(model, layer_idx, mt, model_type)
            U, S, Vh = decompose_weight_svd(originals[mt], device)
            W_pruned, _ = reconstruct_keeping_chunks(
                U, S, Vh, keep_chunks, chunk_size, total_components
            )
            update_layer_with_svd(model, layer_idx, W_pruned, mt, model_type)

        # Evaluate
        mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        pruned_accuracy = float(np.mean([r['is_correct'] for r in mcq_eval]))
        acc_change = pruned_accuracy - baseline_accuracy

        print(f"  Layer {layer_idx}: remove {len(remove_chunks)}/{info['num_chunks']} chunks → "
              f"acc={pruned_accuracy*100:.1f}% ({acc_change*100:+.1f}pp)")

        per_layer_results.append({
            'layer': layer_idx,
            'n_removed': len(remove_chunks),
            'n_kept': len(keep_chunks),
            'n_total': info['num_chunks'],
            'pruned_accuracy': pruned_accuracy,
            'accuracy_change': acc_change,
        })

        # Restore
        for mt in matrix_types:
            restore_original_weight(model, layer_idx, originals[mt], mt, model_type)

    return {
        'mode': 'per_layer',
        'baseline_accuracy': baseline_accuracy,
        'per_layer': per_layer_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 21: Structured Pruning via Importance Maps")
    parser.add_argument('--importance-dir', required=True,
                        help='Path to exp 20 importance sweep results directory')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (auto-detected from importance config if not set)')
    parser.add_argument('--model-type', type=str, default="llama")
    parser.add_argument('--eval-set', type=str, required=True,
                        help='Path to MCQ evaluation set')
    parser.add_argument('--per-layer', action='store_true',
                        help='Run per-layer pruning only')
    parser.add_argument('--both', action='store_true',
                        help='Run both all-at-once and per-layer')
    parser.add_argument('--flip-threshold', type=int, default=None,
                        help='Override flip threshold k for re-classification (uses raw flip counts from exp 20)')
    parser.add_argument('--output-dir', default='results/structured_pruning/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    seed_everything(42)

    importance_dir = Path(args.importance_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load importance map
    importance_map, config_info = load_importance_map(importance_dir, flip_threshold_override=args.flip_threshold)

    # Auto-detect model from config
    model_name = args.model
    if model_name is None:
        config_path = importance_dir / 'sweep_config.json'
        if config_path.exists():
            with open(config_path) as f:
                sweep_cfg = json.load(f)
            model_name = sweep_cfg.get('model_name', 'meta-llama/Llama-2-7b-chat-hf')
            print(f"Auto-detected model: {model_name}")
        else:
            model_name = 'meta-llama/Llama-2-7b-chat-hf'
            print(f"No sweep config found, using default: {model_name}")

    # Load eval set
    mcq_samples = load_mcq_eval_set(args.eval_set)

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=args.device)
    print("Model loaded.")

    # Baseline
    print("\nEvaluating baseline...")
    baseline_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, args.device)
    baseline_accuracy = float(np.mean([r['is_correct'] for r in baseline_eval]))
    baseline_entropy = float(np.nanmean([r['entropy'] for r in baseline_eval]))
    print(f"  Baseline: {baseline_accuracy*100:.1f}% accuracy, {baseline_entropy:.4f} entropy")

    k_used = args.flip_threshold if args.flip_threshold is not None else config_info.get('flip_threshold')
    results = {
        'config': {
            'importance_dir': str(importance_dir),
            'model_name': model_name,
            'eval_set': args.eval_set,
            'flip_threshold': k_used,
            'timestamp': datetime.now().isoformat(),
        },
        'baseline_accuracy': baseline_accuracy,
        'baseline_entropy': baseline_entropy,
    }

    # Run modes
    if args.per_layer and not args.both:
        results['per_layer'] = run_per_layer(
            model, tokenizer, mcq_samples,
            baseline_accuracy, baseline_entropy,
            importance_map, args.model_type, args.device,
        )
    elif args.both:
        results['all_at_once'] = run_all_at_once(
            model, tokenizer, mcq_samples,
            baseline_accuracy, baseline_entropy,
            importance_map, args.model_type, args.device,
        )
        results['per_layer'] = run_per_layer(
            model, tokenizer, mcq_samples,
            baseline_accuracy, baseline_entropy,
            importance_map, args.model_type, args.device,
        )
    else:
        # Default: all-at-once
        results['all_at_once'] = run_all_at_once(
            model, tokenizer, mcq_samples,
            baseline_accuracy, baseline_entropy,
            importance_map, args.model_type, args.device,
        )

    # Save
    imp_name = importance_dir.name
    if args.flip_threshold is not None:
        out_file = output_dir / f"pruning_{imp_name}_k{args.flip_threshold}.pkl"
    else:
        out_file = output_dir / f"pruning_{imp_name}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
