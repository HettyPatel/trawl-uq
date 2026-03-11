"""
Experiment 20: Importance Sweep - 2-Way Chunk Classification

Identical to experiment 15 (full layer sweep) but classifies each SVD chunk as:
    IMPORTANT:   removing this chunk flips >= k question answers
    UNIMPORTANT: removing this chunk flips < k question answers

k is configurable via --flip-threshold (default: 5)

This is cleaner than the 5-way entropy+accuracy classification from exp 15
and easier to communicate in a paper:
    "A component is important if removing it changes the model's prediction
     on at least k questions out of N evaluated."

Baseline correctness per question is computed once at the start (full forward
pass with unmodified weights), then each chunk removal's per-sample is_correct
is compared against it to count flips.

Usage:
    # Llama-2 paired MLP, ARC-500, default k=5
    python experiments/20_importance_sweep.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --layers 0-31 --matrices mlp --paired \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json

    # Try different k values
    python experiments/20_importance_sweep.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --layers 0-31 --matrices mlp --paired \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --flip-threshold 10

    # Resume interrupted run
    python experiments/20_importance_sweep.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --layers 0-31 --matrices mlp --paired \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --resume
"""

import sys
sys.path.append('.')

import json
import csv
import re
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
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
    get_svd_stats,
)


# =============================================================================
# Matrix group definitions (same as exp 15)
# =============================================================================

MATRIX_GROUPS = {
    'mlp': ['mlp_in', 'mlp_out'],
    'mlp_full': ['mlp_in', 'mlp_out', 'gate_proj'],
    'attn': ['attn_q', 'attn_k', 'attn_v', 'attn_o'],
    'attn_value': ['attn_v', 'attn_o'],
    'all': ['mlp_in', 'mlp_out', 'gate_proj', 'attn_q', 'attn_k', 'attn_v', 'attn_o'],
}


# =============================================================================
# Helpers (shared with exp 15)
# =============================================================================

def parse_layer_spec(spec: str, num_layers: int = 32) -> list:
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


def get_original_weight(model, layer_idx, matrix_type, model_type):
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


def reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end):
    keep_indices = torch.cat([
        torch.arange(0, chunk_start, device=S.device),
        torch.arange(chunk_end, len(S), device=S.device)
    ])
    U_k = U[:, keep_indices]
    S_k = S[keep_indices]
    Vh_k = Vh[keep_indices, :]
    W_mod = reconstruct_from_svd(U_k, S_k, Vh_k)
    total_energy = torch.sum(S ** 2).item()
    removed_energy = torch.sum(S[chunk_start:chunk_end] ** 2).item()
    energy_removed = removed_energy / total_energy if total_energy > 0 else 0.0
    return W_mod, energy_removed


# =============================================================================
# Importance classification
# =============================================================================

def count_flips(chunk_per_sample, baseline_correct: dict) -> int:
    """Count how many questions changed correct/incorrect vs baseline."""
    return sum(
        1 for s in chunk_per_sample
        if s['is_correct'] != baseline_correct.get(s['sample_id'], s['is_correct'])
    )


def classify_importance(flip_count: int, flip_threshold: int) -> str:
    return 'important' if flip_count >= flip_threshold else 'unimportant'


# =============================================================================
# Single layer — paired removal (main mode)
# =============================================================================

def run_layer_paired(
    model, tokenizer, mcq_samples,
    baseline_results, baseline_correct,
    baseline_entropy, baseline_nll, baseline_accuracy,
    layer_idx: int, matrix_types: list,
    model_type: str, chunk_size: int,
    flip_threshold: int,
    output_dir: Path, model_short: str,
    device: str = "cuda",
):
    matrix_str = "+".join(matrix_types)
    result_path = output_dir / f"{model_short}_layer{layer_idx}_{matrix_str}.pkl"

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} | PAIRED {matrix_str}")
    print(f"{'='*60}")

    # Decompose all matrices
    original_weights = {}
    decompositions = {}
    svd_stats_all = {}

    for mt in matrix_types:
        original_weights[mt] = get_original_weight(model, layer_idx, mt, model_type)
        U, S, Vh = decompose_weight_svd(original_weights[mt], device)
        decompositions[mt] = {'U': U, 'S': S, 'Vh': Vh}
        svd_stats_all[mt] = get_svd_stats(original_weights[mt], device)
        print(f"  {mt}: shape={svd_stats_all[mt]['shape']}, SVs={len(S)}, "
              f"eff_rank={svd_stats_all[mt]['effective_rank']:.1f}")

    total_components = min(len(decompositions[mt]['S']) for mt in matrix_types)
    num_chunks = (total_components + chunk_size - 1) // chunk_size
    print(f"  Chunks: {num_chunks} (min SVs={total_components}, flip_threshold k={flip_threshold})")

    chunk_results = []
    important_chunks = []
    unimportant_chunks = []
    critical_chunks = []

    for chunk_idx in tqdm(range(num_chunks), desc=f"L{layer_idx}/{matrix_str}", leave=False):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_components)

        # Remove chunk from ALL matrices simultaneously
        energy_per_matrix = {}
        for mt in matrix_types:
            U = decompositions[mt]['U']
            S = decompositions[mt]['S']
            Vh = decompositions[mt]['Vh']
            W_mod, e = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
            energy_per_matrix[mt] = e
            update_layer_with_svd(model, layer_idx, W_mod, mt, model_type)

        avg_energy = float(np.mean(list(energy_per_matrix.values())))

        mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        mcq_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
        mcq_nll = np.nanmean([r['nll'] for r in mcq_eval])
        mcq_accuracy = float(np.mean([r['is_correct'] for r in mcq_eval]))

        is_nan = np.isnan(mcq_entropy)

        if is_nan:
            mcq_entropy_change = float('nan')
            mcq_nll_change = float('nan')
            flip_count = len(mcq_samples)  # treat NaN as maximum disruption
            importance = 'critical'
        else:
            mcq_entropy_change = float(mcq_entropy - baseline_entropy)
            mcq_nll_change = float(mcq_nll - baseline_nll)
            flip_count = count_flips(mcq_eval, baseline_correct)
            importance = classify_importance(flip_count, flip_threshold)

        acc_change = mcq_accuracy - baseline_accuracy

        chunk_results.append({
            'chunk_idx': chunk_idx,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'chunk_size': chunk_end - chunk_start,
            'energy_removed': energy_per_matrix,
            'avg_energy_removed': avg_energy,
            'mcq_entropy': float(mcq_entropy) if not is_nan else float('nan'),
            'mcq_nll': float(mcq_nll) if not is_nan else float('nan'),
            'mcq_accuracy': mcq_accuracy,
            'mcq_entropy_change': mcq_entropy_change,
            'mcq_nll_change': mcq_nll_change,
            'mcq_accuracy_change': acc_change,
            'flip_count': flip_count,
            'importance': importance,
            'mcq_per_sample': mcq_eval,
        })

        if importance == 'critical':
            critical_chunks.append(chunk_idx)
        elif importance == 'important':
            important_chunks.append(chunk_idx)
        else:
            unimportant_chunks.append(chunk_idx)

        # Print progress
        marker = '***' if importance == 'important' else ('!!!' if importance == 'critical' else '   ')
        if is_nan:
            print(f"  {marker} ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"NaN  acc={acc_change*100:+.1f}pp  flips=MAX  → {importance.upper()}")
        else:
            print(f"  {marker} ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"ent={mcq_entropy_change:+.4f}  acc={acc_change*100:+.1f}pp  "
                  f"flips={flip_count:>3}  energy={avg_energy*100:.1f}%  → {importance.upper()}")

        # Restore all matrices
        for mt in matrix_types:
            restore_original_weight(model, layer_idx, original_weights[mt], mt, model_type)

    n_total = len(chunk_results)
    print(f"\n  Summary: {len(important_chunks)} important ({100*len(important_chunks)/n_total:.1f}%), "
          f"{len(unimportant_chunks)} unimportant ({100*len(unimportant_chunks)/n_total:.1f}%), "
          f"{len(critical_chunks)} critical")

    result = {
        'config': {
            'layer': layer_idx,
            'matrix_type': matrix_str,
            'matrix_types_paired': matrix_types,
            'paired': True,
            'model_type': model_type,
            'chunk_size': chunk_size,
            'total_components': total_components,
            'num_chunks': num_chunks,
            'flip_threshold': flip_threshold,
            'experiment': 'exp20_importance_sweep',
        },
        'svd_stats': svd_stats_all,
        'baseline_mcq': {
            'avg_entropy': baseline_entropy,
            'avg_nll': baseline_nll,
            'accuracy': baseline_accuracy,
        },
        'chunk_results': chunk_results,
        'classification': {
            'important': important_chunks,
            'unimportant': unimportant_chunks,
            'critical': critical_chunks,
        },
        'summary': {
            'n_total': n_total,
            'n_important': len(important_chunks),
            'n_unimportant': len(unimportant_chunks),
            'n_critical': len(critical_chunks),
            'important_fraction': len(important_chunks) / n_total if n_total > 0 else 0,
            'mean_flips_important': float(np.mean([c['flip_count'] for c in chunk_results if c['importance'] == 'important'])) if important_chunks else 0.0,
            'mean_flips_unimportant': float(np.mean([c['flip_count'] for c in chunk_results if c['importance'] == 'unimportant'])) if unimportant_chunks else 0.0,
        },
    }

    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"  Saved: {result_path}")
    return result


# =============================================================================
# Summary CSV
# =============================================================================

def generate_summary_csv(output_dir: Path, model_short: str):
    csv_path = output_dir / f"{model_short}_importance_summary.csv"
    rows = []

    for pkl_file in sorted(output_dir.glob(f"{model_short}_layer*.pkl")):
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)

        config = result['config']
        summary = result['summary']
        baseline = result['baseline_mcq']

        rows.append({
            'layer': config['layer'],
            'matrix_type': config['matrix_type'],
            'flip_threshold': config['flip_threshold'],
            'total_components': config['total_components'],
            'num_chunks': config['num_chunks'],
            'baseline_entropy': baseline['avg_entropy'],
            'baseline_accuracy': baseline['accuracy'],
            'n_important': summary['n_important'],
            'n_unimportant': summary['n_unimportant'],
            'n_critical': summary['n_critical'],
            'important_fraction': summary['important_fraction'],
            'mean_flips_important': summary['mean_flips_important'],
            'mean_flips_unimportant': summary['mean_flips_unimportant'],
        })

    if rows:
        rows.sort(key=lambda r: r['layer'])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV saved: {csv_path}")
        print(f"Total completed: {len(rows)} layers")
    else:
        print("No results found.")


# =============================================================================
# Main sweep
# =============================================================================

def run_importance_sweep_paired(
    model_name: str,
    model_type: str = "llama",
    layers: list = None,
    matrix_types: list = None,
    chunk_size: int = 100,
    flip_threshold: int = 5,
    mcq_eval_set_path: str = "data/eval_sets/eval_set_mcq_arc_challenge_500.json",
    device: str = "cuda",
    resume: bool = False,
):
    seed_everything(42)

    if layers is None:
        layers = list(range(32))
    if matrix_types is None:
        matrix_types = MATRIX_GROUPS['mlp']

    matrix_str = "+".join(matrix_types)
    model_short = model_name.split("/")[-1]
    eval_set_stem = Path(mcq_eval_set_path).stem
    output_dir = Path(
        f"results/importance_sweep/{model_short}_paired_{matrix_str}"
        f"_chunk{chunk_size}_k{flip_threshold}_{eval_set_stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    skipped = 0
    for layer in layers:
        result_path = output_dir / f"{model_short}_layer{layer}_{matrix_str}.pkl"
        if resume and result_path.exists():
            skipped += 1
            continue
        jobs.append(layer)

    print("=" * 70)
    print("IMPORTANCE SWEEP (Experiment 20)")
    print("=" * 70)
    print(f"Model:           {model_name}")
    print(f"Layers:          {layers}")
    print(f"Paired matrices: {matrix_types}")
    print(f"Chunk size:      {chunk_size}")
    print(f"Flip threshold:  k={flip_threshold} questions")
    print(f"Eval set:        {mcq_eval_set_path}")
    print(f"Output:          {output_dir}")
    print(f"Jobs:            {len(jobs)} to run, {skipped} already done")
    print("=" * 70)

    if not jobs:
        print("\nAll jobs complete. Generating summary...")
        generate_summary_csv(output_dir, model_short)
        return

    # Load eval set
    mcq_samples = load_mcq_eval_set(mcq_eval_set_path)

    # Save config
    with open(output_dir / 'sweep_config.json', 'w') as f:
        json.dump({
            'experiment': 'exp20_importance_sweep',
            'model_name': model_name,
            'model_type': model_type,
            'layers': layers,
            'matrix_types': matrix_types,
            'paired': True,
            'chunk_size': chunk_size,
            'flip_threshold': flip_threshold,
            'mcq_eval_set_path': mcq_eval_set_path,
            'num_mcq_samples': len(mcq_samples),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        }, f, indent=2)

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # Baseline evaluation
    print("\nEvaluating baseline...")
    baseline_results = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
    baseline_entropy = float(np.nanmean([r['entropy'] for r in baseline_results]))
    baseline_nll = float(np.nanmean([r['nll'] for r in baseline_results]))
    baseline_accuracy = float(np.mean([r['is_correct'] for r in baseline_results]))
    baseline_correct = {r['sample_id']: r['is_correct'] for r in baseline_results}

    print(f"  Baseline: {baseline_accuracy*100:.1f}% accuracy, "
          f"{baseline_entropy:.4f} entropy, {baseline_nll:.4f} NLL")
    print(f"  Baseline correct: {sum(baseline_correct.values())}/{len(baseline_correct)} questions")

    # Save baseline (with per_sample this time)
    with open(output_dir / 'baseline.pkl', 'wb') as f:
        pickle.dump({
            'mcq_entropy': baseline_entropy,
            'mcq_nll': baseline_nll,
            'mcq_accuracy': baseline_accuracy,
            'mcq_per_sample': baseline_results,
            'baseline_correct': baseline_correct,
        }, f)

    # Run layers
    for job_idx, layer in enumerate(jobs):
        print(f"\n[Job {job_idx+1}/{len(jobs)}] Layer {layer}")
        run_layer_paired(
            model=model,
            tokenizer=tokenizer,
            mcq_samples=mcq_samples,
            baseline_results=baseline_results,
            baseline_correct=baseline_correct,
            baseline_entropy=baseline_entropy,
            baseline_nll=baseline_nll,
            baseline_accuracy=baseline_accuracy,
            layer_idx=layer,
            matrix_types=matrix_types,
            model_type=model_type,
            chunk_size=chunk_size,
            flip_threshold=flip_threshold,
            output_dir=output_dir,
            model_short=model_short,
            device=device,
        )

    print("\n" + "=" * 70)
    print("IMPORTANCE SWEEP COMPLETE")
    print("=" * 70)
    generate_summary_csv(output_dir, model_short)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp 20: Importance Sweep")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama")
    parser.add_argument("--layers", type=str, default="0-31")
    parser.add_argument("--matrices", type=str, default="mlp",
                        choices=list(MATRIX_GROUPS.keys()) + ["custom"])
    parser.add_argument("--matrix-list", type=str, nargs="+",
                        choices=["mlp_in", "mlp_out", "gate_proj",
                                 "attn_q", "attn_k", "attn_v", "attn_o"])
    parser.add_argument("--paired", action="store_true", default=True,
                        help="Paired matrix removal (default: True)")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--flip-threshold", type=int, default=5,
                        help="Min question flips to classify as important (default: 5)")
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_arc_challenge_500.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    num_layers = 32
    layers = parse_layer_spec(args.layers, num_layers)

    if args.matrices == "custom":
        if not args.matrix_list:
            parser.error("--matrix-list required when --matrices custom")
        matrix_types = args.matrix_list
    else:
        matrix_types = MATRIX_GROUPS[args.matrices]

    run_importance_sweep_paired(
        model_name=args.model,
        model_type=args.model_type,
        layers=layers,
        matrix_types=matrix_types,
        chunk_size=args.chunk_size,
        flip_threshold=args.flip_threshold,
        mcq_eval_set_path=args.eval_set,
        device=args.device,
        resume=args.resume,
    )
