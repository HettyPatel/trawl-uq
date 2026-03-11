"""
Full Layer Sweep - Spectral Anatomy of Factual Noise

Automates the SVD chunk removal experiment across ALL layers and multiple
matrix types (MLP and/or attention). Builds on experiment 14's core logic
but adds:
  - Multi-layer sweep (loops over layers automatically)
  - Matrix group presets (--matrices mlp, attn, all, or custom list)
  - Per-layer result saving (can resume interrupted runs)
  - Skip already-completed layer/matrix combos
  - Summary CSV output for easy plotting

Each layer+matrix combination is run independently: SVD decompose the matrix,
remove chunks one at a time, measure MCQ entropy and accuracy, classify chunks,
run cumulative noise removal, save results.

Usage:
    # Full MLP sweep, all 32 layers of Llama-2-7b
    python experiments/15_full_layer_sweep.py --layers 0-31 --matrices mlp

    # Full attention sweep
    python experiments/15_full_layer_sweep.py --layers 0-31 --matrices attn

    # Everything (MLP + attention + gate)
    python experiments/15_full_layer_sweep.py --layers 0-31 --matrices all

    # Custom matrix selection
    python experiments/15_full_layer_sweep.py --layers 0-31 --matrices custom --matrix-list attn_v attn_o

    # Specific layers only
    python experiments/15_full_layer_sweep.py --layers 0,1,4,12,31 --matrices mlp

    # Resume interrupted run (skips completed layers)
    python experiments/15_full_layer_sweep.py --layers 0-31 --matrices mlp --resume

    # Quick test
    python experiments/15_full_layer_sweep.py --layers 31 --matrices mlp --test
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
    compute_energy_retention,
    get_svd_stats,
)


# =============================================================================
# Matrix group definitions
# =============================================================================

MATRIX_GROUPS = {
    'mlp': ['mlp_in', 'mlp_out'],
    'mlp_full': ['mlp_in', 'mlp_out', 'gate_proj'],
    'attn': ['attn_q', 'attn_k', 'attn_v', 'attn_o'],
    'attn_value': ['attn_v', 'attn_o'],
    'all': ['mlp_in', 'mlp_out', 'gate_proj', 'attn_q', 'attn_k', 'attn_v', 'attn_o'],
}


# =============================================================================
# Helper functions
# =============================================================================

def parse_layer_spec(spec: str, num_layers: int = 32) -> list:
    """
    Parse layer specification string into list of layer indices.

    Supports:
        "0-31"      → [0, 1, ..., 31]
        "0,1,4,31"  → [0, 1, 4, 31]
        "0-4,12,31" → [0, 1, 2, 3, 4, 12, 31]
        "31"        → [31]
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


def get_original_weight(model, layer_idx, matrix_type, model_type):
    """Get a clone of the original weight matrix from the model."""
    if model_type == "llama":
        weight_map = {
            'mlp_in': lambda: model.model.layers[layer_idx].mlp.up_proj.weight.data,
            'mlp_out': lambda: model.model.layers[layer_idx].mlp.down_proj.weight.data,
            'gate_proj': lambda: model.model.layers[layer_idx].mlp.gate_proj.weight.data,
            'attn_q': lambda: model.model.layers[layer_idx].self_attn.q_proj.weight.data,
            'attn_k': lambda: model.model.layers[layer_idx].self_attn.k_proj.weight.data,
            'attn_v': lambda: model.model.layers[layer_idx].self_attn.v_proj.weight.data,
            'attn_o': lambda: model.model.layers[layer_idx].self_attn.o_proj.weight.data,
        }
    elif model_type == "gpt2":
        weight_map = {
            'mlp_in': lambda: model.transformer.h[layer_idx].mlp.c_fc.weight.data,
            'mlp_out': lambda: model.transformer.h[layer_idx].mlp.c_proj.weight.data,
        }
    elif model_type == "gptj":
        weight_map = {
            'mlp_in': lambda: model.transformer.h[layer_idx].mlp.fc_in.weight.data,
            'mlp_out': lambda: model.transformer.h[layer_idx].mlp.fc_out.weight.data,
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if matrix_type not in weight_map:
        raise ValueError(f"Unknown matrix_type '{matrix_type}' for model_type '{model_type}'")
    return weight_map[matrix_type]().clone()


def reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end):
    """Reconstruct weight matrix with a chunk of singular values dropped."""
    keep_indices = torch.cat([
        torch.arange(0, chunk_start, device=S.device),
        torch.arange(chunk_end, len(S), device=S.device)
    ])
    U_kept = U[:, keep_indices]
    S_kept = S[keep_indices]
    Vh_kept = Vh[keep_indices, :]
    W_modified = reconstruct_from_svd(U_kept, S_kept, Vh_kept)
    total_energy = torch.sum(S ** 2).item()
    removed_energy = torch.sum(S[chunk_start:chunk_end] ** 2).item()
    energy_removed = removed_energy / total_energy if total_energy > 0 else 0.0
    return W_modified, energy_removed


def reconstruct_with_chunks_removed(U, S, Vh, chunk_ranges):
    """Reconstruct weight matrix with multiple chunks removed."""
    remove_indices = set()
    for start, end in chunk_ranges:
        remove_indices.update(range(start, end))
    keep_indices = sorted(set(range(len(S))) - remove_indices)
    keep_indices = torch.tensor(keep_indices, device=S.device, dtype=torch.long)
    U_kept = U[:, keep_indices]
    S_kept = S[keep_indices]
    Vh_kept = Vh[keep_indices, :]
    W_modified = reconstruct_from_svd(U_kept, S_kept, Vh_kept)
    total_energy = torch.sum(S ** 2).item()
    remove_indices_t = torch.tensor(sorted(remove_indices), device=S.device, dtype=torch.long)
    removed_energy = torch.sum(S[remove_indices_t] ** 2).item()
    energy_removed = removed_energy / total_energy if total_energy > 0 else 0.0
    return W_modified, energy_removed


def get_result_path(output_dir: Path, model_short: str, layer: int, matrix_type: str) -> Path:
    """Get the result file path for a specific layer+matrix combo."""
    return output_dir / f"{model_short}_layer{layer}_{matrix_type}.pkl"


def is_completed(output_dir: Path, model_short: str, layer: int, matrix_type: str) -> bool:
    """Check if a layer+matrix combo has already been completed."""
    result_path = get_result_path(output_dir, model_short, layer, matrix_type)
    return result_path.exists()


# =============================================================================
# Single layer+matrix experiment
# =============================================================================

def run_single_layer_matrix(
    model, tokenizer, mcq_samples,
    baseline_mcq, baseline_mcq_entropy, baseline_mcq_nll, baseline_mcq_accuracy,
    layer_idx: int, matrix_type: str,
    model_type: str, chunk_size: int,
    output_dir: Path, model_short: str,
    device: str = "cuda",
):
    """
    Run SVD chunk removal for a single layer + single matrix type.
    Saves results to a per-layer-matrix pickle file.
    """
    result_path = get_result_path(output_dir, model_short, layer_idx, matrix_type)

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} | {matrix_type}")
    print(f"{'='*60}")

    # Decompose
    original_weight = get_original_weight(model, layer_idx, matrix_type, model_type)
    U, S, Vh = decompose_weight_svd(original_weight, device)
    stats = get_svd_stats(original_weight, device)
    total_components = len(S)
    num_chunks = (total_components + chunk_size - 1) // chunk_size

    print(f"  Shape: {stats['shape']}, SVs: {total_components}, "
          f"Effective rank: {stats['effective_rank']:.1f}, Chunks: {num_chunks}")

    # Chunk removal loop
    chunk_results = []
    for chunk_idx in tqdm(range(num_chunks), desc=f"L{layer_idx}/{matrix_type}", leave=False):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_components)

        W_modified, energy_removed = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
        update_layer_with_svd(model, layer_idx, W_modified, matrix_type, model_type)

        mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        mcq_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
        mcq_nll = np.nanmean([r['nll'] for r in mcq_eval])
        mcq_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

        # Handle NaN (model produces NaN logits when destruction is too severe)
        if np.isnan(mcq_entropy):
            mcq_entropy = float('nan')
            mcq_entropy_change = float('nan')
            mcq_nll_change = float('nan')
        else:
            mcq_entropy_change = mcq_entropy - baseline_mcq_entropy
            mcq_nll_change = mcq_nll - baseline_mcq_nll

        acc_change = mcq_accuracy - baseline_mcq_accuracy

        chunk_results.append({
            'chunk_idx': chunk_idx,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'chunk_size': chunk_end - chunk_start,
            'energy_removed': energy_removed,
            'mcq_entropy': mcq_entropy,
            'mcq_nll': mcq_nll,
            'mcq_accuracy': mcq_accuracy,
            'mcq_entropy_change': mcq_entropy_change,
            'mcq_nll_change': mcq_nll_change,
            'mcq_accuracy_change': acc_change,
            'mcq_per_sample': mcq_eval,
        })

        # Quick classification label for printing
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

        # Print every chunk so progress is visible
        if np.isnan(mcq_entropy_change):
            print(f"    ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"ent=NaN  acc={acc_change*100:+.1f}pp  energy={energy_removed*100:.1f}%  → {label}")
        else:
            print(f"    ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"ent={mcq_entropy_change:+.4f}  acc={acc_change*100:+.1f}pp  "
                  f"energy={energy_removed*100:.1f}%  → {label}")

        # Restore
        restore_original_weight(model, layer_idx, original_weight, matrix_type, model_type)

    # Classify chunks
    # NaN = model broke when chunk removed = critical signal (most important components)
    true_noise = []
    confident_wrong = []
    true_signal = []
    uncertain_right = []
    critical_signal = []  # NaN logits — removing these breaks the model entirely

    for c in chunk_results:
        if np.isnan(c['mcq_entropy_change']):
            critical_signal.append(c['chunk_idx'])
            continue
        if c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] > 0:
            true_noise.append(c['chunk_idx'])
        elif c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] <= 0:
            confident_wrong.append(c['chunk_idx'])
        elif c['mcq_entropy_change'] >= 0 and c['mcq_accuracy_change'] <= 0:
            true_signal.append(c['chunk_idx'])
        else:
            uncertain_right.append(c['chunk_idx'])

    crit_str = f", {len(critical_signal)} critical" if critical_signal else ""
    print(f"  Classification: {len(true_noise)} noise, {len(true_signal)} signal, "
          f"{len(confident_wrong)} conf_wrong, {len(uncertain_right)} unc_right{crit_str}")

    # Cumulative noise removal
    cumulative_results = []
    if true_noise:
        noise_sorted = sorted(true_noise)
        chunk_range_map = {c['chunk_idx']: (c['chunk_start'], c['chunk_end']) for c in chunk_results}
        stacked_ranges = []

        for i, ci in enumerate(noise_sorted):
            stacked_ranges.append(chunk_range_map[ci])
            W_modified, energy_removed = reconstruct_with_chunks_removed(U, S, Vh, stacked_ranges)
            update_layer_with_svd(model, layer_idx, W_modified, matrix_type, model_type)

            mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
            cum_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
            cum_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

            # Handle NaN in cumulative removal (rare but possible if too much removed)
            if np.isnan(cum_entropy):
                cum_entropy_change = float('nan')
            else:
                cum_entropy_change = cum_entropy - baseline_mcq_entropy

            cumulative_results.append({
                'step': i,
                'num_chunks_removed': i + 1,
                'chunk_indices_removed': list(noise_sorted[:i+1]),
                'ranges_removed': list(stacked_ranges),
                'energy_removed': energy_removed,
                'mcq_entropy': cum_entropy,
                'mcq_accuracy': cum_accuracy,
                'mcq_entropy_change': cum_entropy_change,
                'mcq_accuracy_change': cum_accuracy - baseline_mcq_accuracy,
            })

            restore_original_weight(model, layer_idx, original_weight, matrix_type, model_type)

    # Compute summary stats (excluding NaN chunks)
    valid_chunks = [c for c in chunk_results if not np.isnan(c['mcq_entropy_change'])]
    sum_entropy_change = sum(c['mcq_entropy_change'] for c in valid_chunks)
    sum_accuracy_change = sum(c['mcq_accuracy_change'] for c in valid_chunks)
    num_classifiable = len(valid_chunks)
    # Include critical_signal as signal for fraction calculation
    total_signal = len(true_signal) + len(critical_signal)
    noise_fraction = len(true_noise) / len(chunk_results) if chunk_results else 0

    # Save
    result = {
        'config': {
            'layer': layer_idx,
            'matrix_type': matrix_type,
            'model_type': model_type,
            'chunk_size': chunk_size,
            'total_components': total_components,
            'num_chunks': num_chunks,
        },
        'svd_stats': stats,
        'baseline_mcq': {
            'avg_entropy': baseline_mcq_entropy,
            'avg_nll': baseline_mcq_nll,
            'accuracy': baseline_mcq_accuracy,
        },
        'chunk_results': chunk_results,
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
            'signal_fraction': total_signal / len(chunk_results) if chunk_results else 0,
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

    print(f"  Saved: {result_path}")
    print(f"  Net entropy: {sum_entropy_change:+.4f}, Net accuracy: {sum_accuracy_change*100:+.1f}pp, "
          f"Noise fraction: {noise_fraction:.1%}")

    return result


# =============================================================================
# Paired matrix removal (remove same chunk from multiple matrices at once)
# =============================================================================

def run_single_layer_paired(
    model, tokenizer, mcq_samples,
    baseline_mcq, baseline_mcq_entropy, baseline_mcq_nll, baseline_mcq_accuracy,
    layer_idx: int, matrix_types: list,
    model_type: str, chunk_size: int,
    output_dir: Path, model_short: str,
    device: str = "cuda",
):
    """
    Run SVD chunk removal for a single layer with PAIRED matrix removal.
    Removes the same chunk index from ALL specified matrices simultaneously.
    This amplifies the perturbation and gives cleaner noise/signal separation.
    """
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

    # Use the minimum component count across all matrices
    total_components = min(len(decompositions[mt]['S']) for mt in matrix_types)
    num_chunks = (total_components + chunk_size - 1) // chunk_size
    print(f"  Chunks: {num_chunks} (based on min SVs={total_components})")

    # Chunk removal loop — remove same chunk from ALL matrices
    chunk_results = []
    for chunk_idx in tqdm(range(num_chunks), desc=f"L{layer_idx}/{matrix_str}", leave=False):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_components)

        # Remove chunk from ALL matrices simultaneously
        energy_removed_per_matrix = {}
        for mt in matrix_types:
            U = decompositions[mt]['U']
            S = decompositions[mt]['S']
            Vh = decompositions[mt]['Vh']
            W_modified, e_removed = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
            energy_removed_per_matrix[mt] = e_removed
            update_layer_with_svd(model, layer_idx, W_modified, mt, model_type)

        avg_energy_removed = np.mean(list(energy_removed_per_matrix.values()))

        mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        mcq_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
        mcq_nll = np.nanmean([r['nll'] for r in mcq_eval])
        mcq_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

        # Handle NaN
        if np.isnan(mcq_entropy):
            mcq_entropy_change = float('nan')
            mcq_nll_change = float('nan')
        else:
            mcq_entropy_change = mcq_entropy - baseline_mcq_entropy
            mcq_nll_change = mcq_nll - baseline_mcq_nll

        acc_change = mcq_accuracy - baseline_mcq_accuracy

        chunk_results.append({
            'chunk_idx': chunk_idx,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'chunk_size': chunk_end - chunk_start,
            'energy_removed': energy_removed_per_matrix,
            'avg_energy_removed': avg_energy_removed,
            'mcq_entropy': mcq_entropy,
            'mcq_nll': mcq_nll,
            'mcq_accuracy': mcq_accuracy,
            'mcq_entropy_change': mcq_entropy_change,
            'mcq_nll_change': mcq_nll_change,
            'mcq_accuracy_change': acc_change,
            'mcq_per_sample': mcq_eval,
        })

        # Print label
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

        if np.isnan(mcq_entropy_change):
            print(f"    ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"ent=NaN  acc={acc_change*100:+.1f}pp  energy={avg_energy_removed*100:.1f}%  → {label}")
        else:
            print(f"    ch{chunk_idx:>2} [{chunk_start:>4}:{chunk_end:>4}] "
                  f"ent={mcq_entropy_change:+.4f}  acc={acc_change*100:+.1f}pp  "
                  f"energy={avg_energy_removed*100:.1f}%  → {label}")

        # Restore ALL matrices
        for mt in matrix_types:
            restore_original_weight(model, layer_idx, original_weights[mt], mt, model_type)

    # Classify chunks
    true_noise = []
    confident_wrong = []
    true_signal = []
    uncertain_right = []
    critical_signal = []

    for c in chunk_results:
        if np.isnan(c['mcq_entropy_change']):
            critical_signal.append(c['chunk_idx'])
            continue
        if c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] > 0:
            true_noise.append(c['chunk_idx'])
        elif c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] <= 0:
            confident_wrong.append(c['chunk_idx'])
        elif c['mcq_entropy_change'] >= 0 and c['mcq_accuracy_change'] <= 0:
            true_signal.append(c['chunk_idx'])
        else:
            uncertain_right.append(c['chunk_idx'])

    crit_str = f", {len(critical_signal)} critical" if critical_signal else ""
    print(f"  Classification: {len(true_noise)} noise, {len(true_signal)} signal, "
          f"{len(confident_wrong)} conf_wrong, {len(uncertain_right)} unc_right{crit_str}")

    # Cumulative noise removal (remove from ALL matrices simultaneously)
    cumulative_results = []
    if true_noise:
        noise_sorted = sorted(true_noise)
        chunk_range_map = {c['chunk_idx']: (c['chunk_start'], c['chunk_end']) for c in chunk_results}
        stacked_ranges = []

        for i, ci in enumerate(noise_sorted):
            stacked_ranges.append(chunk_range_map[ci])

            # Remove stacked chunks from ALL matrices
            for mt in matrix_types:
                U = decompositions[mt]['U']
                S = decompositions[mt]['S']
                Vh = decompositions[mt]['Vh']
                W_modified, e_removed = reconstruct_with_chunks_removed(U, S, Vh, stacked_ranges)
                update_layer_with_svd(model, layer_idx, W_modified, mt, model_type)

            mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
            cum_entropy = np.nanmean([r['entropy'] for r in mcq_eval])
            cum_accuracy = np.mean([r['is_correct'] for r in mcq_eval])

            if np.isnan(cum_entropy):
                cum_entropy_change = float('nan')
            else:
                cum_entropy_change = cum_entropy - baseline_mcq_entropy

            cumulative_results.append({
                'step': i,
                'num_chunks_removed': i + 1,
                'chunk_indices_removed': list(noise_sorted[:i+1]),
                'ranges_removed': list(stacked_ranges),
                'mcq_entropy': cum_entropy,
                'mcq_accuracy': cum_accuracy,
                'mcq_entropy_change': cum_entropy_change,
                'mcq_accuracy_change': cum_accuracy - baseline_mcq_accuracy,
            })

            # Restore ALL matrices
            for mt in matrix_types:
                restore_original_weight(model, layer_idx, original_weights[mt], mt, model_type)

    # Summary stats
    valid_chunks = [c for c in chunk_results if not np.isnan(c['mcq_entropy_change'])]
    sum_entropy_change = sum(c['mcq_entropy_change'] for c in valid_chunks)
    sum_accuracy_change = sum(c['mcq_accuracy_change'] for c in valid_chunks)
    total_signal = len(true_signal) + len(critical_signal)
    noise_fraction = len(true_noise) / len(chunk_results) if chunk_results else 0

    result = {
        'config': {
            'layer': layer_idx,
            'matrix_type': matrix_str,  # e.g. "mlp_in+mlp_out"
            'matrix_types_paired': matrix_types,
            'paired': True,
            'model_type': model_type,
            'chunk_size': chunk_size,
            'total_components': total_components,
            'num_chunks': num_chunks,
        },
        'svd_stats': svd_stats_all,
        'baseline_mcq': {
            'avg_entropy': baseline_mcq_entropy,
            'avg_nll': baseline_mcq_nll,
            'accuracy': baseline_mcq_accuracy,
        },
        'chunk_results': chunk_results,
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
            'signal_fraction': total_signal / len(chunk_results) if chunk_results else 0,
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

    print(f"  Saved: {result_path}")
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
        baseline = result['baseline_mcq']

        rows.append({
            'layer': config['layer'],
            'matrix_type': config['matrix_type'],
            'total_components': config['total_components'],
            'num_chunks': config['num_chunks'],
            'chunk_size': config['chunk_size'],
            'baseline_entropy': baseline['avg_entropy'],
            'baseline_accuracy': baseline['accuracy'],
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
        rows.sort(key=lambda r: (r['layer'], r['matrix_type']))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV saved to: {csv_path}")
        print(f"Total completed: {len(rows)} layer-matrix combinations")
    else:
        print("No results found to summarize.")


# =============================================================================
# Main sweep
# =============================================================================

def run_full_sweep(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    layers: list = None,
    matrix_types: list = None,
    chunk_size: int = 100,
    mcq_eval_set_path: str = "data/eval_sets/eval_set_mcq_arc_challenge_500.json",
    device: str = "cuda",
    resume: bool = False,
):
    """
    Run full layer sweep across specified layers and matrix types.
    """
    seed_everything(42)

    if layers is None:
        layers = list(range(32))
    if matrix_types is None:
        matrix_types = MATRIX_GROUPS['mlp']

    model_short = model_name.split("/")[-1]
    # Include eval set name in output dir to avoid overwrites
    eval_set_stem = Path(mcq_eval_set_path).stem  # e.g. "eval_set_mcq_arc_challenge_500"
    output_dir = Path(f"results/full_sweep/{model_short}_chunk{chunk_size}_{eval_set_stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    skipped = 0
    for layer in layers:
        for mt in matrix_types:
            if resume and is_completed(output_dir, model_short, layer, mt):
                skipped += 1
                continue
            jobs.append((layer, mt))

    total_jobs = len(jobs) + skipped
    print("=" * 70)
    print("FULL LAYER SWEEP - Spectral Anatomy of Factual Noise")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"Matrices: {matrix_types}")
    print(f"Chunk size: {chunk_size}")
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
        'matrix_types': matrix_types,
        'chunk_size': chunk_size,
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

    # Baseline evaluation (once for all jobs)
    print("\nEvaluating baseline MCQ...")
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
    for job_idx, (layer, mt) in enumerate(jobs):
        print(f"\n[Job {job_idx+1}/{len(jobs)}] Layer {layer}, {mt}")

        run_single_layer_matrix(
            model=model,
            tokenizer=tokenizer,
            mcq_samples=mcq_samples,
            baseline_mcq=baseline_mcq,
            baseline_mcq_entropy=baseline_mcq_entropy,
            baseline_mcq_nll=baseline_mcq_nll,
            baseline_mcq_accuracy=baseline_mcq_accuracy,
            layer_idx=layer,
            matrix_type=mt,
            model_type=model_type,
            chunk_size=chunk_size,
            output_dir=output_dir,
            model_short=model_short,
            device=device,
        )

    # Generate summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    generate_summary_csv(output_dir, model_short)


def run_full_sweep_paired(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    layers: list = None,
    matrix_types: list = None,
    chunk_size: int = 100,
    mcq_eval_set_path: str = "data/eval_sets/eval_set_mcq_arc_challenge_500.json",
    device: str = "cuda",
    resume: bool = False,
):
    """
    Run full layer sweep with PAIRED matrix removal.
    For each layer, removes the same chunk from all specified matrices simultaneously.
    One job per layer (not per layer-matrix combo).
    """
    seed_everything(42)

    if layers is None:
        layers = list(range(32))
    if matrix_types is None:
        matrix_types = MATRIX_GROUPS['mlp']

    matrix_str = "+".join(matrix_types)
    model_short = model_name.split("/")[-1]
    # Include eval set name in output dir to avoid overwrites
    eval_set_stem = Path(mcq_eval_set_path).stem  # e.g. "eval_set_mcq_arc_challenge_500"
    output_dir = Path(f"results/full_sweep/{model_short}_paired_{matrix_str}_chunk{chunk_size}_{eval_set_stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job list (one per layer)
    jobs = []
    skipped = 0
    for layer in layers:
        result_path = output_dir / f"{model_short}_layer{layer}_{matrix_str}.pkl"
        if resume and result_path.exists():
            skipped += 1
            continue
        jobs.append(layer)

    total_jobs = len(jobs) + skipped
    print("=" * 70)
    print("FULL LAYER SWEEP (PAIRED) - Spectral Anatomy of Factual Noise")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print(f"Paired matrices: {matrix_types} (removed together)")
    print(f"Chunk size: {chunk_size}")
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
        'matrix_types': matrix_types,
        'paired': True,
        'chunk_size': chunk_size,
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

    # Baseline evaluation
    print("\nEvaluating baseline MCQ...")
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

    # Run jobs (one per layer)
    for job_idx, layer in enumerate(jobs):
        print(f"\n[Job {job_idx+1}/{len(jobs)}] Layer {layer}, paired {matrix_str}")

        run_single_layer_paired(
            model=model,
            tokenizer=tokenizer,
            mcq_samples=mcq_samples,
            baseline_mcq=baseline_mcq,
            baseline_mcq_entropy=baseline_mcq_entropy,
            baseline_mcq_nll=baseline_mcq_nll,
            baseline_mcq_accuracy=baseline_mcq_accuracy,
            layer_idx=layer,
            matrix_types=matrix_types,
            model_type=model_type,
            chunk_size=chunk_size,
            output_dir=output_dir,
            model_short=model_short,
            device=device,
        )

    # Generate summary
    print("\n" + "=" * 70)
    print("PAIRED SWEEP COMPLETE")
    print("=" * 70)
    generate_summary_csv(output_dir, model_short)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full Layer Sweep - Spectral Anatomy of Factual Noise"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama",
                        choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--layers", type=str, default="0-31",
                        help="Layer specification: '0-31', '0,1,4,31', '0-4,12,31'")
    parser.add_argument("--matrices", type=str, default="mlp",
                        choices=list(MATRIX_GROUPS.keys()) + ["custom"],
                        help="Matrix group preset or 'custom' with --matrix-list")
    parser.add_argument("--matrix-list", type=str, nargs="+",
                        choices=["mlp_in", "mlp_out", "gate_proj",
                                 "attn_q", "attn_k", "attn_v", "attn_o"],
                        help="Custom matrix list (used when --matrices custom)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of singular values per chunk (default: 100)")
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_arc_challenge_500.json",
                        help="Path to MCQ evaluation set JSON")
    parser.add_argument("--paired", action="store_true",
                        help="Remove same chunk from all matrices simultaneously (paired mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed layer/matrix combos")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: chunk-size=1024, single layer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")

    args = parser.parse_args()

    # Determine number of layers from model
    num_layers = 32  # default for Llama-2-7b and Llama-3-8B
    if "gpt2" in args.model.lower() and "xl" not in args.model.lower():
        num_layers = 12
    elif "gpt-j" in args.model.lower() or "gptj" in args.model.lower():
        num_layers = 28

    layers = parse_layer_spec(args.layers, num_layers)

    # Determine matrix types
    if args.matrices == "custom":
        if not args.matrix_list:
            parser.error("--matrix-list required when --matrices custom")
        matrix_types = args.matrix_list
    else:
        matrix_types = MATRIX_GROUPS[args.matrices]

    chunk_size = 1024 if args.test else args.chunk_size
    if args.test:
        print("TEST MODE: chunk-size=1024")

    if args.paired:
        run_full_sweep_paired(
            model_name=args.model,
            model_type=args.model_type,
            layers=layers,
            matrix_types=matrix_types,
            chunk_size=chunk_size,
            mcq_eval_set_path=args.eval_set,
            device=args.device,
            resume=args.resume,
        )
    else:
        run_full_sweep(
            model_name=args.model,
            model_type=args.model_type,
            layers=layers,
            matrix_types=matrix_types,
            chunk_size=chunk_size,
            mcq_eval_set_path=args.eval_set,
            device=args.device,
            resume=args.resume,
        )
