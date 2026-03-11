"""
Experiment 26: Error Propagation Test

Tests whether early-layer SVD chunk importance is due to error propagation
vs genuine local importance.

Approach: Remove a chunk from layer L, then measure how much the hidden states
(residual stream) diverge from baseline at every subsequent layer.

If error propagation causes the importance:
  - Divergence grows monotonically through layers L+1 → 31
  - The chunk's local effect is small but amplifies downstream

If the chunk is genuinely important at layer L:
  - Divergence spikes at layer L but stays relatively flat after
  - The chunk contributes something unique that later layers don't amplify

We measure:
  - Cosine similarity between baseline and modified hidden states at each layer
  - L2 distance (normalized by baseline norm) at each layer
  - Final logit KL divergence

Usage:
    python experiments/26_error_propagation.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --importance-dir results/importance_sweep/... \\
        --target-layers 0-20 --paired \\
        --num-questions 50 \\
        --output-dir results/error_propagation/
"""

import sys
sys.path.append('.')

import csv
import json
import torch
import pickle
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
from collections import defaultdict

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.decomposition.svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
)


def load_eval_set(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    return data['samples']


def load_importance_data(importance_dir: Path):
    """Load importance data and return {layer: {chunk_idx: importance_label}}."""
    pkls = sorted(glob.glob(str(importance_dir / "*.pkl")))
    pkls = [p for p in pkls if 'baseline' not in Path(p).name]
    importance_data = {}
    for p in pkls:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        layer = data['config']['layer']
        chunk_map = {}
        for c in data['chunk_results']:
            chunk_map[c['chunk_idx']] = {
                'importance': c['importance'],
                'flip_count': c['flip_count'],
            }
        importance_data[layer] = chunk_map
    print(f"Loaded importance data: {len(importance_data)} layers")
    return importance_data


def get_first_chunk_by_importance(importance_data, layer_idx, label):
    """Find the first chunk index with the given importance label for a layer."""
    if layer_idx not in importance_data:
        return None
    chunks = importance_data[layer_idx]
    matching = [ci for ci, info in chunks.items() if info['importance'] == label]
    if not matching:
        return None
    return min(matching)


def parse_layer_spec(spec: str) -> list:
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


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


class ResidualStreamHook:
    """Captures the residual stream (hidden states) after each transformer layer."""

    def __init__(self):
        self.hidden_states = {}
        self.handles = []

    def make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hidden_states[layer_idx] = output[0][:, -1, :].detach()
            else:
                self.hidden_states[layer_idx] = output[:, -1, :].detach()
        return hook_fn

    def register_all(self, model, num_layers):
        for i in range(num_layers):
            handle = model.model.layers[i].register_forward_hook(self.make_hook(i))
            self.handles.append(handle)

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hidden_states = {}


def compute_divergence(baseline_hidden, modified_hidden):
    """Compute cosine similarity and relative L2 distance between two hidden states."""
    bl = baseline_hidden.float().squeeze(0)
    md = modified_hidden.float().squeeze(0)

    cos_sim = torch.dot(bl, md) / (torch.norm(bl) * torch.norm(md) + 1e-10)
    cos_sim = cos_sim.item()
    l2_dist = torch.norm(bl - md).item()
    bl_norm = torch.norm(bl).item()
    relative_l2 = l2_dist / bl_norm if bl_norm > 0 else 0.0

    return {
        'cosine_sim': cos_sim,
        'l2_dist': l2_dist,
        'relative_l2': relative_l2,
    }


def compute_kl_divergence(baseline_logits, modified_logits):
    """KL(baseline || modified) on the final logits."""
    bl = baseline_logits.float()
    md = modified_logits.float()
    p = torch.softmax(bl, dim=0)
    log_p = torch.log_softmax(bl, dim=0)
    log_q = torch.log_softmax(md, dim=0)
    kl = torch.sum(p * (log_p - log_q)).item()
    return kl


def run_one_removal(model, tokenizer, samples, target_layer, chunk_idx, chunk_size,
                    matrix_types, decompositions, original_weights, num_layers, device):
    """Run chunk removal for one layer+chunk combo, return per-question divergence data."""

    # Prepare modified weights
    modified_weights = {}
    energy_removed_all = {}
    for mt in matrix_types:
        U = decompositions[target_layer][mt]['U']
        S = decompositions[target_layer][mt]['S']
        Vh = decompositions[target_layer][mt]['Vh']
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(S))
        W_mod, energy_rem = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
        modified_weights[mt] = W_mod
        energy_removed_all[mt] = energy_rem

    per_question_results = []

    # Map answer letters to token IDs for correctness checking
    letter_token_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[-1]

    for q_idx, sample in enumerate(samples):
        prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
        correct_letter = sample.get('correct_letter', None)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline run
        hook_bl = ResidualStreamHook()
        hook_bl.register_all(model, num_layers)
        with torch.no_grad():
            outputs_bl = model(**inputs)
        baseline_logits = outputs_bl.logits[0, -1, :].detach()
        baseline_hidden = {k: v.clone() for k, v in hook_bl.hidden_states.items()}
        hook_bl.remove_all()

        # Check baseline correctness (MCQ: which letter has highest logit?)
        baseline_correct = False
        if correct_letter and letter_token_ids:
            letter_logits = {l: baseline_logits[tid].item() for l, tid in letter_token_ids.items()}
            predicted_letter = max(letter_logits, key=letter_logits.get)
            baseline_correct = (predicted_letter == correct_letter)

        # Modified run
        for mt in matrix_types:
            update_layer_with_svd(model, target_layer, modified_weights[mt], mt, 'llama')

        hook_md = ResidualStreamHook()
        hook_md.register_all(model, num_layers)
        with torch.no_grad():
            outputs_md = model(**inputs)
        modified_logits = outputs_md.logits[0, -1, :].detach()
        modified_hidden = {k: v.clone() for k, v in hook_md.hidden_states.items()}
        hook_md.remove_all()

        # Restore
        for mt in matrix_types:
            restore_original_weight(model, target_layer,
                                    original_weights[target_layer][mt], mt, 'llama')

        # Compute divergence at each layer
        layer_divergences = {}
        for li in range(num_layers):
            if li in baseline_hidden and li in modified_hidden:
                layer_divergences[li] = compute_divergence(baseline_hidden[li], modified_hidden[li])

        kl = compute_kl_divergence(baseline_logits, modified_logits)
        bl_top1 = torch.argmax(baseline_logits).item()
        md_top1 = torch.argmax(modified_logits).item()

        per_question_results.append({
            'layer_divergences': layer_divergences,
            'final_kl': kl,
            'prediction_changed': bl_top1 != md_top1,
            'baseline_correct': baseline_correct,
        })

    return per_question_results, energy_removed_all


def main():
    parser = argparse.ArgumentParser(description="Exp 26: Error Propagation Test")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--importance-dir', type=str, required=True,
                        help='Directory with importance sweep results')
    parser.add_argument('--target-layers', type=str, default='0-20',
                        help='Layers to remove chunks FROM')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--paired', action='store_true',
                        help='Remove chunk from both mlp_in and mlp_out simultaneously')
    parser.add_argument('--matrix-type', type=str, default='mlp_out',
                        choices=['mlp_in', 'mlp_out'])
    parser.add_argument('--num-questions', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='results/error_propagation/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layer_spec(args.target_layers)

    # Load importance data
    print("Loading importance data...")
    importance_data = load_importance_data(Path(args.importance_dir))

    # Determine chunks to test per layer
    layer_chunks = {}  # {layer: {'important': chunk_idx, 'unimportant': chunk_idx}}
    for layer_idx in target_layers:
        imp_chunk = get_first_chunk_by_importance(importance_data, layer_idx, 'important')
        unimp_chunk = get_first_chunk_by_importance(importance_data, layer_idx, 'unimportant')
        if imp_chunk is not None and unimp_chunk is not None:
            layer_chunks[layer_idx] = {'important': imp_chunk, 'unimportant': unimp_chunk}
            print(f"  Layer {layer_idx}: important=chunk {imp_chunk}, unimportant=chunk {unimp_chunk}")
        else:
            print(f"  Layer {layer_idx}: SKIPPED (imp={imp_chunk}, unimp={unimp_chunk})")

    # Filter target layers to only those with both chunk types
    target_layers = [l for l in target_layers if l in layer_chunks]
    print(f"\nWill test {len(target_layers)} layers: {target_layers}")

    # Load eval set
    print("\nLoading eval set...")
    all_samples = load_eval_set(args.eval_set)
    samples = all_samples[:args.num_questions]
    print(f"Using {len(samples)} of {len(all_samples)} questions")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    num_layers = len(model.model.layers)
    print(f"Model loaded. {num_layers} layers.")

    # Determine matrix types
    if args.paired:
        matrix_types = ['mlp_in', 'mlp_out']
        print("Paired mode: removing chunks from both mlp_in and mlp_out")
    else:
        matrix_types = [args.matrix_type]

    # Pre-decompose all target layers
    print("\nDecomposing weights for target layers...")
    decompositions = {}
    original_weights = {}
    for layer_idx in target_layers:
        decompositions[layer_idx] = {}
        original_weights[layer_idx] = {}
        for mt in matrix_types:
            if mt == 'mlp_out':
                W = model.model.layers[layer_idx].mlp.down_proj.weight.data.clone()
            else:
                W = model.model.layers[layer_idx].mlp.up_proj.weight.data.clone()
            original_weights[layer_idx][mt] = W
            U, S, Vh = decompose_weight_svd(W, args.device)
            decompositions[layer_idx][mt] = {'U': U, 'S': S, 'Vh': Vh}
        print(f"  Layer {layer_idx}: done")

    # CSV rows
    csv_rows = []

    # Run for each target layer, both important and unimportant chunks
    total_runs = len(target_layers) * 2
    run_idx = 0

    for target_layer in target_layers:
        for chunk_label in ['important', 'unimportant']:
            run_idx += 1
            chunk_idx = layer_chunks[target_layer][chunk_label]
            flip_count = importance_data[target_layer][chunk_idx]['flip_count']

            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] Layer {target_layer} | {chunk_label} chunk {chunk_idx} (flips={flip_count})")
            print(f"{'='*60}")

            per_q_results, energy_removed = run_one_removal(
                model, tokenizer, samples, target_layer, chunk_idx,
                args.chunk_size, matrix_types, decompositions, original_weights,
                num_layers, args.device,
            )

            # Aggregate and write CSV rows — split by baseline correctness
            for baseline_group in ['all', 'correct', 'incorrect']:
                if baseline_group == 'all':
                    filtered = per_q_results
                elif baseline_group == 'correct':
                    filtered = [q for q in per_q_results if q['baseline_correct']]
                else:
                    filtered = [q for q in per_q_results if not q['baseline_correct']]

                if not filtered:
                    continue

                for li in range(num_layers):
                    cos_sims = [q['layer_divergences'][li]['cosine_sim']
                                for q in filtered if li in q['layer_divergences']]
                    rel_l2s = [q['layer_divergences'][li]['relative_l2']
                               for q in filtered if li in q['layer_divergences']]
                    if cos_sims:
                        csv_rows.append({
                            'removed_from_layer': target_layer,
                            'chunk_idx': chunk_idx,
                            'chunk_label': chunk_label,
                            'flip_count': flip_count,
                            'baseline_group': baseline_group,
                            'n_questions': len(filtered),
                            'measured_at_layer': li,
                            'layers_after_removal': li - target_layer,
                            'mean_cos_sim': np.mean(cos_sims),
                            'std_cos_sim': np.std(cos_sims),
                            'mean_rel_l2': np.mean(rel_l2s),
                            'std_rel_l2': np.std(rel_l2s),
                        })

            # Summary stats
            kls = [q['final_kl'] for q in per_q_results if not np.isnan(q['final_kl'])]
            flips = [q['prediction_changed'] for q in per_q_results]
            n_correct = sum(q['baseline_correct'] for q in per_q_results)
            mean_kl = np.mean(kls) if kls else float('nan')
            flip_rate = np.mean(flips)

            print(f"  Final KL: {mean_kl:.4f}  Flip rate: {flip_rate*100:.1f}%  "
                  f"Baseline correct: {n_correct}/{len(per_q_results)}")
            # Show divergence at key points
            for li in [target_layer, min(target_layer + 1, 31), min(target_layer + 5, 31), 31]:
                cos_sims = [q['layer_divergences'][li]['cosine_sim']
                            for q in per_q_results if li in q['layer_divergences']]
                rel_l2s = [q['layer_divergences'][li]['relative_l2']
                           for q in per_q_results if li in q['layer_divergences']]
                if cos_sims:
                    label_str = " (removed)" if li == target_layer else ""
                    print(f"  L{li:>2}: cos={np.mean(cos_sims):.4f} rel_l2={np.mean(rel_l2s):.4f}{label_str}")

    # Save CSV
    model_short = args.model.split('/')[-1]
    paired_str = "paired" if args.paired else args.matrix_type
    csv_file = output_dir / f"error_prop_{model_short}_{paired_str}.csv"

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'removed_from_layer', 'chunk_idx', 'chunk_label', 'flip_count',
            'baseline_group', 'n_questions',
            'measured_at_layer', 'layers_after_removal',
            'mean_cos_sim', 'std_cos_sim', 'mean_rel_l2', 'std_rel_l2',
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_file}")

    # Also save pickle for detailed per-question data
    pkl_file = output_dir / f"error_prop_{model_short}_{paired_str}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'eval_set': args.eval_set,
                'target_layers': target_layers,
                'layer_chunks': layer_chunks,
                'chunk_size': args.chunk_size,
                'paired': args.paired,
                'num_questions': len(samples),
                'timestamp': datetime.now().isoformat(),
            },
            'csv_rows': csv_rows,
        }, f)
    print(f"Saved PKL: {pkl_file}")
    print("Done!")


if __name__ == '__main__':
    main()
