"""
Experiment 27: Error Propagation by Outcome Group

Remove SVD chunk 0 (top components) from each layer L (0-31), run all 500
ARC-Challenge questions through baseline and modified model. Capture residual
stream hidden states at every layer 0-31 and compute divergence metrics.

Groups results by:
  - all: all questions
  - baseline_correct: unmodified model gets it right
  - baseline_incorrect: unmodified model gets it wrong
  - flipped: predicted answer letter changes after chunk removal
  - not_flipped: predicted answer letter stays the same

Uses paired removal (up_proj + down_proj), chunk size 100.

Usage:
    python experiments/27_error_propagation_by_outcome.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \
        --target-layers 0-31 \
        --output-dir results/error_propagation_by_outcome/
"""

import sys
sys.path.append('.')

import csv
import json
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

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
        'relative_l2': relative_l2,
    }


def get_predicted_letter(logits, letter_token_ids):
    """Return the letter (A/B/C/D) with the highest logit."""
    letter_logits = {l: logits[tid].item() for l, tid in letter_token_ids.items()}
    return max(letter_logits, key=letter_logits.get)


def main():
    parser = argparse.ArgumentParser(description="Exp 27: Error Propagation by Outcome Group")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_arc_challenge_500.json')
    parser.add_argument('--target-layers', type=str, default='0-31',
                        help='Layers to remove chunk 0 from')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--chunk-idx', type=int, default=0,
                        help='Which chunk to remove (0 = top SVD components)')
    parser.add_argument('--output-dir', type=str,
                        default='results/error_propagation_by_outcome/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layer_spec(args.target_layers)

    chunk_idx = args.chunk_idx
    chunk_size = args.chunk_size
    chunk_start = chunk_idx * chunk_size
    matrix_types = ['mlp_in', 'mlp_out']  # paired removal

    # Load eval set
    print("Loading eval set...")
    samples = load_eval_set(args.eval_set)
    print(f"Using all {len(samples)} questions")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    num_layers = len(model.model.layers)
    print(f"Model loaded. {num_layers} layers.")

    # Map answer letters to token IDs
    letter_token_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[-1]
    print(f"Letter token IDs: {letter_token_ids}")

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

    # =========================================================================
    # Run baseline once for all questions (no modification)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running baseline inference for all questions...")
    print("=" * 60)

    baseline_data = []  # per-question: {hidden_states, predicted_letter, correct}
    hook_bl = ResidualStreamHook()
    hook_bl.register_all(model, num_layers)

    for q_idx, sample in enumerate(tqdm(samples, desc="Baseline")):
        prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
        correct_letter = sample.get('correct_letter', None)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :].detach()
        predicted = get_predicted_letter(logits, letter_token_ids)
        correct = (predicted == correct_letter) if correct_letter else None

        # Store hidden states (move to CPU to save GPU memory)
        hidden = {k: v.clone().cpu() for k, v in hook_bl.hidden_states.items()}

        baseline_data.append({
            'hidden_states': hidden,
            'predicted_letter': predicted,
            'correct_letter': correct_letter,
            'baseline_correct': correct,
        })

    hook_bl.remove_all()

    n_correct = sum(1 for d in baseline_data if d['baseline_correct'])
    print(f"Baseline accuracy: {n_correct}/{len(samples)} "
          f"({100*n_correct/len(samples):.1f}%)")

    # =========================================================================
    # For each target layer, remove chunk 0 and run all questions
    # =========================================================================
    csv_rows = []
    all_per_question_data = {}  # layer -> list of per-question dicts

    for run_idx, target_layer in enumerate(target_layers):
        print(f"\n{'=' * 60}")
        print(f"[{run_idx + 1}/{len(target_layers)}] Removing chunk {chunk_idx} "
              f"from layer {target_layer}")
        print("=" * 60)

        # Prepare modified weights
        modified_weights = {}
        energy_removed = {}
        for mt in matrix_types:
            U = decompositions[target_layer][mt]['U']
            S = decompositions[target_layer][mt]['S']
            Vh = decompositions[target_layer][mt]['Vh']
            chunk_end = min(chunk_start + chunk_size, len(S))
            W_mod, e_rem = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
            modified_weights[mt] = W_mod
            energy_removed[mt] = e_rem
        print(f"  Energy removed: {', '.join(f'{mt}={e:.4f}' for mt, e in energy_removed.items())}")

        # Apply modified weights
        for mt in matrix_types:
            update_layer_with_svd(model, target_layer, modified_weights[mt], mt, 'llama')

        # Register hooks
        hook_md = ResidualStreamHook()
        hook_md.register_all(model, num_layers)

        per_question = []
        for q_idx, sample in enumerate(tqdm(samples, desc=f"Layer {target_layer}")):
            prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[0, -1, :].detach()
            modified_predicted = get_predicted_letter(logits, letter_token_ids)
            baseline_predicted = baseline_data[q_idx]['predicted_letter']
            flipped = (modified_predicted != baseline_predicted)

            # Compute divergence at each layer
            layer_divergences = {}
            for li in range(num_layers):
                if li in hook_md.hidden_states:
                    bl_h = baseline_data[q_idx]['hidden_states'][li].to(args.device)
                    md_h = hook_md.hidden_states[li]
                    layer_divergences[li] = compute_divergence(bl_h, md_h)

            per_question.append({
                'q_idx': q_idx,
                'baseline_correct': baseline_data[q_idx]['baseline_correct'],
                'baseline_predicted': baseline_predicted,
                'modified_predicted': modified_predicted,
                'flipped': flipped,
                'layer_divergences': layer_divergences,
            })

        hook_md.remove_all()

        # Restore original weights
        for mt in matrix_types:
            restore_original_weight(model, target_layer,
                                    original_weights[target_layer][mt], mt, 'llama')

        all_per_question_data[target_layer] = per_question

        # Aggregate by group
        groups = {
            'all': per_question,
            'baseline_correct': [q for q in per_question if q['baseline_correct']],
            'baseline_incorrect': [q for q in per_question if not q['baseline_correct']],
            'flipped': [q for q in per_question if q['flipped']],
            'not_flipped': [q for q in per_question if not q['flipped']],
        }

        n_flipped = len(groups['flipped'])
        print(f"  Flipped: {n_flipped}/{len(samples)} ({100*n_flipped/len(samples):.1f}%)")
        print(f"  Baseline correct: {len(groups['baseline_correct'])}, "
              f"incorrect: {len(groups['baseline_incorrect'])}")

        for group_name, group_qs in groups.items():
            if not group_qs:
                continue

            for li in range(num_layers):
                cos_sims = [q['layer_divergences'][li]['cosine_sim']
                            for q in group_qs if li in q['layer_divergences']]
                rel_l2s = [q['layer_divergences'][li]['relative_l2']
                           for q in group_qs if li in q['layer_divergences']]

                if not cos_sims:
                    continue

                csv_rows.append({
                    'removed_from_layer': target_layer,
                    'measured_at_layer': li,
                    'layers_after_removal': li - target_layer,
                    'group': group_name,
                    'n_questions': len(group_qs),
                    'mean_cos_sim': np.mean(cos_sims),
                    'std_cos_sim': np.std(cos_sims),
                    'mean_rel_l2': np.mean(rel_l2s),
                    'std_rel_l2': np.std(rel_l2s),
                })

        # Print summary for key layers
        for li in [target_layer, min(target_layer + 1, 31), min(target_layer + 5, 31), 31]:
            cos_sims = [q['layer_divergences'][li]['cosine_sim']
                        for q in per_question if li in q['layer_divergences']]
            rel_l2s = [q['layer_divergences'][li]['relative_l2']
                       for q in per_question if li in q['layer_divergences']]
            if cos_sims:
                tag = " (removed)" if li == target_layer else ""
                print(f"  L{li:>2}: cos={np.mean(cos_sims):.6f} "
                      f"rel_l2={np.mean(rel_l2s):.6f}{tag}")

    # =========================================================================
    # Save results
    # =========================================================================
    model_short = args.model.split('/')[-1]
    csv_file = output_dir / f"error_prop_by_outcome_{model_short}_chunk{chunk_idx}.csv"

    fieldnames = [
        'removed_from_layer', 'measured_at_layer', 'layers_after_removal',
        'group', 'n_questions',
        'mean_cos_sim', 'std_cos_sim', 'mean_rel_l2', 'std_rel_l2',
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_file}")

    # Save pickle with per-question details
    pkl_file = output_dir / f"error_prop_by_outcome_{model_short}_chunk{chunk_idx}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'eval_set': args.eval_set,
                'target_layers': target_layers,
                'chunk_idx': chunk_idx,
                'chunk_size': chunk_size,
                'matrix_types': matrix_types,
                'num_questions': len(samples),
                'timestamp': datetime.now().isoformat(),
            },
            'baseline_summary': {
                'n_correct': n_correct,
                'n_total': len(samples),
                'accuracy': n_correct / len(samples),
                'per_question_predictions': [
                    {
                        'q_idx': i,
                        'predicted_letter': d['predicted_letter'],
                        'correct_letter': d['correct_letter'],
                        'baseline_correct': d['baseline_correct'],
                    }
                    for i, d in enumerate(baseline_data)
                ],
            },
            'per_layer_results': {
                layer: [
                    {k: v for k, v in q.items()}  # keep all per-question data
                    for q in qs
                ]
                for layer, qs in all_per_question_data.items()
            },
            'csv_rows': csv_rows,
        }, f)
    print(f"Saved PKL: {pkl_file}")
    print("Done!")


if __name__ == '__main__':
    main()
