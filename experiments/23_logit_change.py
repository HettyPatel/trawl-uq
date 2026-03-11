"""
Experiment 23: Inference-Based Logit Change Analysis

For targeted SVD chunks (selected by importance from exp 20), remove the chunk,
run inference, and capture how the full output distribution changes. This gives
ground-truth token-level impact per chunk.

For each chunk removal, we compute:
  - KL divergence between baseline and modified distributions
  - Top-k tokens with biggest logit increase/decrease
  - Answer token (A/B/C/D) probability changes
  - Whether the prediction flipped

Unlike exp 22 (weight-only projection through lm_head, which failed), this uses
actual inference to measure real token-level effects.

Chunk selection: top-N most important + N least important per layer (by flip count).

Usage:
    # Quick test (2 layers, 10 questions)
    python experiments/23_logit_change.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --layers 0,5 --num-questions 10 --chunks-per-class 2

    # Full run
    python experiments/23_logit_change.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_mcq_arc_challenge_500.json \\
        --layers 0-15 --num-questions 100 --chunks-per-class 3
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
from src.decomposition.svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
)


# =============================================================================
# Data loading
# =============================================================================

def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def load_importance_data(importance_dir: Path):
    """Load flip counts and importance per chunk from exp 20."""
    pkls = sorted(glob.glob(str(importance_dir / "*.pkl")))
    pkls = [p for p in pkls if 'baseline' not in Path(p).name]

    importance_data = {}

    for p in pkls:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        layer = data['config']['layer']
        chunk_info = []
        for c in data['chunk_results']:
            chunk_info.append({
                'chunk_idx': c['chunk_idx'],
                'flip_count': c['flip_count'],
                'importance': c['importance'],
            })
        importance_data[layer] = {
            'chunks': chunk_info,
            'chunk_size': data['config']['chunk_size'],
            'total_components': data['config']['total_components'],
            'num_chunks': data['config']['num_chunks'],
        }

    print(f"Loaded importance data: {len(importance_data)} layers")
    return importance_data


def select_target_chunks(importance_data, layers, chunks_per_class):
    """
    Select target chunks per layer: top-N most important + N least important.

    Returns:
        dict: {layer_idx: [{'chunk_idx': int, 'importance': str, 'flip_count': int}, ...]}
    """
    targets = {}

    for layer_idx in layers:
        if layer_idx not in importance_data:
            print(f"  WARNING: No importance data for layer {layer_idx}, skipping")
            continue

        chunks = importance_data[layer_idx]['chunks']

        # Sort by flip count descending
        sorted_chunks = sorted(chunks, key=lambda c: c['flip_count'], reverse=True)

        # Top-N important (highest flip count, excluding zero)
        important = [c for c in sorted_chunks if c['flip_count'] > 0][:chunks_per_class]

        # Bottom-N unimportant (lowest flip count = 0)
        unimportant = [c for c in sorted_chunks if c['flip_count'] == 0]
        # Take from the end (highest chunk indices among zero-flip)
        unimportant = unimportant[-chunks_per_class:] if len(unimportant) >= chunks_per_class else unimportant

        selected = important + unimportant
        targets[layer_idx] = selected

        imp_str = ', '.join(f"ch{c['chunk_idx']}(f={c['flip_count']})" for c in important)
        unimp_str = ', '.join(f"ch{c['chunk_idx']}(f={c['flip_count']})" for c in unimportant)
        print(f"  Layer {layer_idx}: {len(important)} important [{imp_str}], "
              f"{len(unimportant)} unimportant [{unimp_str}]")

    return targets


# =============================================================================
# Chunk removal helpers (from exp 15)
# =============================================================================

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


def get_original_weight(model, layer_idx, matrix_type, model_type="llama"):
    """Get a clone of the original weight matrix."""
    if model_type == "llama":
        weight_map = {
            'mlp_in': lambda: model.model.layers[layer_idx].mlp.up_proj.weight.data,
            'mlp_out': lambda: model.model.layers[layer_idx].mlp.down_proj.weight.data,
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return weight_map[matrix_type]().clone()


# =============================================================================
# Inference and logit analysis
# =============================================================================

def get_option_token_ids(tokenizer, options=None):
    """Get token IDs for MCQ answer letters."""
    if options is None:
        options = ['A', 'B', 'C', 'D']

    option_token_ids = {}
    for letter in options:
        token_id = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(token_id) == 1:
            option_token_ids[letter] = token_id[0]
        else:
            token_id = tokenizer.encode(letter, add_special_tokens=False)
            option_token_ids[letter] = token_id[0] if token_id else tokenizer.unk_token_id

    return option_token_ids


def run_inference_batch(model, tokenizer, samples, device):
    """
    Run inference on all samples, return last-token logits.

    Returns:
        list of dicts: [{
            'sample_id': str,
            'logits': tensor (vocab_size,),
            'answer_probs': {A: float, ...},
            'predicted_letter': str,
            'is_correct': bool,
        }]
    """
    model.eval()
    option_ids = get_option_token_ids(tokenizer)
    options = ['A', 'B', 'C', 'D']
    results = []

    for sample in samples:
        inputs = tokenizer(sample['mcq_prompt'], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Answer option probs
        option_logits = torch.tensor([logits[option_ids[l]].item() for l in options])
        option_probs = torch.nn.functional.softmax(option_logits, dim=0)
        answer_probs = {l: option_probs[i].item() for i, l in enumerate(options)}

        predicted_letter = options[torch.argmax(option_probs).item()]

        results.append({
            'sample_id': sample.get('id', sample.get('question', '')[:50]),
            'logits': logits.cpu(),  # save on CPU
            'answer_probs': answer_probs,
            'predicted_letter': predicted_letter,
            'is_correct': predicted_letter == sample['correct_letter'],
            'correct_letter': sample['correct_letter'],
        })

    return results


def compute_logit_changes(baseline_result, modified_result, tokenizer, top_k=100):
    """
    Compare baseline vs modified logits for one question.

    Returns:
        dict with KL divergence, top-k increases/decreases, flip info
    """
    bl = baseline_result['logits'].float()
    md = modified_result['logits'].float()

    # KL divergence: KL(baseline || modified)
    bl_log_probs = torch.log_softmax(bl, dim=0)
    md_log_probs = torch.log_softmax(md, dim=0)
    bl_probs = torch.softmax(bl, dim=0)

    kl_div = torch.sum(bl_probs * (bl_log_probs - md_log_probs)).item()

    # Logit diff
    logit_diff = md - bl  # positive = token boosted by removal

    # Top-k increases (tokens boosted when chunk removed)
    top_inc_vals, top_inc_ids = torch.topk(logit_diff, top_k)
    top_increases = []
    for val, tid in zip(top_inc_vals, top_inc_ids):
        top_increases.append({
            'token_id': tid.item(),
            'token_str': tokenizer.decode([tid.item()]),
            'logit_diff': val.item(),
        })

    # Top-k decreases (tokens suppressed when chunk removed)
    top_dec_vals, top_dec_ids = torch.topk(-logit_diff, top_k)
    top_decreases = []
    for val, tid in zip(top_dec_vals, top_dec_ids):
        top_decreases.append({
            'token_id': tid.item(),
            'token_str': tokenizer.decode([tid.item()]),
            'logit_diff': -val.item(),  # negative = decreased
        })

    return {
        'kl_divergence': kl_div,
        'baseline_correct': baseline_result['is_correct'],
        'modified_correct': modified_result['is_correct'],
        'flipped': baseline_result['is_correct'] != modified_result['is_correct'],
        'baseline_predicted': baseline_result['predicted_letter'],
        'modified_predicted': modified_result['predicted_letter'],
        'answer_probs_baseline': baseline_result['answer_probs'],
        'answer_probs_modified': modified_result['answer_probs'],
        'top_k_increases': top_increases,
        'top_k_decreases': top_decreases,
    }


def aggregate_token_shifts(per_question_results, top_n=20):
    """
    Aggregate which tokens shift most across all questions.

    Returns top-N tokens that are most consistently increased/decreased.
    """
    from collections import defaultdict

    increase_counts = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})
    decrease_counts = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})

    for qr in per_question_results:
        for t in qr['top_k_increases'][:20]:  # top 20 per question
            key = t['token_str']
            increase_counts[key]['total_diff'] += t['logit_diff']
            increase_counts[key]['count'] += 1

        for t in qr['top_k_decreases'][:20]:
            key = t['token_str']
            decrease_counts[key]['total_diff'] += t['logit_diff']
            decrease_counts[key]['count'] += 1

    # Sort by count (how consistently this token shifts)
    top_increases = sorted(
        increase_counts.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:top_n]

    top_decreases = sorted(
        decrease_counts.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:top_n]

    agg_increases = [
        {'token_str': k, 'mean_logit_diff': v['total_diff'] / v['count'], 'count': v['count']}
        for k, v in top_increases
    ]
    agg_decreases = [
        {'token_str': k, 'mean_logit_diff': v['total_diff'] / v['count'], 'count': v['count']}
        for k, v in top_decreases
    ]

    return agg_increases, agg_decreases


# =============================================================================
# Main
# =============================================================================

def parse_layer_spec(spec: str) -> list:
    """Parse layer specification string."""
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(description="Exp 23: Inference-Based Logit Change Analysis")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--importance-dir', type=str, required=True,
                        help='Path to exp 20 importance sweep results')
    parser.add_argument('--eval-set', type=str, required=True,
                        help='Path to MCQ eval set JSON')
    parser.add_argument('--layers', type=str, default='0-15',
                        help='Layer range (e.g., 0-15 or 0,5,10)')
    parser.add_argument('--chunks-per-class', type=int, default=3,
                        help='Number of important + unimportant chunks to test per layer')
    parser.add_argument('--num-questions', type=int, default=100,
                        help='Number of MCQ questions to test (subset of eval set)')
    parser.add_argument('--top-k-save', type=int, default=100,
                        help='Save top-k biggest logit changes per chunk per question')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--paired', action='store_true',
                        help='Remove chunk from both mlp_in and mlp_out (paired removal)')
    parser.add_argument('--output-dir', type=str, default='results/logit_change/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layer_spec(args.layers)

    # Load importance data
    print("Loading importance data...")
    importance_data = load_importance_data(Path(args.importance_dir))

    # Select target chunks
    print(f"\nSelecting {args.chunks_per_class} important + {args.chunks_per_class} unimportant per layer:")
    target_chunks = select_target_chunks(importance_data, layers, args.chunks_per_class)

    # Load eval set (subset)
    print(f"\nLoading eval set...")
    all_samples = load_mcq_eval_set(args.eval_set)
    samples = all_samples[:args.num_questions]
    print(f"Using {len(samples)} of {len(all_samples)} questions")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    print("Model loaded.")

    # Matrix types
    if args.paired:
        matrix_types = ['mlp_in', 'mlp_out']
        print("Mode: PAIRED (mlp_in + mlp_out)")
    else:
        matrix_types = ['mlp_out']
        print("Mode: mlp_out only")

    # Run baseline inference
    print("\nRunning baseline inference...")
    baseline_results = run_inference_batch(model, tokenizer, samples, args.device)
    baseline_acc = np.mean([r['is_correct'] for r in baseline_results])
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%")

    # Build sample_id → baseline_result map
    baseline_map = {r['sample_id']: r for r in baseline_results}

    # Main loop: per layer, per target chunk
    all_layer_results = {}
    total_chunks_tested = 0

    for layer_idx in layers:
        if layer_idx not in target_chunks:
            continue

        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        # SVD decompose + save originals for each matrix type
        original_weights = {}
        decompositions = {}
        for mt in matrix_types:
            original_weights[mt] = get_original_weight(model, layer_idx, mt)
            U, S, Vh = decompose_weight_svd(original_weights[mt], args.device)
            decompositions[mt] = {'U': U, 'S': S, 'Vh': Vh}
            print(f"  {mt}: shape={tuple(original_weights[mt].shape)}, SVs={len(S)}")

        total_components = min(len(d['S']) for d in decompositions.values())
        chunks_to_test = target_chunks[layer_idx]

        layer_chunk_results = []

        for chunk_info in chunks_to_test:
            ci = chunk_info['chunk_idx']
            chunk_start = ci * args.chunk_size
            chunk_end = min(chunk_start + args.chunk_size, total_components)

            print(f"\n  Chunk {ci} [{chunk_start}:{chunk_end}] "
                  f"({chunk_info['importance']}, flips={chunk_info['flip_count']})")

            # Remove chunk from all target matrices
            energy_removed = {}
            for mt in matrix_types:
                U = decompositions[mt]['U']
                S = decompositions[mt]['S']
                Vh = decompositions[mt]['Vh']
                W_modified, e_removed = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
                energy_removed[mt] = e_removed
                update_layer_with_svd(model, layer_idx, W_modified, mt)

            # Run modified inference
            modified_results = run_inference_batch(model, tokenizer, samples, args.device)
            modified_acc = np.mean([r['is_correct'] for r in modified_results])

            # Restore original weights
            for mt in matrix_types:
                restore_original_weight(model, layer_idx, original_weights[mt], mt)

            # Compute per-question logit changes
            per_question = []
            for bl_r, md_r in zip(baseline_results, modified_results):
                changes = compute_logit_changes(bl_r, md_r, tokenizer, args.top_k_save)
                changes['sample_id'] = bl_r['sample_id']
                per_question.append(changes)

            # Aggregate
            n_flipped = sum(1 for q in per_question if q['flipped'])
            mean_kl = np.mean([q['kl_divergence'] for q in per_question])
            acc_change = modified_acc - baseline_acc

            agg_increases, agg_decreases = aggregate_token_shifts(per_question)

            print(f"    Acc: {modified_acc*100:.1f}% ({acc_change*100:+.1f}pp)")
            print(f"    KL divergence: {mean_kl:.4f}")
            print(f"    Flipped: {n_flipped}/{len(per_question)}")
            print(f"    Energy removed: {np.mean(list(energy_removed.values()))*100:.1f}%")

            # Print top shifted tokens
            if agg_increases:
                inc_str = ', '.join(
                    f"{t['token_str']!r}({t['mean_logit_diff']:+.2f}, n={t['count']})"
                    for t in agg_increases[:5]
                )
                print(f"    Top increased: {inc_str}")
            if agg_decreases:
                dec_str = ', '.join(
                    f"{t['token_str']!r}({t['mean_logit_diff']:+.2f}, n={t['count']})"
                    for t in agg_decreases[:5]
                )
                print(f"    Top decreased: {dec_str}")

            # Strip full logits from per_question to save space
            for q in per_question:
                # Keep only top_k_save items (already done in compute_logit_changes)
                pass

            layer_chunk_results.append({
                'chunk_idx': ci,
                'importance': chunk_info['importance'],
                'flip_count': chunk_info['flip_count'],
                'energy_removed': energy_removed,
                'modified_accuracy': modified_acc,
                'accuracy_change': acc_change,
                'mean_kl_div': mean_kl,
                'n_flipped': n_flipped,
                'per_question': per_question,
                'aggregate_top_increases': agg_increases,
                'aggregate_top_decreases': agg_decreases,
            })

            total_chunks_tested += 1

        all_layer_results[layer_idx] = {'chunks_tested': layer_chunk_results}

    # Save results
    results = {
        'config': {
            'model': args.model,
            'importance_dir': args.importance_dir,
            'eval_set': args.eval_set,
            'layers': layers,
            'chunks_per_class': args.chunks_per_class,
            'num_questions': len(samples),
            'top_k_save': args.top_k_save,
            'chunk_size': args.chunk_size,
            'paired': args.paired,
            'matrix_types': matrix_types,
            'baseline_accuracy': baseline_acc,
            'timestamp': datetime.now().isoformat(),
        },
        'baseline': {
            r['sample_id']: {
                'predicted_letter': r['predicted_letter'],
                'is_correct': r['is_correct'],
                'answer_probs': r['answer_probs'],
            }
            for r in baseline_results
        },
        'per_layer': all_layer_results,
    }

    model_short = args.model.split('/')[-1]
    mode_str = 'paired' if args.paired else 'mlp_out'
    out_file = output_dir / f"logit_change_{model_short}_{mode_str}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_file}")

    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total chunks tested: {total_chunks_tested}")
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%")

    # Compare important vs unimportant
    imp_kls = []
    unimp_kls = []
    imp_flips = []
    unimp_flips = []

    for layer_idx, layer_data in all_layer_results.items():
        for chunk in layer_data['chunks_tested']:
            if chunk['importance'] == 'important':
                imp_kls.append(chunk['mean_kl_div'])
                imp_flips.append(chunk['n_flipped'])
            elif chunk['importance'] == 'unimportant':
                unimp_kls.append(chunk['mean_kl_div'])
                unimp_flips.append(chunk['n_flipped'])

    if imp_kls and unimp_kls:
        print(f"\n{'Metric':<25} {'Important':>12} {'Unimportant':>12} {'Ratio':>10}")
        print("-" * 62)
        mk_i, mk_u = np.mean(imp_kls), np.mean(unimp_kls)
        print(f"{'Mean KL divergence':<25} {mk_i:>12.4f} {mk_u:>12.4f} {mk_i/mk_u if mk_u > 0 else float('inf'):>10.1f}x")
        mf_i, mf_u = np.mean(imp_flips), np.mean(unimp_flips)
        print(f"{'Mean # flipped':<25} {mf_i:>12.1f} {mf_u:>12.1f} {mf_i/mf_u if mf_u > 0 else float('inf'):>10.1f}x")
        print(f"\n  n_important={len(imp_kls)}, n_unimportant={len(unimp_kls)}")

    # Print most consistently shifted tokens across all important chunks
    print("\n" + "=" * 70)
    print("TOKENS MOST AFFECTED BY IMPORTANT CHUNK REMOVAL")
    print("=" * 70)

    from collections import defaultdict
    imp_increases = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})
    imp_decreases = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})

    for layer_idx, layer_data in all_layer_results.items():
        for chunk in layer_data['chunks_tested']:
            if chunk['importance'] != 'important':
                continue
            for t in chunk['aggregate_top_increases'][:10]:
                imp_increases[t['token_str']]['total_diff'] += t['mean_logit_diff']
                imp_increases[t['token_str']]['count'] += t['count']
            for t in chunk['aggregate_top_decreases'][:10]:
                imp_decreases[t['token_str']]['total_diff'] += t['mean_logit_diff']
                imp_decreases[t['token_str']]['count'] += t['count']

    top_inc = sorted(imp_increases.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
    top_dec = sorted(imp_decreases.items(), key=lambda x: x[1]['count'], reverse=True)[:15]

    print("\nMost consistently INCREASED tokens (boosted by chunk removal):")
    for token, info in top_inc:
        print(f"  {token!r:<20} count={info['count']:>5}  mean_diff={info['total_diff']/max(info['count'],1):+.3f}")

    print("\nMost consistently DECREASED tokens (suppressed by chunk removal):")
    for token, info in top_dec:
        print(f"  {token!r:<20} count={info['count']:>5}  mean_diff={info['total_diff']/max(info['count'],1):+.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
