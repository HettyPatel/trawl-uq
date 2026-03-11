"""
Experiment 24: Inference-Based Logit Change Analysis — Open-Ended QA

Same approach as exp 23 but for open-ended questions (Natural Questions).
Instead of tracking A/B/C/D answer shifts, we track:
  - What token the model would generate (top-1)
  - Whether chunk removal changes the top-1 token
  - Which content tokens shift most (are they factual/answer-related?)
  - KL divergence of full distribution

This tests whether the "important chunks encode factual tokens" or just
"important chunks encode MCQ formatting" (which would be circular).

Usage:
    python experiments/24_logit_change_openqa.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_nq_open_200.json \\
        --layers 0-15 --num-questions 50 --chunks-per-class 3
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
from collections import defaultdict

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.decomposition.svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
)


# =============================================================================
# Data loading (reused from exp 23)
# =============================================================================

def load_eval_set(filepath: str):
    """Load open-ended QA eval set."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
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
    """Select target chunks per layer: top-N important + N unimportant."""
    targets = {}
    for layer_idx in layers:
        if layer_idx not in importance_data:
            continue
        chunks = importance_data[layer_idx]['chunks']
        sorted_chunks = sorted(chunks, key=lambda c: c['flip_count'], reverse=True)
        important = [c for c in sorted_chunks if c['flip_count'] > 0][:chunks_per_class]
        unimportant = [c for c in sorted_chunks if c['flip_count'] == 0]
        unimportant = unimportant[-chunks_per_class:] if len(unimportant) >= chunks_per_class else unimportant
        selected = important + unimportant
        targets[layer_idx] = selected

        imp_str = ', '.join(f"ch{c['chunk_idx']}(f={c['flip_count']})" for c in important)
        unimp_str = ', '.join(f"ch{c['chunk_idx']}(f={c['flip_count']})" for c in unimportant)
        print(f"  Layer {layer_idx}: {len(important)} imp [{imp_str}], "
              f"{len(unimportant)} unimp [{unimp_str}]")
    return targets


# =============================================================================
# Chunk removal helpers
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
    weight_map = {
        'mlp_in': lambda: model.model.layers[layer_idx].mlp.up_proj.weight.data,
        'mlp_out': lambda: model.model.layers[layer_idx].mlp.down_proj.weight.data,
    }
    return weight_map[matrix_type]().clone()


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, tokenizer, samples, device):
    """
    Run inference on open-ended QA samples.
    Returns last-token logits and top-1 predicted token.
    """
    model.eval()
    results = []

    for sample in samples:
        prompt = sample['prompt']
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Top-1 predicted token
        top1_id = torch.argmax(logits).item()
        top1_str = tokenizer.decode([top1_id])

        # Top-5 for context
        top5_vals, top5_ids = torch.topk(logits, 5)
        top5 = [(tokenizer.decode([tid.item()]), tv.item()) for tid, tv in zip(top5_ids, top5_vals)]

        results.append({
            'sample_id': sample['id'],
            'question': sample['question'],
            'gold_answer': sample['answer'],
            'all_answers': sample.get('all_answers', [sample['answer']]),
            'logits': logits.cpu(),
            'top1_id': top1_id,
            'top1_str': top1_str,
            'top5': top5,
        })

    return results


def compute_logit_changes(baseline_result, modified_result, tokenizer, top_k=100):
    """Compare baseline vs modified logits for one question."""
    bl = baseline_result['logits'].float()
    md = modified_result['logits'].float()

    # KL divergence
    bl_log_probs = torch.log_softmax(bl, dim=0)
    md_log_probs = torch.log_softmax(md, dim=0)
    bl_probs = torch.softmax(bl, dim=0)
    kl_div = torch.sum(bl_probs * (bl_log_probs - md_log_probs)).item()

    # Logit diff
    logit_diff = md - bl

    # Top-1 changed?
    mod_top1_id = torch.argmax(md).item()
    top1_changed = (mod_top1_id != baseline_result['top1_id'])

    # Top-k increases
    top_inc_vals, top_inc_ids = torch.topk(logit_diff, top_k)
    top_increases = [
        {'token_id': tid.item(), 'token_str': tokenizer.decode([tid.item()]),
         'logit_diff': val.item()}
        for val, tid in zip(top_inc_vals, top_inc_ids)
    ]

    # Top-k decreases
    top_dec_vals, top_dec_ids = torch.topk(-logit_diff, top_k)
    top_decreases = [
        {'token_id': tid.item(), 'token_str': tokenizer.decode([tid.item()]),
         'logit_diff': -val.item()}
        for val, tid in zip(top_dec_vals, top_dec_ids)
    ]

    # Check if any gold answer tokens were affected
    gold_token_changes = []
    for ans in baseline_result.get('all_answers', [baseline_result['gold_answer']]):
        # Tokenize the answer
        ans_tokens = tokenizer.encode(ans, add_special_tokens=False)
        # Also try with leading space
        ans_tokens_sp = tokenizer.encode(" " + ans, add_special_tokens=False)

        for tid in set(ans_tokens + ans_tokens_sp):
            token_str = tokenizer.decode([tid])
            gold_token_changes.append({
                'token_id': tid,
                'token_str': token_str,
                'baseline_logit': bl[tid].item(),
                'modified_logit': md[tid].item(),
                'logit_diff': logit_diff[tid].item(),
                'baseline_rank': (bl > bl[tid]).sum().item(),
                'modified_rank': (md > md[tid]).sum().item(),
            })

    return {
        'kl_divergence': kl_div,
        'top1_changed': top1_changed,
        'baseline_top1': baseline_result['top1_str'],
        'modified_top1': tokenizer.decode([mod_top1_id]),
        'modified_top1_id': mod_top1_id,
        'top_k_increases': top_increases,
        'top_k_decreases': top_decreases,
        'gold_token_changes': gold_token_changes,
    }


def aggregate_token_shifts(per_question_results, top_n=20):
    """Aggregate which tokens shift most across all questions."""
    increase_counts = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})
    decrease_counts = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})

    for qr in per_question_results:
        for t in qr['top_k_increases'][:20]:
            key = t['token_str']
            increase_counts[key]['total_diff'] += t['logit_diff']
            increase_counts[key]['count'] += 1
        for t in qr['top_k_decreases'][:20]:
            key = t['token_str']
            decrease_counts[key]['total_diff'] += t['logit_diff']
            decrease_counts[key]['count'] += 1

    agg_increases = sorted(
        [{'token_str': k, 'mean_logit_diff': v['total_diff'] / v['count'], 'count': v['count']}
         for k, v in increase_counts.items()],
        key=lambda x: x['count'], reverse=True
    )[:top_n]

    agg_decreases = sorted(
        [{'token_str': k, 'mean_logit_diff': v['total_diff'] / v['count'], 'count': v['count']}
         for k, v in decrease_counts.items()],
        key=lambda x: x['count'], reverse=True
    )[:top_n]

    return agg_increases, agg_decreases


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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 24: Logit Change — Open-Ended QA")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--importance-dir', type=str, required=True)
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--layers', type=str, default='0-15')
    parser.add_argument('--chunks-per-class', type=int, default=3)
    parser.add_argument('--num-questions', type=int, default=50)
    parser.add_argument('--top-k-save', type=int, default=100)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/logit_change_openqa/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layer_spec(args.layers)

    # Load data
    print("Loading importance data...")
    importance_data = load_importance_data(Path(args.importance_dir))

    print(f"\nSelecting chunks:")
    target_chunks = select_target_chunks(importance_data, layers, args.chunks_per_class)

    print(f"\nLoading eval set...")
    all_samples = load_eval_set(args.eval_set)
    samples = all_samples[:args.num_questions]
    print(f"Using {len(samples)} of {len(all_samples)} questions")

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    print("Model loaded.")

    matrix_types = ['mlp_in', 'mlp_out'] if args.paired else ['mlp_out']
    mode_str = 'paired' if args.paired else 'mlp_out'
    print(f"Mode: {mode_str}")

    # Baseline inference
    print("\nRunning baseline inference...")
    baseline_results = run_inference(model, tokenizer, samples, args.device)

    # Print some baseline predictions
    print("\nBaseline predictions (first 10):")
    for r in baseline_results[:10]:
        print(f"  Q: {r['question'][:60]}")
        print(f"    Gold: {r['gold_answer']}")
        print(f"    Pred top-1: {r['top1_str']!r}")
        print(f"    Top-5: {[t[0] for t in r['top5']]}")

    # Main loop
    all_layer_results = {}
    total_chunks_tested = 0

    for layer_idx in layers:
        if layer_idx not in target_chunks:
            continue

        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        original_weights = {}
        decompositions = {}
        for mt in matrix_types:
            original_weights[mt] = get_original_weight(model, layer_idx, mt)
            U, S, Vh = decompose_weight_svd(original_weights[mt], args.device)
            decompositions[mt] = {'U': U, 'S': S, 'Vh': Vh}

        total_components = min(len(d['S']) for d in decompositions.values())
        layer_chunk_results = []

        for chunk_info in target_chunks[layer_idx]:
            ci = chunk_info['chunk_idx']
            chunk_start = ci * args.chunk_size
            chunk_end = min(chunk_start + args.chunk_size, total_components)

            print(f"\n  Chunk {ci} [{chunk_start}:{chunk_end}] "
                  f"({chunk_info['importance']}, flips={chunk_info['flip_count']})")

            # Remove chunk
            energy_removed = {}
            for mt in matrix_types:
                U, S, Vh = decompositions[mt]['U'], decompositions[mt]['S'], decompositions[mt]['Vh']
                W_modified, e_removed = reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end)
                energy_removed[mt] = e_removed
                update_layer_with_svd(model, layer_idx, W_modified, mt)

            # Modified inference
            modified_results = run_inference(model, tokenizer, samples, args.device)

            # Restore
            for mt in matrix_types:
                restore_original_weight(model, layer_idx, original_weights[mt], mt)

            # Compute changes
            per_question = []
            for bl_r, md_r in zip(baseline_results, modified_results):
                changes = compute_logit_changes(bl_r, md_r, tokenizer, args.top_k_save)
                changes['sample_id'] = bl_r['sample_id']
                changes['question'] = bl_r['question']
                changes['gold_answer'] = bl_r['gold_answer']
                per_question.append(changes)

            # Stats
            n_top1_changed = sum(1 for q in per_question if q['top1_changed'])
            mean_kl = np.nanmean([q['kl_divergence'] for q in per_question])

            # Gold answer token impact
            all_gold_diffs = []
            for q in per_question:
                for gtc in q['gold_token_changes']:
                    all_gold_diffs.append(gtc['logit_diff'])
            mean_gold_diff = np.mean(all_gold_diffs) if all_gold_diffs else 0

            agg_increases, agg_decreases = aggregate_token_shifts(per_question)

            print(f"    KL divergence: {mean_kl:.4f}")
            print(f"    Top-1 changed: {n_top1_changed}/{len(per_question)}")
            print(f"    Gold answer token mean logit diff: {mean_gold_diff:+.3f}")

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

            # Show a few example top-1 changes
            changed_examples = [q for q in per_question if q['top1_changed']][:3]
            for ex in changed_examples:
                print(f"    Example: {ex['question'][:50]}...")
                print(f"      {ex['baseline_top1']!r} → {ex['modified_top1']!r} "
                      f"(gold: {ex['gold_answer']})")

            layer_chunk_results.append({
                'chunk_idx': ci,
                'importance': chunk_info['importance'],
                'flip_count': chunk_info['flip_count'],
                'energy_removed': energy_removed,
                'mean_kl_div': mean_kl,
                'n_top1_changed': n_top1_changed,
                'mean_gold_answer_logit_diff': mean_gold_diff,
                'per_question': per_question,
                'aggregate_top_increases': agg_increases,
                'aggregate_top_decreases': agg_decreases,
            })
            total_chunks_tested += 1

        all_layer_results[layer_idx] = {'chunks_tested': layer_chunk_results}

    # Save
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
            'timestamp': datetime.now().isoformat(),
        },
        'baseline': {
            r['sample_id']: {
                'question': r['question'],
                'gold_answer': r['gold_answer'],
                'top1_str': r['top1_str'],
                'top5': r['top5'],
            }
            for r in baseline_results
        },
        'per_layer': all_layer_results,
    }

    model_short = args.model.split('/')[-1]
    out_file = output_dir / f"logit_change_openqa_{model_short}_{mode_str}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_file}")

    # =========================================================================
    # Overall summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    imp_kls = []
    unimp_kls = []
    imp_top1 = []
    unimp_top1 = []
    imp_gold_diffs = []
    unimp_gold_diffs = []

    for layer_data in all_layer_results.values():
        for chunk in layer_data['chunks_tested']:
            kl = chunk['mean_kl_div']
            if np.isnan(kl):
                continue
            if chunk['importance'] == 'important':
                imp_kls.append(kl)
                imp_top1.append(chunk['n_top1_changed'])
                imp_gold_diffs.append(chunk['mean_gold_answer_logit_diff'])
            elif chunk['importance'] == 'unimportant':
                unimp_kls.append(kl)
                unimp_top1.append(chunk['n_top1_changed'])
                unimp_gold_diffs.append(chunk['mean_gold_answer_logit_diff'])

    if imp_kls and unimp_kls:
        print(f"\n{'Metric':<35} {'Important':>12} {'Unimportant':>12} {'Ratio':>8}")
        print("-" * 70)
        mk_i, mk_u = np.mean(imp_kls), np.mean(unimp_kls)
        print(f"{'Mean KL divergence':<35} {mk_i:>12.4f} {mk_u:>12.4f} {mk_i/mk_u if mk_u > 0 else float('inf'):>7.1f}x")
        mt_i, mt_u = np.mean(imp_top1), np.mean(unimp_top1)
        print(f"{'Mean # top-1 changed':<35} {mt_i:>12.1f} {mt_u:>12.1f} {mt_i/mt_u if mt_u > 0 else float('inf'):>7.1f}x")
        mg_i, mg_u = np.mean(imp_gold_diffs), np.mean(unimp_gold_diffs)
        print(f"{'Mean gold answer logit diff':<35} {mg_i:>+12.3f} {mg_u:>+12.3f}")

    # Aggregate tokens across important chunks
    print("\n" + "=" * 70)
    print("TOKENS MOST AFFECTED BY IMPORTANT CHUNK REMOVAL (open QA)")
    print("=" * 70)

    all_inc = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})
    all_dec = defaultdict(lambda: {'total_diff': 0.0, 'count': 0})

    for layer_data in all_layer_results.values():
        for chunk in layer_data['chunks_tested']:
            if chunk['importance'] != 'important':
                continue
            for t in chunk['aggregate_top_increases'][:10]:
                all_inc[t['token_str']]['total_diff'] += t['mean_logit_diff'] * t['count']
                all_inc[t['token_str']]['count'] += t['count']
            for t in chunk['aggregate_top_decreases'][:10]:
                all_dec[t['token_str']]['total_diff'] += t['mean_logit_diff'] * t['count']
                all_dec[t['token_str']]['count'] += t['count']

    print("\nMost consistently INCREASED:")
    top_inc = sorted(all_inc.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
    for token, info in top_inc:
        mean = info['total_diff'] / max(info['count'], 1)
        print(f"  {token!r:<20} count={info['count']:>5}  mean_diff={mean:+.3f}")

    print("\nMost consistently DECREASED:")
    top_dec = sorted(all_dec.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
    for token, info in top_dec:
        mean = info['total_diff'] / max(info['count'], 1)
        print(f"  {token!r:<20} count={info['count']:>5}  mean_diff={mean:+.3f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
