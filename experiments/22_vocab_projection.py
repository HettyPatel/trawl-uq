"""
Experiment 22: SVD Vocabulary Projection Analysis (Logit Lens)

For each layer's MLP down_proj, decompose via SVD and project the right
singular vectors (Vh rows) through the lm_head to get a "vocabulary
fingerprint" for each SVD component. This reveals what tokens each
component encodes in the residual stream.

Key idea:
  - down_proj has shape (4096, 11008) for Llama-2
  - SVD: W = U @ diag(S) @ Vh
  - U columns are in 4096-dim residual stream (output space)
  - Vh rows are in 11008-dim intermediate space (input space)
  - lm_head maps 4096 → 32000 (vocab)
  - So U columns @ lm_head.T → which tokens each SV component activates

Combined with exp 20 importance data, we can compare:
  - Do important chunks (high flip count) project onto different tokens
    than unimportant chunks?
  - Are important chunks "sharper" (concentrated on fewer tokens)?
  - How does this change across layers?

Needs GPU for model loading but no inference — just matrix multiplies.

Usage:
    python experiments/22_vocab_projection.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --output-dir results/vocab_projection/
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
from src.decomposition.svd import decompose_weight_svd


# =============================================================================
# Load importance data from exp 20
# =============================================================================

def load_importance_data(importance_dir: Path):
    """Load flip counts and importance per chunk from exp 20."""
    pkls = sorted(glob.glob(str(importance_dir / "*.pkl")))
    pkls = [p for p in pkls if 'baseline' not in Path(p).name]

    importance_data = {}
    config_info = None

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
                'energy_removed': c.get('avg_energy_removed', 0),
                'acc_change': c.get('mcq_accuracy_change', 0),
            })
        importance_data[layer] = {
            'chunks': chunk_info,
            'chunk_size': data['config']['chunk_size'],
            'total_components': data['config']['total_components'],
            'num_chunks': data['config']['num_chunks'],
        }
        if config_info is None:
            config_info = data['config']

    print(f"Loaded importance data: {len(importance_data)} layers")
    return importance_data, config_info


# =============================================================================
# Vocabulary projection
# =============================================================================

def project_chunk_to_vocab(U_chunk, S_chunk, lm_head_weight):
    """
    Project a chunk's U columns through lm_head to get token activations.

    down_proj has shape (4096, 11008), so:
      - U columns are in the 4096-dim residual stream (output space)
      - Vh rows are in the 11008-dim intermediate space (input space)

    We need U (output space) to project through lm_head (4096 → 32000).

    For a chunk of SVD components:
      weighted_direction = sum(s_i * u_i) for each SV in the chunk
      token_logits = weighted_direction @ lm_head.T

    Args:
        U_chunk: (hidden_dim, chunk_size) — columns of U for this chunk
        S_chunk: (chunk_size,) — singular values for this chunk
        lm_head_weight: (vocab_size, hidden_dim)

    Returns:
        token_logits: (vocab_size,) — how strongly this chunk activates each token
    """
    # Weight each U column by its singular value
    # U_chunk is (hidden_dim, chunk_size), transpose to (chunk_size, hidden_dim)
    weighted = S_chunk.unsqueeze(1) * U_chunk.T  # (chunk_size, hidden_dim)

    # Sum to get the chunk's net direction in residual stream
    chunk_direction = weighted.sum(dim=0)  # (hidden_dim,)

    # Project through lm_head: direction @ lm_head.T
    token_logits = chunk_direction.float() @ lm_head_weight.float().T  # (vocab_size,)

    return token_logits


def compute_projection_stats(token_logits, tokenizer, top_k=20):
    """
    Compute statistics about a chunk's vocabulary projection.

    Returns top-k tokens, entropy (sharpness), and category breakdown.
    """
    # Softmax to get probabilities
    probs = torch.softmax(token_logits.float(), dim=0)
    log_probs = torch.log_softmax(token_logits.float(), dim=0)

    # Entropy — lower = sharper/more concentrated
    entropy = -torch.sum(probs * log_probs).item()

    # Max probability — how concentrated on single token
    max_prob = probs.max().item()

    # Top-k tokens
    topk_vals, topk_ids = torch.topk(probs, top_k)
    top_tokens = []
    for val, tid in zip(topk_vals, topk_ids):
        token_str = tokenizer.decode([tid.item()])
        top_tokens.append({
            'token_id': tid.item(),
            'token_str': token_str,
            'prob': val.item(),
        })

    # Bottom-k tokens (most suppressed)
    botk_vals, botk_ids = torch.topk(probs, top_k, largest=False)
    bottom_tokens = []
    for val, tid in zip(botk_vals, botk_ids):
        token_str = tokenizer.decode([tid.item()])
        bottom_tokens.append({
            'token_id': tid.item(),
            'token_str': token_str,
            'prob': val.item(),
        })

    # L2 norm of the projection (overall magnitude)
    projection_norm = torch.norm(token_logits.float()).item()

    return {
        'entropy': entropy,
        'max_prob': max_prob,
        'projection_norm': projection_norm,
        'top_tokens': top_tokens,
        'bottom_tokens': bottom_tokens,
    }


# =============================================================================
# Main analysis
# =============================================================================

def analyze_layer(model, layer_idx, lm_head_weight, tokenizer, importance_info,
                  chunk_size, device, top_k=20):
    """Analyze all chunks in one layer."""

    # Get down_proj weight
    W = model.model.layers[layer_idx].mlp.down_proj.weight.data

    # SVD
    U, S, Vh = decompose_weight_svd(W, device)
    total_components = len(S)
    num_chunks = (total_components + chunk_size - 1) // chunk_size

    chunk_results = []

    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total_components)

        U_chunk = U[:, start:end]  # (4096, chunk_size)
        S_chunk = S[start:end]

        # Project through lm_head (using U columns — residual stream space)
        token_logits = project_chunk_to_vocab(U_chunk, S_chunk, lm_head_weight)

        # Get stats
        stats = compute_projection_stats(token_logits, tokenizer, top_k=top_k)

        # Get importance info if available
        flip_count = 0
        importance = 'unknown'
        if importance_info:
            for c in importance_info['chunks']:
                if c['chunk_idx'] == ci:
                    flip_count = c['flip_count']
                    importance = c['importance']
                    break

        # Energy fraction for this chunk
        chunk_energy = torch.sum(S_chunk ** 2).item()
        total_energy = torch.sum(S ** 2).item()
        energy_fraction = chunk_energy / total_energy if total_energy > 0 else 0

        chunk_results.append({
            'chunk_idx': ci,
            'chunk_start': start,
            'chunk_end': end,
            'flip_count': flip_count,
            'importance': importance,
            'energy_fraction': energy_fraction,
            'sv_range': (S[start].item(), S[end - 1].item()),
            **stats,
        })

    return chunk_results


def analyze_layer_per_sv(model, layer_idx, lm_head_weight, tokenizer, importance_info,
                         chunk_size, device, top_k=10):
    """Analyze individual singular vectors in one layer."""

    W = model.model.layers[layer_idx].mlp.down_proj.weight.data

    # SVD
    U, S, Vh = decompose_weight_svd(W, device)
    total_components = len(S)
    num_chunks = (total_components + chunk_size - 1) // chunk_size

    # Project each individual U column through lm_head
    # U[:, i] is a 4096-dim direction in residual stream
    # We project the RAW direction (not scaled by S) — the direction matters,
    # not the magnitude. S-weighting makes single SVs too small for softmax
    # to differentiate.
    lm_float = lm_head_weight.float()
    U_float = U.float()
    S_float = S.float()

    # Batch: (total_components, 4096) @ (4096, 32000) = (total_components, 32000)
    all_logits = U_float.T @ lm_float.T  # (total_components, 32000)

    # Compute per-SV stats
    all_probs = torch.softmax(all_logits, dim=1)  # (total_components, 32000)
    all_log_probs = torch.log_softmax(all_logits, dim=1)

    # Entropy per SV
    sv_entropy = -torch.sum(all_probs * all_log_probs, dim=1).cpu().numpy()  # (total_components,)
    sv_max_prob = all_probs.max(dim=1).values.cpu().numpy()
    sv_norm = torch.norm(all_logits.float(), dim=1).cpu().numpy()

    # Top-k tokens per SV
    topk_vals, topk_ids = torch.topk(all_probs, top_k, dim=1)  # (total_components, top_k)

    sv_results = []
    for i in range(total_components):
        top_tokens = []
        for j in range(top_k):
            tid = topk_ids[i, j].item()
            top_tokens.append({
                'token_id': tid,
                'token_str': tokenizer.decode([tid]),
                'prob': topk_vals[i, j].item(),
            })
        sv_results.append({
            'sv_idx': i,
            'singular_value': S_float[i].item(),
            'entropy': sv_entropy[i],
            'max_prob': sv_max_prob[i],
            'projection_norm': sv_norm[i],
            'top_tokens': top_tokens,
        })

    # Aggregate per chunk
    chunk_sv_stats = []
    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total_components)

        chunk_entropies = sv_entropy[start:end]
        chunk_max_probs = sv_max_prob[start:end]
        chunk_norms = sv_norm[start:end]

        # Get importance info
        flip_count = 0
        importance = 'unknown'
        if importance_info:
            for c in importance_info['chunks']:
                if c['chunk_idx'] == ci:
                    flip_count = c['flip_count']
                    importance = c['importance']
                    break

        # Energy fraction
        chunk_energy = torch.sum(S_float[start:end] ** 2).item()
        total_energy = torch.sum(S_float ** 2).item()

        # Find sharpest SV in this chunk (lowest entropy)
        sharpest_idx = start + np.argmin(chunk_entropies)
        sharpest_sv = sv_results[sharpest_idx]

        chunk_sv_stats.append({
            'chunk_idx': ci,
            'chunk_start': start,
            'chunk_end': end,
            'flip_count': flip_count,
            'importance': importance,
            'energy_fraction': chunk_energy / total_energy if total_energy > 0 else 0,
            'mean_sv_entropy': np.mean(chunk_entropies),
            'min_sv_entropy': np.min(chunk_entropies),
            'max_sv_entropy': np.max(chunk_entropies),
            'mean_sv_max_prob': np.mean(chunk_max_probs),
            'max_sv_max_prob': np.max(chunk_max_probs),
            'mean_sv_norm': np.mean(chunk_norms),
            'sharpest_sv_idx': sharpest_idx,
            'sharpest_sv_entropy': sharpest_sv['entropy'],
            'sharpest_sv_top_tokens': sharpest_sv['top_tokens'][:5],
        })

    return chunk_sv_stats, sv_results


def main():
    parser = argparse.ArgumentParser(description="Exp 22: SVD Vocabulary Projection")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--importance-dir', type=str, default=None,
                        help='Path to exp 20 importance sweep results (optional)')
    parser.add_argument('--layers', type=str, default='0-31',
                        help='Layer range (e.g. 0-31 or 0,5,10,31)')
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top/bottom tokens to save per chunk')
    parser.add_argument('--per-sv', action='store_true',
                        help='Analyze individual singular vectors (not just chunks)')
    parser.add_argument('--output-dir', type=str, default='results/vocab_projection/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse layer range
    if '-' in args.layers:
        start, end = args.layers.split('-')
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    # Load importance data if provided
    importance_data = None
    importance_config = None
    if args.importance_dir:
        importance_data, importance_config = load_importance_data(Path(args.importance_dir))

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    print("Model loaded.")

    # Get lm_head weight
    lm_head_weight = model.lm_head.weight.data  # (vocab_size, hidden_dim)
    print(f"lm_head shape: {lm_head_weight.shape}")
    print(f"Vocab size: {lm_head_weight.shape[0]}")

    # =========================================================================
    # Per-SV analysis mode
    # =========================================================================
    if args.per_sv:
        print("\n" + "=" * 70)
        print("PER-SINGULAR-VECTOR ANALYSIS")
        print("=" * 70)

        all_chunk_sv_stats = {}
        all_sv_results = {}

        for layer_idx in layers:
            print(f"\n--- Layer {layer_idx} ---")
            imp_info = importance_data.get(layer_idx) if importance_data else None

            chunk_sv_stats, sv_results = analyze_layer_per_sv(
                model, layer_idx, lm_head_weight, tokenizer,
                imp_info, args.chunk_size, args.device, args.top_k,
            )

            all_chunk_sv_stats[layer_idx] = chunk_sv_stats
            # Save only per-SV entropy/max_prob/top_tokens (not full results to save space)
            all_sv_results[layer_idx] = sv_results

            # Print per-chunk summary with per-SV stats
            imp_chunks = [c for c in chunk_sv_stats if c['importance'] == 'important']
            unimp_chunks = [c for c in chunk_sv_stats if c['importance'] == 'unimportant']

            if imp_chunks and unimp_chunks:
                ent_i = np.mean([c['mean_sv_entropy'] for c in imp_chunks])
                ent_u = np.mean([c['mean_sv_entropy'] for c in unimp_chunks])
                min_ent_i = np.mean([c['min_sv_entropy'] for c in imp_chunks])
                min_ent_u = np.mean([c['min_sv_entropy'] for c in unimp_chunks])
                maxp_i = np.mean([c['max_sv_max_prob'] for c in imp_chunks])
                maxp_u = np.mean([c['max_sv_max_prob'] for c in unimp_chunks])

                print(f"  Important ({len(imp_chunks)} chunks):")
                print(f"    Mean SV entropy:    {ent_i:.2f}")
                print(f"    Min SV entropy:     {min_ent_i:.2f}  (sharpest SV in chunk)")
                print(f"    Max SV max_prob:    {maxp_i:.4f}")
                print(f"  Unimportant ({len(unimp_chunks)} chunks):")
                print(f"    Mean SV entropy:    {ent_u:.2f}")
                print(f"    Min SV entropy:     {min_ent_u:.2f}  (sharpest SV in chunk)")
                print(f"    Max SV max_prob:    {maxp_u:.4f}")

            # Print sharpest SV from the highest-flip chunk
            if chunk_sv_stats:
                top_chunk = max(chunk_sv_stats, key=lambda c: c['flip_count'])
                if top_chunk['flip_count'] > 0:
                    tokens_str = ', '.join(
                        repr(t['token_str']) + f"({t['prob']:.3f})"
                        for t in top_chunk['sharpest_sv_top_tokens']
                    )
                    print(f"  Highest-flip chunk {top_chunk['chunk_idx']} (flips={top_chunk['flip_count']}):")
                    print(f"    Sharpest SV idx={top_chunk['sharpest_sv_idx']}, entropy={top_chunk['sharpest_sv_entropy']:.2f}")
                    print(f"    Top tokens: {tokens_str}")

        # Save per-SV results
        results = {
            'config': {
                'model': args.model,
                'importance_dir': args.importance_dir,
                'layers': layers,
                'chunk_size': args.chunk_size,
                'top_k': args.top_k,
                'mode': 'per_sv',
                'timestamp': datetime.now().isoformat(),
            },
            'per_layer_chunk_stats': all_chunk_sv_stats,
            'per_layer_sv_results': all_sv_results,
        }

        model_short = args.model.split('/')[-1]
        out_file = output_dir / f"vocab_projection_per_sv_{model_short}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved: {out_file}")

        # Overall summary
        print("\n" + "=" * 70)
        print("PER-SV OVERALL SUMMARY")
        print("=" * 70)

        all_imp_ent = []
        all_unimp_ent = []
        all_imp_min_ent = []
        all_unimp_min_ent = []
        all_imp_maxp = []
        all_unimp_maxp = []

        for layer_idx in layers:
            for c in all_chunk_sv_stats[layer_idx]:
                if c['importance'] == 'important':
                    all_imp_ent.append(c['mean_sv_entropy'])
                    all_imp_min_ent.append(c['min_sv_entropy'])
                    all_imp_maxp.append(c['max_sv_max_prob'])
                elif c['importance'] == 'unimportant':
                    all_unimp_ent.append(c['mean_sv_entropy'])
                    all_unimp_min_ent.append(c['min_sv_entropy'])
                    all_unimp_maxp.append(c['max_sv_max_prob'])

        if all_imp_ent and all_unimp_ent:
            print(f"\n{'Metric':<30} {'Important':>12} {'Unimportant':>12} {'Diff':>10}")
            print("-" * 67)

            me_i, me_u = np.mean(all_imp_ent), np.mean(all_unimp_ent)
            print(f"{'Mean SV entropy':<30} {me_i:>12.2f} {me_u:>12.2f} {me_i-me_u:>+10.2f}")

            mme_i, mme_u = np.mean(all_imp_min_ent), np.mean(all_unimp_min_ent)
            print(f"{'Min SV entropy (sharpest)':<30} {mme_i:>12.2f} {mme_u:>12.2f} {mme_i-mme_u:>+10.2f}")

            mp_i, mp_u = np.mean(all_imp_maxp), np.mean(all_unimp_maxp)
            print(f"{'Max SV max_prob':<30} {mp_i:>12.4f} {mp_u:>12.4f} {mp_i-mp_u:>+10.4f}")

            print(f"\n  n_important={len(all_imp_ent)}, n_unimportant={len(all_unimp_ent)}")

        # Print the globally sharpest SVs
        print("\n" + "=" * 70)
        print("TOP 20 SHARPEST SINGULAR VECTORS (lowest entropy)")
        print("=" * 70)

        all_svs = []
        for layer_idx in layers:
            for sv in all_sv_results[layer_idx]:
                chunk_idx = sv['sv_idx'] // args.chunk_size
                # Find importance for this chunk
                imp = 'unknown'
                flips = 0
                if importance_data and layer_idx in importance_data:
                    for c in importance_data[layer_idx]['chunks']:
                        if c['chunk_idx'] == chunk_idx:
                            imp = c['importance']
                            flips = c['flip_count']
                            break
                all_svs.append({
                    'layer': layer_idx,
                    'sv_idx': sv['sv_idx'],
                    'chunk_idx': chunk_idx,
                    'entropy': sv['entropy'],
                    'max_prob': sv['max_prob'],
                    'singular_value': sv['singular_value'],
                    'importance': imp,
                    'flip_count': flips,
                    'top_tokens': sv['top_tokens'],
                })

        all_svs.sort(key=lambda x: x['entropy'])
        for i, sv in enumerate(all_svs[:20]):
            tokens_str = ', '.join(
                repr(t['token_str']) + f"({t['prob']:.3f})"
                for t in sv['top_tokens'][:5]
            )
            print(f"  {i+1:>2}. Layer {sv['layer']:>2} SV {sv['sv_idx']:>4} "
                  f"(chunk {sv['chunk_idx']:>2}, {sv['importance']:<11}, flips={sv['flip_count']:>3}) "
                  f"entropy={sv['entropy']:.2f} maxP={sv['max_prob']:.4f} "
                  f"S={sv['singular_value']:.1f}")
            print(f"      tokens: {tokens_str}")

        print("\nDone!")

    # =========================================================================
    # Chunk-level analysis mode (original)
    # =========================================================================
    else:
        all_results = {}

        for layer_idx in layers:
            print(f"\n--- Layer {layer_idx} ---")
            imp_info = importance_data.get(layer_idx) if importance_data else None

            chunk_results = analyze_layer(
                model, layer_idx, lm_head_weight, tokenizer,
                imp_info, args.chunk_size, args.device, args.top_k,
            )

            all_results[layer_idx] = chunk_results

            # Print summary for this layer
            imp_chunks = [c for c in chunk_results if c['importance'] == 'important']
            unimp_chunks = [c for c in chunk_results if c['importance'] == 'unimportant']

            if imp_chunks and unimp_chunks:
                avg_entropy_imp = np.mean([c['entropy'] for c in imp_chunks])
                avg_entropy_unimp = np.mean([c['entropy'] for c in unimp_chunks])
                avg_norm_imp = np.mean([c['projection_norm'] for c in imp_chunks])
                avg_norm_unimp = np.mean([c['projection_norm'] for c in unimp_chunks])
                avg_maxp_imp = np.mean([c['max_prob'] for c in imp_chunks])
                avg_maxp_unimp = np.mean([c['max_prob'] for c in unimp_chunks])

                print(f"  Important ({len(imp_chunks)} chunks):")
                print(f"    Avg entropy:    {avg_entropy_imp:.2f}")
                print(f"    Avg proj norm:  {avg_norm_imp:.2f}")
                print(f"    Avg max prob:   {avg_maxp_imp:.4f}")
                print(f"  Unimportant ({len(unimp_chunks)} chunks):")
                print(f"    Avg entropy:    {avg_entropy_unimp:.2f}")
                print(f"    Avg proj norm:  {avg_norm_unimp:.2f}")
                print(f"    Avg max prob:   {avg_maxp_unimp:.4f}")

            # Print top tokens for highest flip-count chunk
            if chunk_results:
                top_chunk = max(chunk_results, key=lambda c: c['flip_count'])
                if top_chunk['flip_count'] > 0:
                    print(f"  Highest-flip chunk (idx={top_chunk['chunk_idx']}, flips={top_chunk['flip_count']}):")
                    print(f"    Top tokens: {', '.join(repr(t['token_str']) for t in top_chunk['top_tokens'][:10])}")

                zero_flip = [c for c in chunk_results if c['flip_count'] == 0]
                if zero_flip:
                    example = zero_flip[0]
                    print(f"  Zero-flip chunk example (idx={example['chunk_idx']}):")
                    print(f"    Top tokens: {', '.join(repr(t['token_str']) for t in example['top_tokens'][:10])}")

        # Save results
        results = {
            'config': {
                'model': args.model,
                'importance_dir': args.importance_dir,
                'layers': layers,
                'chunk_size': args.chunk_size,
                'top_k': args.top_k,
                'timestamp': datetime.now().isoformat(),
            },
            'per_layer': all_results,
        }

        model_short = args.model.split('/')[-1]
        out_file = output_dir / f"vocab_projection_{model_short}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved: {out_file}")

        # Print overall summary
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)

        all_imp_entropies = []
        all_unimp_entropies = []
        all_imp_norms = []
        all_unimp_norms = []
        all_imp_maxp = []
        all_unimp_maxp = []

        for layer_idx in layers:
            for c in all_results[layer_idx]:
                if c['importance'] == 'important':
                    all_imp_entropies.append(c['entropy'])
                    all_imp_norms.append(c['projection_norm'])
                    all_imp_maxp.append(c['max_prob'])
                elif c['importance'] == 'unimportant':
                    all_unimp_entropies.append(c['entropy'])
                    all_unimp_norms.append(c['projection_norm'])
                    all_unimp_maxp.append(c['max_prob'])

        if all_imp_entropies and all_unimp_entropies:
            print(f"\n{'Metric':<25} {'Important':>12} {'Unimportant':>12} {'Diff':>10}")
            print("-" * 62)

            me_i, me_u = np.mean(all_imp_entropies), np.mean(all_unimp_entropies)
            print(f"{'Projection entropy':<25} {me_i:>12.2f} {me_u:>12.2f} {me_i-me_u:>+10.2f}")

            mn_i, mn_u = np.mean(all_imp_norms), np.mean(all_unimp_norms)
            print(f"{'Projection norm':<25} {mn_i:>12.2f} {mn_u:>12.2f} {mn_i-mn_u:>+10.2f}")

            mp_i, mp_u = np.mean(all_imp_maxp), np.mean(all_unimp_maxp)
            print(f"{'Max token prob':<25} {mp_i:>12.4f} {mp_u:>12.4f} {mp_i-mp_u:>+10.4f}")

            print(f"\n  n_important={len(all_imp_entropies)}, n_unimportant={len(all_unimp_entropies)}")

        print("\nDone!")


if __name__ == '__main__':
    main()
