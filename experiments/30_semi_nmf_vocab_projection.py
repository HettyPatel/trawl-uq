"""
Experiment 30: Semi-NMF Activation-Based Vocab Projection

Same idea as Exp 25 (activation-based SVD chunk projection) but using Semi-NMF
instead of SVD to decompose down_proj.

Semi-NMF factorizes: W ≈ F @ G where G >= 0
  - F: (4096, r) — basis vectors in residual stream (mixed sign, like U in SVD)
  - G: (r, 11008) — non-negative coefficients in intermediate space

Why Semi-NMF over SVD for vocab projection:
  - Non-negative G means components are *additive* — no cancellation between parts
  - Each component captures a distinct "part" of the intermediate representation
  - More interpretable: a component either contributes or doesn't (no negative mixing)
  - SVD components are orthogonal but can cancel; Semi-NMF components are additive parts

Flow:
  1. Semi-NMF decompose down_proj: W ≈ F @ G (G >= 0)
  2. Run inference, capture intermediate activation h (11008-dim before down_proj)
  3. For each component k:
       activation_k = h * G[k, :]        (element-wise, how much h activates component k)
       contribution_k = F[:, k] * sum(activation_k)   (project back to residual stream)
       OR more precisely:
       contribution_k = F[:, k] * (G[k, :] @ h)       (dot product in intermediate space)
  4. Project through lm_head: chunk_logits_k = contribution_k @ lm_head.T
  5. Softmax → which tokens does this component push toward?

For chunked analysis (groups of components):
  contribution_chunk = F[:, chunk] @ diag(G[chunk, :] @ h)...
  Actually simpler: contribution_chunk = F[:, chunk] @ (G[chunk, :] @ h)
  Where (G[chunk, :] @ h) gives a vector of per-component activations.
  Then: contribution_chunk = F[:, chunk] @ coeffs   → (4096,)

Usage:
    python experiments/30_semi_nmf_vocab_projection.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --eval-set data/eval_sets/eval_set_nq_open_200.json \\
        --layers 0,5,10,15 --num-questions 20

    # Full run with importance data
    python experiments/30_semi_nmf_vocab_projection.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_nq_open_200.json \\
        --layers 0-15 --num-questions 50
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
from src.decomposition.semi_nmf import (
    decompose_weight_semi_nmf,
    get_semi_nmf_stats,
)


# =============================================================================
# Data loading (same as Exp 25)
# =============================================================================

def load_eval_set(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    return data['samples']


def load_importance_data(importance_dir: Path):
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
        importance_data[layer] = {'chunks': chunk_info}
    print(f"Loaded importance data: {len(importance_data)} layers")
    return importance_data


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
# Activation capture via hooks (same as Exp 25)
# =============================================================================

class MLPIntermediateHook:
    """
    Hook to capture the intermediate activation before down_proj.

    In LlamaMLP.forward:
        intermediate = act_fn(gate_proj(x)) * up_proj(x)   # (batch, seq_len, 11008)
        output = down_proj(intermediate)                     # (batch, seq_len, 4096)
    """

    def __init__(self):
        self.activation = None
        self.handle = None

    def hook_fn(self, module, input, output):
        self.activation = input[0][:, -1, :].detach()  # (batch, 11008)

    def register(self, model, layer_idx):
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        self.handle = down_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


# =============================================================================
# Core analysis: Semi-NMF component projection
# =============================================================================

def analyze_component_projections(
    intermediate,       # (11008,) — intermediate activation at last token
    F,                  # (4096, r) — Semi-NMF basis in residual stream
    G,                  # (r, 11008) — Semi-NMF non-negative coefficients
    lm_head_weight,     # (32000, 4096)
    tokenizer,
    chunk_size,
    top_k=20,
):
    """
    For each Semi-NMF component group (chunk), compute its contribution to
    the output and project through lm_head to get token logits.

    For down_proj W ≈ F @ G:
      output = W @ h ≈ F @ G @ h = F @ c
      where c = G @ h is a vector of per-component coefficients (non-negative
      since G >= 0 and h = act_fn(gate) * up, which is typically non-negative
      after SiLU gating).

    For chunk k (components k*chunk_size to (k+1)*chunk_size-1):
      c_k = G[chunk, :] @ h              → (chunk_size,) per-component coefficients
      contribution_k = F[:, chunk] @ c_k  → (4096,) residual stream contribution
      chunk_logits_k = contribution_k @ lm_head.T  → (32000,)
    """
    n_components = F.shape[1]
    num_chunks = (n_components + chunk_size - 1) // chunk_size

    intermediate = intermediate.float()
    lm_float = lm_head_weight.float()
    F_float = F.float()
    G_float = G.float()

    # Compute per-component coefficients: how much each component is activated
    # c = G @ h → (r,) — these are non-negative (G >= 0)
    coefficients = G_float @ intermediate  # (r,)

    # Compute full contribution (all components)
    full_contribution = F_float @ coefficients  # (4096,)
    full_logits = full_contribution @ lm_float.T  # (32000,)
    full_probs = torch.softmax(full_logits, dim=0)
    full_log_probs = torch.log_softmax(full_logits, dim=0)
    full_entropy = -torch.sum(full_probs * full_log_probs).item()
    full_max_prob = full_probs.max().item()
    full_norm = torch.norm(full_contribution).item()

    full_topk_vals, full_topk_ids = torch.topk(full_probs, top_k)
    full_top_tokens = []
    for val, tid in zip(full_topk_vals, full_topk_ids):
        full_top_tokens.append({
            'token_id': tid.item(),
            'token_str': tokenizer.decode([tid.item()]),
            'prob': val.item(),
        })

    full_info = {
        'entropy': full_entropy,
        'max_prob': full_max_prob,
        'contribution_norm': full_norm,
        'top_tokens': full_top_tokens,
    }

    chunk_results = []

    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_components)

        F_k = F_float[:, start:end]       # (4096, chunk_size)
        c_k = coefficients[start:end]      # (chunk_size,) per-component activation

        # Contribution of this chunk to residual stream
        contribution = F_k @ c_k  # (4096,)

        contribution_norm = torch.norm(contribution).item()

        # Project through lm_head
        chunk_logits = contribution @ lm_float.T  # (32000,)

        chunk_probs = torch.softmax(chunk_logits, dim=0)
        chunk_log_probs = torch.log_softmax(chunk_logits, dim=0)

        entropy = -torch.sum(chunk_probs * chunk_log_probs).item()
        max_prob = chunk_probs.max().item()

        # Top-k tokens
        topk_vals, topk_ids = torch.topk(chunk_probs, top_k)
        top_tokens = []
        for val, tid in zip(topk_vals, topk_ids):
            top_tokens.append({
                'token_id': tid.item(),
                'token_str': tokenizer.decode([tid.item()]),
                'prob': val.item(),
            })

        # Bottom-k (most suppressed)
        botk_vals, botk_ids = torch.topk(chunk_logits, top_k, largest=False)
        bottom_tokens = []
        for val, tid in zip(botk_vals, botk_ids):
            bottom_tokens.append({
                'token_id': tid.item(),
                'token_str': tokenizer.decode([tid.item()]),
                'logit': val.item(),
            })

        # Per-component coefficient stats (unique to Semi-NMF)
        coeff_mean = c_k.mean().item()
        coeff_max = c_k.max().item()
        coeff_sparsity = (c_k < 1e-6).float().mean().item()

        chunk_results.append({
            'chunk_idx': ci,
            'chunk_start': start,
            'chunk_end': end,
            'contribution_norm': contribution_norm,
            'entropy': entropy,
            'max_prob': max_prob,
            'top_tokens': top_tokens,
            'bottom_tokens': bottom_tokens,
            # Semi-NMF specific: coefficient activation pattern
            'coeff_mean': coeff_mean,
            'coeff_max': coeff_max,
            'coeff_sparsity': coeff_sparsity,
        })

    return chunk_results, full_info, coefficients.cpu().numpy()


# =============================================================================
# Static (weight-only) projection analysis
# =============================================================================

def analyze_static_projections(
    F,                  # (4096, r) — Semi-NMF basis
    G,                  # (r, 11008) — Semi-NMF non-negative coefficients
    lm_head_weight,     # (32000, 4096)
    tokenizer,
    chunk_size,
    top_k=20,
):
    """
    Static analysis: project each F column through lm_head to see what
    vocabulary each Semi-NMF basis vector corresponds to (no input needed).

    Unlike SVD's U columns (which are orthonormal), F columns have varying
    norms and are not orthogonal — their magnitude matters.
    """
    n_components = F.shape[1]
    lm_float = lm_head_weight.float()
    F_float = F.float()
    G_float = G.float()

    # Project all F columns through lm_head at once
    # F.T @ lm_head.T → (r, 32000)
    all_logits = F_float.T @ lm_float.T  # (r, 32000)
    all_probs = torch.softmax(all_logits, dim=1)
    all_log_probs = torch.log_softmax(all_logits, dim=1)

    # Per-component entropy
    comp_entropy = -torch.sum(all_probs * all_log_probs, dim=1).cpu().numpy()
    comp_max_prob = all_probs.max(dim=1).values.cpu().numpy()

    # F column norms (component strength in residual stream)
    f_norms = torch.norm(F_float, dim=0).cpu().numpy()
    # G row norms (component "receptive field" size in intermediate space)
    g_norms = torch.norm(G_float, dim=1).cpu().numpy()
    # G row sparsity
    g_sparsity = (G_float < 1e-6).float().mean(dim=1).cpu().numpy()

    # Top-k per component
    topk_vals, topk_ids = torch.topk(all_probs, top_k, dim=1)

    component_results = []
    for i in range(n_components):
        top_tokens = []
        for j in range(top_k):
            tid = topk_ids[i, j].item()
            top_tokens.append({
                'token_id': tid,
                'token_str': tokenizer.decode([tid]),
                'prob': topk_vals[i, j].item(),
            })
        component_results.append({
            'component_idx': i,
            'entropy': float(comp_entropy[i]),
            'max_prob': float(comp_max_prob[i]),
            'f_norm': float(f_norms[i]),
            'g_norm': float(g_norms[i]),
            'g_sparsity': float(g_sparsity[i]),
            'strength': float(f_norms[i] * g_norms[i]),
            'top_tokens': top_tokens,
        })

    return component_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 30: Semi-NMF Activation-Based Vocab Projection")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--importance-dir', type=str, default=None)
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--layers', type=str, default='0-15')
    parser.add_argument('--num-questions', type=int, default=50)
    parser.add_argument('--n-components', type=int, default=4096,
                        help='Number of Semi-NMF components (rank). Use full rank '
                             '(4096 for Llama-2-7b) for faithful decomposition; '
                             'interpretability comes from G>=0, not rank reduction.')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Components per chunk for grouped analysis')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--nmf-max-iter', type=int, default=500,
                        help='Max iterations for Semi-NMF convergence')
    parser.add_argument('--output-dir', type=str, default='results/semi_nmf_projection/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mcq', action='store_true',
                        help='Eval set is MCQ format')
    parser.add_argument('--static-only', action='store_true',
                        help='Only do static (weight-only) projection, skip inference')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layer_spec(args.layers)

    # Load importance data
    importance_data = None
    if args.importance_dir:
        print("Loading importance data...")
        importance_data = load_importance_data(Path(args.importance_dir))

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    print("Model loaded.")

    lm_head_weight = model.lm_head.weight.data  # (32000, 4096)
    print(f"lm_head shape: {lm_head_weight.shape}")

    # Pre-decompose down_proj for all target layers via Semi-NMF
    print(f"\nSemi-NMF decomposing down_proj for target layers "
          f"(r={args.n_components}, max_iter={args.nmf_max_iter})...")
    decompositions = {}
    decomposition_stats = {}

    for layer_idx in layers:
        W = model.model.layers[layer_idx].mlp.down_proj.weight.data
        print(f"  Layer {layer_idx}: down_proj shape={tuple(W.shape)}, "
              f"decomposing into r={args.n_components}...")

        F, G = decompose_weight_semi_nmf(
            W, args.n_components, args.device,
            max_iter=args.nmf_max_iter, verbose=True,
        )

        stats = get_semi_nmf_stats(W, F, G)
        decompositions[layer_idx] = {'F': F, 'G': G}
        decomposition_stats[layer_idx] = stats

        print(f"    relative_error={stats['relative_error']:.4f}, "
              f"G_sparsity={stats['g_sparsity']:.3f}")

    # =========================================================================
    # Static analysis (weight-only, no inference needed)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STATIC ANALYSIS: per-component vocabulary fingerprints")
    print("=" * 70)

    static_results = {}
    for layer_idx in layers:
        F = decompositions[layer_idx]['F']
        G = decompositions[layer_idx]['G']

        comp_results = analyze_static_projections(
            F, G, lm_head_weight, tokenizer, args.chunk_size, args.top_k,
        )
        static_results[layer_idx] = comp_results

        # Print the sharpest components
        sorted_by_entropy = sorted(comp_results, key=lambda c: c['entropy'])
        print(f"\n  Layer {layer_idx} — Top 5 sharpest components:")
        for c in sorted_by_entropy[:5]:
            tokens_str = ', '.join(
                f"{t['token_str']!r}({t['prob']:.3f})"
                for t in c['top_tokens'][:5]
            )
            print(f"    comp {c['component_idx']:>3}: "
                  f"ent={c['entropy']:.2f} maxP={c['max_prob']:.4f} "
                  f"|F|={c['f_norm']:.2f} |G|={c['g_norm']:.2f} "
                  f"G_sparse={c['g_sparsity']:.2f}")
            print(f"      tokens: {tokens_str}")

        # Compare with SVD-based importance if available
        if importance_data and layer_idx in importance_data:
            # Map Semi-NMF components to SVD chunks for rough comparison
            print(f"    (importance data available for cross-reference)")

    if args.static_only:
        # Save and exit
        results = {
            'config': {
                'model': args.model,
                'layers': layers,
                'n_components': args.n_components,
                'chunk_size': args.chunk_size,
                'top_k': args.top_k,
                'nmf_max_iter': args.nmf_max_iter,
                'mode': 'static_only',
                'timestamp': datetime.now().isoformat(),
            },
            'decomposition_stats': decomposition_stats,
            'static_results': static_results,
        }

        model_short = args.model.split('/')[-1]
        out_file = output_dir / f"semi_nmf_static_{model_short}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved: {out_file}")
        print("Done (static-only mode)!")
        return

    # =========================================================================
    # Activation-based analysis (with inference)
    # =========================================================================
    print("\nLoading eval set...")
    all_samples = load_eval_set(args.eval_set)
    samples = all_samples[:args.num_questions]
    print(f"Using {len(samples)} of {len(all_samples)} questions")

    all_results = {}

    for q_idx, sample in enumerate(samples):
        # Get prompt
        if args.mcq:
            prompt = sample['mcq_prompt']
            question = sample.get('question', prompt[:60])
            gold_answer = sample.get('correct_letter', '?')
            all_answers = [gold_answer]
        else:
            prompt = sample['prompt']
            question = sample['question']
            gold_answer = sample['answer']
            all_answers = sample.get('all_answers', [gold_answer])

        sample_id = sample.get('id', f'q_{q_idx}')

        print(f"\n--- Q{q_idx}: {question[:60]} ---")
        print(f"    Gold: {gold_answer}")

        per_layer_results = {}

        for layer_idx in layers:
            # Register hook
            hook = MLPIntermediateHook()
            hook.register(model, layer_idx)

            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)

            final_logits = outputs.logits[0, -1, :]
            top1_id = torch.argmax(final_logits).item()
            top1_str = tokenizer.decode([top1_id])

            intermediate = hook.activation[0]  # (11008,)
            hook.remove()

            # Analyze with Semi-NMF components
            F = decompositions[layer_idx]['F']
            G = decompositions[layer_idx]['G']

            chunk_results, full_info, coefficients = analyze_component_projections(
                intermediate, F, G, lm_head_weight,
                tokenizer, args.chunk_size, args.top_k,
            )

            # Add importance info if available
            if importance_data and layer_idx in importance_data:
                # Note: importance data is from SVD chunks (size 100), Semi-NMF chunks
                # may have different size. We store it for reference but it's not 1:1.
                imp_chunks = {c['chunk_idx']: c
                              for c in importance_data[layer_idx]['chunks']}
                for cr in chunk_results:
                    ci = cr['chunk_idx']
                    if ci in imp_chunks:
                        cr['svd_flip_count'] = imp_chunks[ci]['flip_count']
                        cr['svd_importance'] = imp_chunks[ci]['importance']

            # Check if gold answer appears in any chunk's top tokens
            gold_token_ids = set()
            for ans in all_answers:
                gold_token_ids.update(
                    tokenizer.encode(ans, add_special_tokens=False))
                gold_token_ids.update(
                    tokenizer.encode(" " + ans, add_special_tokens=False))

            for cr in chunk_results:
                cr['has_gold_in_top'] = any(
                    t['token_id'] in gold_token_ids for t in cr['top_tokens']
                )

            full_info['has_gold_in_top'] = any(
                t['token_id'] in gold_token_ids for t in full_info['top_tokens']
            )

            per_layer_results[layer_idx] = {
                'chunks': chunk_results,
                'full_semi_nmf': full_info,
                'coefficient_stats': {
                    'mean': float(np.mean(coefficients)),
                    'std': float(np.std(coefficients)),
                    'max': float(np.max(coefficients)),
                    'sparsity': float(np.mean(coefficients < 1e-6)),
                    'top5_components': list(np.argsort(coefficients)[::-1][:5].astype(int)),
                },
            }

            # Print summary
            if q_idx < 5 or layer_idx in [0, 1, 5, 10, 15]:
                full_tokens_str = ', '.join(
                    f"{t['token_str']!r}({t['prob']:.3f})"
                    for t in full_info['top_tokens'][:5]
                )
                print(f"  L{layer_idx} FULL Semi-NMF: ent={full_info['entropy']:.2f} "
                      f"maxP={full_info['max_prob']:.4f} "
                      f"norm={full_info['contribution_norm']:.1f}")
                print(f"    tokens: {full_tokens_str}")

                # Highest-norm chunk
                top_chunk = max(chunk_results, key=lambda c: c['contribution_norm'])
                tc_tokens = ', '.join(
                    f"{t['token_str']!r}({t['prob']:.3f})"
                    for t in top_chunk['top_tokens'][:5]
                )
                print(f"    top chunk {top_chunk['chunk_idx']}: "
                      f"ent={top_chunk['entropy']:.2f} "
                      f"norm={top_chunk['contribution_norm']:.1f} "
                      f"coeff_sparsity={top_chunk['coeff_sparsity']:.2f}")
                print(f"      tokens: {tc_tokens}")

        all_results[sample_id] = {
            'question': question,
            'gold_answer': gold_answer,
            'all_answers': all_answers,
            'model_top1': top1_str,
            'per_layer': per_layer_results,
        }

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        'config': {
            'model': args.model,
            'importance_dir': args.importance_dir,
            'eval_set': args.eval_set,
            'layers': layers,
            'num_questions': len(samples),
            'n_components': args.n_components,
            'chunk_size': args.chunk_size,
            'top_k': args.top_k,
            'nmf_max_iter': args.nmf_max_iter,
            'mcq': args.mcq,
            'timestamp': datetime.now().isoformat(),
        },
        'decomposition_stats': decomposition_stats,
        'static_results': static_results,
        'per_question': all_results,
    }

    model_short = args.model.split('/')[-1]
    out_file = output_dir / f"semi_nmf_projection_{model_short}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_file}")

    # =========================================================================
    # Overall summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    # Coefficient sparsity across layers (Semi-NMF specific)
    print(f"\nPer-layer coefficient activation patterns:")
    print(f"{'Layer':>5} {'Mean coeff':>12} {'Sparsity':>10} {'Top5 comps':>30}")
    print("-" * 60)

    for layer_idx in layers:
        layer_means = []
        layer_sparsity = []
        layer_top5 = defaultdict(int)

        for qdata in all_results.values():
            if layer_idx not in qdata['per_layer']:
                continue
            cs = qdata['per_layer'][layer_idx]['coefficient_stats']
            layer_means.append(cs['mean'])
            layer_sparsity.append(cs['sparsity'])
            for comp_idx in cs['top5_components']:
                layer_top5[comp_idx] += 1

        if layer_means:
            # Most frequently appearing top-5 components
            common_comps = sorted(layer_top5.items(), key=lambda x: -x[1])[:5]
            comps_str = ', '.join(f"c{c}({n})" for c, n in common_comps)
            print(f"{layer_idx:>5} {np.mean(layer_means):>12.4f} "
                  f"{np.mean(layer_sparsity):>10.3f} {comps_str:>30}")

    # Entropy comparison: chunks with gold answer vs without
    gold_ent = []
    no_gold_ent = []
    for qdata in all_results.values():
        for layer_data in qdata['per_layer'].values():
            for c in layer_data['chunks']:
                if c['has_gold_in_top']:
                    gold_ent.append(c['entropy'])
                else:
                    no_gold_ent.append(c['entropy'])

    if gold_ent and no_gold_ent:
        print(f"\nChunks with gold in top-{args.top_k}: "
              f"n={len(gold_ent)}, mean_ent={np.mean(gold_ent):.2f}")
        print(f"Chunks without gold:                   "
              f"n={len(no_gold_ent)}, mean_ent={np.mean(no_gold_ent):.2f}")

    # Sharpest chunks overall
    print("\n" + "=" * 70)
    print(f"TOP 20 SHARPEST CHUNK PROJECTIONS (lowest entropy)")
    print("=" * 70)

    all_chunk_obs = []
    for sample_id, qdata in all_results.items():
        for layer_idx, layer_data in qdata['per_layer'].items():
            for c in layer_data['chunks']:
                all_chunk_obs.append({
                    'sample_id': sample_id,
                    'question': qdata['question'][:50],
                    'gold': qdata['gold_answer'][:20] if isinstance(qdata['gold_answer'], str) else str(qdata['gold_answer']),
                    'layer': layer_idx,
                    **c,
                })

    all_chunk_obs.sort(key=lambda x: x['entropy'])
    for i, obs in enumerate(all_chunk_obs[:20]):
        tokens_str = ', '.join(
            f"{t['token_str']!r}({t['prob']:.3f})"
            for t in obs['top_tokens'][:5]
        )
        gold_flag = " *GOLD*" if obs['has_gold_in_top'] else ""
        print(f"  {i+1:>2}. L{obs['layer']:>2} ch{obs['chunk_idx']:>2} "
              f"ent={obs['entropy']:.2f} maxP={obs['max_prob']:.4f} "
              f"norm={obs['contribution_norm']:.1f} "
              f"coeff_sparse={obs['coeff_sparsity']:.2f}{gold_flag}")
        print(f"      Q: {obs['question']} | Gold: {obs['gold']}")
        print(f"      tokens: {tokens_str}")

    # Decomposition quality summary
    print("\n" + "=" * 70)
    print("DECOMPOSITION QUALITY")
    print("=" * 70)
    for layer_idx in layers:
        s = decomposition_stats[layer_idx]
        print(f"  Layer {layer_idx:>2}: relative_error={s['relative_error']:.4f} "
              f"G_sparsity={s['g_sparsity']:.3f} "
              f"mean_strength={s['mean_component_strength']:.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
