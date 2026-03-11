"""
Experiment 25: Activation-Based SVD Chunk Projection

For each layer, capture the intermediate MLP activation (11008-dim vector
before down_proj), then pass it through individual SVD chunks of down_proj
and project through lm_head to get per-chunk token logits.

This answers: "What tokens does each SVD chunk push toward for a given input?"

Unlike Exp 22 (static weight projection, failed), this uses actual input-dependent
activations, so the projections should be meaningful.

Flow:
  1. Run inference with forward hooks capturing intermediate activations
     (the 11008-dim vector: act_fn(gate_proj(x)) * up_proj(x))
  2. SVD decompose down_proj into chunks
  3. For each chunk k: contribution_k = intermediate @ (U_k @ S_k @ Vh_k).T
  4. Project through lm_head: chunk_logits_k = contribution_k @ lm_head.T
  5. Softmax → which tokens does this chunk push toward?
  6. Cross-reference with importance from exp 20

Usage:
    # Quick test
    python experiments/25_activation_chunk_projection.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --importance-dir results/importance_sweep/... \\
        --eval-set data/eval_sets/eval_set_nq_open_200.json \\
        --layers 0,5,10,15 --num-questions 20

    # Full run
    python experiments/25_activation_chunk_projection.py \\
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
from src.decomposition.svd import decompose_weight_svd


# =============================================================================
# Data loading
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
# Activation capture via hooks
# =============================================================================

class MLPIntermediateHook:
    """
    Hook to capture the intermediate activation before down_proj.

    In LlamaMLP.forward:
        intermediate = act_fn(gate_proj(x)) * up_proj(x)   # (batch, seq_len, 11008)
        output = down_proj(intermediate)                     # (batch, seq_len, 4096)

    We hook down_proj's forward to capture its input (the intermediate).
    """

    def __init__(self):
        self.activation = None
        self.handle = None

    def hook_fn(self, module, input, output):
        # input is a tuple, input[0] is the intermediate activation
        # Shape: (batch, seq_len, 11008)
        # We only need the last token position
        self.activation = input[0][:, -1, :].detach()  # (batch, 11008)

    def register(self, model, layer_idx):
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        self.handle = down_proj.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()


# =============================================================================
# Core analysis
# =============================================================================

def analyze_chunk_projections(
    intermediate,    # (11008,) — intermediate activation at last token
    U, S, Vh,        # SVD of down_proj
    lm_head_weight,  # (32000, 4096)
    tokenizer,
    chunk_size,
    top_k=20,
):
    """
    For each SVD chunk, compute its contribution to the output and project
    through lm_head to get token logits.

    intermediate @ down_proj.T = intermediate @ (U @ S @ Vh).T
                               = intermediate @ Vh.T @ S @ U.T

    For chunk k (SVs k*100 to k*100+99):
        contribution_k = intermediate @ Vh_k.T @ S_k @ U_k.T   → (4096,)
        chunk_logits_k = contribution_k @ lm_head.T             → (32000,)
    """
    total_components = len(S)
    num_chunks = (total_components + chunk_size - 1) // chunk_size

    intermediate = intermediate.float()
    lm_float = lm_head_weight.float()
    U_float = U.float()
    S_float = S.float()
    Vh_float = Vh.float()

    # Compute full MLP contribution (all SVD components combined) as baseline
    full_proj = intermediate @ Vh_float.T          # (total_components,)
    full_scaled = full_proj * S_float               # (total_components,)
    full_contribution = U_float @ full_scaled       # (4096,)
    full_logits = full_contribution @ lm_float.T    # (32000,)
    full_probs = torch.softmax(full_logits, dim=0)
    full_log_probs = torch.log_softmax(full_logits, dim=0)
    full_entropy = -torch.sum(full_probs * full_log_probs).item()
    full_max_prob = full_probs.max().item()
    full_norm = torch.norm(full_contribution).item()

    # Full MLP top-k tokens
    full_topk_vals, full_topk_ids = torch.topk(full_probs, top_k)
    full_top_tokens = []
    for val, tid in zip(full_topk_vals, full_topk_ids):
        full_top_tokens.append({
            'token_id': tid.item(),
            'token_str': tokenizer.decode([tid.item()]),
            'prob': val.item(),
        })

    full_mlp_info = {
        'entropy': full_entropy,
        'max_prob': full_max_prob,
        'contribution_norm': full_norm,
        'top_tokens': full_top_tokens,
    }

    chunk_results = []

    for ci in range(num_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, total_components)

        # Get chunk components
        U_k = U_float[:, start:end]       # (4096, chunk_size)
        S_k = S_float[start:end]          # (chunk_size,)
        Vh_k = Vh_float[start:end, :]     # (chunk_size, 11008)

        # Compute chunk's contribution to residual stream
        # intermediate @ Vh_k.T → (chunk_size,)
        # multiply by S_k → (chunk_size,)
        # multiply by U_k → (4096,)
        proj_intermediate = intermediate @ Vh_k.T           # (chunk_size,)
        scaled = proj_intermediate * S_k                     # (chunk_size,)
        contribution = U_k @ scaled                          # (4096,)

        # Project through lm_head
        chunk_logits = contribution @ lm_float.T             # (32000,)

        # Also compute the contribution norm (how much this chunk contributes)
        contribution_norm = torch.norm(contribution).item()

        # Get probabilities and top tokens
        chunk_probs = torch.softmax(chunk_logits, dim=0)
        chunk_log_probs = torch.log_softmax(chunk_logits, dim=0)

        # Entropy
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

        chunk_results.append({
            'chunk_idx': ci,
            'chunk_start': start,
            'chunk_end': end,
            'contribution_norm': contribution_norm,
            'entropy': entropy,
            'max_prob': max_prob,
            'top_tokens': top_tokens,
            'bottom_tokens': bottom_tokens,
        })

    return chunk_results, full_mlp_info


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 25: Activation-Based SVD Chunk Projection")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--importance-dir', type=str, default=None)
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--layers', type=str, default='0-15')
    parser.add_argument('--num-questions', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='results/activation_projection/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mcq', action='store_true',
                        help='Eval set is MCQ format (use mcq_prompt field)')

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

    # Load eval set
    print("Loading eval set...")
    all_samples = load_eval_set(args.eval_set)
    samples = all_samples[:args.num_questions]
    print(f"Using {len(samples)} of {len(all_samples)} questions")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
    print("Model loaded.")

    lm_head_weight = model.lm_head.weight.data  # (32000, 4096)
    print(f"lm_head shape: {lm_head_weight.shape}")

    # Pre-decompose down_proj for all target layers
    print("\nDecomposing down_proj for target layers...")
    decompositions = {}
    for layer_idx in layers:
        W = model.model.layers[layer_idx].mlp.down_proj.weight.data
        U, S, Vh = decompose_weight_svd(W, args.device)
        decompositions[layer_idx] = {'U': U, 'S': S, 'Vh': Vh}
        print(f"  Layer {layer_idx}: down_proj shape={tuple(W.shape)}, SVs={len(S)}")

    # Process each question
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

        # For each layer, hook and run inference
        per_layer_results = {}

        for layer_idx in layers:
            # Register hook
            hook = MLPIntermediateHook()
            hook.register(model, layer_idx)

            # Run inference
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Get the model's prediction (last token logits)
            final_logits = outputs.logits[0, -1, :]
            top1_id = torch.argmax(final_logits).item()
            top1_str = tokenizer.decode([top1_id])

            # Get captured intermediate activation
            intermediate = hook.activation[0]  # (11008,)
            hook.remove()

            # Analyze chunk projections
            U = decompositions[layer_idx]['U']
            S = decompositions[layer_idx]['S']
            Vh = decompositions[layer_idx]['Vh']

            chunk_results, full_mlp_info = analyze_chunk_projections(
                intermediate, U, S, Vh, lm_head_weight,
                tokenizer, args.chunk_size, args.top_k,
            )

            # Add importance info
            if importance_data and layer_idx in importance_data:
                imp_chunks = {c['chunk_idx']: c for c in importance_data[layer_idx]['chunks']}
                for cr in chunk_results:
                    ci = cr['chunk_idx']
                    if ci in imp_chunks:
                        cr['flip_count'] = imp_chunks[ci]['flip_count']
                        cr['importance'] = imp_chunks[ci]['importance']
                    else:
                        cr['flip_count'] = 0
                        cr['importance'] = 'unknown'

            # Check if gold answer tokens appear in any chunk's top tokens
            gold_token_ids = set()
            for ans in all_answers:
                gold_token_ids.update(tokenizer.encode(ans, add_special_tokens=False))
                gold_token_ids.update(tokenizer.encode(" " + ans, add_special_tokens=False))

            for cr in chunk_results:
                cr['has_gold_in_top'] = any(
                    t['token_id'] in gold_token_ids for t in cr['top_tokens']
                )

            # Also check gold in full MLP top tokens
            full_mlp_info['has_gold_in_top'] = any(
                t['token_id'] in gold_token_ids for t in full_mlp_info['top_tokens']
            )

            per_layer_results[layer_idx] = {
                'chunks': chunk_results,
                'full_mlp': full_mlp_info,
            }

            # Print summary for this layer
            if q_idx < 5 or layer_idx in [0, 1, 5, 10, 15]:  # Print selectively
                # Full MLP baseline
                full_tokens_str = ', '.join(
                    f"{t['token_str']!r}({t['prob']:.3f})"
                    for t in full_mlp_info['top_tokens'][:5]
                )
                print(f"  L{layer_idx} FULL MLP: ent={full_mlp_info['entropy']:.2f} "
                      f"maxP={full_mlp_info['max_prob']:.4f} "
                      f"norm={full_mlp_info['contribution_norm']:.1f}")
                print(f"    full tokens: {full_tokens_str}")

                imp_chunks_r = [c for c in chunk_results if c.get('importance') == 'important']

                if imp_chunks_r:
                    top_chunk = max(imp_chunks_r, key=lambda c: c['max_prob'])
                    tokens_str = ', '.join(
                        f"{t['token_str']!r}({t['prob']:.3f})"
                        for t in top_chunk['top_tokens'][:5]
                    )
                    print(f"  L{layer_idx} top imp chunk {top_chunk['chunk_idx']} "
                          f"(flips={top_chunk.get('flip_count', '?')}): "
                          f"entropy={top_chunk['entropy']:.2f} norm={top_chunk['contribution_norm']:.1f}")
                    print(f"    chunk tokens: {tokens_str}")

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
            'chunk_size': args.chunk_size,
            'top_k': args.top_k,
            'mcq': args.mcq,
            'timestamp': datetime.now().isoformat(),
        },
        'per_question': all_results,
    }

    model_short = args.model.split('/')[-1]
    out_file = output_dir / f"activation_projection_{model_short}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_file}")

    # =========================================================================
    # Overall summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    # Compare entropy and max_prob for important vs unimportant chunks
    imp_entropies = []
    unimp_entropies = []
    imp_max_probs = []
    unimp_max_probs = []
    imp_norms = []
    unimp_norms = []
    imp_gold_hits = 0
    imp_total = 0
    unimp_gold_hits = 0
    unimp_total = 0

    full_mlp_entropies = []
    for sample_id, qdata in all_results.items():
        for layer_idx, layer_data in qdata['per_layer'].items():
            full_mlp_entropies.append(layer_data['full_mlp']['entropy'])
            for c in layer_data['chunks']:
                if c.get('importance') == 'important':
                    imp_entropies.append(c['entropy'])
                    imp_max_probs.append(c['max_prob'])
                    imp_norms.append(c['contribution_norm'])
                    imp_total += 1
                    if c['has_gold_in_top']:
                        imp_gold_hits += 1
                elif c.get('importance') == 'unimportant':
                    unimp_entropies.append(c['entropy'])
                    unimp_max_probs.append(c['max_prob'])
                    unimp_norms.append(c['contribution_norm'])
                    unimp_total += 1
                    if c['has_gold_in_top']:
                        unimp_gold_hits += 1

    if imp_entropies and unimp_entropies:
        print(f"\n{'Metric':<35} {'Important':>12} {'Unimportant':>12}")
        print("-" * 62)
        print(f"{'Mean entropy':<35} {np.mean(imp_entropies):>12.2f} {np.mean(unimp_entropies):>12.2f}")
        print(f"{'Mean max_prob':<35} {np.mean(imp_max_probs):>12.4f} {np.mean(unimp_max_probs):>12.4f}")
        print(f"{'Mean contribution norm':<35} {np.mean(imp_norms):>12.1f} {np.mean(unimp_norms):>12.1f}")
        print(f"{'Gold answer in top-{args.top_k}':<35} "
              f"{imp_gold_hits}/{imp_total} ({100*imp_gold_hits/imp_total:.1f}%) "
              f"{unimp_gold_hits}/{unimp_total} ({100*unimp_gold_hits/unimp_total:.1f}%)")
        print(f"\n{'Full MLP mean entropy':<35} {np.mean(full_mlp_entropies):>12.2f}")
        print(f"  (for reference: uniform over 32k ≈ 10.37)")

    # Per-layer entropy comparison
    print(f"\nPer-layer entropy (important vs unimportant):")
    print(f"{'Layer':>5} {'Imp ent':>10} {'Unimp ent':>10} {'Imp maxP':>10} {'Unimp maxP':>10} {'Imp norm':>10} {'Unimp norm':>10}")
    print("-" * 70)

    for layer_idx in layers:
        l_imp_ent = []
        l_unimp_ent = []
        l_imp_mp = []
        l_unimp_mp = []
        l_imp_n = []
        l_unimp_n = []

        for qdata in all_results.values():
            if layer_idx not in qdata['per_layer']:
                continue
            for c in qdata['per_layer'][layer_idx]['chunks']:
                if c.get('importance') == 'important':
                    l_imp_ent.append(c['entropy'])
                    l_imp_mp.append(c['max_prob'])
                    l_imp_n.append(c['contribution_norm'])
                elif c.get('importance') == 'unimportant':
                    l_unimp_ent.append(c['entropy'])
                    l_unimp_mp.append(c['max_prob'])
                    l_unimp_n.append(c['contribution_norm'])

        if l_imp_ent and l_unimp_ent:
            print(f"{layer_idx:>5} {np.mean(l_imp_ent):>10.2f} {np.mean(l_unimp_ent):>10.2f} "
                  f"{np.mean(l_imp_mp):>10.4f} {np.mean(l_unimp_mp):>10.4f} "
                  f"{np.mean(l_imp_n):>10.1f} {np.mean(l_unimp_n):>10.1f}")
        elif l_imp_ent:
            print(f"{layer_idx:>5} {np.mean(l_imp_ent):>10.2f}        -   "
                  f"{np.mean(l_imp_mp):>10.4f}          - "
                  f"{np.mean(l_imp_n):>10.1f}          -")

    # Show the most "specific" chunks (lowest entropy) and their top tokens
    print("\n" + "=" * 70)
    print(f"TOP 20 SHARPEST CHUNK PROJECTIONS (lowest entropy across all questions)")
    print("=" * 70)

    all_chunk_obs = []
    for sample_id, qdata in all_results.items():
        for layer_idx, layer_data in qdata['per_layer'].items():
            for c in layer_data['chunks']:
                all_chunk_obs.append({
                    'sample_id': sample_id,
                    'question': qdata['question'][:50],
                    'gold': qdata['gold_answer'][:20],
                    'layer': layer_idx,
                    **c,
                })

    all_chunk_obs.sort(key=lambda x: x['entropy'])
    for i, obs in enumerate(all_chunk_obs[:20]):
        tokens_str = ', '.join(
            f"{t['token_str']!r}({t['prob']:.3f})"
            for t in obs['top_tokens'][:5]
        )
        gold_flag = " ★GOLD" if obs['has_gold_in_top'] else ""
        print(f"  {i+1:>2}. L{obs['layer']:>2} ch{obs['chunk_idx']:>2} "
              f"({obs.get('importance', '?'):<11} flips={obs.get('flip_count', '?'):>3}) "
              f"ent={obs['entropy']:.2f} maxP={obs['max_prob']:.4f} "
              f"norm={obs['contribution_norm']:.1f}{gold_flag}")
        print(f"      Q: {obs['question']} | Gold: {obs['gold']}")
        print(f"      tokens: {tokens_str}")

    print("\nDone!")


if __name__ == '__main__':
    main()
