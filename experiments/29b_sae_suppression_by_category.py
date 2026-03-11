"""
Experiment 29b: SAE Feature Suppression by Category

Suppress features grouped by their quadrant classification:
  - pure_uncertainty: features that encode uncertainty regardless of correctness (C vs A)
  - pure_incorrectness: features that encode incorrectness regardless of confidence (B vs A)
  - both: features confounded — significant in both comparisons

This disentangles whether the accuracy improvement from suppression in Exp 29
was driven by genuine uncertainty features or incorrectness-correlated features.

Usage:
    python experiments/29b_sae_suppression_by_category.py \
        --device cuda:0 --load-in-8bit
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
import gc

from src.generation.generate import seed_everything


def load_eval_set(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    return data['samples']


def get_predicted_letter(logits, letter_token_ids):
    letter_logits = {l: logits[tid].item() for l, tid in letter_token_ids.items()}
    return max(letter_logits, key=letter_logits.get)


def compute_entropy(logits, letter_token_ids):
    abcd_logits = torch.tensor([logits[tid].item() for tid in letter_token_ids.values()])
    probs = torch.softmax(abcd_logits, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return entropy


def load_quadrant_features(quadrant_pkl, layers, top_n=5, both_ranking='unc'):
    """Load features per category from quadrant analysis."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    features_by_category = {}  # {layer: {category: [feature_indices]}}

    for layer in layers:
        if layer not in data['all_comparisons']:
            print(f"  WARNING: Layer {layer} not in quadrant analysis, skipping")
            continue

        comp = data['all_comparisons'][layer]

        # Get significant features from each pure comparison
        sig_pure_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_pure_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}

        pure_unc_only = sig_pure_unc - sig_pure_inc
        pure_inc_only = sig_pure_inc - sig_pure_unc
        both = sig_pure_unc & sig_pure_inc

        # Rank by effect size within each category, take top N
        unc_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in pure_unc_only],
            key=lambda x: x['effect_size'], reverse=True
        )
        inc_ranked = sorted(
            [r for r in comp['pure_incorrectness'] if r['feature_idx'] in pure_inc_only],
            key=lambda x: x['effect_size'], reverse=True
        )
        inc_effect_lookup = {r['feature_idx']: r['effect_size']
                             for r in comp['pure_incorrectness']}
        if both_ranking == 'min':
            both_key = lambda x: min(x['effect_size'], inc_effect_lookup.get(x['feature_idx'], 0))
        else:
            both_key = lambda x: x['effect_size']
        both_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in both],
            key=both_key, reverse=True
        )

        features_by_category[layer] = {
            'pure_uncertainty': [r['feature_idx'] for r in unc_ranked[:top_n]],
            'pure_incorrectness': [r['feature_idx'] for r in inc_ranked[:top_n]],
            'both': [r['feature_idx'] for r in both_ranked[:top_n]],
        }

        for cat, feats in features_by_category[layer].items():
            print(f"  Layer {layer} [{cat}]: {feats} ({len(feats)} features)")

    return features_by_category


class SAESuppressionHook:
    """Forward hook that suppresses specified SAE features in the residual stream."""

    def __init__(self, sae, feature_indices, scale=0.0):
        self.sae = sae
        self.feature_indices = torch.tensor(feature_indices) if feature_indices else torch.tensor([], dtype=torch.long)
        self.scale = scale
        self.handle = None

    def hook_fn(self, module, input, output):
        if len(self.feature_indices) == 0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        last_token = hidden[:, -1:, :].float()

        with torch.no_grad():
            feat_acts = self.sae.encode(last_token)
            feat_acts_modified = feat_acts.clone()
            feat_acts_modified[:, :, self.feature_indices] *= self.scale
            reconstructed = self.sae.decode(feat_acts_modified)
            original_reconstructed = self.sae.decode(feat_acts)
            delta = reconstructed - original_reconstructed
            modified_last = hidden[:, -1:, :] + delta.to(hidden.dtype)

        hidden_modified = hidden.clone()
        hidden_modified[:, -1:, :] = modified_last

        if rest is not None:
            return (hidden_modified,) + rest
        return hidden_modified

    def register(self, model, layer_idx):
        self.handle = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def run_eval(model, tokenizer, samples, letter_token_ids, device, desc="Eval"):
    results = []
    for sample in tqdm(samples, desc=desc):
        prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
        correct_letter = sample.get('correct_letter', None)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].detach()
        predicted = get_predicted_letter(logits, letter_token_ids)
        correct = (predicted == correct_letter) if correct_letter else None
        entropy = compute_entropy(logits, letter_token_ids)
        results.append({
            'predicted': predicted,
            'correct_letter': correct_letter,
            'correct': correct,
            'entropy': entropy,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 29b: SAE Suppression by Feature Category")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_arc_challenge_500.json')
    parser.add_argument('--quadrant-pkl', type=str,
                        default='results/sae_quadrant/quadrant_analysis.pkl')
    parser.add_argument('--suppress-layers', type=str, default='20,24,28,31',
                        help='Layers to suppress features at (comma-separated)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Max features per category per layer')
    parser.add_argument('--scale', type=float, default=0.0,
                        help='Suppression scale (0.0=full suppression)')
    parser.add_argument('--sae-expansion', type=str, default='8x')
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_suppression_by_category/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--both-ranking', type=str, default='unc', choices=['unc', 'min'],
                        help='Ranking for "both" features: unc=by uncertainty effect, min=by min(unc,inc) effect')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suppress_layers = [int(x) for x in args.suppress_layers.split(',')]

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # Load quadrant-classified features
    print("\nLoading quadrant-classified features...")
    features_by_category = load_quadrant_features(
        args.quadrant_pkl, suppress_layers, args.top_n, args.both_ranking
    )

    # Load model
    print(f"\nLoading model: {args.model}")
    if args.load_in_8bit:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": args.device},
            low_cpu_mem_usage=True,
        )
        model.eval()
    else:
        from src.generation.generate import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    letter_token_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            letter_token_ids[letter] = ids[-1]
    print(f"Letter token IDs: {letter_token_ids}")

    csv_file = output_dir / "suppression_by_category.csv"

    def save_partial(results):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # =========================================================================
    # Baseline
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (no suppression)")
    print(f"{'='*60}")
    baseline_results = run_eval(model, tokenizer, samples, letter_token_ids,
                                args.device, "Baseline")
    baseline_acc = sum(r['correct'] for r in baseline_results) / len(baseline_results)
    baseline_entropy = np.mean([r['entropy'] for r in baseline_results])
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%  mean entropy: {baseline_entropy:.4f}")

    # =========================================================================
    # Category suppression tests
    # =========================================================================
    from sae_lens import SAE

    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    all_results = [{
        'condition': 'baseline',
        'category': None,
        'layer': None,
        'scale': None,
        'n_features': 0,
        'features': '',
        'accuracy': baseline_acc,
        'acc_delta': 0.0,
        'mean_entropy': baseline_entropy,
        'n_flipped_to_correct': 0,
        'n_flipped_to_incorrect': 0,
    }]

    # --- Per-layer, per-category tests ---
    for layer_idx in suppress_layers:
        if layer_idx not in features_by_category:
            continue

        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        print(f"\nLoading SAE for layer {layer_idx}...")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)

        for category in categories:
            feat_indices = features_by_category[layer_idx][category]
            if not feat_indices:
                print(f"\n  L{layer_idx} [{category}]: no features, skipping")
                continue

            print(f"\n{'='*60}")
            print(f"Layer {layer_idx} | {category} | scale={args.scale} | "
                  f"{len(feat_indices)} features: {feat_indices}")
            print(f"{'='*60}")

            hook = SAESuppressionHook(sae, feat_indices, scale=args.scale)
            hook.register(model, layer_idx)

            suppressed_results = run_eval(
                model, tokenizer, samples, letter_token_ids, args.device,
                f"L{layer_idx} {category}"
            )

            hook.remove()

            acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
            mean_ent = np.mean([r['entropy'] for r in suppressed_results])
            flipped_to_correct = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                     if not bl['correct'] and sp['correct'])
            flipped_to_incorrect = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                       if bl['correct'] and not sp['correct'])

            delta = (acc - baseline_acc) * 100
            print(f"  Accuracy: {acc*100:.1f}% (delta: {delta:+.1f}%)")
            print(f"  Mean entropy: {mean_ent:.4f}")
            print(f"  Flipped to correct: {flipped_to_correct}, "
                  f"flipped to incorrect: {flipped_to_incorrect}")

            all_results.append({
                'condition': f'L{layer_idx}_{category}',
                'category': category,
                'layer': layer_idx,
                'scale': args.scale,
                'n_features': len(feat_indices),
                'features': str(feat_indices),
                'accuracy': acc,
                'acc_delta': delta,
                'mean_entropy': mean_ent,
                'n_flipped_to_correct': flipped_to_correct,
                'n_flipped_to_incorrect': flipped_to_incorrect,
            })
            save_partial(all_results)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    # --- All layers simultaneously, per category ---
    print(f"\n{'='*60}")
    print("All layers simultaneously, by category")
    print(f"{'='*60}")

    # Load all SAEs
    saes = {}
    for layer_idx in suppress_layers:
        if layer_idx not in features_by_category:
            continue
        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)
        saes[layer_idx] = sae

    for category in categories:
        # Check if any layer has features for this category
        any_features = any(
            features_by_category.get(l, {}).get(category, [])
            for l in suppress_layers
        )
        if not any_features:
            print(f"\n  All layers [{category}]: no features in any layer, skipping")
            continue

        hooks = []
        total_feats = 0
        all_feat_ids = {}
        for layer_idx in suppress_layers:
            if layer_idx not in features_by_category or layer_idx not in saes:
                continue
            feat_indices = features_by_category[layer_idx][category]
            if not feat_indices:
                continue
            hook = SAESuppressionHook(saes[layer_idx], feat_indices, scale=args.scale)
            hook.register(model, layer_idx)
            hooks.append(hook)
            total_feats += len(feat_indices)
            all_feat_ids[layer_idx] = feat_indices

        print(f"\n{'='*60}")
        print(f"All layers | {category} | scale={args.scale} | {total_feats} total features")
        for l, f in all_feat_ids.items():
            print(f"  L{l}: {f}")
        print(f"{'='*60}")

        suppressed_results = run_eval(
            model, tokenizer, samples, letter_token_ids, args.device,
            f"All {category}"
        )

        for h in hooks:
            h.remove()

        acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
        mean_ent = np.mean([r['entropy'] for r in suppressed_results])
        flipped_to_correct = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                 if not bl['correct'] and sp['correct'])
        flipped_to_incorrect = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                   if bl['correct'] and not sp['correct'])

        delta = (acc - baseline_acc) * 100
        print(f"  Accuracy: {acc*100:.1f}% (delta: {delta:+.1f}%)")
        print(f"  Mean entropy: {mean_ent:.4f}")
        print(f"  Flipped to correct: {flipped_to_correct}, "
              f"flipped to incorrect: {flipped_to_incorrect}")

        all_results.append({
            'condition': f'all_layers_{category}',
            'category': category,
            'layer': 'all',
            'scale': args.scale,
            'n_features': total_feats,
            'features': str(all_feat_ids),
            'accuracy': acc,
            'acc_delta': delta,
            'mean_entropy': mean_ent,
            'n_flipped_to_correct': flipped_to_correct,
            'n_flipped_to_incorrect': flipped_to_incorrect,
        })
        save_partial(all_results)

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'Cat':<18} {'#F':>3} {'Acc':>6} {'Delta':>7} "
          f"{'->Corr':>7} {'->Wrong':>8}")
    print("-" * 70)
    for r in all_results:
        cat = r['category'] or ''
        print(f"{r['condition']:<35} {cat:<18} {r['n_features']:>3} "
              f"{r['accuracy']*100:>5.1f}% {r['acc_delta']:>+6.1f}% "
              f"{r['n_flipped_to_correct']:>7} {r['n_flipped_to_incorrect']:>8}")

    # Final save (CSV already written incrementally)
    save_partial(all_results)
    print(f"\nSaved: {csv_file}")

    pkl_file = output_dir / "suppression_by_category.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'suppress_layers': suppress_layers,
                'top_n': args.top_n,
                'scale': args.scale,
                'features_by_category': features_by_category,
                'timestamp': datetime.now().isoformat(),
            },
            'results': all_results,
            'baseline_results': baseline_results,
        }, f)
    print(f"Saved: {pkl_file}")


if __name__ == '__main__':
    main()
