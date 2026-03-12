"""
Experiment 30: SAE "Both" Feature Selection via Individual Suppression

For features in the "both" category (confounded certainty+correctness), test suppressing
them one-by-one on the discovery set. Identify which individual features increase accuracy
AND decrease entropy, then suppress those selected features together on the held-out
validation set.

Full pipeline (run in order):
  1. Create MMLU test eval set (14K, disjoint from quadrant analysis which used the val split):
       python scripts/create_mmlu_eval_set.py --split test --samples 14042
  2. Split into discovery / validation halves:
       python scripts/split_eval_set.py --input data/eval_sets/eval_set_mcq_mmlu_test_14042.json
  3. Run SAE uncertainty on discovery:
       python experiments/28_sae_uncertainty_features.py \\
           --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json \\
           --output-dir results/sae_uncertainty_mmlu_discovery --device cuda:0
  4. Run quadrant analysis on discovery:
       python scripts/analyze_sae_quadrant.py \\
           --pickle results/sae_uncertainty_mmlu_discovery/sae_uncertainty_Llama-3.1-8B.pkl \\
           --output-dir results/sae_quadrant_mmlu_discovery --entropy-percentile 25
  5. Run this script:
       python experiments/30_sae_both_feature_selection.py --device cuda:0
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


def load_both_features(quadrant_pkl, layers):
    """Load only the 'both' category features from quadrant analysis (all of them, no top-N limit)."""
    with open(quadrant_pkl, 'rb') as f:
        data = pickle.load(f)

    p_threshold = data['config']['p_threshold']
    both_features = {}  # {layer: [feature_idx, ...]}

    print(f"  p_threshold={p_threshold}")
    for layer in layers:
        if layer not in data['all_comparisons']:
            print(f"  WARNING: Layer {layer} not in quadrant analysis, skipping")
            continue

        comp = data['all_comparisons'][layer]

        sig_pure_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}
        sig_pure_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                       if r['p_value'] < p_threshold and r['effect_size'] > 0}

        both = sig_pure_unc & sig_pure_inc

        # Rank by min(unc_effect, inc_effect) — strongest in both comparisons first
        inc_effect_lookup = {r['feature_idx']: r['effect_size']
                             for r in comp['pure_incorrectness']}
        both_ranked = sorted(
            [r for r in comp['pure_uncertainty'] if r['feature_idx'] in both],
            key=lambda x: min(x['effect_size'], inc_effect_lookup.get(x['feature_idx'], 0)),
            reverse=True
        )

        both_features[layer] = [(r['feature_idx'], r['effect_size'],
                                  inc_effect_lookup.get(r['feature_idx'], 0))
                                 for r in both_ranked]

        print(f"  Layer {layer}: {len(both_features[layer])} 'both' features: "
              f"{[f[0] for f in both_features[layer]]}")

    return both_features


class SAESuppressionHook:
    """Forward hook that suppresses specified SAE features in the residual stream."""

    def __init__(self, sae, feature_indices, scale=0.0):
        self.sae = sae
        self.feature_indices = (torch.tensor(feature_indices, dtype=torch.long)
                                if len(feature_indices) > 0
                                else torch.tensor([], dtype=torch.long))
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
    for sample in tqdm(samples, desc=desc, leave=False):
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


def summarise(results):
    acc = sum(r['correct'] for r in results) / len(results)
    ent = np.mean([r['entropy'] for r in results])
    return acc, ent


def main():
    parser = argparse.ArgumentParser(
        description="Exp 30: SAE 'Both' Feature Selection via Individual Suppression"
    )
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_discovery.json',
                        help='Discovery set — used for individual feature suppression.')
    parser.add_argument('--validation-set', type=str,
                        default='data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json',
                        help='Held-out validation set — used for final combined suppression.')
    parser.add_argument('--quadrant-pkl', type=str,
                        default='results/sae_quadrant_mmlu_discovery/quadrant_analysis.pkl')
    parser.add_argument('--suppress-layers', type=str, default='20,24,28,31',
                        help='Layers to load both features from (comma-separated)')
    parser.add_argument('--scale', type=float, default=0.0,
                        help='Suppression scale (0.0=full suppression)')
    parser.add_argument('--sae-expansion', type=str, default='8x')
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_both_feature_selection/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suppress_layers = [int(x) for x in args.suppress_layers.split(',')]

    # =========================================================================
    # Load discovery (feature selection) and validation (held-out) sets
    # =========================================================================
    selection_set = load_eval_set(args.eval_set)
    validation_set = load_eval_set(args.validation_set)
    print(f"Discovery set:  {len(selection_set)} samples")
    print(f"Validation set: {len(validation_set)} samples")

    # =========================================================================
    # Load both features
    # =========================================================================
    print("\nLoading 'both' category features...")
    both_features = load_both_features(args.quadrant_pkl, suppress_layers)

    total_both = sum(len(v) for v in both_features.values())
    print(f"\nTotal 'both' features across all layers: {total_both}")
    if total_both == 0:
        print("No 'both' features found. Exiting.")
        return

    # =========================================================================
    # Load model
    # =========================================================================
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

    # =========================================================================
    # Baseline on selection set
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (selection set, no suppression)")
    print(f"{'='*60}")
    baseline_sel = run_eval(model, tokenizer, selection_set, letter_token_ids,
                            args.device, "Baseline (selection)")
    base_sel_acc, base_sel_ent = summarise(baseline_sel)
    print(f"Baseline accuracy: {base_sel_acc*100:.2f}%  mean entropy: {base_sel_ent:.4f}")

    # =========================================================================
    # Individual feature suppression on selection set
    # =========================================================================
    from sae_lens import SAE

    individual_results = []  # list of dicts
    selected_features = {}   # {layer: [feature_idx, ...]} — beneficial features

    for layer_idx in suppress_layers:
        if layer_idx not in both_features or not both_features[layer_idx]:
            print(f"\nLayer {layer_idx}: no 'both' features, skipping")
            continue

        print(f"\nLoading SAE for layer {layer_idx}...")
        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)

        layer_selected = []

        for feat_idx, unc_effect, inc_effect in both_features[layer_idx]:
            desc = f"L{layer_idx} feat {feat_idx}"
            hook = SAESuppressionHook(sae, [feat_idx], scale=args.scale)
            hook.register(model, layer_idx)

            res = run_eval(model, tokenizer, selection_set, letter_token_ids,
                           args.device, desc)
            hook.remove()

            acc, ent = summarise(res)
            acc_delta = (acc - base_sel_acc) * 100
            ent_delta = ent - base_sel_ent
            beneficial = (acc >= base_sel_acc) and (ent < base_sel_ent)

            print(f"  L{layer_idx} feat {feat_idx:>5}: "
                  f"acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
                  f"ent={ent:.4f} ({ent_delta:+.4f})  "
                  f"{'✓ SELECTED' if beneficial else ''}")

            individual_results.append({
                'layer': layer_idx,
                'feature_idx': feat_idx,
                'unc_effect': unc_effect,
                'inc_effect': inc_effect,
                'acc_selection': acc,
                'acc_delta_selection': acc_delta,
                'ent_selection': ent,
                'ent_delta_selection': ent_delta,
                'beneficial': beneficial,
            })

            if beneficial:
                layer_selected.append(feat_idx)

        del sae
        torch.cuda.empty_cache()
        gc.collect()

        if layer_selected:
            selected_features[layer_idx] = layer_selected

    # Save individual results CSV
    ind_csv = output_dir / "individual_suppression_selection.csv"
    if individual_results:
        with open(ind_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=individual_results[0].keys())
            writer.writeheader()
            writer.writerows(individual_results)
        print(f"\nSaved individual results: {ind_csv}")

    # Summary of selection
    n_total = sum(len(v) for v in both_features.values())
    n_selected = sum(len(v) for v in selected_features.values())
    print(f"\n{'='*60}")
    print(f"Feature selection summary: {n_selected}/{n_total} features selected")
    for layer_idx, feats in selected_features.items():
        print(f"  Layer {layer_idx}: {feats}")
    print(f"{'='*60}")

    if n_selected == 0:
        print("No beneficial features found. Skipping validation run.")
        # Still save pkl with what we have
        pkl_file = output_dir / "both_feature_selection.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump({
                'config': vars(args),
                'both_features': both_features,
                'selected_features': selected_features,
                'selection_set_size': len(selection_set),
                'validation_set_size': len(validation_set),
                'baseline_selection': {'accuracy': base_sel_acc, 'mean_entropy': base_sel_ent},
                'individual_results': individual_results,
                'validation_results': None,
                'timestamp': datetime.now().isoformat(),
            }, f)
        print(f"Saved: {pkl_file}")
        return

    # =========================================================================
    # Baseline on validation set
    # =========================================================================
    print(f"\n{'='*60}")
    print("Baseline (validation set, no suppression)")
    print(f"{'='*60}")
    baseline_val = run_eval(model, tokenizer, validation_set, letter_token_ids,
                            args.device, "Baseline (validation)")
    base_val_acc, base_val_ent = summarise(baseline_val)
    print(f"Baseline accuracy: {base_val_acc*100:.2f}%  mean entropy: {base_val_ent:.4f}")

    # =========================================================================
    # Load all needed SAEs once
    # =========================================================================
    saes = {}
    for layer_idx in selected_features:
        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        print(f"  Loading SAE for layer {layer_idx}...")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)
        saes[layer_idx] = sae

    def run_suppression(layer_feats_dict, desc):
        """Suppress given {layer: [feats]} on validation set, return (acc, ent, results)."""
        hooks = []
        for lidx, feats in layer_feats_dict.items():
            h = SAESuppressionHook(saes[lidx], feats, scale=args.scale)
            h.register(model, lidx)
            hooks.append(h)
        res = run_eval(model, tokenizer, validation_set, letter_token_ids, args.device, desc)
        for h in hooks:
            h.remove()
        acc, ent = summarise(res)
        return acc, ent, res

    validation_rows = []

    def record(condition, layer, feats, acc, ent, results):
        acc_delta = (acc - base_val_acc) * 100
        ent_delta = ent - base_val_ent
        flipped_correct   = sum(1 for b, s in zip(baseline_val, results) if not b['correct'] and s['correct'])
        flipped_incorrect = sum(1 for b, s in zip(baseline_val, results) if b['correct'] and not s['correct'])
        print(f"  {condition}: acc={acc*100:.2f}% ({acc_delta:+.2f}%)  "
              f"ent={ent:.4f} ({ent_delta:+.4f})  "
              f"->correct={flipped_correct}  ->wrong={flipped_incorrect}")
        validation_rows.append({
            'condition': condition,
            'layer': layer,
            'features': str(feats),
            'n_features': len(feats) if isinstance(feats, list) else sum(len(v) for v in feats.values()),
            'accuracy': acc,
            'acc_delta': acc_delta,
            'mean_entropy': ent,
            'ent_delta': ent_delta,
            'n_flipped_to_correct': flipped_correct,
            'n_flipped_to_incorrect': flipped_incorrect,
        })

    # --- Per-layer ---
    for layer_idx, feats in selected_features.items():
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx} selected features: {feats}")
        acc, ent, res = run_suppression({layer_idx: feats}, f"L{layer_idx} combined")
        record(f"L{layer_idx}_combined", layer_idx, feats, acc, ent, res)

    # --- All layers combined ---
    print(f"\n{'='*60}")
    print(f"All layers combined: {selected_features}")
    total_selected = sum(len(v) for v in selected_features.values())
    acc, ent, res = run_suppression(selected_features, "All layers combined")
    record("all_layers_combined", "all", selected_features, acc, ent, res)

    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Save results
    # =========================================================================
    val_csv = output_dir / "validation_suppression.csv"
    with open(val_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)
    print(f"\nSaved: {val_csv}")

    pkl_file = output_dir / "both_feature_selection.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'both_features': both_features,
            'selected_features': selected_features,
            'selection_set_size': len(selection_set),
            'validation_set_size': len(validation_set),
            'baseline_selection': {'accuracy': base_sel_acc, 'mean_entropy': base_sel_ent},
            'baseline_validation': {'accuracy': base_val_acc, 'mean_entropy': base_val_ent},
            'individual_results': individual_results,
            'validation_rows': validation_rows,
            'timestamp': datetime.now().isoformat(),
        }, f)
    print(f"Saved: {pkl_file}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
