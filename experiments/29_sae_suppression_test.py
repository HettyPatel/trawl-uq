"""
Experiment 29: SAE Uncertainty Feature Suppression Test

Quick test: suppress top uncertainty features identified in Experiment 28
and measure whether accuracy improves.

Approach:
  1. Load model + SAE for a target layer
  2. Register a forward hook that:
     a. Encodes the residual stream through the SAE
     b. Zeros out specified uncertainty features
     c. Decodes back and replaces the residual stream
  3. Run all 500 MCQ questions and compare accuracy to baseline

Usage:
    python experiments/29_sae_suppression_test.py \
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


def load_top_uncertainty_features(diff_csv, layers, top_n=5):
    """Load top uncertainty features per layer from experiment 28 results."""
    with open(diff_csv) as f:
        rows = list(csv.DictReader(f))
    features = {}
    for layer in layers:
        feats = [r for r in rows
                 if int(r['layer']) == layer
                 and float(r['p_value']) < 0.05
                 and r['direction'] == 'uncertainty']
        feats.sort(key=lambda r: float(r['effect_size']), reverse=True)
        features[layer] = [int(f['feature_idx']) for f in feats[:top_n]]
        print(f"  Layer {layer}: suppressing features {features[layer]}")
    return features


class SAESuppressionHook:
    """Forward hook that suppresses specified SAE features in the residual stream."""

    def __init__(self, sae, feature_indices, scale=0.0):
        """
        Args:
            sae: Loaded SAE model
            feature_indices: List of feature indices to suppress
            scale: Scale factor (0.0 = full suppression, 0.5 = half, 1.0 = no change)
        """
        self.sae = sae
        self.feature_indices = torch.tensor(feature_indices)
        self.scale = scale
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Only modify the last token position
        last_token = hidden[:, -1:, :].float()  # [1, 1, d_model]

        with torch.no_grad():
            # Encode through SAE
            feat_acts = self.sae.encode(last_token)  # [1, 1, d_sae]

            # Suppress uncertainty features
            feat_acts_modified = feat_acts.clone()
            feat_acts_modified[:, :, self.feature_indices] *= self.scale

            # Decode back
            reconstructed = self.sae.decode(feat_acts_modified)
            original_reconstructed = self.sae.decode(feat_acts)

            # Compute the delta and apply it to the original hidden state
            # This preserves information not captured by the SAE
            delta = reconstructed - original_reconstructed
            modified_last = hidden[:, -1:, :] + delta.to(hidden.dtype)

        # Replace last token
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
    """Run MCQ eval and return per-question results."""
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
    parser = argparse.ArgumentParser(description="Exp 29: SAE Feature Suppression Test")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str,
                        default='data/eval_sets/eval_set_mcq_arc_challenge_500.json')
    parser.add_argument('--diff-csv', type=str,
                        default='results/sae_uncertainty/differential_features_Llama-3.1-8B.csv')
    parser.add_argument('--suppress-layers', type=str, default='20,24,28,31',
                        help='Layers to suppress features at (comma-separated)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top uncertainty features to suppress per layer')
    parser.add_argument('--scales', type=str, default='0.0,0.5,1.0',
                        help='Suppression scales to test (comma-separated, 1.0=SAE passthrough control)')
    parser.add_argument('--sae-expansion', type=str, default='8x')
    parser.add_argument('--output-dir', type=str,
                        default='results/sae_suppression/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-in-8bit', action='store_true')

    args = parser.parse_args()
    seed_everything(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suppress_layers = [int(x) for x in args.suppress_layers.split(',')]
    scales = [float(x) for x in args.scales.split(',')]

    # Load eval set
    samples = load_eval_set(args.eval_set)

    # Load uncertainty features
    print("\nLoading uncertainty features from experiment 28...")
    uncertainty_features = load_top_uncertainty_features(
        args.diff_csv, suppress_layers, args.top_n
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

    # =========================================================================
    # Baseline run (no suppression)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Baseline (no suppression)")
    print(f"{'=' * 60}")
    baseline_results = run_eval(model, tokenizer, samples, letter_token_ids,
                                args.device, "Baseline")
    baseline_acc = sum(r['correct'] for r in baseline_results) / len(baseline_results)
    baseline_entropy = np.mean([r['entropy'] for r in baseline_results])
    print(f"Baseline accuracy: {baseline_acc*100:.1f}%  mean entropy: {baseline_entropy:.4f}")

    # =========================================================================
    # Suppression experiments
    # =========================================================================
    from sae_lens import SAE

    all_results = [{
        'condition': 'baseline',
        'layer': None,
        'scale': None,
        'n_features_suppressed': 0,
        'accuracy': baseline_acc,
        'mean_entropy': baseline_entropy,
        'n_flipped_to_correct': 0,
        'n_flipped_to_incorrect': 0,
    }]

    # Test each layer individually
    for layer_idx in suppress_layers:
        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        print(f"\nLoading SAE for layer {layer_idx}...")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)

        feat_indices = uncertainty_features[layer_idx]

        for scale in scales:
            print(f"\n{'=' * 60}")
            print(f"Layer {layer_idx} | scale={scale} | suppressing {len(feat_indices)} features")
            print(f"{'=' * 60}")

            hook = SAESuppressionHook(sae, feat_indices, scale=scale)
            hook.register(model, layer_idx)

            suppressed_results = run_eval(
                model, tokenizer, samples, letter_token_ids, args.device,
                f"L{layer_idx} scale={scale}"
            )

            hook.remove()

            acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
            mean_ent = np.mean([r['entropy'] for r in suppressed_results])

            # Count flips
            flipped_to_correct = 0
            flipped_to_incorrect = 0
            for bl, sp in zip(baseline_results, suppressed_results):
                if bl['correct'] != sp['correct']:
                    if sp['correct']:
                        flipped_to_correct += 1
                    else:
                        flipped_to_incorrect += 1

            print(f"  Accuracy: {acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%, "
                  f"delta: {(acc-baseline_acc)*100:+.1f}%)")
            print(f"  Mean entropy: {mean_ent:.4f} (baseline: {baseline_entropy:.4f})")
            print(f"  Flipped to correct: {flipped_to_correct}, "
                  f"flipped to incorrect: {flipped_to_incorrect}")

            all_results.append({
                'condition': f'L{layer_idx}_scale{scale}',
                'layer': layer_idx,
                'scale': scale,
                'n_features_suppressed': len(feat_indices),
                'accuracy': acc,
                'mean_entropy': mean_ent,
                'n_flipped_to_correct': flipped_to_correct,
                'n_flipped_to_incorrect': flipped_to_incorrect,
            })

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    # Test all layers simultaneously
    print(f"\n{'=' * 60}")
    print("All layers simultaneously")
    print(f"{'=' * 60}")

    saes = {}
    hooks = []
    for layer_idx in suppress_layers:
        sae_id = f"l{layer_idx}r_{args.sae_expansion}"
        release = f"llama_scope_lxr_{args.sae_expansion}"
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)
        saes[layer_idx] = sae

    for scale in scales:
        # Remove any previous hooks
        for h in hooks:
            h.remove()
        hooks = []

        for layer_idx in suppress_layers:
            feat_indices = uncertainty_features[layer_idx]
            hook = SAESuppressionHook(saes[layer_idx], feat_indices, scale=scale)
            hook.register(model, layer_idx)
            hooks.append(hook)

        total_feats = sum(len(uncertainty_features[l]) for l in suppress_layers)
        print(f"\nAll layers, scale={scale}, {total_feats} total features suppressed")

        suppressed_results = run_eval(
            model, tokenizer, samples, letter_token_ids, args.device,
            f"All layers scale={scale}"
        )

        acc = sum(r['correct'] for r in suppressed_results) / len(suppressed_results)
        mean_ent = np.mean([r['entropy'] for r in suppressed_results])
        flipped_to_correct = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                 if not bl['correct'] and sp['correct'])
        flipped_to_incorrect = sum(1 for bl, sp in zip(baseline_results, suppressed_results)
                                   if bl['correct'] and not sp['correct'])

        print(f"  Accuracy: {acc*100:.1f}% (delta: {(acc-baseline_acc)*100:+.1f}%)")
        print(f"  Mean entropy: {mean_ent:.4f}")
        print(f"  Flipped to correct: {flipped_to_correct}, "
              f"flipped to incorrect: {flipped_to_incorrect}")

        all_results.append({
            'condition': f'all_layers_scale{scale}',
            'layer': 'all',
            'scale': scale,
            'n_features_suppressed': total_feats,
            'accuracy': acc,
            'mean_entropy': mean_ent,
            'n_flipped_to_correct': flipped_to_correct,
            'n_flipped_to_incorrect': flipped_to_incorrect,
        })

    for h in hooks:
        h.remove()
    for sae in saes.values():
        del sae
    del saes
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<30} {'Acc':>6} {'Delta':>7} {'->Corr':>7} {'->Wrong':>8}")
    print("-" * 60)
    for r in all_results:
        delta = (r['accuracy'] - baseline_acc) * 100
        print(f"{r['condition']:<30} {r['accuracy']*100:>5.1f}% {delta:>+6.1f}% "
              f"{r['n_flipped_to_correct']:>7} {r['n_flipped_to_incorrect']:>8}")

    csv_file = output_dir / "suppression_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_file}")

    pkl_file = output_dir / "suppression_results.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'model': args.model,
                'suppress_layers': suppress_layers,
                'top_n': args.top_n,
                'scales': scales,
                'uncertainty_features': uncertainty_features,
                'timestamp': datetime.now().isoformat(),
            },
            'results': all_results,
            'baseline_results': baseline_results,
        }, f)
    print(f"Saved: {pkl_file}")


if __name__ == '__main__':
    main()
