"""
Entropy-Based SVD Truncation Experiment (LASER-style)

This script measures the effect of SVD rank reduction on answer entropy and NLL.
Much simpler and faster than the MDUQ-based approach (04_svd_truncation.py).

Experiment flow:
1. Load fixed evaluation set
2. Measure baseline entropy/NLL on all samples
3. For each reduction %: truncate SVD, measure entropy/NLL, restore
4. Save results (plotting done separately)

Key differences from 04_svd_truncation.py:
- No generation needed (single forward pass per sample)
- No NLI model or knowledge extractor
- Uses entropy instead of blockiness-based uncertainty
- Much faster (~60-100x)
"""

import sys
sys.path.append('.')

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.evaluation.metrics import compute_answer_nll_and_entropy
from src.decomposition.svd import (
    decompose_weight_svd,
    truncate_svd,
    reconstruct_from_svd,
    compute_energy_retention,
    get_svd_stats,
    update_layer_with_svd,
    restore_original_weight,
    reduction_to_keep_ratio,
    LASER_REDUCTION_PERCENTAGES
)


def format_short_answer_prompt(question: str) -> str:
    """Format question for short answer evaluation."""
    return f"Question: {question}\nAnswer (short):"


def load_eval_set(filepath: str):
    """Load evaluation set from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} samples from {filepath}")
    return data['samples']


def generate_answer(model, tokenizer, prompt_text, device="cuda", max_new_tokens=20):
    """Generate a short answer from the model."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_on_samples(model, tokenizer, samples, device="cuda", generate_answers=True):
    """
    Evaluate entropy and NLL on all samples, optionally generating answers.

    Returns:
        List of dicts with metrics for each sample
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating", leave=False):
        question = sample['question']
        gold_answer = sample['answer']
        all_answers = sample.get('all_answers', [gold_answer])
        prompt_text = format_short_answer_prompt(question)

        metrics = compute_answer_nll_and_entropy(
            question=question,
            gold_answer=gold_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=prompt_text,
            all_answers=all_answers
        )

        result = {
            'sample_id': sample['id'],
            'question': question,
            'answer': gold_answer,
            **metrics
        }

        # Generate model's actual answer
        if generate_answers:
            result['generated_answer'] = generate_answer(model, tokenizer, prompt_text, device)

        results.append(result)

    return results


def run_entropy_svd_truncation(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_nq_open_200.json",
    target_layer: int = 6,
    matrix_type: str = "mlp_out",
    reduction_percentages: list = None,
    device: str = "cuda",
    checkpoint_every: int = 5,
    generate_answers: bool = True
):
    """
    Run entropy-based SVD truncation experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama' or 'gpt2')
        eval_set_path: Path to fixed evaluation set JSON
        target_layer: Which layer to apply SVD
        matrix_type: 'mlp_in' or 'mlp_out'
        reduction_percentages: List of reduction percentages to test
        device: Device for computation
        checkpoint_every: Save checkpoint every N reduction levels
    """
    if reduction_percentages is None:
        reduction_percentages = LASER_REDUCTION_PERCENTAGES

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_model_name = model_name.split("/")[-1]
    output_dir = Path(f"results/entropy_svd/{short_model_name}_layer{target_layer}_{matrix_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Entropy-Based SVD Truncation (LASER-style)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Matrix: {matrix_type}")
    print(f"Reduction levels: {len(reduction_percentages)}")
    print(f"Eval set: {eval_set_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    seed_everything(42)

    # Load evaluation set
    samples = load_eval_set(eval_set_path)

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # Get original weight
    print(f"\nGetting weight from layer {target_layer} {matrix_type}...")
    if model_type == "llama":
        if matrix_type == "mlp_in":
            original_weight = model.model.layers[target_layer].mlp.up_proj.weight.data.clone()
        else:
            original_weight = model.model.layers[target_layer].mlp.down_proj.weight.data.clone()
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            original_weight = model.transformer.h[target_layer].mlp.c_fc.weight.data.clone()
        else:
            original_weight = model.transformer.h[target_layer].mlp.c_proj.weight.data.clone()
    elif model_type == "gptj":
        if matrix_type == "mlp_in":
            original_weight = model.transformer.h[target_layer].mlp.fc_in.weight.data.clone()
        else:
            original_weight = model.transformer.h[target_layer].mlp.fc_out.weight.data.clone()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Get SVD stats
    svd_stats = get_svd_stats(original_weight, device)
    print(f"Matrix shape: {svd_stats['shape']}")
    print(f"Max rank: {svd_stats['max_rank']}")
    print(f"Effective rank: {svd_stats['effective_rank']:.1f}")

    # Decompose once
    print("\nPerforming SVD decomposition...")
    U, S, Vh = decompose_weight_svd(original_weight, device)
    print(f"SVD complete: {len(S)} singular values")

    # ========== Phase 1: Baseline ==========
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE (Original Model)")
    print("=" * 70)

    baseline_results = evaluate_on_samples(model, tokenizer, samples, device, generate_answers)

    # Filter out inf values and use nanmean
    baseline_entropies = np.array([r['entropy'] for r in baseline_results])
    baseline_nlls = np.array([r['nll'] for r in baseline_results])
    baseline_entropies[np.isinf(baseline_entropies)] = np.nan
    baseline_nlls[np.isinf(baseline_nlls)] = np.nan

    baseline_avg_entropy = np.nanmean(baseline_entropies)
    baseline_avg_nll = np.nanmean(baseline_nlls)
    n_excluded_baseline = np.sum(np.isnan(baseline_entropies))

    print(f"Baseline avg entropy: {baseline_avg_entropy:.4f}")
    print(f"Baseline avg NLL: {baseline_avg_nll:.4f}")
    if n_excluded_baseline > 0:
        print(f"  ({n_excluded_baseline} samples excluded due to truncation)")

    # ========== Phase 2: Truncation Sweep ==========
    print("\n" + "=" * 70)
    print(f"PHASE 2: TRUNCATION SWEEP ({len(reduction_percentages)} levels)")
    print("=" * 70)

    truncation_results = []

    for idx, reduction_pct in enumerate(reduction_percentages):
        keep_ratio = reduction_to_keep_ratio(reduction_pct)
        print(f"\n--- Reduction {reduction_pct}% (keep {keep_ratio*100:.1f}%) ---")

        # Truncate SVD
        U_trunc, S_trunc, Vh_trunc = truncate_svd(U, S, Vh, keep_ratio)
        weight_lr = reconstruct_from_svd(U_trunc, S_trunc, Vh_trunc)

        energy_retention = compute_energy_retention(S, S_trunc)
        components_kept = len(S_trunc)

        print(f"  Kept {components_kept}/{len(S)} components")
        print(f"  Energy retention: {energy_retention*100:.1f}%")

        # Update model
        update_layer_with_svd(model, target_layer, weight_lr, matrix_type, model_type)

        # Evaluate
        trunc_eval = evaluate_on_samples(model, tokenizer, samples, device, generate_answers)

        # Filter out inf values and use nanmean
        trunc_entropies = np.array([r['entropy'] for r in trunc_eval])
        trunc_nlls = np.array([r['nll'] for r in trunc_eval])
        trunc_entropies[np.isinf(trunc_entropies)] = np.nan
        trunc_nlls[np.isinf(trunc_nlls)] = np.nan

        avg_entropy = np.nanmean(trunc_entropies)
        avg_nll = np.nanmean(trunc_nlls)
        entropy_change = avg_entropy - baseline_avg_entropy
        nll_change = avg_nll - baseline_avg_nll
        n_excluded = int(np.sum(np.isnan(trunc_entropies)))

        truncation_results.append({
            'reduction_percent': reduction_pct,
            'keep_ratio': keep_ratio,
            'components_kept': components_kept,
            'total_components': len(S),
            'energy_retention': energy_retention,
            'avg_entropy': avg_entropy,
            'avg_nll': avg_nll,
            'entropy_change': entropy_change,
            'nll_change': nll_change,
            'n_excluded': n_excluded,
            'per_sample': trunc_eval
        })

        excluded_str = f" ({n_excluded} excluded)" if n_excluded > 0 else ""
        print(f"  Entropy: {avg_entropy:.4f} ({entropy_change:+.4f}){excluded_str}")
        print(f"  NLL: {avg_nll:.4f} ({nll_change:+.4f})")

        # Restore original weight
        restore_original_weight(model, target_layer, original_weight, matrix_type, model_type)

        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': {
                    'model_name': model_name,
                    'target_layer': target_layer,
                    'matrix_type': matrix_type
                },
                'baseline': baseline_results,
                'truncation_results': truncation_results
            }
            with open(output_dir / f"checkpoint_{idx + 1}.pkl", 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  Checkpoint saved.")

        torch.cuda.empty_cache()

    # ========== Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'config': {
            'model_name': model_name,
            'model_type': model_type,
            'eval_set_path': eval_set_path,
            'target_layer': target_layer,
            'matrix_type': matrix_type,
            'reduction_percentages': reduction_percentages,
            'num_samples': len(samples),
            'timestamp': timestamp
        },
        'svd_stats': {
            'shape': svd_stats['shape'],
            'max_rank': svd_stats['max_rank'],
            'effective_rank': svd_stats['effective_rank'],
            'singular_values': svd_stats['singular_values'].tolist() if hasattr(svd_stats['singular_values'], 'tolist') else svd_stats['singular_values']
        },
        'baseline': {
            'avg_entropy': baseline_avg_entropy,
            'avg_nll': baseline_avg_nll,
            'per_sample': baseline_results
        },
        'truncation_results': truncation_results
    }

    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Baseline entropy: {baseline_avg_entropy:.4f}")
    print(f"Baseline NLL: {baseline_avg_nll:.4f}")

    # Find best reduction level (lowest entropy while NLL doesn't spike)
    valid_results = [r for r in truncation_results if r['nll_change'] < 1.0]  # NLL didn't spike too much
    if valid_results:
        best = min(valid_results, key=lambda x: x['avg_entropy'])
        print(f"Best reduction: {best['reduction_percent']}% (entropy {best['entropy_change']:+.4f}, NLL {best['nll_change']:+.4f})")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entropy-based SVD truncation experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str, default="data/eval_sets/eval_set_nq_open_200.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--matrix", type=str, default="mlp_out", choices=["mlp_in", "mlp_out"])
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation (faster)")
    parser.add_argument("--test", action="store_true", help="Quick test with fewer reduction levels")

    args = parser.parse_args()

    generate = not args.no_generate

    if args.test:
        print("Running quick test...")
        run_entropy_svd_truncation(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            matrix_type=args.matrix,
            reduction_percentages=[10, 50, 90],  # Just 3 levels for quick test
            checkpoint_every=1,
            generate_answers=generate
        )
    else:
        run_entropy_svd_truncation(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            matrix_type=args.matrix,
            checkpoint_every=args.checkpoint_every,
            generate_answers=generate
        )
