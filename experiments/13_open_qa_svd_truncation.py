"""
Open-ended QA Generation with SVD Truncation Experiment

Tests if Layer 31 MLP is also redundant for open-ended generation tasks.
Similar to experiment 08 but evaluates on generation quality instead of MCQ.

Experiment flow:
1. Load open-ended QA evaluation set
2. Measure baseline generation quality (exact match accuracy)
3. For each reduction %: truncate SVD, measure generation quality, restore
4. Compare with MCQ results to see if Layer 31 redundancy holds for generation

Usage:
    python experiments/13_open_qa_svd_truncation.py --model meta-llama/Llama-2-7b-chat-hf --layer 31 --matrix mlp_out
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
from src.decomposition.svd import (
    decompose_weight_svd,
    truncate_svd,
    select_svd_components,
    reconstruct_from_svd,
    get_svd_stats,
    update_layer_with_svd,
    restore_original_weight,
    compute_energy_retention,
    LASER_REDUCTION_PERCENTAGES,
    reduction_to_keep_ratio
)


def load_qa_eval_set(dataset_name: str = "nq_open", split: str = "validation", num_samples: int = 200):
    """Load open-ended QA dataset."""
    from src.generation.datasets import get_dataset

    print(f"Loading {dataset_name} {split} set...")
    dataset = get_dataset(dataset_name, split=split, num_samples=num_samples)
    dataset.load(None)

    samples = []
    for i, item in enumerate(dataset.data):
        samples.append({
            'id': f"{dataset_name}_{i}",
            'question': item['question'],
            'answer': item['answer'],
            'all_answers': item.get('all_answers', [item['answer']])
        })

    print(f"Loaded {len(samples)} QA samples")
    return samples


def check_answer_match(generated_text: str, gold_answers: list) -> bool:
    """
    Check if gold answer appears in generated text with intelligent matching.

    Handles:
    - Number equivalence: "1" matches "one", "2" matches "two", etc.
    - Partial date matching: "1972" matches "14 December 1972 UTC"
    - Word boundaries: "one" doesn't match "anyone"
    - Multi-word phrases: "South Carolina" matches if both words present

    Args:
        generated_text: Model's generated answer
        gold_answers: List of acceptable gold answers

    Returns:
        True if any gold answer appears in generated text
    """
    import re

    gen_lower = generated_text.lower().strip()

    # Number word mapping
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'twenty': '20', 'thirty': '30'
    }

    # Also reverse mapping
    word_to_number = number_words
    number_to_word = {v: k for k, v in number_words.items()}

    for gold in gold_answers:
        gold_lower = gold.lower().strip()

        # Direct substring match (for multi-word or specific phrases)
        if gold_lower in gen_lower:
            return True

        # Check if gold is in generated with word boundaries (avoid "one" in "anyone")
        # Use word boundary for single words only
        gold_words = gold_lower.split()
        if len(gold_words) == 1:
            # Single word - check with word boundaries
            pattern = r'\b' + re.escape(gold_lower) + r'\b'
            if re.search(pattern, gen_lower):
                return True

            # Also check number/word equivalence
            # If gold is "one", check for "1"
            if gold_lower in word_to_number:
                number_pattern = r'\b' + re.escape(word_to_number[gold_lower]) + r'\b'
                if re.search(number_pattern, gen_lower):
                    return True

            # If gold is "1", check for "one"
            if gold_lower in number_to_word:
                word_pattern = r'\b' + re.escape(number_to_word[gold_lower]) + r'\b'
                if re.search(word_pattern, gen_lower):
                    return True

        # For dates/numbers: check if any number in gold appears in generated
        # e.g., "1972" in "14 December 1972 UTC" should match "1972"
        gold_numbers = re.findall(r'\b\d{4}\b|\b\d+\b', gold_lower)
        gen_numbers = re.findall(r'\b\d{4}\b|\b\d+\b', gen_lower)

        # Check for 4-digit years (high signal)
        gold_years = [n for n in gold_numbers if len(n) == 4]
        gen_years = [n for n in gen_numbers if len(n) == 4]
        if gold_years and gen_years:
            if any(gy in gold_years for gy in gen_years):
                return True

    return False


def evaluate_generation_on_samples(model, tokenizer, samples, max_new_tokens=20, device="cuda"):
    """
    Evaluate open-ended generation on all samples.

    Returns:
        List of dicts with metrics for each sample
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating generation", leave=False):
        question = sample['question']
        gold_answers = sample['all_answers']

        # Format prompt - instruct for concise answers
        prompt = f"Answer this question concisely in 1-5 words.\n\nQuestion: {question}\nAnswer:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        # Check if correct
        is_correct = check_answer_match(generated_text, gold_answers)

        results.append({
            'sample_id': sample['id'],
            'question': question,
            'gold_answers': gold_answers,
            'generated_text': generated_text,
            'is_correct': is_correct,
            'generated_length': len(generated_ids[0]) - prompt_len
        })

    return results


def run_open_qa_svd_truncation(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    dataset_name: str = "nq_open",
    num_samples: int = 200,
    target_layer: int = 31,
    matrix_type: str = "mlp_out",
    reduction_percentages: list = None,
    component_counts: list = None,
    mode: str = "top",
    num_trials: int = 5,
    random_seed_start: int = 100,
    max_new_tokens: int = 10,
    device: str = "cuda",
    checkpoint_every: int = 5
):
    """
    Run open-ended QA generation with SVD truncation experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture type
        dataset_name: QA dataset name
        num_samples: Number of samples to evaluate
        target_layer: Layer to apply SVD truncation
        matrix_type: Which matrix to truncate ("mlp_in", "mlp_out", "gate_proj")
        reduction_percentages: List of reduction percentages (e.g. [0, 50, 90, 95])
        component_counts: Alternative to percentages - specify exact k components to keep
        mode: Component selection mode ("top", "bottom", "random")
        num_trials: Number of random trials (only used if mode="random")
        random_seed_start: Starting seed for random trials
        max_new_tokens: Max tokens to generate per answer
        device: Device for computation
        checkpoint_every: Save checkpoint every N reduction levels
    """
    seed_everything(42)

    # Use default reduction percentages if not specified
    if reduction_percentages is None and component_counts is None:
        reduction_percentages = LASER_REDUCTION_PERCENTAGES

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    matrix_suffix = f"_layer{target_layer}_{matrix_type}"
    mode_suffix = f"_{mode}" if mode != "top" else ""
    output_dir = Path(f"results/open_qa_svd/{model_short}{matrix_suffix}{mode_suffix}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPEN-ENDED QA SVD TRUNCATION EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} ({num_samples} samples)")
    print(f"Target layer: {target_layer}")
    print(f"Matrix type: {matrix_type}")
    print(f"Mode: {mode}")
    print(f"Max new tokens: {max_new_tokens}")
    if reduction_percentages:
        print(f"Reduction %: {reduction_percentages}")
    if component_counts:
        print(f"Component counts: {component_counts}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load evaluation set
    samples = load_qa_eval_set(dataset_name, num_samples=num_samples)

    # Save config
    config = {
        'experiment_type': 'open_qa_svd_truncation',
        'model_name': model_name,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'num_samples': len(samples),
        'target_layer': target_layer,
        'matrix_type': matrix_type,
        'reduction_percentages': reduction_percentages,
        'component_counts': component_counts,
        'mode': mode,
        'num_trials': num_trials if mode == "random" else 1,
        'random_seed_start': random_seed_start,
        'max_new_tokens': max_new_tokens,
        'timestamp': timestamp
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    # Extract target weight
    print(f"\nExtracting weight from layer {target_layer}, matrix type: {matrix_type}")

    if model_type == "llama":
        if matrix_type == "mlp_in":
            original_weight = model.model.layers[target_layer].mlp.up_proj.weight.data.clone()
        elif matrix_type == "mlp_out":
            original_weight = model.model.layers[target_layer].mlp.down_proj.weight.data.clone()
        elif matrix_type == "gate_proj":
            original_weight = model.model.layers[target_layer].mlp.gate_proj.weight.data.clone()
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"Weight shape: {original_weight.shape}")

    # ========== Baseline Evaluation ==========
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (no truncation)")
    print("=" * 70)

    baseline_results = evaluate_generation_on_samples(
        model, tokenizer, samples, max_new_tokens=max_new_tokens, device=device
    )

    baseline_accuracy = sum(r['is_correct'] for r in baseline_results) / len(baseline_results)
    baseline_avg_length = np.mean([r['generated_length'] for r in baseline_results])

    print(f"Baseline Accuracy: {baseline_accuracy*100:.1f}%")
    print(f"Baseline Avg Length: {baseline_avg_length:.1f} tokens")

    # Show examples
    print("\nBaseline Examples:")
    for i, r in enumerate(baseline_results[:3]):
        print(f"\n{i+1}. Q: {r['question']}")
        print(f"   Gold: {r['gold_answers'][0]}")
        print(f"   Generated: {r['generated_text']}")
        print(f"   Correct: {r['is_correct']}")

    # ========== SVD Decomposition ==========
    print("\n" + "=" * 70)
    print("SVD DECOMPOSITION")
    print("=" * 70)

    U, S, Vt = decompose_weight_svd(original_weight)
    svd_stats = get_svd_stats(original_weight)

    print(f"Weight shape: {original_weight.shape}")
    print(f"Rank: {len(S)}")
    print(f"Effective rank: {svd_stats['effective_rank']:.1f}")
    print(f"Top singular value: {svd_stats['top_singular_value']:.2f}")
    print(f"Top 10 singular values: {S[:10].cpu().numpy()}")

    # Determine reduction levels
    if component_counts:
        test_configs = [{'k': k, 'reduction_pct': None} for k in component_counts]
    else:
        test_configs = []
        for red_pct in reduction_percentages:
            keep_ratio = reduction_to_keep_ratio(red_pct)
            k = max(1, int(keep_ratio * len(S)))
            test_configs.append({'k': k, 'reduction_pct': red_pct})

    print(f"\nTesting {len(test_configs)} reduction levels")

    # ========== SVD Truncation Experiments ==========
    print("\n" + "=" * 70)
    print("RUNNING SVD TRUNCATION EXPERIMENTS")
    print("=" * 70)

    all_results = {
        'baseline': {
            'results': baseline_results,
            'accuracy': baseline_accuracy,
            'avg_length': baseline_avg_length,
            'k': len(S),
            'reduction_pct': 0.0,
            'energy_retention': 1.0
        }
    }

    for test_idx, test_config in enumerate(test_configs):
        k = test_config['k']
        reduction_pct = test_config['reduction_pct']

        print(f"\n--- Test {test_idx+1}/{len(test_configs)} ---")
        print(f"k = {k} components ({k/len(S)*100:.1f}% retained)")
        if reduction_pct is not None:
            print(f"Reduction: {reduction_pct}%")

        # Determine number of trials
        if mode == "random":
            trials = num_trials
        else:
            trials = 1

        trial_results = []

        for trial_idx in range(trials):
            if mode == "random":
                seed = random_seed_start + trial_idx
                print(f"  Trial {trial_idx+1}/{trials} (seed={seed})")
            else:
                seed = None

            # Select components
            keep_ratio = k / len(S)
            U_trunc, S_trunc, Vt_trunc, indices = select_svd_components(
                U, S, Vt, keep_ratio=keep_ratio, mode=mode, seed=seed
            )

            # Reconstruct
            W_compressed = reconstruct_from_svd(U_trunc, S_trunc, Vt_trunc)
            energy_retention = compute_energy_retention(S, S_trunc)

            # Update model
            update_layer_with_svd(
                model=model,
                layer_idx=target_layer,
                weight_lr=W_compressed,
                matrix_type=matrix_type,
                model_type=model_type
            )

            # Evaluate
            results = evaluate_generation_on_samples(
                model, tokenizer, samples, max_new_tokens=max_new_tokens, device=device
            )

            accuracy = sum(r['is_correct'] for r in results) / len(results)
            avg_length = np.mean([r['generated_length'] for r in results])

            trial_results.append({
                'results': results,
                'accuracy': accuracy,
                'avg_length': avg_length,
                'k': k,
                'reduction_pct': reduction_pct,
                'energy_retention': energy_retention,
                'seed': seed
            })

            print(f"  Accuracy: {accuracy*100:.1f}% (Δ = {(accuracy - baseline_accuracy)*100:+.1f}pp)")
            print(f"  Avg Length: {avg_length:.1f} tokens")
            print(f"  Energy Retention: {energy_retention*100:.1f}%")

            # Restore original weight
            restore_original_weight(
                model=model,
                model_type=model_type,
                layer_idx=target_layer,
                matrix_type=matrix_type,
                original_weight=original_weight
            )

        # Store results
        key = f"k{k}_red{reduction_pct}" if reduction_pct else f"k{k}"
        all_results[key] = trial_results if mode == "random" else trial_results[0]

        # Checkpoint
        if (test_idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': config,
                'svd_stats': svd_stats,
                'all_results': all_results,
                'completed_tests': test_idx + 1
            }
            with open(output_dir / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"\n  Checkpoint saved ({test_idx + 1}/{len(test_configs)} tests)")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Reduction %':>12} {'k':>6} {'Accuracy':>10} {'Δ Acc':>8} {'Avg Len':>9} {'Energy %':>10}")
    print("-" * 70)

    # Baseline
    print(f"{'Baseline':>12} {len(S):>6} {baseline_accuracy*100:>9.1f}% {'---':>8} {baseline_avg_length:>8.1f} {100.0:>9.1f}%")

    # Truncated
    for key in sorted(all_results.keys()):
        if key == 'baseline':
            continue

        result = all_results[key]
        if isinstance(result, list):  # Random mode with multiple trials
            accs = [r['accuracy'] for r in result]
            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            lens = [r['avg_length'] for r in result]
            len_mean = np.mean(lens)
            energy = result[0]['energy_retention']
            k = result[0]['k']
            red_pct = result[0]['reduction_pct'] or 0

            delta = (acc_mean - baseline_accuracy) * 100
            print(f"{red_pct:>11.1f}% {k:>6} {acc_mean*100:>9.1f}% {delta:>+7.1f} {len_mean:>8.1f} {energy*100:>9.1f}%")
            print(f"{'':>12} {'':>6} (±{acc_std*100:.1f}%)")
        else:
            acc = result['accuracy']
            avg_len = result['avg_length']
            energy = result['energy_retention']
            k = result['k']
            red_pct = result['reduction_pct'] or 0

            delta = (acc - baseline_accuracy) * 100
            print(f"{red_pct:>11.1f}% {k:>6} {acc*100:>9.1f}% {delta:>+7.1f} {avg_len:>8.1f} {energy*100:>9.1f}%")

    # ========== Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    final_results = {
        'config': config,
        'svd_stats': svd_stats,
        'all_results': all_results
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(final_results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Open-ended QA generation with SVD truncation"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama")
    parser.add_argument("--dataset", type=str, default="nq_open",
                        choices=["nq_open", "hotpotqa", "coqa"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--layer", type=int, default=31,
                        help="Target layer for SVD truncation")
    parser.add_argument("--matrix", type=str, default="mlp_out",
                        choices=["mlp_in", "mlp_out", "gate_proj"],
                        help="Which MLP matrix to truncate")
    parser.add_argument("--mode", type=str, default="top",
                        choices=["top", "bottom", "random"],
                        help="Component selection mode")
    parser.add_argument("--reduction-pct", type=float, nargs="+",
                        help="Reduction percentages to test (e.g., 0 50 90 95 99)")
    parser.add_argument("--component-counts", type=int, nargs="+",
                        help="Specific k values to test (alternative to reduction-pct)")
    parser.add_argument("--num-trials", type=int, default=5,
                        help="Number of random trials (for mode=random)")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate per answer")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--test", action="store_true",
                        help="Quick test with 10 samples")

    args = parser.parse_args()

    kwargs = dict(
        model_name=args.model,
        model_type=args.model_type,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        target_layer=args.layer,
        matrix_type=args.matrix,
        mode=args.mode,
        num_trials=args.num_trials,
        max_new_tokens=args.max_new_tokens,
        checkpoint_every=args.checkpoint_every
    )

    if args.reduction_pct:
        kwargs['reduction_percentages'] = args.reduction_pct
    if args.component_counts:
        kwargs['component_counts'] = args.component_counts

    if args.test:
        print("Running quick test with 10 samples...")
        kwargs['num_samples'] = 10
        kwargs['checkpoint_every'] = 2

    run_open_qa_svd_truncation(**kwargs)
