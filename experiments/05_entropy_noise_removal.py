"""
Entropy-Based Noise Removal Experiment (CP/Tucker Decomposition)

This script measures the effect of component removal on answer entropy and NLL.
Much simpler and faster than the MDUQ-based approach (03_noise_removal.py).

Experiment flow:
1. Load fixed evaluation set
2. Measure baseline entropy/NLL on all samples
3. For each component: remove it, measure entropy/NLL, restore
4. Save results (plotting done separately)

Key differences from 03_noise_removal.py:
- No generation needed (single forward pass per sample)
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
from src.decomposition.tucker import (
    decompose_fc_layer,
    get_fc_layer_weights,
    remove_component,
    reconstruct_weights
)
from src.decomposition.cp import (
    decompose_fc_layer_cp,
    remove_cp_component,
    reconstruct_weights_cp
)
from src.decomposition.model_utils import update_fc_layer_weights


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


def run_entropy_noise_removal(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_nq_open_200.json",
    target_layer: int = 6,
    rank: int = 40,
    decomposition_type: str = "cp",
    device: str = "cuda",
    checkpoint_every: int = 10,
    generate_answers: bool = True
):
    """
    Run entropy-based noise removal experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama' or 'gpt2')
        eval_set_path: Path to fixed evaluation set JSON
        target_layer: Which layer to decompose
        rank: Decomposition rank
        decomposition_type: 'cp' or 'tucker'
        device: Device for computation
        checkpoint_every: Save checkpoint every N components
    """
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_model_name = model_name.split("/")[-1]
    output_dir = Path(f"results/entropy_noise_removal/{short_model_name}_layer{target_layer}_rank{rank}_{decomposition_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Entropy-Based Noise Removal ({decomposition_type.upper()})")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Rank: {rank}")
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

    # Get original weights
    fc_in_weight, fc_out_weight = get_fc_layer_weights(model, target_layer, model_type=model_type)
    original_fc_in = fc_in_weight.clone()
    original_fc_out = fc_out_weight.clone()

    # Decompose
    print(f"\nDecomposing layer {target_layer} with {decomposition_type.upper()} rank {rank}...")
    if decomposition_type == "cp":
        cp_result = decompose_fc_layer_cp(fc_in_weight, fc_out_weight, rank=rank, device=device)
        decomp_weights = cp_result[0]
        decomp_factors = cp_result[1:]
    else:
        decomp_weights, decomp_factors = decompose_fc_layer(fc_in_weight, fc_out_weight, rank=rank, device=device)
    print("Decomposition complete.")

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

    # ========== Phase 2: Component Search ==========
    print("\n" + "=" * 70)
    print(f"PHASE 2: COMPONENT SEARCH ({rank} components)")
    print("=" * 70)

    component_results = []

    for comp_idx in range(rank):
        print(f"\n--- Component {comp_idx + 1}/{rank} ---")

        # Remove component and reconstruct
        if decomposition_type == "cp":
            modified_weights, _ = remove_cp_component(decomp_weights, decomp_factors, comp_idx)
            fc_in_recon, fc_out_recon = reconstruct_weights_cp(modified_weights, decomp_factors)
        else:
            modified_core = remove_component(decomp_weights, comp_idx, dimension=1)
            fc_in_recon, fc_out_recon = reconstruct_weights(modified_core, decomp_factors)

        # Update model
        update_fc_layer_weights(model, target_layer, fc_in_recon, fc_out_recon, model_type=model_type)

        # Evaluate
        comp_eval = evaluate_on_samples(model, tokenizer, samples, device, generate_answers)

        # Filter out inf values
        comp_entropies = np.array([r['entropy'] for r in comp_eval])
        comp_nlls = np.array([r['nll'] for r in comp_eval])
        comp_entropies[np.isinf(comp_entropies)] = np.nan
        comp_nlls[np.isinf(comp_nlls)] = np.nan

        avg_entropy = np.nanmean(comp_entropies)
        avg_nll = np.nanmean(comp_nlls)
        entropy_change = avg_entropy - baseline_avg_entropy
        nll_change = avg_nll - baseline_avg_nll
        n_excluded = np.sum(np.isnan(comp_entropies))

        component_results.append({
            'component_idx': comp_idx,
            'avg_entropy': avg_entropy,
            'avg_nll': avg_nll,
            'entropy_change': entropy_change,
            'nll_change': nll_change,
            'n_excluded': int(n_excluded),
            'per_sample': comp_eval
        })

        excluded_str = f" ({n_excluded} excluded)" if n_excluded > 0 else ""
        print(f"  Entropy: {avg_entropy:.4f} ({entropy_change:+.4f}){excluded_str}")
        print(f"  NLL: {avg_nll:.4f} ({nll_change:+.4f})")

        # Restore original weights
        update_fc_layer_weights(model, target_layer, original_fc_in, original_fc_out, model_type=model_type)

        # Checkpoint
        if (comp_idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': {
                    'model_name': model_name,
                    'target_layer': target_layer,
                    'rank': rank,
                    'decomposition_type': decomposition_type
                },
                'baseline': baseline_results,
                'component_results': component_results
            }
            with open(output_dir / f"checkpoint_{comp_idx + 1}.pkl", 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  Checkpoint saved.")

        torch.cuda.empty_cache()

    # ========== Phase 3: Rank Components ==========
    print("\n" + "=" * 70)
    print("PHASE 3: RANKING COMPONENTS")
    print("=" * 70)

    # Sort by entropy change (most negative = most noise)
    sorted_by_entropy = sorted(component_results, key=lambda x: x['entropy_change'])

    noise_components = [c for c in sorted_by_entropy if c['entropy_change'] < 0]
    signal_components = [c for c in sorted_by_entropy if c['entropy_change'] >= 0]

    print(f"Noise components (entropy decreased): {len(noise_components)}")
    print(f"Signal components (entropy increased): {len(signal_components)}")

    print("\nTop 10 noise components:")
    for i, c in enumerate(noise_components[:10]):
        print(f"  {i+1}. Component {c['component_idx']}: entropy {c['entropy_change']:+.4f}, NLL {c['nll_change']:+.4f}")

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
            'rank': rank,
            'decomposition_type': decomposition_type,
            'num_samples': len(samples),
            'timestamp': timestamp
        },
        'baseline': {
            'avg_entropy': baseline_avg_entropy,
            'avg_nll': baseline_avg_nll,
            'per_sample': baseline_results
        },
        'component_results': component_results,
        'rankings': {
            'by_entropy': [(c['component_idx'], c['entropy_change']) for c in sorted_by_entropy],
            'noise_components': [c['component_idx'] for c in noise_components],
            'signal_components': [c['component_idx'] for c in signal_components]
        }
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
    print(f"Noise components: {len(noise_components)}/{rank}")
    print(f"Signal components: {len(signal_components)}/{rank}")
    if noise_components:
        best = noise_components[0]
        print(f"Best noise component: {best['component_idx']} (entropy {best['entropy_change']:+.4f})")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entropy-based noise removal experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str, default="data/eval_sets/eval_set_nq_open_200.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=40)
    parser.add_argument("--decomposition", type=str, default="cp", choices=["cp", "tucker"])
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-generate", action="store_true", help="Skip answer generation (faster)")
    parser.add_argument("--test", action="store_true", help="Quick test with rank=5")

    args = parser.parse_args()

    generate = not args.no_generate

    if args.test:
        print("Running quick test...")
        run_entropy_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            rank=5,
            decomposition_type=args.decomposition,
            checkpoint_every=2,
            generate_answers=generate
        )
    else:
        run_entropy_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            rank=args.rank,
            decomposition_type=args.decomposition,
            checkpoint_every=args.checkpoint_every,
            generate_answers=generate
        )
