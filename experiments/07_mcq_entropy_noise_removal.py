"""
MCQ Entropy-Based Noise Removal Experiment (CP/Tucker Decomposition)

Uses multiple choice format for cleaner entropy measurements:
- Entropy over 4 options (A, B, C, D) instead of full vocabulary
- Max entropy = 1.386 nats (uniform over 4)
- Cleaner signal without vocabulary noise

Experiment flow:
1. Load MCQ evaluation set
2. Measure baseline MCQ entropy/accuracy on all samples
3. For each component: remove it, measure MCQ entropy/accuracy, restore
4. Classify components based on entropy and accuracy changes:
   - TRUE NOISE: entropy↓ accuracy↑ (removing helps)
   - Confident Wrong: entropy↓ accuracy↓ (removing makes confident but wrong)
   - TRUE SIGNAL: entropy↑ accuracy↓ (removing hurts)
   - Uncertain Right: entropy↑ accuracy↑ (removing adds uncertainty but helps)
5. Save results (plotting done separately)
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
from src.evaluation.metrics import compute_mcq_entropy_and_nll
from src.decomposition.tucker import (
    decompose_fc_layer,
    reconstruct_weights,
    remove_component,
    get_fc_layer_weights
)
from src.decomposition.cp import (
    decompose_fc_layer_cp,
    reconstruct_weights_cp,
    remove_cp_component
)
from src.decomposition.model_utils import update_fc_layer_weights


def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def evaluate_mcq_on_samples(model, tokenizer, samples, device="cuda"):
    """
    Evaluate MCQ entropy and NLL on all samples.

    Returns:
        List of dicts with metrics for each sample
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating MCQ", leave=False):
        mcq_prompt = sample['mcq_prompt']
        correct_letter = sample['correct_letter']

        metrics = compute_mcq_entropy_and_nll(
            mcq_prompt=mcq_prompt,
            correct_letter=correct_letter,
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        results.append({
            'sample_id': sample['id'],
            'question': sample['question'],
            'correct_answer': sample['correct_answer'],
            'correct_letter': correct_letter,
            **metrics
        })

    return results


def run_mcq_entropy_noise_removal(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_mcq_nq_open_200.json",
    target_layer: int = 6,
    rank: int = 40,
    decomposition_type: str = "cp",
    device: str = "cuda",
    checkpoint_every: int = 10
):
    """
    Run MCQ entropy-based noise removal experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama', 'gpt2', 'gptj')
        eval_set_path: Path to MCQ evaluation set JSON
        target_layer: Which layer to decompose
        rank: Decomposition rank
        decomposition_type: 'cp' or 'tucker'
        device: Device for computation
        checkpoint_every: Save checkpoint every N components
    """
    seed_everything(42)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    output_dir = Path(f"results/mcq_entropy_noise_removal/{model_short}_layer{target_layer}_rank{rank}_{decomposition_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MCQ ENTROPY NOISE REMOVAL EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Rank: {rank}")
    print(f"Decomposition: {decomposition_type.upper()}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load eval set
    samples = load_mcq_eval_set(eval_set_path)
    print(f"Using {len(samples)} MCQ samples")

    # Save config
    config = {
        'model_name': model_name,
        'model_type': model_type,
        'eval_set_path': eval_set_path,
        'target_layer': target_layer,
        'rank': rank,
        'decomposition_type': decomposition_type,
        'num_samples': len(samples),
        'timestamp': timestamp
    }

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

    baseline_results = evaluate_mcq_on_samples(model, tokenizer, samples, device)

    # Calculate averages
    baseline_avg_entropy = np.mean([r['entropy'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['nll'] for r in baseline_results])
    baseline_accuracy = np.mean([r['is_correct'] for r in baseline_results])

    print(f"Baseline avg entropy: {baseline_avg_entropy:.4f} (max: 1.386)")
    print(f"Baseline avg NLL: {baseline_avg_nll:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")

    # ========== Phase 2: Component Removal ==========
    print("\n" + "=" * 70)
    print(f"PHASE 2: COMPONENT REMOVAL ({rank} components)")
    print("=" * 70)

    component_results = []

    for comp_idx in tqdm(range(rank), desc="Testing components"):
        # Remove component
        if decomposition_type == "cp":
            modified_weights, modified_factors = remove_cp_component(decomp_weights, decomp_factors, comp_idx)
            fc_in_recon, fc_out_recon = reconstruct_weights_cp(modified_weights, modified_factors)
        else:
            modified_core = remove_component(decomp_weights, comp_idx)
            fc_in_recon, fc_out_recon = reconstruct_weights(modified_core, decomp_factors)

        # Update model
        update_fc_layer_weights(model, target_layer, fc_in_recon, fc_out_recon, model_type=model_type)

        # Evaluate
        comp_eval = evaluate_mcq_on_samples(model, tokenizer, samples, device)

        avg_entropy = np.mean([r['entropy'] for r in comp_eval])
        avg_nll = np.mean([r['nll'] for r in comp_eval])
        accuracy = np.mean([r['is_correct'] for r in comp_eval])
        entropy_change = avg_entropy - baseline_avg_entropy
        nll_change = avg_nll - baseline_avg_nll
        accuracy_change = accuracy - baseline_accuracy

        component_results.append({
            'component_idx': comp_idx,
            'avg_entropy': avg_entropy,
            'avg_nll': avg_nll,
            'accuracy': accuracy,
            'entropy_change': entropy_change,
            'nll_change': nll_change,
            'accuracy_change': accuracy_change,
            'per_sample': comp_eval
        })

        if (comp_idx + 1) % 10 == 0:
            print(f"  Component {comp_idx}: entropy {entropy_change:+.4f}, acc {accuracy_change*100:+.1f}pp ({accuracy*100:.1f}%)")

        # Restore original weights
        update_fc_layer_weights(model, target_layer, original_fc_in, original_fc_out, model_type=model_type)

        # Checkpoint
        if (comp_idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': config,
                'baseline': {
                    'avg_entropy': baseline_avg_entropy,
                    'avg_nll': baseline_avg_nll,
                    'accuracy': baseline_accuracy,
                    'per_sample': baseline_results
                },
                'component_results': component_results,
                'completed_components': comp_idx + 1
            }
            with open(output_dir / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

    # ========== Phase 3: Ranking ==========
    print("\n" + "=" * 70)
    print("PHASE 3: RANKING COMPONENTS")
    print("=" * 70)

    # Rank by entropy change
    by_entropy = sorted(
        [(c['component_idx'], c['entropy_change']) for c in component_results],
        key=lambda x: x[1]
    )

    # Rank by accuracy change (descending - best improvements first)
    by_accuracy = sorted(
        [(c['component_idx'], c['accuracy_change']) for c in component_results],
        key=lambda x: x[1],
        reverse=True
    )

    # Classify components using entropy and ACCURACY
    # With MCQ, accuracy is more interpretable than NLL (since NLL and entropy are highly correlated)
    true_noise = []      # entropy down, accuracy up (removing helps - less uncertain AND more accurate)
    confident_wrong = [] # entropy down, accuracy down (removing makes confident but wrong)
    true_signal = []     # entropy up, accuracy down (removing hurts - more uncertain AND less accurate)
    uncertain_right = [] # entropy up, accuracy up (removing adds uncertainty but improves accuracy)

    for c in component_results:
        if c['entropy_change'] < 0 and c['accuracy_change'] > 0:
            true_noise.append(c['component_idx'])
        elif c['entropy_change'] < 0 and c['accuracy_change'] <= 0:
            confident_wrong.append(c['component_idx'])
        elif c['entropy_change'] >= 0 and c['accuracy_change'] <= 0:
            true_signal.append(c['component_idx'])
        else:
            uncertain_right.append(c['component_idx'])

    print(f"TRUE NOISE (entropy↓ accuracy↑): {len(true_noise)}")
    print(f"Confident but wrong (entropy↓ accuracy↓): {len(confident_wrong)}")
    print(f"TRUE SIGNAL (entropy↑ accuracy↓): {len(true_signal)}")
    print(f"Uncertain but right (entropy↑ accuracy↑): {len(uncertain_right)}")

    if true_noise:
        print(f"\nTop 10 TRUE NOISE components:")
        for idx in true_noise[:10]:
            c = component_results[idx]
            print(f"  Component {idx}: entropy {c['entropy_change']:+.4f}, accuracy {c['accuracy_change']*100:+.1f}pp")

    # ========== Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'config': config,
        'baseline': {
            'avg_entropy': baseline_avg_entropy,
            'avg_nll': baseline_avg_nll,
            'accuracy': baseline_accuracy,
            'per_sample': baseline_results
        },
        'component_results': component_results,
        'rankings': {
            'by_entropy': by_entropy,
            'by_accuracy': by_accuracy,
            'true_noise': true_noise,
            'confident_wrong': confident_wrong,
            'true_signal': true_signal,
            'uncertain_right': uncertain_right
        }
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Baseline entropy: {baseline_avg_entropy:.4f}")
    print(f"Baseline NLL: {baseline_avg_nll:.4f}")
    print(f"Baseline accuracy: {baseline_accuracy*100:.1f}%")
    print(f"TRUE NOISE components: {len(true_noise)}/{rank}")
    print(f"Confident-wrong components: {len(confident_wrong)}/{rank}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCQ entropy-based noise removal experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str, default="data/eval_sets/eval_set_mcq_nq_open_200.json")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=40)
    parser.add_argument("--decomposition", type=str, default="cp", choices=["cp", "tucker"])
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--test", action="store_true", help="Quick test with rank=5")

    args = parser.parse_args()

    if args.test:
        print("Running quick test...")
        run_mcq_entropy_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            rank=5,
            decomposition_type=args.decomposition,
            checkpoint_every=2
        )
    else:
        run_mcq_entropy_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            eval_set_path=args.eval_set,
            target_layer=args.layer,
            rank=args.rank,
            decomposition_type=args.decomposition,
            checkpoint_every=args.checkpoint_every
        )
