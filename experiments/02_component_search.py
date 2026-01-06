"""
Component Search Experiment: Tucker and CP Decomposition

This script systematically removes components from Tucker or CP decomposition
and measures the impact on uncertainty metrics.

Goal: Identify which components encode noise vs. signal
"""

import sys
sys.path.append('.')

import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.generation.datasets import get_dataset
from src.generation.generate import (
    load_model_and_tokenizer,
    generate_responses,
    seed_everything
)
from src.uncertainty.similarity import (
    NLISimilarityCalculator,
    build_semantic_similarity_matrix
)
from src.uncertainty.knowledge import KnowledgeExtractor
from src.uncertainty.metrics import (
    compute_blockiness_score,
    compute_uncertainty_score
)
from src.decomposition.cp import (
    decompose_fc_layer_cp,
    remove_cp_component,
    reconstruct_weights_cp
)
from src.decomposition.tucker import (
    decompose_fc_layer,
    get_fc_layer_weights,
    remove_component,
    reconstruct_weights
)
from src.decomposition.model_utils import update_fc_layer_weights


def measure_uncertainty_and_blockiness(
        responses,
        question,
        nli_calculator,
        knowledge_extractor,
        decomposition_type="cp",
        device="cuda",
        ranks=[5, 10, 15, 20]
):
    """
    Measure uncertainty and blockiness for a set of responses

    Args:
        responses: List of generated responses
        question: Original question
        nli_calculator: NLISimilarityCalculator instance
        knowledge_extractor: KnowledgeExtractor instance
        decomposition_type: Type of decomposition ('tucker' or 'cp')
        device: Device for computation
        ranks: List of ranks for blockiness measurement

    Returns:
        Dictionary with metrics
    """
    # Build semantic similarity matrix
    S_semantic = build_semantic_similarity_matrix(
        responses=responses,
        nli_calculator=nli_calculator,
        device=device
    )

    # Extract knowledge using KnowledgeExtractor
    knowledge_responses = knowledge_extractor.extract_knowledge_batch(
        question=question,
        responses=responses
    )

    # Build knowledge similarity matrix
    S_knowledge = build_semantic_similarity_matrix(
        responses=knowledge_responses,
        nli_calculator=nli_calculator,
        device=device
    )

    # Compute blockiness at multiple ranks
    blockiness_results = {}
    for rank in ranks:
        blockiness = compute_blockiness_score(
            S_semantic=S_semantic,
            S_knowledge=S_knowledge,
            rank=rank,
            decomposition_type=decomposition_type
        )
        blockiness_results[f'rank_{rank}'] = blockiness

    # Compute uncertainty (using rank = 10 as default)
    uncertainty = compute_uncertainty_score(blockiness_results['rank_10'])

    return {
        'semantic_similarity': S_semantic,
        'knowledge_similarity': S_knowledge,
        'knowledge_responses': knowledge_responses,
        'blockiness_by_rank': blockiness_results,
        'uncertainty_score': uncertainty
    }


def run_component_search(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        dataset_name="coqa",
        num_samples=10,
        target_layer=6,
        rank=80,
        num_generations=20,
        nli_model="microsoft/deberta-large-mnli",
        device="cuda",
        checkpoint_every=5,
        decomposition_type="cp",
):
    """
    Run component search experiment using Tucker or CP decomposition

    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset to use (coqa, hotpotqa)
        num_samples: Number of dataset samples
        target_layer: Which layer to decompose
        rank: Rank for decomposition
        num_generations: Responses per sample
        nli_model: NLI model for semantic similarity
        device: Device to run on
        checkpoint_every: Save results every N components
        decomposition_type: Type of decomposition ('tucker' or 'cp')
    """

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/component_search_{decomposition_type}/{dataset_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Component Search Experiment ({decomposition_type.upper()} Decomposition)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Num samples: {num_samples}")
    print(f"Target layer: {target_layer}")
    print(f"{decomposition_type.upper()} rank: {rank}")
    print(f"Generations per sample: {num_generations}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Set seed
    seed_everything(42)

    # ========== Load Dataset ==========
    print("\nLoading dataset...")
    dataset = get_dataset(dataset_name, split="validation", num_samples=num_samples)
    dataset.load(None)
    samples = dataset.data
    print(f"Loaded {len(samples)} samples")

    # ========== Load Main Model ==========
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded")

    # ========== Load NLI Calculator ==========
    print(f"\nLoading NLI model: {nli_model}")
    nli_calculator = NLISimilarityCalculator(model_name=nli_model, device=device)
    print("NLI model loaded")

    # ========== Load Knowledge Extractor ==========
    print(f"\nLoading knowledge extractor: {model_name}")
    knowledge_extractor = KnowledgeExtractor(model_name=model_name, device=device)
    print("Knowledge extractor loaded")

    # ========== Decompose Target Layer ==========
    print(f"\nDecomposing layer {target_layer} with {decomposition_type.upper()} rank {rank}...")
    fc_in_weight, fc_out_weight = get_fc_layer_weights(model, target_layer, model_type="llama")

    if decomposition_type == "cp":
        # CP returns [weights, factor_0, factor_1, factor_2]
        cp_result = decompose_fc_layer_cp(fc_in_weight, fc_out_weight, rank=rank, device=device)
        decomp_weights = cp_result[0]  # [rank]
        decomp_factors = cp_result[1:]  # [factor_0, factor_1, factor_2]
        print(f"Decomposed: weights shape = {decomp_weights.shape}, {rank} components")
    else:  # tucker
        core, factors = decompose_fc_layer(fc_in_weight, fc_out_weight, rank=rank, device=device)
        decomp_weights = core
        decomp_factors = factors
        print(f"Decomposed: core shape = {core.shape}, {rank} components")

    # Store original weights for resetting after each component test
    original_fc_in = fc_in_weight.clone()
    original_fc_out = fc_out_weight.clone()

    # ========== Baseline Measurement ==========
    print("\n" + "="*80)
    print("BASELINE MEASUREMENT (No Component Removal)")
    print("="*80)

    baseline_results = []

    for idx, sample in enumerate(tqdm(samples, desc="Baseline")):
        question = sample['question']

        # Generate responses
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            num_generations=num_generations,
            max_new_tokens=100,
            temperature=1.0,
            top_p=0.95
        )

        # Measure uncertainty
        metrics = measure_uncertainty_and_blockiness(
            responses=responses,
            question=question,
            nli_calculator=nli_calculator,
            knowledge_extractor=knowledge_extractor,
            decomposition_type=decomposition_type,
            device=device
        )

        baseline_results.append({
            'sample_id': idx,
            'question': question,
            'answer': sample['answer'],
            'responses': responses,
            **metrics
        })

    # Save baseline
    with open(output_dir / "baseline.pkl", 'wb') as f:
        pickle.dump(baseline_results, f)
    print(f"\nSaved baseline to {output_dir / 'baseline.pkl'}")

    # ========== Component Search Loop ==========
    print("\n" + "="*80)
    print(f"COMPONENT SEARCH ({rank} components)")
    print("="*80)

    all_results = []

    for component_idx in range(rank):
        print(f"\n--- Component {component_idx + 1}/{rank} ---")

        if decomposition_type == "cp":
            # Remove component from CP (zeros the weight)
            modified_weights, _ = remove_cp_component(decomp_weights, decomp_factors, component_idx)
            # Reconstruct weights
            fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights_cp(modified_weights, decomp_factors)
        else:  # tucker
            # Remove component from core
            modified_core = remove_component(decomp_weights, component_idx, dimension=1)
            # Reconstruct weights
            fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights(modified_core, decomp_factors)

        # Update model with reconstructed weights
        update_fc_layer_weights(
            model,
            target_layer,
            fc_in_reconstructed,
            fc_out_reconstructed,
            model_type="llama"
        )

        # Measure on all samples
        component_results = []

        for idx, sample in enumerate(tqdm(samples, desc=f"Component {component_idx}", leave=False)):
            question = sample['question']

            # Generate responses with modified model
            responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                num_generations=num_generations,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.95
            )

            # Measure uncertainty
            metrics = measure_uncertainty_and_blockiness(
                responses=responses,
                question=question,
                nli_calculator=nli_calculator,
                knowledge_extractor=knowledge_extractor,
                decomposition_type=decomposition_type,
                device=device
            )

            # Compute changes from baseline
            baseline = baseline_results[idx]
            uncertainty_change = metrics['uncertainty_score'] - baseline['uncertainty_score']
            blockiness_change = (
                metrics['blockiness_by_rank']['rank_10']['reconstruction_fit'] -
                baseline['blockiness_by_rank']['rank_10']['reconstruction_fit']
            )

            component_results.append({
                'sample_id': idx,
                'component_removed': component_idx,
                'uncertainty_score': metrics['uncertainty_score'],
                'uncertainty_change': uncertainty_change,
                'blockiness_change': blockiness_change,
                'metrics': metrics
            })

        all_results.append({
            'component_idx': component_idx,
            'results': component_results
        })

        # Checkpoint
        if (component_idx + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"checkpoint_{component_idx + 1}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Reload original weights for next component
        update_fc_layer_weights(
            model,
            target_layer,
            original_fc_in,
            original_fc_out,
            model_type="llama"
        )

        # Clear memory
        torch.cuda.empty_cache()

    # ========== Save Final Results ==========
    final_results = {
        'config': {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'num_samples': num_samples,
            'target_layer': target_layer,
            'rank': rank,
            'num_generations': num_generations,
            'nli_model': nli_model,
            'timestamp': timestamp,
            'decomposition_type': decomposition_type
        },
        'baseline': baseline_results,
        'component_search': all_results
    }

    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir / 'results.pkl'}")
    print(f"Total components tested: {rank}")
    print(f"Total samples: {num_samples}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Component Search Experiment')
    parser.add_argument('--layer', type=int, default=30, help='Target layer (default: 30)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples (default: 10)')
    parser.add_argument('--rank', type=int, default=80, help='Decomposition rank (default: 80)')
    parser.add_argument('--dataset', type=str, default='coqa', choices=['coqa', 'hotpotqa'],
                        help='Dataset to use (default: coqa)')
    parser.add_argument('--decomposition', type=str, default='cp', choices=['tucker', 'cp'],
                        help='Decomposition type (default: cp)')
    parser.add_argument('--test', action='store_true', help='Run quick test (1 sample, 3 components)')

    args = parser.parse_args()

    if args.test:
        # Quick test
        print("Running QUICK TEST mode...")
        run_component_search(
            dataset_name=args.dataset,
            num_samples=1,
            target_layer=args.layer,
            rank=3,
            checkpoint_every=1,
            decomposition_type=args.decomposition
        )
    else:
        # Full run
        print(f"Running FULL experiment on layer {args.layer}...")
        run_component_search(
            dataset_name=args.dataset,
            num_samples=args.samples,
            target_layer=args.layer,
            rank=args.rank,
            checkpoint_every=5,
            decomposition_type=args.decomposition
        )
