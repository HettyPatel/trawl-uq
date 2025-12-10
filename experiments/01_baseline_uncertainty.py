"""
Baseline Uncertainty Measurement Experiment

This script measrues uncertainty without any model intervention.
Establishes baseline metrics for comparison with component removal experiments.
"""

import sys
sys.path.append(".")  # Ensure src/ is in path

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.logging import setup_logger, get_timestamp
from src.generation.datasets import get_dataset
from src.generation.generate import (
    load_model_and_tokenizer,
    generate_for_dataset,
    seed_everything
)
from src.uncertainty.similarity import (
    NLISimilarityCalculator,
    build_semantic_similarity_matrix
)
from src.uncertainty.knowledge import KnowledgeExtractor
from src.uncertainty.metrics import (
    compute_blockiness_score,
    compute_uncertainty_score,
)


def main():
    # ========== Configuration ==========
    SEED = 42
    DATASET_NAME = "coqa"
    NUM_SAMPLES = 10  # Use small subset for testing
    NUM_GENERATIONS = 20

    # Models
    MAIN_MODEL = "gpt2" # For testing meta-llama/Llama2-7b-hf for larger runs
    KNOWLEDGE_MODEL = "gpt2" # For testing use meta-llama/Llama2-7b-hf for larger runs
    NLI_MODEL = "microsoft/deberta-large-mnli"

    DEVICE = "cuda"  # or "cpu"

    # output
    timestamp = get_timestamp()
    output_dir = Path(f"results/baseline/{DATASET_NAME}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # +========== Setup Logger ==========
    logger = setup_logger(
        "baseline",
        log_file=str(output_dir / "experiment.log")
    )

    logger.info("="*60)
    logger.info("Baseline Uncertainty Measurement Experiment")
    logger.info("="*60)
    logger.info(f"Dataset: {DATASET_NAME} (num_samples={NUM_SAMPLES})")
    logger.info(f"Generation per sample: {NUM_GENERATIONS}")
    logger.info(f"Main Model: {MAIN_MODEL}")
    logger.info(f"Knowledge Model: {KNOWLEDGE_MODEL}")
    logger.info(f"NLI Model: {NLI_MODEL}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*60)

    # set seed
    seed_everything(SEED)
    logger.info(f"Random seed set to {SEED}")

    # ========== Load Dataset ==========
    logger.info(f"\nLoading dataset {DATASET_NAME}...")
    dataset = get_dataset(DATASET_NAME, num_samples=NUM_SAMPLES)
    dataset.load(None)
    logger.info(f"Loaded {len(dataset)} samples from {DATASET_NAME} dataset.")

    # ========== Load Models ==========
    logger.info(f"\nLoading main model {MAIN_MODEL}...")
    model, tokenizer = load_model_and_tokenizer(MAIN_MODEL, device=DEVICE)
    logger.info("Main model loaded.")

    ## ========= Generate Response =========
    logger.info(f"\nGenerating {NUM_GENERATIONS} responses per sample...")

    results = generate_for_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=100,
        temperature=1.0,
        top_p=0.95,
        save_every=5,
        output_file=str(output_dir / "generations.pkl")
    )
    logger.info(f"Generation completed for {len(results)} samples.")

    # Save generations
    with open(output_dir / "generations.pkl", "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Generations saved to {output_dir / 'generations.pkl'}")

    # ========== Build Semantic Similarity Matrices ==========
    logger.info(f"\nComputing semantic similarity matrices...")

    nli_calculator = NLISimilarityCalculator(model_name=NLI_MODEL, device=DEVICE)

    for result in results:
        S_semantic = build_semantic_similarity_matrix(
            responses=result['responses'],
            nli_calculator=nli_calculator,
            device=DEVICE
        )
        result['semantic_similarity'] = S_semantic
    logger.info("Semantic similarity matrices computed.")

    #========== Extract Knowledge and Build Knowledge Similarity ==========
    logger.info("\nExtracting knowledge from responses and computing knowledge similarity matrices...")
    knowledge_extractor = KnowledgeExtractor(model_name=KNOWLEDGE_MODEL, device=DEVICE)

    for i, result in enumerate(results):
        logger.info(f"Processing sample {i+1}/{len(results)}: {result['id']}")
        
        # Extract knowledge
        knowledge_responses = knowledge_extractor.extract_knowledge_batch(
            question=result['question'],
            responses=result['responses']
        )
        result['knowledge_responses'] = knowledge_responses
        
        # Build knowledge similarity matrix
        S_knowledge = build_semantic_similarity_matrix(
            responses=knowledge_responses,
            nli_calculator=nli_calculator,
            device=DEVICE
        )
        result['knowledge_similarity'] = S_knowledge

    logger.info("✓ Extracted knowledge and computed knowledge similarity matrices")

    # ========== Compute Uncertainty Metrics ==========
    logger.info(f"\nComputing uncertainty metrics...")

    for result in results:
        blockiness = compute_blockiness_score(
            S_semantic=result['semantic_similarity'],
            S_knowledge=result['knowledge_similarity'],
            rank=10,
            decomposition_type='tucker'
        )

        uncertainty = compute_uncertainty_score(blockiness)

        result['blockiness_metrics'] = blockiness
        result['uncertainty_score'] = uncertainty

    logger.info("Uncertainty metrics computed.")

    # ========== Save Final Results ==========
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Final results saved to {output_dir / 'results.pkl'}")

    # ========== Summary Statistics ==========
    logger.info("\n" + "="*60)
    logger.info("Summary Statistics")
    logger.info("="*60)

    uncertainties = [r['uncertainty_score'] for r in results]
    recon_fits = [r['blockiness_metrics']['reconstruction_fit'] for r in results]

    logger.info(f"Uncertainty Scores:")
    logger.info(f"  Mean: {np.mean(uncertainties):.3f}")
    logger.info(f"  Std: {np.std(uncertainties):.3f}")
    logger.info(f"  Min: {np.min(uncertainties):.3f}")
    logger.info(f"  Max: {np.max(uncertainties):.3f}")

    logger.info(f"\nReconstruction Fit:")
    logger.info(f"  Mean: {np.mean(recon_fits):.3f}")
    logger.info(f"  Std: {np.std(recon_fits):.3f}")

    logger.info("\n" + "="*60)
    logger.info("Baseline experiment completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()