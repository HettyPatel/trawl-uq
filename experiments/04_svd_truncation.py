"""
SVD Truncation Experiment (LASER-style)

This script implements LASER-style rank reduction using SVD and measures its effect
on uncertainty quantification metrics.

Based on: "The Truth is in There: Improving Reasoning in Language Models
with Layer-Selective Rank Reduction" (Sharma et al., 2023)

Key differences from CP/Tucker noise removal:
- SVD operates on individual weight matrices (not stacked tensors)
- Components are naturally ordered by singular value magnitude
- We sweep over reduction percentages (keep top X% of singular values)
- Can target either MLP input or output matrix separately

Experiment flow:
1. Baseline: Original model performance
2. SVD Analysis: Analyze singular value distribution
3. Progressive Truncation: For each reduction %, measure uncertainty/F1/NLL
4. Generate analysis figures and report
"""

import sys
sys.path.append('.')

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    build_semantic_similarity_matrix,
    build_knowledge_similarity_matrix
)
from src.uncertainty.knowledge import (
    KnowledgeExtractor,
    compute_valid_ratio
)
from src.uncertainty.metrics import (
    compute_blockiness_score,
    compute_uncertainty_score
)
from src.evaluation.metrics import compute_evaluation_metrics
from src.decomposition.svd import (
    decompose_weight_svd,
    truncate_svd,
    reconstruct_from_svd,
    low_rank_approximation,
    compute_energy_retention,
    compute_reconstruction_error_svd,
    get_svd_stats,
    apply_svd_to_layer,
    update_layer_with_svd,
    restore_original_weight,
    reduction_to_keep_ratio,
    LASER_REDUCTION_PERCENTAGES
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def format_short_answer_prompt(question: str) -> str:
    """Format question as a prompt that encourages short answers (for evaluation)."""
    return f"Question: {question}\nAnswer (short):"


def measure_uncertainty_and_blockiness(
        responses,
        question,
        nli_calculator,
        knowledge_extractor,
        device="cuda",
        ranks=[5, 10, 15, 20]
):
    """
    Measure uncertainty and blockiness for a set of responses.

    Args:
        responses: List of generated responses
        question: Original question
        nli_calculator: NLISimilarityCalculator instance
        knowledge_extractor: KnowledgeExtractor instance
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

    # Check valid ratio - if too many responses are invalid, return max uncertainty
    valid_ratio = compute_valid_ratio(knowledge_responses)

    # Build knowledge similarity matrix (with invalid response handling)
    S_knowledge, knowledge_stats = build_knowledge_similarity_matrix(
        knowledge_responses=knowledge_responses,
        nli_calculator=nli_calculator,
        device=device
    )

    # Add valid_ratio to stats
    knowledge_stats['valid_ratio'] = valid_ratio

    # If less than 50% valid responses, model outputs are unreliable
    if valid_ratio < 0.5:
        return {
            'semantic_similarity': S_semantic,
            'knowledge_similarity': S_knowledge,
            'knowledge_responses': knowledge_responses,
            'knowledge_stats': knowledge_stats,
            'blockiness_by_rank': {f'rank_{r}': None for r in ranks},
            'uncertainty_score': 1.0,  # Maximum uncertainty - outputs unreliable
            'unreliable_reason': f'valid_ratio={valid_ratio:.2f} < 0.5'
        }

    # Compute blockiness at multiple ranks (using CP as decomposition type for consistency)
    blockiness_results = {}
    for rank in ranks:
        blockiness = compute_blockiness_score(
            S_semantic=S_semantic,
            S_knowledge=S_knowledge,
            rank=rank,
            decomposition_type="cp"
        )
        blockiness_results[f'rank_{rank}'] = blockiness

    # Compute uncertainty (using rank = 10 as default)
    uncertainty = compute_uncertainty_score(blockiness_results['rank_10'])

    return {
        'semantic_similarity': S_semantic,
        'knowledge_similarity': S_knowledge,
        'knowledge_responses': knowledge_responses,
        'knowledge_stats': knowledge_stats,
        'blockiness_by_rank': blockiness_results,
        'uncertainty_score': uncertainty
    }


def generate_svd_analysis_figures(
        svd_stats,
        output_dir,
        layer_name,
        matrix_type
):
    """Generate figures analyzing singular value distribution."""
    output_dir = Path(output_dir)

    # Plot 1: Singular value spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    singular_values = svd_stats['singular_values']

    # Linear scale
    axes[0].plot(singular_values, 'b-', linewidth=1)
    axes[0].set_xlabel('Component Index', fontsize=12)
    axes[0].set_ylabel('Singular Value', fontsize=12)
    axes[0].set_title(f'{layer_name} {matrix_type}: Singular Value Spectrum (Linear)', fontsize=14)
    axes[0].grid(alpha=0.3)

    # Log scale
    axes[1].semilogy(singular_values, 'b-', linewidth=1)
    axes[1].set_xlabel('Component Index', fontsize=12)
    axes[1].set_ylabel('Singular Value (log)', fontsize=12)
    axes[1].set_title(f'{layer_name} {matrix_type}: Singular Value Spectrum (Log)', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'singular_value_spectrum.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'singular_value_spectrum.png'}")
    plt.close()

    # Plot 2: Cumulative energy
    fig, ax = plt.subplots(figsize=(10, 6))

    cumulative_energy = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
    ax.plot(cumulative_energy * 100, 'b-', linewidth=2)

    # Mark key thresholds
    for thresh, color in [(90, 'green'), (95, 'orange'), (99, 'red')]:
        k = np.searchsorted(cumulative_energy, thresh / 100) + 1
        ax.axhline(y=thresh, color=color, linestyle='--', alpha=0.5, label=f'{thresh}% at k={k}')
        ax.axvline(x=k, color=color, linestyle='--', alpha=0.5)

    ax.set_xlabel('Number of Components Kept', fontsize=12)
    ax.set_ylabel('Cumulative Energy (%)', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: Cumulative Energy vs Components', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_energy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_energy.png'}")
    plt.close()


def generate_truncation_figures(
        truncation_results,
        baseline_results,
        output_dir,
        layer_name,
        matrix_type
):
    """Generate figures for SVD truncation experiment."""
    output_dir = Path(output_dir)

    # Extract data
    reductions = [r['reduction_percent'] for r in truncation_results]
    uncertainties = [r['avg_uncertainty'] for r in truncation_results]
    f1_scores = [r['avg_f1'] for r in truncation_results]
    nlls = [r['avg_nll'] for r in truncation_results]
    energy_retentions = [r['energy_retention'] * 100 for r in truncation_results]

    baseline_uncertainty = np.mean([r['uncertainty_score'] for r in baseline_results])
    baseline_f1 = np.mean([r['evaluation_metrics']['f1'] for r in baseline_results])
    baseline_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in baseline_results])

    # Plot 1: Uncertainty vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_uncertainty, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_uncertainty:.4f})')
    ax.plot(reductions, uncertainties, 'b-o', linewidth=2, markersize=6, label='After SVD Truncation')

    # Shade improvement region
    ax.fill_between(reductions, baseline_uncertainty, uncertainties,
                    where=[u < baseline_uncertainty for u in uncertainties],
                    alpha=0.3, color='green', label='Improvement')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average Uncertainty', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: Uncertainty vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # Higher reduction on left (more aggressive)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_vs_reduction.png'}")
    plt.close()

    # Plot 2: F1 Score vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_f1:.3f})')
    ax.plot(reductions, f1_scores, 'g-o', linewidth=2, markersize=6, label='After SVD Truncation')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: F1 Score vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'f1_vs_reduction.png'}")
    plt.close()

    # Plot 3: Answer NLL vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_nll, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_nll:.2f})')
    ax.plot(reductions, nlls, 'm-o', linewidth=2, markersize=6, label='After SVD Truncation')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average Answer NLL', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: Answer NLL vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'nll_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nll_vs_reduction.png'}")
    plt.close()

    # Plot 4: Energy Retention vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(reductions, energy_retentions, 'c-o', linewidth=2, markersize=6)

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Energy Retention (%)', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: Energy Retention vs SVD Reduction', fontsize=14)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_retention.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'energy_retention.png'}")
    plt.close()

    # Plot 5: Combined - Uncertainty vs F1 tradeoff
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax1.set_ylabel('Uncertainty', fontsize=12, color='blue')
    line1 = ax1.plot(reductions, uncertainties, 'b-o', linewidth=2, markersize=6, label='Uncertainty')
    ax1.axhline(y=baseline_uncertainty, color='blue', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', fontsize=12, color='green')
    line2 = ax2.plot(reductions, f1_scores, 'g-s', linewidth=2, markersize=6, label='F1 Score')
    ax2.axhline(y=baseline_f1, color='green', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='green')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11)

    ax1.set_title(f'{layer_name} {matrix_type}: Uncertainty vs F1 Tradeoff', fontsize=14)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_f1_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_f1_tradeoff.png'}")
    plt.close()

    # Plot 6: Percentage change from baseline
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert reduction % to keep % for clearer x-axis
    keep_pcts = [100 - r for r in reductions]

    # Both use (new - baseline) / baseline * 100
    # Positive = increased, Negative = decreased
    unc_pct_change = [(u - baseline_uncertainty) / baseline_uncertainty * 100 for u in uncertainties]
    f1_pct_change = [(f - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0 for f in f1_scores]

    ax.plot(keep_pcts, unc_pct_change, 'b-o', linewidth=2, markersize=6, label='Uncertainty Change %')
    ax.plot(keep_pcts, f1_pct_change, 'g-s', linewidth=2, markersize=6, label='F1 Change %')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Components Kept (%)', fontsize=12)
    ax.set_ylabel('Change from Baseline (%)', fontsize=12)
    ax.set_title(f'{layer_name} {matrix_type}: Percentage Change from Baseline', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # Left = 90% kept (mild), Right = 0.25% kept (aggressive)

    plt.tight_layout()
    plt.savefig(output_dir / 'percentage_change.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'percentage_change.png'}")
    plt.close()


def generate_report(
        svd_stats,
        truncation_results,
        baseline_results,
        config,
        output_dir
):
    """Generate text report with findings."""
    output_dir = Path(output_dir)

    # Baseline stats
    baseline_avg_unc = np.mean([r['uncertainty_score'] for r in baseline_results])
    baseline_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in baseline_results])

    report = []
    report.append("=" * 80)
    report.append("SVD TRUNCATION EXPERIMENT REPORT (LASER-style)")
    report.append("=" * 80)
    report.append("")

    report.append("EXPERIMENT CONFIGURATION")
    report.append("-" * 80)
    report.append(f"Model: {config.get('model_name', 'N/A')}")
    report.append(f"Dataset: {config.get('dataset_name', 'N/A')}")
    report.append(f"Layer: {config.get('target_layer', 'N/A')}")
    report.append(f"Matrix Type: {config.get('matrix_type', 'N/A')}")
    report.append(f"Samples: {config.get('num_samples', 'N/A')}")
    report.append("")

    report.append("SVD ANALYSIS")
    report.append("-" * 80)
    report.append(f"Matrix Shape: {svd_stats['shape']}")
    report.append(f"Max Rank: {svd_stats['max_rank']}")
    report.append(f"Effective Rank: {svd_stats['effective_rank']:.1f}")
    report.append(f"Top Singular Value: {svd_stats['top_singular_value']:.4f}")
    report.append(f"Bottom Singular Value: {svd_stats['bottom_singular_value']:.6f}")
    report.append(f"Singular Value Ratio (max/min): {svd_stats['singular_value_ratio']:.1f}")
    report.append(f"Components for 90% energy: {svd_stats['k_for_90_energy']}")
    report.append(f"Components for 95% energy: {svd_stats['k_for_95_energy']}")
    report.append(f"Components for 99% energy: {svd_stats['k_for_99_energy']}")
    report.append("")

    report.append("BASELINE PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Uncertainty: {baseline_avg_unc:.4f}")
    report.append(f"F1 Score: {baseline_avg_f1:.4f}")
    report.append(f"Answer NLL: {baseline_avg_nll:.2f}")
    report.append("")

    report.append("TRUNCATION RESULTS")
    report.append("=" * 80)
    report.append(f"{'Reduction %':<12} {'Keep %':<10} {'Uncertainty':<12} {'Unc Δ':<10} {'F1':<10} {'F1 Δ':<10} {'NLL':<10} {'Energy %':<10}")
    report.append("-" * 80)

    for tr in truncation_results:
        unc_change = tr['avg_uncertainty'] - baseline_avg_unc
        f1_change = tr['avg_f1'] - baseline_avg_f1
        report.append(
            f"{tr['reduction_percent']:<12.1f} "
            f"{tr['keep_ratio']*100:<10.1f} "
            f"{tr['avg_uncertainty']:<12.4f} "
            f"{unc_change:+.4f}    "
            f"{tr['avg_f1']:<10.3f} "
            f"{f1_change:+.3f}    "
            f"{tr['avg_nll']:<10.2f} "
            f"{tr['energy_retention']*100:<10.1f}"
        )

    report.append("")

    # Find best result
    best_by_unc = min(truncation_results, key=lambda x: x['avg_uncertainty'])
    unc_reduction = (baseline_avg_unc - best_by_unc['avg_uncertainty']) / baseline_avg_unc * 100
    f1_retention = best_by_unc['avg_f1'] / baseline_avg_f1 * 100 if baseline_avg_f1 > 0 else 100

    report.append("OPTIMAL RESULT")
    report.append("-" * 80)
    report.append(f"Best reduction: {best_by_unc['reduction_percent']:.1f}%")
    report.append(f"Uncertainty reduction: {unc_reduction:.1f}%")
    report.append(f"F1 retention: {f1_retention:.1f}%")
    report.append(f"Energy retention: {best_by_unc['energy_retention']*100:.1f}%")
    report.append("")

    report.append("INTERPRETATION (LASER)")
    report.append("-" * 80)
    report.append("Lower-order components (larger singular values):")
    report.append("  - Contain core model knowledge")
    report.append("  - Should be kept for model quality")
    report.append("")
    report.append("Higher-order components (smaller singular values):")
    report.append("  - May contain noise or conflicting information")
    report.append("  - Removing them can REDUCE uncertainty (denoising effect)")
    report.append("  - LASER paper found this especially effective in later MLP layers")
    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_dir / 'analysis_report.txt'}")


def run_svd_truncation(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_type="llama",
        dataset_name="hotpotqa",
        num_samples=10,
        target_layer=31,
        matrix_type="mlp_in",
        reduction_percentages=None,
        num_generations=20,
        nli_model="microsoft/deberta-large-mnli",
        device="cuda",
        checkpoint_every=3,
):
    """
    Run SVD truncation experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture type ('llama' or 'gpt2')
        dataset_name: Dataset to use (coqa, hotpotqa, nq_open)
        num_samples: Number of dataset samples
        target_layer: Which layer to truncate
        matrix_type: Which matrix to truncate ('mlp_in' or 'mlp_out')
        reduction_percentages: List of reduction percentages to test
        num_generations: Responses per sample
        nli_model: NLI model for semantic similarity
        device: Device to run on
        checkpoint_every: Save results every N reduction levels
    """
    # Default reduction percentages (LASER-style)
    if reduction_percentages is None:
        reduction_percentages = LASER_REDUCTION_PERCENTAGES

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_model_name = model_name.split("/")[-1]
    output_dir = Path(f"results/svd_truncation/{short_model_name}_{dataset_name}_layer{target_layer}_{matrix_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_name = f"Layer {target_layer}"

    print("=" * 80)
    print("SVD Truncation Experiment (LASER-style)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset_name}")
    print(f"Num samples: {num_samples}")
    print(f"Target layer: {target_layer}")
    print(f"Matrix type: {matrix_type}")
    print(f"Reduction percentages: {reduction_percentages}")
    print(f"Generations per sample: {num_generations}")
    print(f"Output: {output_dir}")
    print("=" * 80)

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
    # Uses Mistral-7B-Instruct by default (best for fact extraction)
    print("\nLoading knowledge extractor (Mistral-7B-Instruct)...")
    knowledge_extractor = KnowledgeExtractor(device=device)
    print("Knowledge extractor loaded")

    # ========== Get Original Weight and SVD Stats ==========
    print(f"\nAnalyzing SVD of layer {target_layer} {matrix_type}...")

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

    svd_stats = get_svd_stats(original_weight, device)
    print(f"Matrix shape: {svd_stats['shape']}")
    print(f"Max rank: {svd_stats['max_rank']}")
    print(f"Effective rank: {svd_stats['effective_rank']:.1f}")
    print(f"Components for 90% energy: {svd_stats['k_for_90_energy']}")
    print(f"Components for 99% energy: {svd_stats['k_for_99_energy']}")

    # Generate SVD analysis figures
    generate_svd_analysis_figures(svd_stats, output_dir, layer_name, matrix_type)

    # Save SVD stats
    with open(output_dir / "svd_stats.pkl", 'wb') as f:
        pickle.dump(svd_stats, f)

    # ========== PHASE 1: Baseline Measurement ==========
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE MEASUREMENT (Original Model)")
    print("=" * 80)

    baseline_results = []

    for idx, sample in enumerate(tqdm(samples, desc="Baseline")):
        question = sample['question']
        gold_answer = sample['answer']

        # Generate sampled responses for uncertainty measurement
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            num_generations=num_generations,
            max_new_tokens=100,
            temperature=1.0,
            top_p=0.95
        )

        # Generate greedy response for quality evaluation
        greedy_response = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=format_short_answer_prompt(question),
            num_generations=1,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False
        )[0]

        # Measure uncertainty
        metrics = measure_uncertainty_and_blockiness(
            responses=responses,
            question=question,
            nli_calculator=nli_calculator,
            knowledge_extractor=knowledge_extractor,
            device=device
        )

        # Measure downstream task performance
        eval_metrics = compute_evaluation_metrics(
            response=greedy_response,
            gold_answer=gold_answer,
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device,
            nll_prompt_text=format_short_answer_prompt(question)
        )

        baseline_results.append({
            'sample_id': idx,
            'question': question,
            'answer': gold_answer,
            'responses': responses,
            'greedy_response': greedy_response,
            **metrics,
            'evaluation_metrics': eval_metrics
        })

    # Save baseline
    with open(output_dir / "baseline.pkl", 'wb') as f:
        pickle.dump(baseline_results, f)
    print(f"\nSaved baseline to {output_dir / 'baseline.pkl'}")

    baseline_avg_unc = np.mean([r['uncertainty_score'] for r in baseline_results])
    baseline_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in baseline_results])
    print(f"Baseline avg uncertainty: {baseline_avg_unc:.4f}")
    print(f"Baseline avg F1: {baseline_avg_f1:.4f}")
    print(f"Baseline avg answer NLL: {baseline_avg_nll:.2f}")

    # ========== PHASE 2: Progressive SVD Truncation ==========
    print("\n" + "=" * 80)
    print(f"PHASE 2: SVD TRUNCATION ({len(reduction_percentages)} reduction levels)")
    print("=" * 80)

    # Decompose ONCE and reuse for all reduction levels
    print("\nPerforming SVD decomposition (once)...")
    U, S, Vh = decompose_weight_svd(original_weight, device)
    print(f"  SVD complete: {len(S)} singular values")

    truncation_results = []

    for reduction_idx, reduction_pct in enumerate(reduction_percentages):
        keep_ratio = reduction_to_keep_ratio(reduction_pct)
        print(f"\n--- Reduction {reduction_pct}% (keep {keep_ratio*100:.1f}%) ---")

        # Truncate SVD components (reusing precomputed U, S, Vh)
        U_trunc, S_trunc, Vh_trunc = truncate_svd(U, S, Vh, keep_ratio)

        # Reconstruct low-rank weight from truncated components
        weight_lr = reconstruct_from_svd(U_trunc, S_trunc, Vh_trunc)

        # Compute stats
        energy_retention = compute_energy_retention(S, S_trunc)
        recon_error = compute_reconstruction_error_svd(original_weight, weight_lr)

        print(f"  Kept {len(S_trunc)}/{len(S)} components")
        print(f"  Energy retention: {energy_retention*100:.1f}%")
        print(f"  Reconstruction error: {recon_error:.4f}")

        # Update model weight
        update_layer_with_svd(model, target_layer, weight_lr, matrix_type, model_type)

        # Measure on all samples
        reduction_results = []

        for idx, sample in enumerate(tqdm(samples, desc=f"Reduction {reduction_pct}%", leave=False)):
            question = sample['question']
            gold_answer = sample['answer']

            # Generate sampled responses
            responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                num_generations=num_generations,
                max_new_tokens=100,
                temperature=1.0,
                top_p=0.95
            )

            # Generate greedy response
            greedy_response = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=format_short_answer_prompt(question),
                num_generations=1,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False
            )[0]

            # Measure uncertainty
            metrics = measure_uncertainty_and_blockiness(
                responses=responses,
                question=question,
                nli_calculator=nli_calculator,
                knowledge_extractor=knowledge_extractor,
                device=device
            )

            # Measure downstream task performance
            eval_metrics = compute_evaluation_metrics(
                response=greedy_response,
                gold_answer=gold_answer,
                model=model,
                tokenizer=tokenizer,
                question=question,
                device=device,
                nll_prompt_text=format_short_answer_prompt(question)
            )

            # Compute changes from baseline
            baseline = baseline_results[idx]
            uncertainty_change = metrics['uncertainty_score'] - baseline['uncertainty_score']
            f1_change = eval_metrics['f1'] - baseline['evaluation_metrics']['f1']
            nll_change = eval_metrics['answer_nll'] - baseline['evaluation_metrics']['answer_nll']

            reduction_results.append({
                'sample_id': idx,
                'reduction_percent': reduction_pct,
                'keep_ratio': keep_ratio,
                'uncertainty_score': metrics['uncertainty_score'],
                'uncertainty_change': uncertainty_change,
                'f1': eval_metrics['f1'],
                'f1_change': f1_change,
                'answer_nll': eval_metrics['answer_nll'],
                'nll_change': nll_change,
                'responses': responses,
                'greedy_response': greedy_response,
                'metrics': metrics,
                'evaluation_metrics': eval_metrics
            })

        # Compute averages
        avg_uncertainty = np.mean([r['uncertainty_score'] for r in reduction_results])
        avg_uncertainty_change = np.mean([r['uncertainty_change'] for r in reduction_results])
        avg_f1 = np.mean([r['f1'] for r in reduction_results])
        avg_f1_change = np.mean([r['f1_change'] for r in reduction_results])
        avg_nll = np.mean([r['answer_nll'] for r in reduction_results])
        avg_nll_change = np.mean([r['nll_change'] for r in reduction_results])

        truncation_results.append({
            'reduction_percent': reduction_pct,
            'keep_ratio': keep_ratio,
            'components_kept': len(S_trunc),
            'total_components': len(S),
            'energy_retention': energy_retention,
            'reconstruction_error': recon_error,
            'avg_uncertainty': avg_uncertainty,
            'avg_uncertainty_change': avg_uncertainty_change,
            'avg_f1': avg_f1,
            'avg_f1_change': avg_f1_change,
            'avg_nll': avg_nll,
            'avg_nll_change': avg_nll_change,
            'results': reduction_results
        })

        print(f"  Uncertainty: {avg_uncertainty:.4f} ({avg_uncertainty_change:+.4f})")
        print(f"  F1: {avg_f1:.3f} ({avg_f1_change:+.3f})")
        print(f"  Answer NLL: {avg_nll:.2f} ({avg_nll_change:+.2f})")

        # Restore original weight for next iteration
        restore_original_weight(model, target_layer, original_weight, matrix_type, model_type)

        # Checkpoint
        if (reduction_idx + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"checkpoint_{reduction_idx + 1}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(truncation_results, f)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Clear memory
        torch.cuda.empty_cache()

    # Save truncation results
    with open(output_dir / "truncation_results.pkl", 'wb') as f:
        pickle.dump(truncation_results, f)
    print(f"\nSaved truncation results to {output_dir / 'truncation_results.pkl'}")

    # ========== Generate Figures ==========
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    generate_truncation_figures(
        truncation_results,
        baseline_results,
        output_dir,
        layer_name,
        matrix_type
    )

    # ========== Generate Report ==========
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    config = {
        'model_name': model_name,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'num_samples': num_samples,
        'target_layer': target_layer,
        'matrix_type': matrix_type,
        'reduction_percentages': reduction_percentages,
        'num_generations': num_generations,
        'nli_model': nli_model,
        'timestamp': timestamp,
        'experiment_type': 'svd_truncation'
    }

    generate_report(
        svd_stats,
        truncation_results,
        baseline_results,
        config,
        output_dir
    )

    # ========== Save Final Results ==========
    final_results = {
        'config': config,
        'svd_stats': svd_stats,
        'baseline': baseline_results,
        'truncation_results': truncation_results
    }

    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Layer: {target_layer}")
    print(f"Matrix: {matrix_type}")
    print(f"Reduction levels tested: {len(reduction_percentages)}")
    print(f"Samples: {num_samples}")
    print("\nFiles created:")
    print("  - results.pkl (all data)")
    print("  - baseline.pkl")
    print("  - svd_stats.pkl")
    print("  - truncation_results.pkl")
    print("  - singular_value_spectrum.png")
    print("  - cumulative_energy.png")
    print("  - uncertainty_vs_reduction.png")
    print("  - f1_vs_reduction.png")
    print("  - nll_vs_reduction.png")
    print("  - energy_retention.png")
    print("  - uncertainty_f1_tradeoff.png")
    print("  - percentage_change.png")
    print("  - analysis_report.txt")
    print("=" * 80)

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SVD Truncation Experiment (LASER-style)')
    parser.add_argument('--layer', type=int, default=31, help='Target layer (default: 31)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples (default: 10)')
    parser.add_argument('--matrix', type=str, default='mlp_in', choices=['mlp_in', 'mlp_out'],
                        help='Matrix to truncate (default: mlp_in)')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['coqa', 'hotpotqa', 'nq_open'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='HuggingFace model name (default: meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--model-type', type=str, default='llama', choices=['llama', 'gpt2'],
                        help='Model architecture type (default: llama). Use gpt2 for GPT-2/GPT-J')
    parser.add_argument('--test', action='store_true', help='Run quick test (1 sample, 3 reduction levels)')

    args = parser.parse_args()

    if args.test:
        # Quick test
        print("Running QUICK TEST mode...")
        run_svd_truncation(
            model_name=args.model,
            model_type=args.model_type,
            dataset_name=args.dataset,
            num_samples=1,
            target_layer=args.layer,
            matrix_type=args.matrix,
            reduction_percentages=[50, 90, 99],  # Just 3 levels for quick test
            checkpoint_every=1,
        )
    else:
        # Full run
        print(f"Running FULL experiment on layer {args.layer} {args.matrix}...")
        run_svd_truncation(
            model_name=args.model,
            model_type=args.model_type,
            dataset_name=args.dataset,
            num_samples=args.samples,
            target_layer=args.layer,
            matrix_type=args.matrix,
            checkpoint_every=3,
        )
