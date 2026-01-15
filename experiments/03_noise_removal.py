"""
Noise Removal Experiment: Tucker and CP Decomposition

This script:
1. Runs component search to identify noise vs signal components
2. Ranks components by uncertainty impact
3. Progressively removes top-k noise components and measures uncertainty reduction
4. Generates analysis figures and reports

Goal: Remove top-1 noise, top-2 noise, ..., top-N noise components and measure uncertainty reduction
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
        'knowledge_stats': knowledge_stats,  # Stats about invalid/degenerate responses
        'blockiness_by_rank': blockiness_results,
        'uncertainty_score': uncertainty
    }


def generate_analysis_figures(
        component_search_results,
        baseline_results,
        output_dir,
        decomposition_type,
        layer_name,
        rank
):
    """Generate analysis figures similar to analysis_report.py"""

    output_dir = Path(output_dir)
    decomp_label = decomposition_type.upper()

    # Extract data
    indices = [c['component_idx'] for c in component_search_results]
    unc_change = [c['avg_uncertainty_change'] for c in component_search_results]

    # Also compute blockiness changes (handle None values from high invalid rate)
    block_change = []
    for comp in component_search_results:
        changes = []
        for r in comp['results']:
            sample_id = r['sample_id']
            metrics_block = r['metrics']['blockiness_by_rank'].get('rank_10')
            baseline_block = baseline_results[sample_id]['blockiness_by_rank'].get('rank_10')

            # Skip if either blockiness is None (high invalid rate)
            if metrics_block is None or baseline_block is None:
                # Component caused model breakdown - treat as large negative change
                # (removing it would help, but we can't measure blockiness)
                continue

            change = metrics_block['reconstruction_fit'] - baseline_block['reconstruction_fit']
            changes.append(change)

        # If all samples had None blockiness, use NaN
        block_change.append(np.mean(changes) if changes else np.nan)

    # Plot 1: Uncertainty Change by Component
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
    ax.bar(indices, unc_change, color=['green' if x < 0 else 'orange' for x in unc_change], alpha=0.7)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Uncertainty Change', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Component Removal Effect on Uncertainty\n(Green = Noise, Orange = Signal)', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_change_by_component.png'}")
    plt.close()

    # Plot 2: Scatter - Uncertainty vs Blockiness Change (filter out NaN values)
    fig, ax = plt.subplots(figsize=(10, 8))
    valid_mask = ~np.isnan(block_change)
    valid_unc = np.array(unc_change)[valid_mask]
    valid_block = np.array(block_change)[valid_mask]
    valid_indices = np.array(indices)[valid_mask]
    if len(valid_unc) > 0:
        sc = ax.scatter(valid_unc, valid_block, alpha=0.6, s=100, c=valid_indices, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='Component Index')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Uncertainty Change', fontsize=12)
    ax.set_ylabel('Blockiness Change', fontsize=12)
    n_skipped = np.sum(~valid_mask)
    title = f'{layer_name} ({decomp_label}): Uncertainty vs Blockiness Change'
    if n_skipped > 0:
        title += f'\n({n_skipped} components with broken model excluded)'
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_blockiness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_vs_blockiness.png'}")
    plt.close()

    # Plot 3: Distribution of Changes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(unc_change, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Uncertainty Change', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{layer_name}: Distribution of Uncertainty Changes', fontsize=14)
    axes[0].grid(alpha=0.3)

    valid_block_hist = [b for b in block_change if not np.isnan(b)]
    axes[1].hist(valid_block_hist, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Blockiness Change', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    title_block = f'{layer_name}: Distribution of Blockiness Changes'
    if len(block_change) - len(valid_block_hist) > 0:
        title_block += f' ({len(block_change) - len(valid_block_hist)} excluded)'
    axes[1].set_title(title_block, fontsize=14)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'change_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'change_distributions.png'}")
    plt.close()

    return unc_change, block_change


def generate_topk_removal_figures(
        progressive_results,
        baseline_avg_uncertainty,
        baseline_results,
        decomposed_baseline_results,
        output_dir,
        decomposition_type,
        layer_name
):
    """Generate figures for top-k noise component removal"""

    output_dir = Path(output_dir)
    decomp_label = decomposition_type.upper()

    if not progressive_results:
        print("No progressive results to plot")
        return

    # Decomposed baseline stats (K=0)
    decomp_avg_unc = np.mean([r['uncertainty_score'] for r in decomposed_baseline_results])
    decomp_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in decomposed_baseline_results])
    decomp_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in decomposed_baseline_results])

    # Extract data and prepend K=0 (decomposed baseline)
    k_values = [0] + [pr['num_noise_removed'] for pr in progressive_results]
    avg_uncertainties = [decomp_avg_unc] + [pr['avg_uncertainty'] for pr in progressive_results]
    avg_changes = [decomp_avg_unc - baseline_avg_uncertainty] + [pr['avg_uncertainty_change'] for pr in progressive_results]
    avg_f1s = [decomp_avg_f1] + [pr['avg_f1'] for pr in progressive_results]
    avg_nlls = [decomp_avg_nll] + [pr['avg_nll'] for pr in progressive_results]

    # Original baseline evaluation metrics
    baseline_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in baseline_results])

    # Plot 1: Uncertainty vs Number of Noise Components Removed
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_avg_uncertainty, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_avg_uncertainty:.4f})')
    ax.plot(k_values, avg_uncertainties, 'b-o', linewidth=2, markersize=8, label='After Noise Removal')
    ax.fill_between(k_values, baseline_avg_uncertainty, avg_uncertainties,
                    alpha=0.3, color='green', where=[u < baseline_avg_uncertainty for u in avg_uncertainties])
    ax.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax.set_ylabel('Average Uncertainty', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Uncertainty vs Top-K Noise Removal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_uncertainty_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_uncertainty_reduction.png'}")
    plt.close()

    # Plot 2: Uncertainty Change vs K
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if c < 0 else 'orange' for c in avg_changes]
    ax.bar(k_values, avg_changes, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax.set_ylabel('Uncertainty Change from Baseline', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Uncertainty Change vs Top-K Noise Removal\n(Green = Reduced, Orange = Increased)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_uncertainty_change.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_uncertainty_change.png'}")
    plt.close()

    # Plot 3: Cumulative effect - percentage reduction
    pct_changes = [(baseline_avg_uncertainty - u) / baseline_avg_uncertainty * 100 for u in avg_uncertainties]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k_values, pct_changes, 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax.set_ylabel('Uncertainty Reduction (%)', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Percentage Uncertainty Reduction vs Top-K', fontsize=14)
    ax.grid(alpha=0.3)

    # Annotate max reduction
    max_idx = np.argmax(pct_changes)
    ax.annotate(f'Max: {pct_changes[max_idx]:.1f}% at K={k_values[max_idx]}',
                xy=(k_values[max_idx], pct_changes[max_idx]),
                xytext=(k_values[max_idx] + 1, pct_changes[max_idx] + 2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(output_dir / 'topk_percentage_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_percentage_reduction.png'}")
    plt.close()

    # Plot 4: F1 Score vs K
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_avg_f1, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline F1 ({baseline_avg_f1:.3f})')
    ax.plot(k_values, avg_f1s, 'b-o', linewidth=2, markersize=8, label='F1 After Noise Removal')
    ax.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax.set_ylabel('Average F1 Score', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): F1 Score vs Top-K Noise Removal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_f1_score.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_f1_score.png'}")
    plt.close()

    # Plot 5: Answer NLL vs K
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_avg_nll, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline NLL ({baseline_avg_nll:.2f})')
    ax.plot(k_values, avg_nlls, 'b-o', linewidth=2, markersize=8, label='NLL After Noise Removal')
    ax.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax.set_ylabel('Average Answer NLL', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Answer NLL vs Top-K Noise Removal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_nll.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_nll.png'}")
    plt.close()

    # Plot 6: Combined metrics - Uncertainty vs F1 tradeoff
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Uncertainty on left axis
    ax1.set_xlabel('Number of Noise Components Removed (Top-K)', fontsize=12)
    ax1.set_ylabel('Uncertainty', fontsize=12, color='blue')
    line1 = ax1.plot(k_values, avg_uncertainties, 'b-o', linewidth=2, markersize=8, label='Uncertainty')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(alpha=0.3)

    # F1 on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', fontsize=12, color='green')
    line2 = ax2.plot(k_values, avg_f1s, 'g-s', linewidth=2, markersize=8, label='F1 Score')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11)

    ax1.set_title(f'{layer_name} ({decomp_label}): Uncertainty vs F1 Tradeoff', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_uncertainty_f1_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_uncertainty_f1_tradeoff.png'}")
    plt.close()


def generate_report(
        component_search_results,
        rankings_info,
        progressive_results,
        baseline_results,
        decomposed_baseline_results,
        config,
        output_dir,
        decomposition_type,
        layer_name
):
    """Generate text report with findings"""

    output_dir = Path(output_dir)
    rank = config.get('rank', 40)
    decomp_label = decomposition_type.upper()

    # Baseline stats
    baseline_avg = np.mean([r['uncertainty_score'] for r in baseline_results])
    baseline_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in baseline_results])
    baseline_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in baseline_results])

    # Decomposed baseline stats
    decomp_avg = np.mean([r['uncertainty_score'] for r in decomposed_baseline_results])
    decomp_avg_f1 = np.mean([r['evaluation_metrics']['f1'] for r in decomposed_baseline_results])
    decomp_avg_nll = np.mean([r['evaluation_metrics']['answer_nll'] for r in decomposed_baseline_results])

    # Categorize components
    noise_comps = [(c['component_idx'], c['avg_uncertainty_change'])
                   for c in component_search_results if c['avg_uncertainty_change'] < 0]
    signal_comps = [(c['component_idx'], c['avg_uncertainty_change'])
                    for c in component_search_results if c['avg_uncertainty_change'] >= 0]

    # Sort
    noise_sorted = sorted(noise_comps, key=lambda x: x[1])
    signal_sorted = sorted(signal_comps, key=lambda x: x[1], reverse=True)

    report = []
    report.append("="*80)
    report.append(f"NOISE REMOVAL EXPERIMENT REPORT ({decomp_label})")
    report.append("="*80)
    report.append("")

    report.append("EXPERIMENT CONFIGURATION")
    report.append("-"*80)
    report.append(f"Decomposition: {decomp_label}")
    report.append(f"Layer: {layer_name}")
    report.append(f"Rank: {rank}")
    report.append(f"Samples: {config.get('num_samples', 'N/A')}")
    report.append(f"Model: {config.get('model_name', 'N/A')}")
    report.append(f"Dataset: {config.get('dataset_name', 'N/A')}")
    report.append("")

    report.append("BASELINE COMPARISON")
    report.append("-"*80)
    report.append(f"{'Metric':<20} {'Original':<15} {'Decomposed':<15} {'Change':<15}")
    report.append("-"*80)
    report.append(f"{'Uncertainty':<20} {baseline_avg:<15.4f} {decomp_avg:<15.4f} {decomp_avg - baseline_avg:+.4f}")
    report.append(f"{'F1 Score':<20} {baseline_avg_f1:<15.4f} {decomp_avg_f1:<15.4f} {decomp_avg_f1 - baseline_avg_f1:+.4f}")
    report.append(f"{'Answer NLL':<20} {baseline_avg_nll:<15.2f} {decomp_avg_nll:<15.2f} {decomp_avg_nll - baseline_avg_nll:+.2f}")
    report.append("")

    report.append("COMPONENT CATEGORIZATION")
    report.append("-"*80)
    report.append(f"Noise components:  {len(noise_comps)}/{rank} ({len(noise_comps)/rank*100:.1f}%)")
    report.append(f"Signal components: {len(signal_comps)}/{rank} ({len(signal_comps)/rank*100:.1f}%)")
    report.append("")

    report.append("TOP 10 NOISE COMPONENTS (safe to remove)")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty Change':<20}")
    report.append("-"*80)
    for i, (comp_idx, change) in enumerate(noise_sorted[:10], 1):
        report.append(f"{i:<6} {comp_idx:<12} {change:+.4f}")
    report.append("")

    report.append("TOP 10 SIGNAL COMPONENTS (important to keep)")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty Change':<20}")
    report.append("-"*80)
    for i, (comp_idx, change) in enumerate(signal_sorted[:10], 1):
        report.append(f"{i:<6} {comp_idx:<12} {change:+.4f}")
    report.append("")

    report.append("TOP-K NOISE REMOVAL RESULTS")
    report.append("="*80)

    # Compute baseline evaluation metrics (already computed above)
    report.append(f"{'K':<6} {'Uncertainty':<14} {'Unc Δ':<10} {'F1':<10} {'F1 Δ':<10} {'NLL':<10} {'NLL Δ':<10}")
    report.append("-"*80)

    if progressive_results:
        # Show key milestones
        milestones = [1, 2, 3, 5, 10, 15, 20, 25, 30]
        for pr in progressive_results:
            k = pr['num_noise_removed']
            if k in milestones or k == len(progressive_results):
                report.append(
                    f"{k:<6} "
                    f"{pr['avg_uncertainty']:<14.4f} "
                    f"{pr['avg_uncertainty_change']:+.4f}     "
                    f"{pr['avg_f1']:<10.3f} "
                    f"{pr['avg_f1_change']:+.3f}     "
                    f"{pr['avg_nll']:<10.2f} "
                    f"{pr['avg_nll_change']:+.2f}"
                )

        # Find optimal K
        best_k = min(progressive_results, key=lambda x: x['avg_uncertainty'])
        best_reduction = (baseline_avg - best_k['avg_uncertainty']) / baseline_avg * 100
        f1_retention = (best_k['avg_f1'] / baseline_avg_f1) * 100
        report.append("")
        report.append(f"OPTIMAL: K={best_k['num_noise_removed']} achieves {best_reduction:.1f}% uncertainty reduction with {f1_retention:.1f}% F1 retention")
    else:
        report.append("No noise components found for removal")

    report.append("")
    report.append("INTERPRETATION")
    report.append("-"*80)
    report.append("Noise components (Uncertainty Change < 0):")
    report.append("  - Removing them REDUCES uncertainty")
    report.append("  - Safe to compress without hurting model quality")
    report.append("")
    report.append("Signal components (Uncertainty Change > 0):")
    report.append("  - Removing them INCREASES uncertainty")
    report.append("  - Important to keep for model quality")
    report.append("")
    report.append("="*80)

    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {output_dir / 'analysis_report.txt'}")


def run_noise_removal(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_type="llama",
        dataset_name="hotpotqa",
        num_samples=10,
        target_layer=6,
        rank=40,
        num_generations=20,
        nli_model="microsoft/deberta-large-mnli",
        device="cuda",
        checkpoint_every=5,
        decomposition_type="cp",
        max_k=None,
):
    """
    Run noise removal experiment using Tucker or CP decomposition

    Phase 1: Baseline measurement
    Phase 2: Component search (individual removal)
    Phase 3: Rank components by noise impact
    Phase 4: Progressive top-k noise removal

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture type ('llama' or 'gpt2')
        dataset_name: Dataset to use (coqa, hotpotqa)
        num_samples: Number of dataset samples
        target_layer: Which layer to decompose
        rank: Rank for decomposition
        num_generations: Responses per sample
        nli_model: NLI model for semantic similarity
        device: Device to run on
        checkpoint_every: Save results every N components
        decomposition_type: Type of decomposition ('tucker' or 'cp')
        max_k: Maximum number of noise components to remove (None = all noise components)
    """

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract short model name (e.g., "Llama-3.1-8B-Instruct" from "meta-llama/Llama-3.1-8B-Instruct")
    short_model_name = model_name.split("/")[-1]
    output_dir = Path(f"results/noise_removal_{decomposition_type}/{short_model_name}_{dataset_name}_layer{target_layer}_rank{rank}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_name = f"Layer {target_layer}"

    print("="*80)
    print(f"Noise Removal Experiment ({decomposition_type.upper()} Decomposition)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Model type: {model_type}")
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
    # Uses Mistral-7B-Instruct by default (best for fact extraction)
    print("\nLoading knowledge extractor (Mistral-7B-Instruct)...")
    knowledge_extractor = KnowledgeExtractor(device=device)
    print("Knowledge extractor loaded")

    # ========== Decompose Target Layer ==========
    print(f"\nDecomposing layer {target_layer} with {decomposition_type.upper()} rank {rank}...")
    fc_in_weight, fc_out_weight = get_fc_layer_weights(model, target_layer, model_type=model_type)

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

    # ========== PHASE 1: Baseline Measurement ==========
    print("\n" + "="*80)
    print("PHASE 1: BASELINE MEASUREMENT (No Component Removal)")
    print("="*80)

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

        # Generate greedy response for quality evaluation (short answer prompt)
        greedy_response = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=format_short_answer_prompt(question),
            num_generations=1,
            max_new_tokens=50,  # Shorter since we expect short answers
            temperature=0.0,
            do_sample=False
        )[0]

        # Measure uncertainty
        metrics = measure_uncertainty_and_blockiness(
            responses=responses,
            question=question,
            nli_calculator=nli_calculator,
            knowledge_extractor=knowledge_extractor,
            decomposition_type=decomposition_type,
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

    # ========== PHASE 1.5: Decomposed Baseline (All Components, No Removal) ==========
    print("\n" + "="*80)
    print("PHASE 1.5: DECOMPOSED BASELINE (Reconstruct with ALL components)")
    print("="*80)

    # Reconstruct weights with all components (no removal)
    if decomposition_type == "cp":
        fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights_cp(decomp_weights, decomp_factors)
    else:  # tucker
        fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights(decomp_weights, decomp_factors)

    # Update model with reconstructed weights
    update_fc_layer_weights(
        model,
        target_layer,
        fc_in_reconstructed,
        fc_out_reconstructed,
        model_type=model_type
    )

    decomposed_baseline_results = []

    for idx, sample in enumerate(tqdm(samples, desc="Decomposed Baseline")):
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

        # Generate greedy response for quality evaluation (short answer prompt)
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
            decomposition_type=decomposition_type,
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

        # Compute changes from original baseline
        orig_baseline = baseline_results[idx]
        uncertainty_change = metrics['uncertainty_score'] - orig_baseline['uncertainty_score']
        f1_change = eval_metrics['f1'] - orig_baseline['evaluation_metrics']['f1']
        nll_change = eval_metrics['answer_nll'] - orig_baseline['evaluation_metrics']['answer_nll']

        decomposed_baseline_results.append({
            'sample_id': idx,
            'question': question,
            'answer': gold_answer,
            'responses': responses,
            'greedy_response': greedy_response,
            'uncertainty_change_from_original': uncertainty_change,
            'f1_change_from_original': f1_change,
            'nll_change_from_original': nll_change,
            **metrics,
            'evaluation_metrics': eval_metrics
        })

    # Save decomposed baseline
    with open(output_dir / "decomposed_baseline.pkl", 'wb') as f:
        pickle.dump(decomposed_baseline_results, f)

    # Print summary
    avg_unc_decomposed = np.mean([r['uncertainty_score'] for r in decomposed_baseline_results])
    avg_f1_decomposed = np.mean([r['evaluation_metrics']['f1'] for r in decomposed_baseline_results])
    avg_nll_decomposed = np.mean([r['evaluation_metrics']['answer_nll'] for r in decomposed_baseline_results])
    avg_unc_original = np.mean([r['uncertainty_score'] for r in baseline_results])

    print(f"\nDecomposed baseline saved to {output_dir / 'decomposed_baseline.pkl'}")
    print(f"Original baseline avg uncertainty: {avg_unc_original:.4f}")
    print(f"Decomposed baseline avg uncertainty: {avg_unc_decomposed:.4f} (change: {avg_unc_decomposed - avg_unc_original:+.4f})")
    print(f"Decomposed baseline avg F1: {avg_f1_decomposed:.4f}")
    print(f"Decomposed baseline avg answer NLL: {avg_nll_decomposed:.2f}")

    # Restore original weights before component search
    update_fc_layer_weights(
        model,
        target_layer,
        original_fc_in,
        original_fc_out,
        model_type=model_type
    )

    # ========== PHASE 2: Component Search (Individual Removal) ==========
    print("\n" + "="*80)
    print(f"PHASE 2: COMPONENT SEARCH ({rank} components)")
    print("="*80)

    component_search_results = []

    for component_idx in range(rank):
        print(f"\n--- Component {component_idx + 1}/{rank} ---")

        if decomposition_type == "cp":
            # Remove single component from CP (zeros the weight)
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
            model_type=model_type
        )

        # Measure on all samples
        component_results = []

        for idx, sample in enumerate(tqdm(samples, desc=f"Component {component_idx}", leave=False)):
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

            # Generate greedy response for quality evaluation (short answer prompt)
            greedy_response = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt=format_short_answer_prompt(question),
                num_generations=1,
                max_new_tokens=50,  # Shorter since we expect short answers
                temperature=0.0,
                do_sample=False
            )[0]

            # Measure uncertainty
            metrics = measure_uncertainty_and_blockiness(
                responses=responses,
                question=question,
                nli_calculator=nli_calculator,
                knowledge_extractor=knowledge_extractor,
                decomposition_type=decomposition_type,
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

            component_results.append({
                'sample_id': idx,
                'component_removed': component_idx,
                'uncertainty_score': metrics['uncertainty_score'],
                'uncertainty_change': uncertainty_change,
                'f1': eval_metrics['f1'],
                'f1_change': f1_change,
                'answer_nll': eval_metrics['answer_nll'],
                'nll_change': nll_change,
                'responses': responses,  # Save responses for debugging
                'greedy_response': greedy_response,  # Save greedy response
                'knowledge_stats': metrics.get('knowledge_stats', {}),  # Degenerate/invalid stats
                'metrics': metrics,
                'evaluation_metrics': eval_metrics
            })

        # Compute average uncertainty change for this component
        avg_uncertainty_change = np.mean([r['uncertainty_change'] for r in component_results])

        # Compute degenerate/invalid response stats for this component
        total_degenerate = sum(r.get('knowledge_stats', {}).get('degenerate_count', 0) for r in component_results)
        total_no_facts = sum(r.get('knowledge_stats', {}).get('no_facts_count', 0) for r in component_results)
        total_responses = sum(r.get('knowledge_stats', {}).get('total_count', 0) for r in component_results)
        invalid_rate = (total_degenerate + total_no_facts) / total_responses if total_responses > 0 else 0

        component_search_results.append({
            'component_idx': component_idx,
            'avg_uncertainty_change': avg_uncertainty_change,
            'degenerate_count': total_degenerate,
            'no_facts_count': total_no_facts,
            'total_responses': total_responses,
            'invalid_rate': invalid_rate,
            'results': component_results
        })

        print(f"Component {component_idx}: avg uncertainty change = {avg_uncertainty_change:.4f}, "
              f"invalid rate = {invalid_rate:.1%} ({total_degenerate} degenerate, {total_no_facts} no_facts)")

        # Checkpoint
        if (component_idx + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"phase2_checkpoint_{component_idx + 1}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(component_search_results, f)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Reload original weights for next component
        update_fc_layer_weights(
            model,
            target_layer,
            original_fc_in,
            original_fc_out,
            model_type=model_type
        )

        # Clear memory
        torch.cuda.empty_cache()

    # Save component search results
    with open(output_dir / "component_search.pkl", 'wb') as f:
        pickle.dump(component_search_results, f)
    print(f"\nSaved component search to {output_dir / 'component_search.pkl'}")

    # ========== Generate Analysis Figures (Component Search) ==========
    print("\n" + "="*80)
    print("GENERATING COMPONENT SEARCH FIGURES")
    print("="*80)

    generate_analysis_figures(
        component_search_results,
        baseline_results,
        output_dir,
        decomposition_type,
        layer_name,
        rank
    )

    # ========== PHASE 3: Rank Components by Noise (Uncertainty Impact) ==========
    print("\n" + "="*80)
    print("PHASE 3: RANKING COMPONENTS BY UNCERTAINTY IMPACT")
    print("="*80)

    # Noise components: removing them DECREASES uncertainty (negative change)
    # Signal components: removing them INCREASES uncertainty (positive change)
    # Sort by uncertainty change (most negative first = most noise)
    component_rankings = sorted(
        component_search_results,
        key=lambda x: x['avg_uncertainty_change']
    )

    # Only keep components with NEGATIVE uncertainty change (actual noise)
    noise_components = [c for c in component_rankings if c['avg_uncertainty_change'] < 0]
    signal_components = [c for c in component_rankings if c['avg_uncertainty_change'] >= 0]

    noise_component_order = [c['component_idx'] for c in noise_components]
    num_noise_components = len(noise_component_order)

    print(f"\nFound {num_noise_components} noise components (negative uncertainty change)")
    print(f"Found {len(signal_components)} signal components (positive/zero uncertainty change)")
    print("\nTop noise components:")
    for i, c in enumerate(noise_components[:10]):  # Show top 10
        print(f"  {i+1}. Component {c['component_idx']}: avg change = {c['avg_uncertainty_change']:.4f}")
    if len(noise_components) > 10:
        print("  ...")

    # Save rankings
    rankings_info = {
        'noise_component_order': noise_component_order,
        'num_noise_components': num_noise_components,
        'num_signal_components': len(signal_components),
        'component_rankings': [(c['component_idx'], c['avg_uncertainty_change']) for c in component_rankings],
        'noise_components': [(c['component_idx'], c['avg_uncertainty_change']) for c in noise_components],
        'signal_components': [(c['component_idx'], c['avg_uncertainty_change']) for c in signal_components]
    }
    with open(output_dir / "component_rankings.pkl", 'wb') as f:
        pickle.dump(rankings_info, f)

    # ========== PHASE 4: Progressive Noise Removal ==========
    # Determine how many noise components to remove
    if max_k is not None:
        num_to_remove = min(max_k, num_noise_components)
    else:
        num_to_remove = num_noise_components

    print("\n" + "="*80)
    print(f"PHASE 4: PROGRESSIVE NOISE REMOVAL (1 to {num_to_remove} noise components)")
    print("="*80)

    if num_noise_components == 0:
        print("No noise components found! Skipping Phase 4.")
        progressive_results = []
    else:
        progressive_results = []

        for num_noise_to_remove in range(1, num_to_remove + 1):
            print(f"\n--- Removing top {num_noise_to_remove} noise components ---")

            # Get components to remove
            components_to_remove = noise_component_order[:num_noise_to_remove]

            if decomposition_type == "cp":
                # Create modified weights by zeroing out all noise components
                modified_weights = decomp_weights.clone()
                for comp_idx in components_to_remove:
                    modified_weights[comp_idx] = 0.0
                # Reconstruct weights (signal-only reconstruction)
                fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights_cp(modified_weights, decomp_factors)
            else:  # tucker
                # For Tucker, we need to zero out multiple components
                modified_core = decomp_weights.clone()
                for comp_idx in components_to_remove:
                    # Zero out the slice for this component
                    modified_core[:, comp_idx, :] = 0.0
                # Reconstruct weights
                fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights(modified_core, decomp_factors)

            # Update model with signal-only weights
            update_fc_layer_weights(
                model,
                target_layer,
                fc_in_reconstructed,
                fc_out_reconstructed,
                model_type=model_type
            )

            # Measure on all samples
            removal_results = []

            for idx, sample in enumerate(tqdm(samples, desc=f"Remove top {num_noise_to_remove}", leave=False)):
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

                # Generate greedy response for quality evaluation (short answer prompt)
                greedy_response = generate_responses(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=format_short_answer_prompt(question),
                    num_generations=1,
                    max_new_tokens=50,  # Shorter since we expect short answers
                    temperature=0.0,
                    do_sample=False
                )[0]

                # Measure uncertainty
                metrics = measure_uncertainty_and_blockiness(
                    responses=responses,
                    question=question,
                    nli_calculator=nli_calculator,
                    knowledge_extractor=knowledge_extractor,
                    decomposition_type=decomposition_type,
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

                removal_results.append({
                    'sample_id': idx,
                    'num_noise_removed': num_noise_to_remove,
                    'components_removed': components_to_remove.copy(),
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

            # Compute average uncertainty for this removal level
            avg_uncertainty = np.mean([r['uncertainty_score'] for r in removal_results])
            avg_uncertainty_change = np.mean([r['uncertainty_change'] for r in removal_results])
            avg_f1 = np.mean([r['f1'] for r in removal_results])
            avg_f1_change = np.mean([r['f1_change'] for r in removal_results])
            avg_nll = np.mean([r['answer_nll'] for r in removal_results])
            avg_nll_change = np.mean([r['nll_change'] for r in removal_results])

            progressive_results.append({
                'num_noise_removed': num_noise_to_remove,
                'components_removed': components_to_remove.copy(),
                'avg_uncertainty': avg_uncertainty,
                'avg_uncertainty_change': avg_uncertainty_change,
                'avg_f1': avg_f1,
                'avg_f1_change': avg_f1_change,
                'avg_nll': avg_nll,
                'avg_nll_change': avg_nll_change,
                'results': removal_results
            })

            print(f"Top {num_noise_to_remove} noise removed: uncertainty={avg_uncertainty:.4f} ({avg_uncertainty_change:+.4f}), F1={avg_f1:.3f} ({avg_f1_change:+.3f}), NLL={avg_nll:.2f} ({avg_nll_change:+.2f})")

            # Checkpoint
            if num_noise_to_remove % checkpoint_every == 0:
                checkpoint_path = output_dir / f"phase4_checkpoint_{num_noise_to_remove}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(progressive_results, f)
                print(f"Checkpoint saved: {checkpoint_path}")

            # Reload original weights for next iteration
            update_fc_layer_weights(
                model,
                target_layer,
                original_fc_in,
                original_fc_out,
                model_type=model_type
            )

            # Clear memory
            torch.cuda.empty_cache()

    # ========== Generate Top-K Removal Figures ==========
    print("\n" + "="*80)
    print("GENERATING TOP-K REMOVAL FIGURES")
    print("="*80)

    baseline_avg = np.mean([r['uncertainty_score'] for r in baseline_results])
    generate_topk_removal_figures(
        progressive_results,
        baseline_avg,
        baseline_results,
        decomposed_baseline_results,
        output_dir,
        decomposition_type,
        layer_name
    )

    # ========== Generate Report ==========
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)

    config = {
        'model_name': model_name,
        'model_type': model_type,
        'dataset_name': dataset_name,
        'num_samples': num_samples,
        'target_layer': target_layer,
        'rank': rank,
        'num_generations': num_generations,
        'nli_model': nli_model,
        'timestamp': timestamp,
        'decomposition_type': decomposition_type,
        'experiment_type': 'noise_removal'
    }

    generate_report(
        component_search_results,
        rankings_info,
        progressive_results,
        baseline_results,
        decomposed_baseline_results,
        config,
        output_dir,
        decomposition_type,
        layer_name
    )

    # ========== Save Final Results ==========
    final_results = {
        'config': config,
        'baseline': baseline_results,
        'component_search': component_search_results,
        'component_rankings': rankings_info,
        'progressive_removal': progressive_results,
        'decomposed_baseline': decomposed_baseline_results
    }

    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    # ========== Summary ==========
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Total components: {rank}")
    print(f"Noise components: {num_noise_components}")
    print(f"Signal components: {len(signal_components)}")
    print(f"Total samples: {num_samples}")
    print("\nFiles created:")
    print("  - results.pkl (all data)")
    print("  - baseline.pkl")
    print("  - decomposed_baseline.pkl")
    print("  - component_search.pkl")
    print("  - component_rankings.pkl")
    print("  - uncertainty_change_by_component.png")
    print("  - uncertainty_vs_blockiness.png")
    print("  - change_distributions.png")
    print("  - topk_uncertainty_reduction.png")
    print("  - topk_uncertainty_change.png")
    print("  - topk_percentage_reduction.png")
    print("  - topk_f1_score.png")
    print("  - topk_nll.png")
    print("  - topk_uncertainty_f1_tradeoff.png")
    print("  - analysis_report.txt")
    print("="*80)

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Noise Removal Experiment')
    parser.add_argument('--layer', type=int, default=30, help='Target layer (default: 30)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples (default: 10)')
    parser.add_argument('--rank', type=int, default=40, help='Decomposition rank (default: 40)')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['coqa', 'hotpotqa', 'nq_open'],
                        help='Dataset to use (default: hotpotqa)')
    parser.add_argument('--decomposition', type=str, default='cp', choices=['tucker', 'cp'],
                        help='Decomposition type (default: cp)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='HuggingFace model name (default: meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--model-type', type=str, default='llama', choices=['llama', 'gpt2'],
                        help='Model architecture type (default: llama). Use gpt2 for GPT-2/GPT-J')
    parser.add_argument('--max-k', type=int, default=None,
                        help='Maximum number of noise components to remove (default: all)')
    parser.add_argument('--test', action='store_true', help='Run quick test (1 sample, 5 components)')

    args = parser.parse_args()

    if args.test:
        # Quick test
        print("Running QUICK TEST mode...")
        run_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            dataset_name=args.dataset,
            num_samples=1,
            target_layer=args.layer,
            rank=5,
            checkpoint_every=1,
            decomposition_type=args.decomposition,
            max_k=args.max_k
        )
    else:
        # Full run
        print(f"Running FULL experiment on layer {args.layer}...")
        run_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            dataset_name=args.dataset,
            num_samples=args.samples,
            target_layer=args.layer,
            rank=args.rank,
            checkpoint_every=5,
            decomposition_type=args.decomposition,
            max_k=args.max_k
        )
