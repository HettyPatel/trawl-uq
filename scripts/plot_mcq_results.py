"""
Plot MCQ entropy experiment results.

Focuses on Accuracy and Entropy (not NLL) since with 4 options:
- Accuracy is most interpretable
- Entropy measures confidence
- NLL is redundant with both

Usage:
    python scripts/plot_mcq_results.py --results results/mcq_entropy_noise_removal/.../results.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path: str):
    """Load results from pickle file."""
    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def detect_experiment_type(results: dict) -> str:
    """Detect whether this is noise_removal or svd_truncation."""
    if 'component_results' in results:
        return 'noise_removal'
    elif 'truncation_results' in results:
        return 'svd_truncation'
    else:
        raise ValueError("Unknown experiment type")


def plot_noise_removal(results: dict, output_dir: Path):
    """Plot MCQ noise removal experiment results."""
    output_dir = Path(output_dir) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']
    baseline = results['baseline']
    component_results = results['component_results']
    rankings = results['rankings']

    model_name = config['model_name'].split('/')[-1]
    layer = config['target_layer']
    decomp = config['decomposition_type'].upper()

    baseline_entropy = baseline['avg_entropy']
    baseline_accuracy = baseline['accuracy']

    # Extract data
    components = [c['component_idx'] for c in component_results]
    entropies = [c['avg_entropy'] for c in component_results]
    entropy_changes = [c['entropy_change'] for c in component_results]
    accuracies = [c['accuracy'] for c in component_results]
    # Use stored accuracy_change if available, otherwise compute it
    accuracy_changes = [c.get('accuracy_change', c['accuracy'] - baseline_accuracy) for c in component_results]

    # Plot 1: Entropy Change by Component
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['green' if ec < 0 else 'orange' for ec in entropy_changes]
    ax.bar(components, entropy_changes, color=colors, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Entropy Change', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy Change by Component\n(Green = Lower entropy, Orange = Higher entropy)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_change_by_component.png'}")
    plt.close()

    # Plot 2: Accuracy Change by Component
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['red' if ac < 0 else 'green' for ac in accuracy_changes]
    ax.bar(components, [ac * 100 for ac in accuracy_changes], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Accuracy Change by Component\n(Green = Improved, Red = Degraded)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_change_by_component.png'}")
    plt.close()

    # Plot 3: Entropy vs Accuracy Scatter (key plot!)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by classification
    true_noise = set(rankings.get('true_noise', []))
    confident_wrong = set(rankings.get('confident_wrong', []))
    true_signal = set(rankings.get('true_signal', []))
    uncertain_right = set(rankings.get('uncertain_right', []))

    for c in component_results:
        idx = c['component_idx']
        ec = c['entropy_change']
        ac = (c['accuracy'] - baseline_accuracy) * 100

        if idx in true_noise:
            color, label = 'green', 'True Noise (E↓ Acc↑)'
        elif idx in confident_wrong:
            color, label = 'red', 'Confident Wrong (E↓ Acc↓)'
        elif idx in true_signal:
            color, label = 'blue', 'True Signal (E↑ Acc↓)'
        else:
            color, label = 'purple', 'Uncertain Right (E↑ Acc↑)'

        ax.scatter(ec, ac, c=color, alpha=0.6, s=50)

    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Add quadrant labels
    ax.text(0.02, 0.98, 'Uncertain Right\n(E↑ Acc↑)', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='purple', fontweight='bold')
    ax.text(0.98, 0.98, 'True Signal\n(E↑ Acc↓)', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color='blue', fontweight='bold')
    ax.text(0.02, 0.02, 'True Noise\n(E↓ Acc↑)', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', color='green', fontweight='bold')
    ax.text(0.98, 0.02, 'Confident Wrong\n(E↓ Acc↓)', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', color='red', fontweight='bold')

    ax.set_xlabel('Entropy Change (when component removed)', fontsize=12)
    ax.set_ylabel('Accuracy Change % (when component removed)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy vs Accuracy\nBaseline: Entropy={baseline_entropy:.3f}, Accuracy={baseline_accuracy*100:.1f}%', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_vs_accuracy_scatter.png'}")
    plt.close()

    # Plot 4: Classification Summary (pie chart)
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = ['True Noise\n(E↓ Acc↑)', 'Confident Wrong\n(E↓ Acc↓)',
              'True Signal\n(E↑ Acc↓)', 'Uncertain Right\n(E↑ Acc↑)']
    sizes = [len(true_noise), len(confident_wrong), len(true_signal), len(uncertain_right)]
    colors_pie = ['green', 'red', 'blue', 'purple']

    # Only show non-zero slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels_nz, sizes_nz, colors_nz = zip(*non_zero)
        ax.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Component Classification\n({len(component_results)} total components)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_pie.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'classification_pie.png'}")
    plt.close()

    # Plot 5: Combined Entropy and Accuracy
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Sort by entropy change for better visualization
    sorted_indices = np.argsort(entropy_changes)
    sorted_components = [components[i] for i in sorted_indices]
    sorted_entropy_changes = [entropy_changes[i] for i in sorted_indices]
    sorted_accuracy_changes = [accuracy_changes[i] * 100 for i in sorted_indices]

    x = range(len(sorted_components))

    ax1.set_xlabel('Components (sorted by entropy change)', fontsize=12)
    ax1.set_ylabel('Entropy Change', fontsize=12, color='blue')
    ax1.plot(x, sorted_entropy_changes, 'b-', linewidth=1.5, label='Entropy Change')
    ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy Change (%)', fontsize=12, color='red')
    ax2.plot(x, sorted_accuracy_changes, 'r-', linewidth=1.5, label='Accuracy Change')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy vs Accuracy Changes', fontsize=14)
    ax1.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_accuracy_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_accuracy_combined.png'}")
    plt.close()

    print(f"\nMCQ noise removal plots saved to: {output_dir}")


def plot_svd_truncation(results: dict, output_dir: Path):
    """Plot MCQ SVD truncation experiment results."""
    output_dir = Path(output_dir) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']
    baseline = results['baseline']
    truncation_results = results['truncation_results']

    model_name = config['model_name'].split('/')[-1]
    layer = config['target_layer']
    matrix = config['matrix_type']

    baseline_entropy = baseline['avg_entropy']
    baseline_accuracy = baseline['accuracy']

    # Extract data
    reductions = [r['reduction_percent'] for r in truncation_results]
    entropies = [r['avg_entropy'] for r in truncation_results]
    accuracies = [r['accuracy'] * 100 for r in truncation_results]
    energy_retentions = [r['energy_retention'] * 100 for r in truncation_results]

    # Plot 1: Entropy vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(y=baseline_entropy, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_entropy:.4f})')
    ax.plot(reductions, entropies, 'b-o', linewidth=2, markersize=6, label='After Truncation')

    # Shade improved region
    ax.fill_between(reductions, entropies, baseline_entropy,
                    where=[e < baseline_entropy for e in entropies],
                    alpha=0.3, color='green', label='Lower Entropy')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average Entropy (max=1.386)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Entropy vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_vs_reduction.png'}")
    plt.close()

    # Plot 2: Accuracy vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhline(y=baseline_accuracy * 100, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_accuracy*100:.1f}%)')
    ax.plot(reductions, accuracies, 'g-o', linewidth=2, markersize=6, label='After Truncation')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Accuracy vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_vs_reduction.png'}")
    plt.close()

    # Plot 3: Entropy vs Accuracy Tradeoff (dual axis)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12, color='blue')
    line1 = ax1.plot(reductions, entropies, 'b-o', linewidth=2, markersize=6, label='Entropy')
    ax1.axhline(y=baseline_entropy, color='blue', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color='green')
    line2 = ax2.plot(reductions, accuracies, 'g-s', linewidth=2, markersize=6, label='Accuracy')
    ax2.axhline(y=baseline_accuracy * 100, color='green', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 100)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)

    ax1.set_title(f'{model_name} Layer {layer} {matrix}: Entropy vs Accuracy Tradeoff', fontsize=14)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_accuracy_tradeoff.png'}")
    plt.close()

    # Plot 4: Energy Retention
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(reductions, energy_retentions, 'c-o', linewidth=2, markersize=6)
    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Energy Retention (%)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Energy Retention vs SVD Reduction', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_retention.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'energy_retention.png'}")
    plt.close()

    # Plot 5: All metrics normalized
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to 0-1 scale
    entropy_norm = [(e - min(entropies)) / (max(entropies) - min(entropies) + 1e-10) for e in entropies]
    accuracy_norm = [a / 100 for a in accuracies]
    energy_norm = [e / 100 for e in energy_retentions]

    ax.plot(reductions, entropy_norm, 'b-o', linewidth=2, markersize=5, label='Entropy (normalized)')
    ax.plot(reductions, accuracy_norm, 'g-s', linewidth=2, markersize=5, label='Accuracy')
    ax.plot(reductions, energy_norm, 'c-^', linewidth=2, markersize=5, label='Energy Retention')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: All Metrics vs Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_metrics_normalized.png'}")
    plt.close()

    print(f"\nMCQ SVD truncation plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot MCQ entropy experiment results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as results)")

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output) if args.output else results_path.parent

    results = load_results(results_path)
    exp_type = detect_experiment_type(results)
    print(f"Detected experiment type: {exp_type}")

    if exp_type == 'noise_removal':
        plot_noise_removal(results, output_dir)
    else:
        plot_svd_truncation(results, output_dir)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
