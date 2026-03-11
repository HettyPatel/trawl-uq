"""
Plot Open-Ended QA SVD Truncation Results

Creates visualizations comparing generation quality across compression levels.

Usage:
    python scripts/plot_open_qa_svd.py --results results/open_qa_svd/.../results.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def plot_open_qa_svd_results(results_data, output_dir):
    """Generate plots for open QA SVD truncation results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = results_data['all_results']
    config = results_data['config']
    model_name = config['model_name'].split('/')[-1]

    # Extract data
    reduction_pcts = []
    k_values = []
    accuracies = []
    energy_retentions = []

    baseline = all_results['baseline']
    baseline_acc = baseline['accuracy']

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    for key, result in items_sorted:
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        reduction_pcts.append(red_pct)
        k_values.append(result['k'])
        accuracies.append(result['accuracy'] * 100)
        energy_retentions.append(result['energy_retention'] * 100)

    reduction_pcts = np.array(reduction_pcts)
    k_values = np.array(k_values)
    accuracies = np.array(accuracies)
    energy_retentions = np.array(energy_retentions)

    # ========== Figure 1: Accuracy vs Reduction % ==========
    plt.figure(figsize=(12, 6))
    plt.plot(reduction_pcts, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.axhline(y=baseline_acc*100, color='red', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)

    plt.xlabel('Reduction %', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'Open-Ended QA Accuracy vs SVD Compression\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Annotate key points
    for i, (red, acc) in enumerate([(10, accuracies[0]), (75, accuracies[6]), (90, accuracies[7])]):
        if i < len(reduction_pcts) and reduction_pcts[i] == red:
            plt.annotate(f'{acc:.1f}%',
                        xy=(red, acc),
                        xytext=(red, acc+1.5),
                        fontsize=10,
                        ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_reduction.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_vs_reduction.png'}")
    plt.close()

    # ========== Figure 2: Accuracy vs k (components) ==========
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8, color='#A23B72')
    plt.axhline(y=baseline_acc*100, color='red', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)

    plt.xlabel('k (number of components kept)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'Open-Ended QA Accuracy vs SVD Components\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_k.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_vs_k.png'}")
    plt.close()

    # ========== Figure 3: Energy Retention vs Reduction % ==========
    plt.figure(figsize=(12, 6))
    plt.plot(reduction_pcts, energy_retentions, 'o-', linewidth=2, markersize=8, color='#F18F01')

    plt.xlabel('Reduction %', fontsize=13)
    plt.ylabel('Energy Retention (%)', fontsize=13)
    plt.title(f'Singular Value Energy Retention\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_retention.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'energy_retention.png'}")
    plt.close()

    # ========== Figure 4: Accuracy vs Energy Retention ==========
    plt.figure(figsize=(12, 6))
    plt.scatter(energy_retentions, accuracies, s=100, alpha=0.6, c=reduction_pcts, cmap='RdYlGn_r')
    plt.colorbar(label='Reduction %')

    # Add baseline point
    plt.scatter([100], [baseline_acc*100], s=200, marker='*', color='red',
                edgecolors='black', linewidths=2, label='Baseline', zorder=10)

    plt.xlabel('Energy Retention (%)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'Accuracy vs Energy Retention Trade-off\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_energy.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_vs_energy.png'}")
    plt.close()

    # ========== Figure 5: Change Analysis ==========
    baseline_results = baseline['results']

    improvements = []
    degradations = []

    for key, result in items_sorted:
        compressed_results = result['results']

        improved = 0
        degraded = 0

        for b, c in zip(baseline_results, compressed_results):
            if not b['is_correct'] and c['is_correct']:
                improved += 1
            elif b['is_correct'] and not c['is_correct']:
                degraded += 1

        improvements.append(improved)
        degradations.append(degraded)

    improvements = np.array(improvements)
    degradations = np.array(degradations)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(reduction_pcts))
    width = 0.35

    plt.bar(x - width/2, improvements, width, label='Improved', color='#06A77D', alpha=0.8)
    plt.bar(x + width/2, degradations, width, label='Degraded', color='#D62246', alpha=0.8)

    plt.xlabel('Reduction %', fontsize=13)
    plt.ylabel('Number of Questions', fontsize=13)
    plt.title(f'Answer Changes by Compression Level\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.xticks(x, [f'{r:.1f}%' for r in reduction_pcts], rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'answer_changes.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'answer_changes.png'}")
    plt.close()

    # ========== Figure 6: Net Change ==========
    net_change = improvements - degradations

    plt.figure(figsize=(12, 6))
    colors = ['#06A77D' if x >= 0 else '#D62246' for x in net_change]
    plt.bar(reduction_pcts, net_change, width=3, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    plt.xlabel('Reduction %', fontsize=13)
    plt.ylabel('Net Change (Improved - Degraded)', fontsize=13)
    plt.title(f'Net Answer Quality Change\n{model_name} - Layer {config["target_layer"]} {config["matrix_type"]}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'net_change.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'net_change.png'}")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot open-ended QA SVD truncation results"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots (default: same as results)")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results_data = load_results(args.results)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        results_path = Path(args.results)
        output_dir = results_path.parent / 'figures'

    print(f"Output directory: {output_dir}")

    # Generate plots
    plot_open_qa_svd_results(results_data, output_dir)


if __name__ == "__main__":
    main()
