"""
Plotting script for entropy-based experiments.

Generates figures from results.pkl files produced by:
- 05_entropy_noise_removal.py
- 06_entropy_svd_truncation.py

Usage:
    python scripts/plot_entropy_results.py --results path/to/results.pkl
    python scripts/plot_entropy_results.py --results path/to/results.pkl --output figures/
"""

import sys
sys.path.append('.')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_results(filepath: str):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_noise_removal_results(results, output_dir):
    """
    Generate plots for noise removal experiment (CP/Tucker).

    Plots:
    1. Entropy change by component
    2. NLL change by component
    3. Entropy vs NLL scatter
    4. Component ranking distribution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']
    baseline = results['baseline']
    component_results = results['component_results']

    layer = config['target_layer']
    decomp = config['decomposition_type'].upper()
    model_name = config['model_name'].split('/')[-1]

    # Extract data
    indices = [c['component_idx'] for c in component_results]
    entropy_changes = [c['entropy_change'] for c in component_results]
    nll_changes = [c['nll_change'] for c in component_results]

    # Plot 1: Entropy Change by Component
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['green' if e < 0 else 'orange' for e in entropy_changes]
    ax.bar(indices, entropy_changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Entropy Change', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy Change by Component\n(Green = Noise, Orange = Signal)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_change_by_component.png'}")
    plt.close()

    # Plot 2: NLL Change by Component
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['green' if n < 0 else 'red' for n in nll_changes]
    ax.bar(indices, nll_changes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('NLL Change', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): NLL Change by Component\n(Green = Improved, Red = Degraded)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'nll_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nll_change_by_component.png'}")
    plt.close()

    # Plot 3: Entropy vs NLL Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(entropy_changes, nll_changes, c=indices, cmap='viridis', s=80, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Component Index')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Entropy Change', fontsize=12)
    ax.set_ylabel('NLL Change', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy vs NLL Change', fontsize=14)

    # Annotate quadrants
    ax.text(0.02, 0.98, 'Lower entropy\nHigher NLL\n(Confident but wrong)', transform=ax.transAxes,
            fontsize=9, va='top', ha='left', color='gray')
    ax.text(0.98, 0.98, 'Higher entropy\nHigher NLL\n(Uncertain and wrong)', transform=ax.transAxes,
            fontsize=9, va='top', ha='right', color='gray')
    ax.text(0.02, 0.02, 'Lower entropy\nLower NLL\n(Confident and correct)', transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left', color='green')
    ax.text(0.98, 0.02, 'Higher entropy\nLower NLL\n(Uncertain but correct)', transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right', color='gray')

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_nll_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_vs_nll_scatter.png'}")
    plt.close()

    # Plot 4: Distribution of Changes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(entropy_changes, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Entropy Change', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Entropy Changes', fontsize=14)
    n_noise = sum(1 for e in entropy_changes if e < 0)
    axes[0].text(0.95, 0.95, f'Noise: {n_noise}\nSignal: {len(entropy_changes) - n_noise}',
                 transform=axes[0].transAxes, fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1].hist(nll_changes, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('NLL Change', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of NLL Changes', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'change_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'change_distributions.png'}")
    plt.close()

    # Plot 5: Combined entropy and NLL on same plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel('Component Index', fontsize=12)
    ax1.set_ylabel('Entropy Change', fontsize=12, color='blue')
    ax1.bar(indices, entropy_changes, alpha=0.5, color='blue', label='Entropy Change')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('NLL Change', fontsize=12, color='red')
    ax2.plot(indices, nll_changes, 'r.-', linewidth=1, markersize=4, label='NLL Change')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    ax1.set_title(f'{model_name} Layer {layer} ({decomp}): Entropy and NLL Change by Component', fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_nll_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_nll_combined.png'}")
    plt.close()

    print(f"\nNoise removal plots saved to: {output_dir}")


def plot_svd_truncation_results(results, output_dir):
    """
    Generate plots for SVD truncation experiment.

    Plots:
    1. Entropy vs reduction %
    2. NLL vs reduction %
    3. Energy retention vs reduction %
    4. Entropy vs NLL tradeoff
    5. Percentage change from baseline
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']
    baseline = results['baseline']
    truncation_results = results['truncation_results']

    layer = config['target_layer']
    matrix = config['matrix_type']
    model_name = config['model_name'].split('/')[-1]

    # Extract data
    reductions = [r['reduction_percent'] for r in truncation_results]
    entropies = [r['avg_entropy'] for r in truncation_results]
    nlls = [r['avg_nll'] for r in truncation_results]
    energy_retentions = [r['energy_retention'] * 100 for r in truncation_results]

    baseline_entropy = baseline['avg_entropy']
    baseline_nll = baseline['avg_nll']

    # Plot 1: Entropy vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_entropy, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_entropy:.4f})')
    ax.plot(reductions, entropies, 'b-o', linewidth=2, markersize=6, label='After Truncation')

    # Shade improvement region
    ax.fill_between(reductions, baseline_entropy, entropies,
                    where=[e < baseline_entropy for e in entropies],
                    alpha=0.3, color='green', label='Improved')

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average Entropy', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Entropy vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # ax.invert_xaxis()  # Removed: left-to-right is more intuitive
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_vs_reduction.png'}")
    plt.close()

    # Plot 2: NLL vs Reduction %
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=baseline_nll, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Baseline ({baseline_nll:.4f})')
    ax.plot(reductions, nlls, 'g-o', linewidth=2, markersize=6, label='After Truncation')
    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Average NLL', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: NLL vs SVD Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # ax.invert_xaxis()  # Removed: left-to-right is more intuitive
    plt.tight_layout()
    plt.savefig(output_dir / 'nll_vs_reduction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'nll_vs_reduction.png'}")
    plt.close()

    # Plot 3: Energy Retention
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(reductions, energy_retentions, 'c-o', linewidth=2, markersize=6)
    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Energy Retention (%)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Energy Retention vs SVD Reduction', fontsize=14)
    ax.grid(alpha=0.3)
    # ax.invert_xaxis()  # Removed: left-to-right is more intuitive
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_retention.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'energy_retention.png'}")
    plt.close()

    # Plot 4: Entropy vs NLL Tradeoff (dual axis)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12, color='blue')
    line1 = ax1.plot(reductions, entropies, 'b-o', linewidth=2, markersize=6, label='Entropy')
    ax1.axhline(y=baseline_entropy, color='blue', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='blue')
    # ax1.invert_xaxis()  # Removed: left-to-right is more intuitive

    ax2 = ax1.twinx()
    ax2.set_ylabel('NLL', fontsize=12, color='green')
    line2 = ax2.plot(reductions, nlls, 'g-s', linewidth=2, markersize=6, label='NLL')
    ax2.axhline(y=baseline_nll, color='green', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='green')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)

    ax1.set_title(f'{model_name} Layer {layer} {matrix}: Entropy vs NLL Tradeoff', fontsize=14)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_nll_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'entropy_nll_tradeoff.png'}")
    plt.close()

    # Plot 5: Percentage Change from Baseline
    fig, ax = plt.subplots(figsize=(12, 6))

    entropy_pct_change = [(e - baseline_entropy) / baseline_entropy * 100 for e in entropies]
    nll_pct_change = [(n - baseline_nll) / baseline_nll * 100 if baseline_nll > 0 else 0 for n in nlls]

    ax.plot(reductions, entropy_pct_change, 'b-o', linewidth=2, markersize=6, label='Entropy Change %')
    ax.plot(reductions, nll_pct_change, 'g-s', linewidth=2, markersize=6, label='NLL Change %')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Reduction % (Components Removed)', fontsize=12)
    ax.set_ylabel('Change from Baseline (%)', fontsize=12)
    ax.set_title(f'{model_name} Layer {layer} {matrix}: Percentage Change from Baseline', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # ax.invert_xaxis()  # Removed: left-to-right is more intuitive
    plt.tight_layout()
    plt.savefig(output_dir / 'percentage_change.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'percentage_change.png'}")
    plt.close()

    print(f"\nSVD truncation plots saved to: {output_dir}")


def detect_experiment_type(results):
    """Detect whether results are from noise removal or SVD truncation."""
    if 'component_results' in results:
        return 'noise_removal'
    elif 'truncation_results' in results:
        return 'svd_truncation'
    else:
        raise ValueError("Unknown experiment type")


def main():
    parser = argparse.ArgumentParser(description="Plot entropy experiment results")
    parser.add_argument("--results", type=str, required=True, help="Path to results.pkl file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots (default: same as results)")

    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return

    # Load results
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = results_path.parent / "figures"

    # Detect experiment type and plot
    exp_type = detect_experiment_type(results)
    print(f"Detected experiment type: {exp_type}")

    if exp_type == 'noise_removal':
        plot_noise_removal_results(results, output_dir)
    elif exp_type == 'svd_truncation':
        plot_svd_truncation_results(results, output_dir)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
