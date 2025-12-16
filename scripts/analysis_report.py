"""
Analysis Script for Component Search Results

Analyzes which components are noise vs. signal and compares layers.
Works with both Tucker and CP decomposition results.
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(result_path):
    """Load pickled results."""
    with open(result_path, 'rb') as f:
        return pickle.load(f)


def get_decomposition_type(results):
    """Detect decomposition type from results config."""
    config = results.get('config', {})
    if 'decomposition_type' in config:
        return config['decomposition_type']
    # Infer from keys
    if 'tucker_rank' in config:
        return 'tucker'
    if 'cp_rank' in config:
        return 'cp'
    return 'unknown'


def get_rank(results):
    """Get rank from results config."""
    config = results.get('config', {})
    return config.get('tucker_rank') or config.get('cp_rank') or 40


def analyze_layer(results, layer_name):
    """Analyze component search results for one layer."""

    # Baseline statistics
    baseline_uncertainties = [s['uncertainty_score'] for s in results['baseline']]
    baseline_blockiness = [s['blockiness_by_rank']['rank_10']['reconstruction_fit']
                          for s in results['baseline']]

    # Component analysis
    component_data = []

    for comp in results['component_search']:
        comp_idx = comp['component_idx']

        # Get all samples for this component
        uncertainty_changes = [r['uncertainty_change'] for r in comp['results']]
        blockiness_changes = [r['blockiness_change'] for r in comp['results']]

        component_data.append({
            'component_idx': comp_idx,
            'mean_uncertainty_change': np.mean(uncertainty_changes),
            'std_uncertainty_change': np.std(uncertainty_changes),
            'mean_blockiness_change': np.mean(blockiness_changes),
            'std_blockiness_change': np.std(blockiness_changes),
            'uncertainty_changes': uncertainty_changes,
            'blockiness_changes': blockiness_changes
        })

    return {
        'layer_name': layer_name,
        'baseline_uncertainty': np.mean(baseline_uncertainties),
        'baseline_uncertainty_std': np.std(baseline_uncertainties),
        'baseline_blockiness': np.mean(baseline_blockiness),
        'component_data': component_data
    }


def plot_single_layer_analysis(analysis, output_dir, decomp_type):
    """Create visualization for a single layer."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_data = analysis['component_data']
    layer_name = analysis['layer_name']

    indices = [c['component_idx'] for c in comp_data]
    unc_change = [c['mean_uncertainty_change'] for c in comp_data]
    block_change = [c['mean_blockiness_change'] for c in comp_data]

    decomp_label = decomp_type.upper()

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

    # Plot 2: Scatter - Uncertainty vs Blockiness Change
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(unc_change, block_change, alpha=0.6, s=100, c=indices, cmap='viridis')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Uncertainty Change', fontsize=12)
    ax.set_ylabel('Blockiness Change', fontsize=12)
    ax.set_title(f'{layer_name} ({decomp_label}): Uncertainty vs Blockiness Change', fontsize=14)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Component Index')

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

    axes[1].hist(block_change, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Blockiness Change', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{layer_name}: Distribution of Blockiness Changes', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'change_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'change_distributions.png'}")
    plt.close()


def plot_component_analysis(analysis_1, analysis_2, output_dir, decomp_type):
    """Create comprehensive visualization plots for two layers."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    comp_1 = analysis_1['component_data']
    comp_2 = analysis_2['component_data']

    layer_1_name = analysis_1['layer_name']
    layer_2_name = analysis_2['layer_name']

    indices_1 = [c['component_idx'] for c in comp_1]
    unc_change_1 = [c['mean_uncertainty_change'] for c in comp_1]
    block_change_1 = [c['mean_blockiness_change'] for c in comp_1]

    indices_2 = [c['component_idx'] for c in comp_2]
    unc_change_2 = [c['mean_uncertainty_change'] for c in comp_2]
    block_change_2 = [c['mean_blockiness_change'] for c in comp_2]

    decomp_label = decomp_type.upper()

    # Plot 1: Uncertainty Change by Component
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Layer 1
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
    axes[0].bar(indices_1, unc_change_1, color=['green' if x < 0 else 'orange' for x in unc_change_1], alpha=0.7)
    axes[0].set_xlabel('Component Index', fontsize=12)
    axes[0].set_ylabel('Uncertainty Change', fontsize=12)
    axes[0].set_title(f'{layer_1_name} ({decomp_label}): Component Removal Effect on Uncertainty\n(Green = Noise, Orange = Signal)', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Layer 2
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
    axes[1].bar(indices_2, unc_change_2, color=['green' if x < 0 else 'orange' for x in unc_change_2], alpha=0.7)
    axes[1].set_xlabel('Component Index', fontsize=12)
    axes[1].set_ylabel('Uncertainty Change', fontsize=12)
    axes[1].set_title(f'{layer_2_name} ({decomp_label}): Component Removal Effect on Uncertainty\n(Green = Noise, Orange = Signal)', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_change_by_component.png'}")
    plt.close()

    # Plot 2: Comparison Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x_offset = 0.35
    x_1 = np.array(indices_1) - x_offset/2
    x_2 = np.array(indices_2) + x_offset/2

    ax.bar(x_1, unc_change_1, width=x_offset, label=layer_1_name, alpha=0.7, color='blue')
    ax.bar(x_2, unc_change_2, width=x_offset, label=layer_2_name, alpha=0.7, color='red')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Uncertainty Change', fontsize=12)
    ax.set_title(f'{layer_1_name} vs {layer_2_name} ({decomp_label}): Component Removal Effects', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison.png'}")
    plt.close()

    # Plot 3: Scatter - Uncertainty vs Blockiness Change
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Layer 1
    axes[0].scatter(unc_change_1, block_change_1, alpha=0.6, s=100, c=indices_1, cmap='viridis')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Uncertainty Change', fontsize=12)
    axes[0].set_ylabel('Blockiness Change', fontsize=12)
    axes[0].set_title(f'{layer_1_name}: Uncertainty vs Blockiness Change', fontsize=14)
    axes[0].grid(alpha=0.3)

    # Layer 2
    sc = axes[1].scatter(unc_change_2, block_change_2, alpha=0.6, s=100, c=indices_2, cmap='viridis')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Uncertainty Change', fontsize=12)
    axes[1].set_ylabel('Blockiness Change', fontsize=12)
    axes[1].set_title(f'{layer_2_name}: Uncertainty vs Blockiness Change', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.colorbar(sc, ax=axes[1], label='Component Index')
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_blockiness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_vs_blockiness.png'}")
    plt.close()

    # Plot 4: Distribution of Changes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Uncertainty changes
    axes[0].hist(unc_change_1, bins=20, alpha=0.5, label=layer_1_name, color='blue', edgecolor='black')
    axes[0].hist(unc_change_2, bins=20, alpha=0.5, label=layer_2_name, color='red', edgecolor='black')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Uncertainty Change', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Uncertainty Changes', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Blockiness changes
    axes[1].hist(block_change_1, bins=20, alpha=0.5, label=layer_1_name, color='blue', edgecolor='black')
    axes[1].hist(block_change_2, bins=20, alpha=0.5, label=layer_2_name, color='red', edgecolor='black')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Blockiness Change', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Blockiness Changes', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'change_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'change_distributions.png'}")
    plt.close()


def generate_single_layer_report(analysis, results, output_dir, decomp_type):
    """Generate text report for a single layer."""

    output_dir = Path(output_dir)
    config = results.get('config', {})
    rank = get_rank(results)

    comp_data = analysis['component_data']
    layer_name = analysis['layer_name']

    # Categorize components
    noise_comps = [c for c in comp_data if c['mean_uncertainty_change'] < 0]
    signal_comps = [c for c in comp_data if c['mean_uncertainty_change'] > 0]

    # Sort
    noise_sorted = sorted(noise_comps, key=lambda x: x['mean_uncertainty_change'])
    signal_sorted = sorted(signal_comps, key=lambda x: x['mean_uncertainty_change'], reverse=True)

    decomp_label = decomp_type.upper()

    report = []
    report.append("="*80)
    report.append(f"COMPONENT SEARCH ANALYSIS REPORT ({decomp_label})")
    report.append("="*80)
    report.append("")

    report.append("EXPERIMENT CONFIGURATION")
    report.append("-"*80)
    report.append(f"Decomposition: {decomp_label}")
    report.append(f"Layer: {layer_name}")
    report.append(f"Rank: {rank}")
    report.append(f"Samples: {config.get('num_samples', 'N/A')}")
    report.append(f"Model: {config.get('model_name', 'N/A')}")
    report.append("")

    report.append("BASELINE UNCERTAINTY")
    report.append("-"*80)
    report.append(f"{layer_name}: {analysis['baseline_uncertainty']:.3f} +/- {analysis['baseline_uncertainty_std']:.3f}")
    report.append("")

    report.append("COMPONENT CATEGORIZATION")
    report.append("-"*80)
    report.append(f"Noise components:  {len(noise_comps)}/{rank} ({len(noise_comps)/rank*100:.1f}%)")
    report.append(f"Signal components: {len(signal_comps)}/{rank} ({len(signal_comps)/rank*100:.1f}%)")
    report.append("")

    report.append(f"TOP 10 NOISE COMPONENTS (safe to remove)")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty D':<20} {'Blockiness D':<20}")
    report.append("-"*80)
    for i, comp in enumerate(noise_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")

    report.append(f"TOP 10 SIGNAL COMPONENTS (important to keep)")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty D':<20} {'Blockiness D':<20}")
    report.append("-"*80)
    for i, comp in enumerate(signal_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")

    report.append("INTERPRETATION")
    report.append("-"*80)
    report.append("Noise components (Uncertainty D < 0):")
    report.append("  - Removing them REDUCES uncertainty")
    report.append("  - Safe to compress without hurting model quality")
    report.append("")
    report.append("Signal components (Uncertainty D > 0):")
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


def generate_report(analysis_1, analysis_2, results_1, output_dir, decomp_type):
    """Generate text report with findings for two layers."""

    output_dir = Path(output_dir)
    config = results_1.get('config', {})
    rank = get_rank(results_1)

    comp_1 = analysis_1['component_data']
    comp_2 = analysis_2['component_data']

    layer_1_name = analysis_1['layer_name']
    layer_2_name = analysis_2['layer_name']

    # Categorize components
    noise_1 = [c for c in comp_1 if c['mean_uncertainty_change'] < 0]
    signal_1 = [c for c in comp_1 if c['mean_uncertainty_change'] > 0]

    noise_2 = [c for c in comp_2 if c['mean_uncertainty_change'] < 0]
    signal_2 = [c for c in comp_2 if c['mean_uncertainty_change'] > 0]

    # Sort
    noise_1_sorted = sorted(noise_1, key=lambda x: x['mean_uncertainty_change'])
    noise_2_sorted = sorted(noise_2, key=lambda x: x['mean_uncertainty_change'])

    decomp_label = decomp_type.upper()

    report = []
    report.append("="*80)
    report.append(f"COMPONENT SEARCH ANALYSIS REPORT ({decomp_label})")
    report.append("="*80)
    report.append("")

    report.append("EXPERIMENT CONFIGURATION")
    report.append("-"*80)
    report.append(f"Decomposition: {decomp_label}")
    report.append(f"Samples: {config.get('num_samples', 'N/A')}")
    report.append(f"Rank: {rank}")
    report.append(f"Layers Tested: {layer_1_name}, {layer_2_name}")
    report.append("")

    report.append("BASELINE UNCERTAINTY")
    report.append("-"*80)
    report.append(f"{layer_1_name}: {analysis_1['baseline_uncertainty']:.3f} +/- {analysis_1['baseline_uncertainty_std']:.3f}")
    report.append(f"{layer_2_name}: {analysis_2['baseline_uncertainty']:.3f} +/- {analysis_2['baseline_uncertainty_std']:.3f}")
    report.append("")

    report.append("COMPONENT CATEGORIZATION")
    report.append("-"*80)
    report.append(f"{layer_1_name}:")
    report.append(f"  Noise components:  {len(noise_1)}/{rank} ({len(noise_1)/rank*100:.1f}%)")
    report.append(f"  Signal components: {len(signal_1)}/{rank} ({len(signal_1)/rank*100:.1f}%)")
    report.append(f"")
    report.append(f"{layer_2_name}:")
    report.append(f"  Noise components:  {len(noise_2)}/{rank} ({len(noise_2)/rank*100:.1f}%)")
    report.append(f"  Signal components: {len(signal_2)}/{rank} ({len(signal_2)/rank*100:.1f}%)")
    report.append("")

    report.append(f"TOP 10 NOISE COMPONENTS - {layer_1_name}")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty D':<20} {'Blockiness D':<20}")
    report.append("-"*80)
    for i, comp in enumerate(noise_1_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")

    report.append(f"TOP 10 NOISE COMPONENTS - {layer_2_name}")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty D':<20} {'Blockiness D':<20}")
    report.append("-"*80)
    for i, comp in enumerate(noise_2_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")

    report.append("KEY FINDINGS")
    report.append("="*80)

    if len(noise_2) > len(noise_1):
        diff = len(noise_2) - len(noise_1)
        report.append(f"* {layer_2_name} has {diff} MORE noise components than {layer_1_name} ({len(noise_2)} vs {len(noise_1)})")
    elif len(noise_1) > len(noise_2):
        diff = len(noise_1) - len(noise_2)
        report.append(f"* {layer_1_name} has {diff} MORE noise components than {layer_2_name} ({len(noise_1)} vs {len(noise_2)})")
    else:
        report.append(f"= Both layers have equal noise components ({len(noise_1)})")

    report.append("")
    report.append("INTERPRETATION")
    report.append("-"*80)
    report.append("Noise components (Uncertainty D < 0):")
    report.append("  - Removing them REDUCES uncertainty")
    report.append("  - Safe to compress without hurting model quality")
    report.append("")
    report.append("Signal components (Uncertainty D > 0):")
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


def main():
    parser = argparse.ArgumentParser(description='Analyze component search results')
    parser.add_argument('--results', type=str, required=True, nargs='+',
                        help='Path(s) to results.pkl file(s). Provide 1 or 2 paths.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis (default: results/analysis/<decomp_type>)')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Layer names for labeling (e.g., --layers "Layer 30" "Layer 31")')

    args = parser.parse_args()

    if len(args.results) > 2:
        print("Error: Maximum 2 result files supported for comparison")
        return

    print("Loading results...")
    results_list = [load_results(Path(p)) for p in args.results]

    # Detect decomposition type from first result
    decomp_type = get_decomposition_type(results_list[0])
    print(f"Detected decomposition type: {decomp_type.upper()}")

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f'results/analysis/component_search_{decomp_type}')

    # Set layer names
    if args.layers:
        layer_names = args.layers
    else:
        # Try to infer from config
        layer_names = []
        for i, results in enumerate(results_list):
            config = results.get('config', {})
            layer = config.get('target_layer', f'Unknown_{i}')
            layer_names.append(f"Layer {layer}")

    if len(results_list) == 1:
        # Single layer analysis
        print(f"Analyzing {layer_names[0]}...")
        analysis = analyze_layer(results_list[0], layer_names[0])

        print("\nCreating visualizations...")
        plot_single_layer_analysis(analysis, output_dir, decomp_type)

        print("\nGenerating report...")
        generate_single_layer_report(analysis, results_list[0], output_dir, decomp_type)

    else:
        # Two layer comparison
        print(f"Analyzing {layer_names[0]}...")
        analysis_1 = analyze_layer(results_list[0], layer_names[0])

        print(f"Analyzing {layer_names[1]}...")
        analysis_2 = analyze_layer(results_list[1], layer_names[1])

        print("\nCreating visualizations...")
        plot_component_analysis(analysis_1, analysis_2, output_dir, decomp_type)

        print("\nGenerating report...")
        generate_report(analysis_1, analysis_2, results_list[0], output_dir, decomp_type)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("\nFiles created:")
    print("  - uncertainty_change_by_component.png")
    if len(results_list) == 2:
        print("  - layer_comparison.png")
    print("  - uncertainty_vs_blockiness.png")
    print("  - change_distributions.png")
    print("  - analysis_report.txt")
    print("="*80)


if __name__ == "__main__":
    main()
