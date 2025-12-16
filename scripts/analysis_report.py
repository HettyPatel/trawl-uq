"""
Analysis Script for Component Search Results

Analyzes which components are noise vs. signal and compares layers.
"""

import pickle
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

def plot_component_analysis(analysis_30, analysis_31, output_dir):
    """Create comprehensive visualization plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    comp_30 = analysis_30['component_data']
    comp_31 = analysis_31['component_data']
    
    indices_30 = [c['component_idx'] for c in comp_30]
    unc_change_30 = [c['mean_uncertainty_change'] for c in comp_30]
    block_change_30 = [c['mean_blockiness_change'] for c in comp_30]
    
    indices_31 = [c['component_idx'] for c in comp_31]
    unc_change_31 = [c['mean_uncertainty_change'] for c in comp_31]
    block_change_31 = [c['mean_blockiness_change'] for c in comp_31]
    
    # Plot 1: Uncertainty Change by Component
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Layer 30
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
    axes[0].bar(indices_30, unc_change_30, color=['green' if x < 0 else 'orange' for x in unc_change_30], alpha=0.7)
    axes[0].set_xlabel('Component Index', fontsize=12)
    axes[0].set_ylabel('Uncertainty Change', fontsize=12)
    axes[0].set_title(f'Layer 30: Component Removal Effect on Uncertainty\n(Green = Noise, Orange = Signal)', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Layer 31
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
    axes[1].bar(indices_31, unc_change_31, color=['green' if x < 0 else 'orange' for x in unc_change_31], alpha=0.7)
    axes[1].set_xlabel('Component Index', fontsize=12)
    axes[1].set_ylabel('Uncertainty Change', fontsize=12)
    axes[1].set_title(f'Layer 31: Component Removal Effect on Uncertainty\n(Green = Noise, Orange = Signal)', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_change_by_component.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_change_by_component.png'}")
    plt.close()
    
    # Plot 2: Comparison Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_offset = 0.35
    x_30 = np.array(indices_30) - x_offset/2
    x_31 = np.array(indices_31) + x_offset/2
    
    ax.bar(x_30, unc_change_30, width=x_offset, label='Layer 30', alpha=0.7, color='blue')
    ax.bar(x_31, unc_change_31, width=x_offset, label='Layer 31', alpha=0.7, color='red')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Component Index', fontsize=12)
    ax.set_ylabel('Uncertainty Change', fontsize=12)
    ax.set_title('Layer 30 vs Layer 31: Component Removal Effects', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison.png'}")
    plt.close()
    
    # Plot 3: Scatter - Uncertainty vs Blockiness Change
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Layer 30
    axes[0].scatter(unc_change_30, block_change_30, alpha=0.6, s=100, c=indices_30, cmap='viridis')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Uncertainty Change', fontsize=12)
    axes[0].set_ylabel('Blockiness Change', fontsize=12)
    axes[0].set_title('Layer 30: Uncertainty vs Blockiness Change', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # Layer 31
    sc = axes[1].scatter(unc_change_31, block_change_31, alpha=0.6, s=100, c=indices_31, cmap='viridis')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Uncertainty Change', fontsize=12)
    axes[1].set_ylabel('Blockiness Change', fontsize=12)
    axes[1].set_title('Layer 31: Uncertainty vs Blockiness Change', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.colorbar(sc, ax=axes[1], label='Component Index')
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_blockiness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_vs_blockiness.png'}")
    plt.close()
    
    # Plot 4: Distribution of Changes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Uncertainty changes
    axes[0].hist(unc_change_30, bins=20, alpha=0.5, label='Layer 30', color='blue', edgecolor='black')
    axes[0].hist(unc_change_31, bins=20, alpha=0.5, label='Layer 31', color='red', edgecolor='black')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Uncertainty Change', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Uncertainty Changes', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Blockiness changes
    axes[1].hist(block_change_30, bins=20, alpha=0.5, label='Layer 30', color='blue', edgecolor='black')
    axes[1].hist(block_change_31, bins=20, alpha=0.5, label='Layer 31', color='red', edgecolor='black')
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

def generate_report(analysis_30, analysis_31, output_dir):
    """Generate text report with findings."""
    
    output_dir = Path(output_dir)
    
    comp_30 = analysis_30['component_data']
    comp_31 = analysis_31['component_data']
    
    # Categorize components
    noise_30 = [c for c in comp_30 if c['mean_uncertainty_change'] < 0]
    signal_30 = [c for c in comp_30 if c['mean_uncertainty_change'] > 0]
    
    noise_31 = [c for c in comp_31 if c['mean_uncertainty_change'] < 0]
    signal_31 = [c for c in comp_31 if c['mean_uncertainty_change'] > 0]
    
    # Sort
    noise_30_sorted = sorted(noise_30, key=lambda x: x['mean_uncertainty_change'])
    noise_31_sorted = sorted(noise_31, key=lambda x: x['mean_uncertainty_change'])
    
    report = []
    report.append("="*80)
    report.append("COMPONENT SEARCH ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("EXPERIMENT CONFIGURATION")
    report.append("-"*80)
    report.append(f"Samples: 10")
    report.append(f"Tucker Rank: 40")
    report.append(f"Layers Tested: 30, 31")
    report.append("")
    
    report.append("BASELINE UNCERTAINTY")
    report.append("-"*80)
    report.append(f"Layer 30: {analysis_30['baseline_uncertainty']:.3f} ± {analysis_30['baseline_uncertainty_std']:.3f}")
    report.append(f"Layer 31: {analysis_31['baseline_uncertainty']:.3f} ± {analysis_31['baseline_uncertainty_std']:.3f}")
    report.append("")
    
    report.append("COMPONENT CATEGORIZATION")
    report.append("-"*80)
    report.append(f"Layer 30:")
    report.append(f"  Noise components:  {len(noise_30)}/40 ({len(noise_30)/40*100:.1f}%)")
    report.append(f"  Signal components: {len(signal_30)}/40 ({len(signal_30)/40*100:.1f}%)")
    report.append(f"")
    report.append(f"Layer 31:")
    report.append(f"  Noise components:  {len(noise_31)}/40 ({len(noise_31)/40*100:.1f}%)")
    report.append(f"  Signal components: {len(signal_31)}/40 ({len(signal_31)/40*100:.1f}%)")
    report.append("")
    
    report.append("TOP 10 NOISE COMPONENTS - LAYER 30")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty Δ':<20} {'Blockiness Δ':<20}")
    report.append("-"*80)
    for i, comp in enumerate(noise_30_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")
    
    report.append("TOP 10 NOISE COMPONENTS - LAYER 31")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'Component':<12} {'Uncertainty Δ':<20} {'Blockiness Δ':<20}")
    report.append("-"*80)
    for i, comp in enumerate(noise_31_sorted[:10], 1):
        report.append(f"{i:<6} {comp['component_idx']:<12} {comp['mean_uncertainty_change']:+.4f} ({comp['std_uncertainty_change']:.4f})    {comp['mean_blockiness_change']:+.4f} ({comp['std_blockiness_change']:.4f})")
    report.append("")
    
    report.append("KEY FINDINGS")
    report.append("="*80)
    
    if len(noise_31) > len(noise_30):
        diff = len(noise_31) - len(noise_30)
        report.append(f"✓ Layer 31 has {diff} MORE noise components than Layer 30 ({len(noise_31)} vs {len(noise_30)})")
        report.append(f"  This supports the hypothesis that last layers have higher uncertainty.")
    elif len(noise_30) > len(noise_31):
        diff = len(noise_30) - len(noise_31)
        report.append(f"✗ Layer 30 has {diff} MORE noise components than Layer 31 ({len(noise_30)} vs {len(noise_31)})")
        report.append(f"  This does NOT support the hypothesis.")
    else:
        report.append(f"= Both layers have equal noise components ({len(noise_30)})")
    
    report.append("")
    report.append("INTERPRETATION")
    report.append("-"*80)
    report.append("Noise components (Uncertainty Δ < 0):")
    report.append("  - Removing them REDUCES uncertainty")
    report.append("  - Safe to compress without hurting model quality")
    report.append("  - This explains TRAWL's accuracy improvements")
    report.append("")
    report.append("Signal components (Uncertainty Δ > 0):")
    report.append("  - Removing them INCREASES uncertainty")
    report.append("  - Important to keep for model quality")
    report.append("  - Should not be compressed")
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_dir / 'analysis_report.txt'}")

def main():
    # Paths
    layer30_path = Path('results/component_search/coqa_20251213_142049/results.pkl')
    layer31_path = Path('results/component_search/coqa_20251213_142059/results.pkl')
    output_dir = Path('results/analysis/component_search')
    
    print("Loading results...")
    results_30 = load_results(layer30_path)
    results_31 = load_results(layer31_path)
    
    print("Analyzing Layer 30...")
    analysis_30 = analyze_layer(results_30, "Layer 30")
    
    print("Analyzing Layer 31...")
    analysis_31 = analyze_layer(results_31, "Layer 31")
    
    print("\nCreating visualizations...")
    plot_component_analysis(analysis_30, analysis_31, output_dir)
    
    print("\nGenerating report...")
    generate_report(analysis_30, analysis_31, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("\nFiles created:")
    print("  - uncertainty_change_by_component.png")
    print("  - layer_comparison.png")
    print("  - uncertainty_vs_blockiness.png")
    print("  - change_distributions.png")
    print("  - analysis_report.txt")
    print("="*80)

if __name__ == "__main__":
    main()