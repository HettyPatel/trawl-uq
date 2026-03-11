"""
Plot results from experiment 20 (Importance Sweep).

Generates:
  1. Importance heatmap — binary important/unimportant per (layer, chunk)
  2. Flip count heatmap — continuous flip count per (layer, chunk)
  3. Per-layer importance fraction bar chart
  4. Flip count distribution histogram
  5. Important chunks vs energy removed scatter

Supports --compare-dir for side-by-side comparison of two sweeps
(e.g. k=5 vs k=10, or ARC vs MMLU).

Usage:
    python scripts/plot_importance_results.py \\
        --results-dir results/importance_sweep/... \\
        --output-dir figures/importance/

    # Compare k=5 vs k=10
    python scripts/plot_importance_results.py \\
        --results-dir results/importance_sweep/..._k5_... \\
        --compare-dir results/importance_sweep/..._k10_... \\
        --output-dir figures/importance/

    # Fixed flip count range for cross-comparison
    python scripts/plot_importance_results.py \\
        --results-dir results/importance_sweep/... \\
        --flip-range 50
"""

import sys
sys.path.append('.')

import pickle
import glob
import numpy as np
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found.")


# =============================================================================
# Load data
# =============================================================================

def load_importance_results(results_dir: Path):
    """Load all layer pickles from an importance sweep directory."""
    results_dir = Path(results_dir)

    # Auto-detect model short from filenames
    layer_pkls = sorted(results_dir.glob("*_layer*.pkl"))
    if not layer_pkls:
        print(f"No layer pkl files found in {results_dir}")
        return {}, None, None

    model_short = layer_pkls[0].stem.split('_layer')[0]

    results = {}
    config_info = None
    for pkl_file in sorted(results_dir.glob(f"{model_short}_layer*.pkl")):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        layer = data['config']['layer']
        matrix_type = data['config']['matrix_type']
        results[layer] = data
        if config_info is None:
            config_info = data['config']

    print(f"Loaded {len(results)} layers from {results_dir.name}")
    return results, model_short, config_info


def build_flip_matrix(results):
    """Build (n_layers, n_chunks) matrix of flip counts."""
    layers = sorted(results.keys())
    n_chunks = max(r['config']['num_chunks'] for r in results.values())

    flip_matrix = np.zeros((len(layers), n_chunks))
    imp_matrix = np.zeros((len(layers), n_chunks), dtype=int)  # 0=unimp, 1=imp, 2=critical

    for li, layer in enumerate(layers):
        for chunk in results[layer]['chunk_results']:
            ci = chunk['chunk_idx']
            flip_matrix[li, ci] = chunk['flip_count']
            if chunk['importance'] == 'important':
                imp_matrix[li, ci] = 1
            elif chunk['importance'] == 'critical':
                imp_matrix[li, ci] = 2

    return flip_matrix, imp_matrix, layers, n_chunks


# =============================================================================
# Summary table
# =============================================================================

def print_summary(results, config_info):
    layers = sorted(results.keys())
    k = config_info.get('flip_threshold', '?')

    total_imp = 0
    total_unimp = 0
    total_crit = 0
    total_chunks = 0

    print(f"\n{'Layer':>5} {'Important':>10} {'Unimportant':>12} {'Critical':>9} {'Imp%':>6} {'MeanFlips(imp)':>15}")
    print("-" * 65)
    for layer in layers:
        s = results[layer]['summary']
        total_imp += s['n_important']
        total_unimp += s['n_unimportant']
        total_crit += s['n_critical']
        total_chunks += s['n_total']
        mf = f"{s['mean_flips_important']:.1f}" if s['n_important'] > 0 else "-"
        print(f"  {layer:>3}  {s['n_important']:>10}  {s['n_unimportant']:>12}  {s['n_critical']:>9}  "
              f"{s['important_fraction']*100:>5.1f}%  {mf:>15}")

    print("-" * 65)
    frac = total_imp / total_chunks * 100 if total_chunks > 0 else 0
    print(f"TOTAL  {total_imp:>10}  {total_unimp:>12}  {total_crit:>9}  {frac:>5.1f}%")
    print(f"  flip threshold k={k}, {total_chunks} total chunks")


# =============================================================================
# Plot 1: Importance heatmap (binary)
# =============================================================================

def plot_importance_heatmap(results, output_dir, label, flip_matrix=None, imp_matrix=None, layers=None, n_chunks=None):
    if not HAS_MPL:
        return
    if flip_matrix is None:
        flip_matrix, imp_matrix, layers, n_chunks = build_flip_matrix(results)

    # Custom colormap: yellow=unimportant, red=important, black=critical
    cmap = mcolors.ListedColormap(['#F5E6AB', '#D94F4F', '#2D2D2D'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.3)))
    im = ax.imshow(imp_matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_xlabel('Chunk Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(f'Importance Map — {label}', fontsize=13)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)

    legend_elements = [
        Patch(facecolor='#F5E6AB', label='Unimportant'),
        Patch(facecolor='#D94F4F', label='Important'),
        Patch(facecolor='#2D2D2D', label='Critical'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: importance_heatmap.png")


# =============================================================================
# Plot 2: Flip count heatmap (continuous)
# =============================================================================

def plot_flip_heatmap(results, output_dir, label, flip_matrix=None, layers=None, n_chunks=None,
                      fixed_flip_range=None):
    if not HAS_MPL:
        return
    if flip_matrix is None:
        flip_matrix, _, layers, n_chunks = build_flip_matrix(results)

    if fixed_flip_range is not None:
        vmax = fixed_flip_range
    else:
        # p98 for range
        flat = flip_matrix[np.isfinite(flip_matrix)]
        vmax = max(np.percentile(flat, 98), 5)

    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.3)))
    im = ax.imshow(flip_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax,
                   interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Flip Count (questions changed)', fontsize=9)

    ax.set_xlabel('Chunk Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(f'Flip Count — {label}', fontsize=13)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'flip_count_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: flip_count_heatmap.png")


# =============================================================================
# Plot 3: Per-layer importance fraction
# =============================================================================

def plot_importance_by_layer(results, output_dir, label):
    if not HAS_MPL:
        return

    layers = sorted(results.keys())
    fracs = [results[l]['summary']['important_fraction'] * 100 for l in layers]
    n_imp = [results[l]['summary']['n_important'] for l in layers]
    n_total = [results[l]['summary']['n_total'] for l in layers]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    bars = ax1.bar(range(len(layers)), fracs, color='#D94F4F', alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Important Chunks (%)', fontsize=11, color='#D94F4F')
    ax1.set_title(f'Per-Layer Importance Fraction — {label}', fontsize=13)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, fontsize=7)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', labelcolor='#D94F4F')

    # Add count labels on bars
    for i, (f, n, t) in enumerate(zip(fracs, n_imp, n_total)):
        if f > 3:
            ax1.text(i, f + 1.5, f'{n}', ha='center', va='bottom', fontsize=6, color='#333')

    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'importance_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: importance_by_layer.png")


# =============================================================================
# Plot 3b: Flip count by layer (mean + max)
# =============================================================================

def plot_flip_count_by_layer(results, output_dir, label):
    if not HAS_MPL:
        return

    layers = sorted(results.keys())
    total_flips = []

    for layer in layers:
        flips = [c['flip_count'] for c in results[layer]['chunk_results']]
        total_flips.append(sum(flips))

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(layers))
    ax.bar(x, total_flips, color='#4A90D9', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Total Flip Count', fontsize=11)
    ax.set_title(f'Total Flip Count by Layer — {label}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=7)

    # Label the top bars
    for i, t in enumerate(total_flips):
        if t > max(total_flips) * 0.1:
            ax.text(i, t + max(total_flips) * 0.01, f'{t}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / 'flip_count_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: flip_count_by_layer.png")


# =============================================================================
# Plot 4: Flip count distribution histogram
# =============================================================================

def plot_flip_distribution(results, output_dir, label):
    if not HAS_MPL:
        return

    all_flips = []
    for layer in sorted(results.keys()):
        for chunk in results[layer]['chunk_results']:
            all_flips.append(chunk['flip_count'])

    all_flips = np.array(all_flips)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    bins = np.arange(0, max(all_flips) + 2) - 0.5
    ax1.hist(all_flips, bins=bins, color='#4A90D9', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Flip Count', fontsize=11)
    ax1.set_ylabel('Number of Chunks', fontsize=11)
    ax1.set_title(f'Flip Count Distribution — {label}', fontsize=12)

    # Zoomed: 0-30 range
    zoomed = all_flips[all_flips <= 30]
    bins_z = np.arange(0, 32) - 0.5
    ax2.hist(zoomed, bins=bins_z, color='#4A90D9', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Flip Count', fontsize=11)
    ax2.set_ylabel('Number of Chunks', fontsize=11)
    ax2.set_title(f'Zoomed (0–30 flips)', fontsize=12)

    # Stats annotation
    stats = (f"n={len(all_flips)}, median={np.median(all_flips):.0f}, "
             f"mean={np.mean(all_flips):.1f}, max={np.max(all_flips)}")
    fig.suptitle(stats, fontsize=9, y=0.02, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'flip_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: flip_distribution.png")


# =============================================================================
# Plot 5: Energy removed vs flip count scatter
# =============================================================================

def plot_energy_vs_flips(results, output_dir, label):
    if not HAS_MPL:
        return

    energies = []
    flips = []
    layers_color = []

    for layer in sorted(results.keys()):
        for chunk in results[layer]['chunk_results']:
            e = chunk.get('avg_energy_removed', 0)
            f = chunk['flip_count']
            energies.append(e * 100)
            flips.append(f)
            layers_color.append(layer)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(energies, flips, c=layers_color, cmap='coolwarm', s=10, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Layer', fontsize=9)

    ax.set_xlabel('Energy Removed (%)', fontsize=11)
    ax.set_ylabel('Flip Count', fontsize=11)
    ax.set_title(f'Energy Removed vs Flip Count — {label}', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_vs_flips.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: energy_vs_flips.png")


# =============================================================================
# Plot 6: Comparison — two sweeps side by side
# =============================================================================

def plot_comparison_importance(results_a, label_a, results_b, label_b, output_dir,
                               fixed_flip_range=None):
    """Side-by-side importance heatmaps and flip count heatmaps."""
    if not HAS_MPL:
        return

    flip_a, imp_a, layers_a, nc_a = build_flip_matrix(results_a)
    flip_b, imp_b, layers_b, nc_b = build_flip_matrix(results_b)

    # Importance heatmaps side by side
    cmap = mcolors.ListedColormap(['#F5E6AB', '#D94F4F', '#2D2D2D'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, len(layers_a) * 0.3)))

    ax1.imshow(imp_a, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax1.set_title(f'{label_a}', fontsize=12)
    ax1.set_xlabel('Chunk Index', fontsize=10)
    ax1.set_ylabel('Layer', fontsize=10)
    ax1.set_yticks(range(len(layers_a)))
    ax1.set_yticklabels(layers_a, fontsize=7)

    ax2.imshow(imp_b, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax2.set_title(f'{label_b}', fontsize=12)
    ax2.set_xlabel('Chunk Index', fontsize=10)
    ax2.set_yticks(range(len(layers_b)))
    ax2.set_yticklabels(layers_b, fontsize=7)

    legend_elements = [
        Patch(facecolor='#F5E6AB', label='Unimportant'),
        Patch(facecolor='#D94F4F', label='Important'),
        Patch(facecolor='#2D2D2D', label='Critical'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    fig.suptitle('Importance Map Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_dir / 'comparison_importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_importance_heatmap.png")

    # Flip count heatmaps side by side
    if fixed_flip_range is not None:
        vmax = fixed_flip_range
    else:
        vmax = max(np.percentile(flip_a, 98), np.percentile(flip_b, 98), 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, len(layers_a) * 0.3)))

    im1 = ax1.imshow(flip_a, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax,
                     interpolation='nearest')
    ax1.set_title(f'{label_a}', fontsize=12)
    ax1.set_xlabel('Chunk Index', fontsize=10)
    ax1.set_ylabel('Layer', fontsize=10)
    ax1.set_yticks(range(len(layers_a)))
    ax1.set_yticklabels(layers_a, fontsize=7)

    im2 = ax2.imshow(flip_b, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax,
                     interpolation='nearest')
    ax2.set_title(f'{label_b}', fontsize=12)
    ax2.set_xlabel('Chunk Index', fontsize=10)
    ax2.set_yticks(range(len(layers_b)))
    ax2.set_yticklabels(layers_b, fontsize=7)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label='Flip Count')

    fig.suptitle('Flip Count Comparison', fontsize=14)
    plt.savefig(output_dir / 'comparison_flip_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_flip_heatmap.png")

    # Per-layer fraction comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    fracs_a = [results_a[l]['summary']['important_fraction'] * 100 for l in layers_a]
    fracs_b = [results_b[l]['summary']['important_fraction'] * 100 for l in layers_b]

    x = np.arange(len(layers_a))
    w = 0.35
    ax.bar(x - w/2, fracs_a, w, label=label_a, color='#D94F4F', alpha=0.8)
    ax.bar(x + w/2, fracs_b, w, label=label_b, color='#4A90D9', alpha=0.8)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Important Chunks (%)', fontsize=11)
    ax.set_title('Per-Layer Importance Fraction Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(layers_a, fontsize=7)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_importance_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_importance_by_layer.png")


# =============================================================================
# Plot 7: Multi-k summary — total important/unimportant by threshold
# =============================================================================

def plot_k_comparison(results_dirs, output_dir):
    """
    Bar chart showing total important vs unimportant for multiple k values.
    results_dirs: list of (label, results_dir_path) tuples.
    """
    if not HAS_MPL:
        return

    labels = []
    n_important = []
    n_unimportant = []
    n_critical = []

    for label, rdir in results_dirs:
        results, _, config = load_importance_results(Path(rdir))
        if not results:
            continue
        ti = sum(r['summary']['n_important'] for r in results.values())
        tu = sum(r['summary']['n_unimportant'] for r in results.values())
        tc = sum(r['summary']['n_critical'] for r in results.values())
        labels.append(label)
        n_important.append(ti)
        n_unimportant.append(tu)
        n_critical.append(tc)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.5), 5))
    bars_u = ax.bar(x - w/2, n_unimportant, w, label='Unimportant', color='#F5E6AB', edgecolor='#ccc')
    bars_i = ax.bar(x + w/2, n_important, w, label='Important', color='#D94F4F', edgecolor='#ccc')

    # Add count + percentage labels
    for i in range(len(labels)):
        total = n_important[i] + n_unimportant[i] + n_critical[i]
        pct_i = 100 * n_important[i] / total if total > 0 else 0
        pct_u = 100 * n_unimportant[i] / total if total > 0 else 0
        ax.text(x[i] + w/2, n_important[i] + 5, f'{n_important[i]}\n({pct_i:.0f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(x[i] - w/2, n_unimportant[i] + 5, f'{n_unimportant[i]}\n({pct_u:.0f}%)',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Sweep', fontsize=11)
    ax.set_ylabel('Number of Chunks', fontsize=11)
    ax.set_title('Important vs Unimportant Chunks by Threshold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'k_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: k_comparison.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot importance sweep results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--compare-dir", type=str, default=None,
                        help="Second results dir for side-by-side comparison")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for this sweep (auto-detected if not set)")
    parser.add_argument("--compare-label", type=str, default=None)
    parser.add_argument("--flip-range", type=float, default=None,
                        help="Fixed max flip count for heatmap colorbar (for cross-comparison)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results, model_short, config = load_importance_results(results_dir)
    if not results:
        return

    k = config.get('flip_threshold', '?')
    eval_stem = Path(config.get('mcq_eval_set_path', '')).stem
    # Build label
    if args.label:
        label = args.label
    else:
        # Extract dataset name from eval set path
        dataset = eval_stem.replace('eval_set_mcq_', '').replace('_', ' ').title()
        label = f"{model_short} — k={k} — {dataset}"

    output_dir = Path(args.output_dir) if args.output_dir else Path(f'figures/importance_{results_dir.name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Label: {label}")
    print_summary(results, config)
    print(f"\nGenerating plots to {output_dir}...")

    flip_matrix, imp_matrix, layers, n_chunks = build_flip_matrix(results)
    plot_importance_heatmap(results, output_dir, label, flip_matrix, imp_matrix, layers, n_chunks)
    plot_flip_heatmap(results, output_dir, label, flip_matrix, layers, n_chunks,
                      fixed_flip_range=args.flip_range)
    plot_importance_by_layer(results, output_dir, label)
    plot_flip_count_by_layer(results, output_dir, label)
    plot_flip_distribution(results, output_dir, label)
    plot_energy_vs_flips(results, output_dir, label)

    # Comparison
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        results_b, model_b, config_b = load_importance_results(compare_dir)
        if results_b:
            k_b = config_b.get('flip_threshold', '?')
            if args.compare_label:
                label_b = args.compare_label
            else:
                eval_b = Path(config_b.get('mcq_eval_set_path', '')).stem
                dataset_b = eval_b.replace('eval_set_mcq_', '').replace('_', ' ').title()
                label_b = f"{model_b} — k={k_b} — {dataset_b}"

            print(f"\nGenerating comparison: {label} vs {label_b}")
            plot_comparison_importance(results, label, results_b, label_b, output_dir,
                                      fixed_flip_range=args.flip_range)

    print("\nDone!")


if __name__ == "__main__":
    main()
