"""
Analyze importance consistency across datasets, matrix types, and models.

Loads importance classification results (from experiment 20) and computes:
  1. Cross-dataset consistency: same model+matrix, different datasets
  2. Cross-matrix consistency: same model+dataset, different weight types
  3. Cross-model consistency: same dataset, different models

Outputs:
  - Printed tables with overlap statistics
  - Figures: heatmaps of per-layer importance fraction, overlap matrices
  - Saved summary pickle

Usage:
    # Run experiment 20 first for each sweep, then:
    python scripts/analyze_importance_consistency.py \\
        --output-dir figures/importance_consistency/

    # Or point to specific importance result files:
    python scripts/analyze_importance_consistency.py \\
        --files results/importance/importance_Llama2_arc500.pkl \\
                results/importance/importance_Llama2_boolq500.pkl \\
        --labels "ARC-500" "BoolQ-500" \\
        --output-dir figures/importance_consistency/
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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found, skipping figures.")


# =============================================================================
# Load importance results
# =============================================================================

def load_importance(filepath: str):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_importance_matrix(result):
    """
    Build a (n_layers x n_chunks) boolean matrix where True = important.
    Returns matrix, layer indices, max_chunks.
    """
    per_layer = result['per_layer']
    layers = sorted(set(l['layer'] for l in per_layer))

    # Find max chunk index
    all_chunks = result['all_chunk_info']
    if not all_chunks:
        return None, layers, 0
    max_chunk = max(c['chunk_idx'] for c in all_chunks) + 1

    matrix = np.zeros((len(layers), max_chunk), dtype=bool)
    layer_to_idx = {l: i for i, l in enumerate(layers)}

    for chunk in all_chunks:
        li = layer_to_idx[chunk['layer']]
        ci = chunk['chunk_idx']
        if chunk['importance'] == 'important':
            matrix[li, ci] = True

    return matrix, layers, max_chunk


def get_importance_fraction_per_layer(result):
    """Return dict: layer -> important_fraction."""
    return {l['layer']: l['important_fraction'] for l in result['per_layer']}


# =============================================================================
# Overlap analysis
# =============================================================================

def compute_overlap(mat_a, mat_b, label_a, label_b):
    """
    Compute chunk-level overlap between two importance matrices.
    Both must have same shape (n_layers, n_chunks).
    Returns overlap statistics.
    """
    # Align to same shape
    n_layers = min(mat_a.shape[0], mat_b.shape[0])
    n_chunks = min(mat_a.shape[1], mat_b.shape[1])
    a = mat_a[:n_layers, :n_chunks]
    b = mat_b[:n_layers, :n_chunks]

    both_important = (a & b).sum()
    either_important = (a | b).sum()
    only_a = (a & ~b).sum()
    only_b = (~a & b).sum()
    jaccard = both_important / either_important if either_important > 0 else 0

    print(f"\n  {label_a} vs {label_b}:")
    print(f"    Important in A:         {a.sum():4d} ({100*a.mean():.1f}%)")
    print(f"    Important in B:         {b.sum():4d} ({100*b.mean():.1f}%)")
    print(f"    Important in BOTH:      {both_important:4d}")
    print(f"    Important in EITHER:    {either_important:4d}")
    print(f"    Only in A:              {only_a:4d}")
    print(f"    Only in B:              {only_b:4d}")
    print(f"    Jaccard similarity:     {jaccard:.3f}")

    # Per-layer correlation of importance fractions
    frac_a = a.mean(axis=1)
    frac_b = b.mean(axis=1)
    corr = np.corrcoef(frac_a, frac_b)[0, 1]
    print(f"    Per-layer frac corr:    r={corr:.3f}")

    return {
        'label_a': label_a,
        'label_b': label_b,
        'n_important_a': int(a.sum()),
        'n_important_b': int(b.sum()),
        'n_both': int(both_important),
        'n_either': int(either_important),
        'n_only_a': int(only_a),
        'n_only_b': int(only_b),
        'jaccard': float(jaccard),
        'per_layer_frac_corr': float(corr),
        'frac_per_layer_a': frac_a.tolist(),
        'frac_per_layer_b': frac_b.tolist(),
    }


# =============================================================================
# Figures
# =============================================================================

def plot_importance_fractions(results_dict, output_dir: Path, title: str, filename: str):
    """Line plot: per-layer importance fraction for multiple sweeps."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    for (label, result), color in zip(results_dict.items(), colors):
        fracs = get_importance_fraction_per_layer(result)
        layers = sorted(fracs.keys())
        vals = [fracs[l] * 100 for l in layers]
        ax.plot(layers, vals, marker='o', markersize=4, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Important Chunks (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_importance_heatmap(result, label: str, output_dir: Path, filename: str):
    """Heatmap: layer x chunk_idx colored by importance (binary)."""
    if not HAS_MPL:
        return

    matrix, layers, max_chunks = get_importance_matrix(result)
    if matrix is None:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        matrix.T.astype(float),
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        interpolation='nearest',
        origin='lower',
    )
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Chunk Index', fontsize=12)
    ax.set_title(f'Importance Map — {label}', fontsize=13)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Unimportant', 'Important'])

    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_overlap_heatmap(overlap_matrix, labels, output_dir: Path, filename: str, title: str):
    """Heatmap of pairwise Jaccard similarities."""
    if not HAS_MPL:
        return

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n*1.2), max(4, n)))

    im = ax.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=12)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if overlap_matrix[i, j] > 0.6 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Jaccard Similarity')
    fig.tight_layout()

    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_layer_correlation(overlaps, output_dir: Path, filename: str, title: str):
    """Line plot of per-layer importance fraction for all pairs."""
    if not HAS_MPL or not overlaps:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(overlaps)))

    for overlap, color in zip(overlaps, colors):
        fa = overlap['frac_per_layer_a']
        fb = overlap['frac_per_layer_b']
        layers = range(len(fa))
        label_a = overlap['label_a']
        label_b = overlap['label_b']
        ax.plot(layers, [v*100 for v in fa], color=color, linestyle='-',
                label=label_a, linewidth=1.5)
        ax.plot(layers, [v*100 for v in fb], color=color, linestyle='--',
                label=label_b, linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Important Chunks (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# =============================================================================
# Run analysis group
# =============================================================================

def run_group_analysis(results_dict: dict, group_name: str, output_dir: Path):
    """
    Analyze a group of importance results.
    results_dict: {label: result_dict}
    """
    print(f"\n{'='*60}")
    print(f"GROUP: {group_name}")
    print(f"{'='*60}")

    labels = list(results_dict.keys())
    results_list = list(results_dict.values())
    n = len(labels)

    # Print overall important fraction per sweep
    print("\nOverall importance fractions:")
    for label, result in results_dict.items():
        tot = result['totals']
        print(f"  {label}: {tot['n_important']}/{tot['total_chunks']} "
              f"({100*tot['important_fraction']:.1f}%)")

    # Build importance matrices
    matrices = {}
    for label, result in results_dict.items():
        mat, layers, max_chunks = get_importance_matrix(result)
        if mat is not None:
            matrices[label] = mat

    # Pairwise overlap
    jaccard_matrix = np.zeros((n, n))
    corr_matrix = np.zeros((n, n))
    overlaps = []

    print("\nPairwise overlap:")
    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard_matrix[i, j] = 1.0
                corr_matrix[i, j] = 1.0
                continue
            if i > j:
                # Already computed
                jaccard_matrix[i, j] = jaccard_matrix[j, i]
                corr_matrix[i, j] = corr_matrix[j, i]
                continue
            if labels[i] in matrices and labels[j] in matrices:
                ov = compute_overlap(matrices[labels[i]], matrices[labels[j]],
                                     labels[i], labels[j])
                jaccard_matrix[i, j] = ov['jaccard']
                corr_matrix[i, j] = ov['per_layer_frac_corr']
                overlaps.append(ov)

    # Figures
    safe_name = group_name.replace(' ', '_').replace('/', '_').lower()

    plot_importance_fractions(
        results_dict, output_dir,
        title=f'Per-Layer Importance Fraction — {group_name}',
        filename=f'{safe_name}_importance_fractions.png',
    )

    plot_overlap_heatmap(
        jaccard_matrix, labels, output_dir,
        filename=f'{safe_name}_jaccard_overlap.png',
        title=f'Jaccard Similarity — {group_name}',
    )

    for label, result in results_dict.items():
        safe_label = label.replace(' ', '_').replace('/', '_').lower()
        plot_importance_heatmap(
            result, label, output_dir,
            filename=f'{safe_name}_{safe_label}_heatmap.png',
        )

    return {
        'group_name': group_name,
        'labels': labels,
        'jaccard_matrix': jaccard_matrix.tolist(),
        'corr_matrix': corr_matrix.tolist(),
        'overlaps': overlaps,
    }


# =============================================================================
# Auto-discover sweeps from results/importance/
# =============================================================================

def auto_discover(importance_dir: Path):
    """Find all importance pkl files and group by analysis type."""
    pkls = sorted(glob.glob(str(importance_dir / "importance_*.pkl")))
    print(f"Found {len(pkls)} importance result files:")
    for p in pkls:
        print(f"  {Path(p).name}")
    return pkls


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze importance consistency")
    parser.add_argument('--importance-dir', default='results/importance/',
                        help='Directory with importance pkl files (from exp 20)')
    parser.add_argument('--files', nargs='+',
                        help='Specific importance pkl files to compare')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for --files (same order)')
    parser.add_argument('--group-name', default='Custom',
                        help='Name for this comparison group')
    parser.add_argument('--output-dir', default='figures/importance_consistency/',
                        help='Output directory for figures')
    parser.add_argument('--run-all', action='store_true',
                        help='Auto-run all three consistency analyses')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        # Manual mode: compare specified files
        labels = args.labels if args.labels else [Path(f).stem for f in args.files]
        results_dict = {}
        for label, filepath in zip(labels, args.files):
            print(f"Loading: {filepath}")
            results_dict[label] = load_importance(filepath)

        run_group_analysis(results_dict, args.group_name, output_dir)

    elif args.run_all:
        # Auto mode: look for standard importance files
        importance_dir = Path(args.importance_dir)
        pkls = auto_discover(importance_dir)
        if not pkls:
            print(f"No importance pkl files found in {importance_dir}")
            print("Run experiment 20 first for each sweep directory.")
            return

        # Load all
        all_results = {}
        for p in pkls:
            name = Path(p).stem
            all_results[name] = load_importance(p)

        # Group 1: Cross-dataset (Llama-2 MLP, different datasets)
        cross_dataset = {k: v for k, v in all_results.items()
                        if 'llama-2' in k.lower() or 'llama2' in k.lower()}
        if len(cross_dataset) >= 2:
            run_group_analysis(cross_dataset, 'Cross-Dataset (Llama-2 MLP)', output_dir)

        # Group 2: Cross-matrix (same dataset)
        cross_matrix = {k: v for k, v in all_results.items()
                       if 'arc' in k.lower() or 'challenge' in k.lower()}
        if len(cross_matrix) >= 2:
            run_group_analysis(cross_matrix, 'Cross-Matrix (ARC-500)', output_dir)

        # Group 3: Cross-model
        cross_model = {k: v for k, v in all_results.items()
                      if 'llama' in k.lower()}
        if len(cross_model) >= 2:
            run_group_analysis(cross_model, 'Cross-Model', output_dir)

    else:
        # Just discover and print
        importance_dir = Path(args.importance_dir)
        pkls = auto_discover(importance_dir)
        if pkls:
            print("\nTo run analysis, use:")
            print("  python scripts/analyze_importance_consistency.py --run-all")
            print("  python scripts/analyze_importance_consistency.py --files <file1> <file2> --labels A B")


if __name__ == '__main__':
    main()
