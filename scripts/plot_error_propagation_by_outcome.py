"""
Plot error propagation by outcome group (Experiment 27).

Generates:
  1. Heatmap of cosine similarity (removed_layer × measured_layer) for each group
  2. Heatmap of relative L2 distance (same layout)
  3. Line plots: divergence trajectory for selected removal layers, grouped by outcome
  4. Summary: flip rate and group sizes per removal layer

Usage:
    python scripts/plot_error_propagation_by_outcome.py \
        --csv results/error_propagation_by_outcome/error_prop_by_outcome_Llama-2-7b-chat-hf_chunk0.csv \
        --output-dir figures/error_propagation_by_outcome/
"""

import sys
sys.path.append('.')

import csv
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# =============================================================================
# Load data
# =============================================================================

def load_csv(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['removed_from_layer'] = int(r['removed_from_layer'])
        r['measured_at_layer'] = int(r['measured_at_layer'])
        r['layers_after_removal'] = int(r['layers_after_removal'])
        r['n_questions'] = int(r['n_questions'])
        r['mean_cos_sim'] = float(r['mean_cos_sim'])
        r['std_cos_sim'] = float(r['std_cos_sim'])
        r['mean_rel_l2'] = float(r['mean_rel_l2'])
        r['std_rel_l2'] = float(r['std_rel_l2'])
    return rows


def build_matrix(rows, group, metric):
    """Build a 32x32 matrix (removed_layer × measured_layer) for a given group+metric."""
    removal_layers = sorted(set(r['removed_from_layer'] for r in rows))
    measure_layers = sorted(set(r['measured_at_layer'] for r in rows))
    n_rem = max(removal_layers) + 1
    n_meas = max(measure_layers) + 1

    mat = np.full((n_rem, n_meas), np.nan)
    for r in rows:
        if r['group'] == group:
            mat[r['removed_from_layer'], r['measured_at_layer']] = r[metric]
    return mat, removal_layers, measure_layers


# =============================================================================
# Plot 1: Heatmaps
# =============================================================================

def plot_heatmaps(rows, output_dir, groups=None):
    """One heatmap per group for cosine sim and relative L2."""
    if groups is None:
        groups = ['all', 'baseline_correct', 'baseline_incorrect', 'flipped', 'not_flipped']

    for metric, label, cmap, vmin, vmax in [
        ('mean_cos_sim', 'Cosine Similarity', 'RdYlGn', None, 1.0),
        ('mean_rel_l2', 'Relative L2 Distance', 'YlOrRd', 0, None),
    ]:
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5),
                                 constrained_layout=True)
        if len(groups) == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            mat, rem_layers, meas_layers = build_matrix(rows, group, metric)

            # Auto vmin/vmax from data if not set
            valid = mat[~np.isnan(mat)]
            _vmin = vmin if vmin is not None else np.percentile(valid, 2)
            _vmax = vmax if vmax is not None else np.percentile(valid, 98)

            im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                           vmin=_vmin, vmax=_vmax, interpolation='nearest')
            ax.set_xlabel('Measured at Layer')
            ax.set_ylabel('Removed from Layer')
            # Get n_questions for this group
            ns = [r['n_questions'] for r in rows
                  if r['group'] == group and r['measured_at_layer'] == 0]
            n_str = f" (n varies)" if len(set(ns)) > 1 else f" (n={ns[0]})" if ns else ""
            ax.set_title(f'{group}{n_str}', fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Draw diagonal line
            ax.plot([0, 31], [0, 31], 'k--', alpha=0.3, linewidth=0.5)

        fig.suptitle(f'{label} — Chunk 0 Removal', fontsize=13, fontweight='bold')
        fname = f"heatmap_{metric.replace('mean_', '')}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# =============================================================================
# Plot 2: Line plots — divergence trajectory for selected removal layers
# =============================================================================

def plot_trajectories(rows, output_dir):
    """For selected removal layers, show how divergence evolves across measurement layers,
    with separate lines per outcome group."""
    selected_layers = [0, 2, 5, 8, 12, 16, 20, 25, 31]
    groups = ['all', 'baseline_correct', 'baseline_incorrect', 'flipped', 'not_flipped']
    colors = {
        'all': '#333333',
        'baseline_correct': '#4A90D9',
        'baseline_incorrect': '#D94F4F',
        'flipped': '#E8A838',
        'not_flipped': '#7BC47F',
    }

    for metric, ylabel in [
        ('mean_cos_sim', 'Cosine Similarity'),
        ('mean_rel_l2', 'Relative L2 Distance'),
    ]:
        n_plots = len(selected_layers)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten()

        for idx, rem_layer in enumerate(selected_layers):
            ax = axes_flat[idx]
            for group in groups:
                subset = [r for r in rows
                          if r['removed_from_layer'] == rem_layer and r['group'] == group]
                if not subset:
                    continue
                subset.sort(key=lambda r: r['measured_at_layer'])
                x = [r['measured_at_layer'] for r in subset]
                y = [r[metric] for r in subset]
                std_key = metric.replace('mean_', 'std_')
                std = [r[std_key] for r in subset]

                label = group if idx == 0 else None
                ax.plot(x, y, color=colors[group], label=label, linewidth=1.5,
                        alpha=0.9)
                ax.fill_between(x,
                                np.array(y) - np.array(std),
                                np.array(y) + np.array(std),
                                color=colors[group], alpha=0.1)

            ax.axvline(rem_layer, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.set_title(f'Remove L{rem_layer}', fontsize=10)
            ax.set_xlabel('Measured Layer')
            if idx % ncols == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, 31)

        # Hide unused axes
        for idx in range(len(selected_layers), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Legend on first axis
        axes_flat[0].legend(fontsize=7, loc='best')

        fig.suptitle(f'{ylabel} Trajectory — Chunk 0 Removal', fontsize=13,
                     fontweight='bold')
        fname = f"trajectories_{metric.replace('mean_', '')}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# =============================================================================
# Plot 2b: Single-line trajectories (all questions only)
# =============================================================================

def plot_trajectories_all_only(rows, output_dir):
    """Same as trajectories but only the 'all' group — one clean line per subplot."""
    selected_layers = [0, 2, 5, 8, 12, 16, 20, 25, 31]

    for metric, ylabel in [
        ('mean_cos_sim', 'Cosine Similarity'),
        ('mean_rel_l2', 'Relative L2 Distance'),
    ]:
        n_plots = len(selected_layers)
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 constrained_layout=True)
        axes_flat = axes.flatten()

        for idx, rem_layer in enumerate(selected_layers):
            ax = axes_flat[idx]
            subset = [r for r in rows
                      if r['removed_from_layer'] == rem_layer and r['group'] == 'all']
            if not subset:
                continue
            subset.sort(key=lambda r: r['measured_at_layer'])
            x = [r['measured_at_layer'] for r in subset]
            y = [r[metric] for r in subset]
            std_key = metric.replace('mean_', 'std_')
            std = [r[std_key] for r in subset]

            ax.plot(x, y, color='#333333', linewidth=1.5)
            ax.fill_between(x,
                            np.array(y) - np.array(std),
                            np.array(y) + np.array(std),
                            color='#333333', alpha=0.15)

            ax.axvline(rem_layer, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.set_title(f'Remove L{rem_layer} (n={subset[0]["n_questions"]})', fontsize=10)
            ax.set_xlabel('Measured Layer')
            if idx % ncols == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, 31)

        for idx in range(len(selected_layers), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f'{ylabel} Trajectory (All Questions) — Chunk 0 Removal',
                     fontsize=13, fontweight='bold')
        fname = f"trajectories_all_{metric.replace('mean_', '')}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# =============================================================================
# Plot 3: Divergence at fixed offsets from removal layer
# =============================================================================

def plot_divergence_at_offsets(rows, output_dir):
    """For each removal layer, plot divergence at the removal layer itself,
    +1, +5, +10 layers later, and at the final layer. Separate lines per group."""
    groups = ['all', 'baseline_correct', 'baseline_incorrect', 'flipped', 'not_flipped']
    colors = {
        'all': '#333333',
        'baseline_correct': '#4A90D9',
        'baseline_incorrect': '#D94F4F',
        'flipped': '#E8A838',
        'not_flipped': '#7BC47F',
    }
    offsets = [0, 1, 5, 10]
    offset_labels = ['At removal layer', '+1 layer', '+5 layers', '+10 layers']

    # Also add "at final layer (31)"
    for metric, ylabel in [
        ('mean_cos_sim', 'Cosine Similarity'),
        ('mean_rel_l2', 'Relative L2 Distance'),
    ]:
        fig, axes = plt.subplots(1, len(offsets) + 1, figsize=(4 * (len(offsets) + 1), 4),
                                 constrained_layout=True)

        all_axes = list(axes)
        all_offsets = offsets + [None]  # None = final layer
        all_labels = offset_labels + ['At final layer (31)']

        for ax, offset, olabel in zip(all_axes, all_offsets, all_labels):
            for group in groups:
                removal_layers = sorted(set(r['removed_from_layer'] for r in rows))
                x_vals = []
                y_vals = []
                for rem_l in removal_layers:
                    if offset is not None:
                        meas_l = rem_l + offset
                    else:
                        meas_l = 31
                    if meas_l > 31:
                        continue
                    match = [r for r in rows
                             if r['removed_from_layer'] == rem_l
                             and r['measured_at_layer'] == meas_l
                             and r['group'] == group]
                    if match:
                        x_vals.append(rem_l)
                        y_vals.append(match[0][metric])

                if x_vals:
                    label = group if ax == all_axes[0] else None
                    ax.plot(x_vals, y_vals, color=colors[group], label=label,
                            linewidth=1.5, marker='.', markersize=3)

            ax.set_xlabel('Removal Layer')
            ax.set_title(olabel, fontsize=10)
            if ax == all_axes[0]:
                ax.set_ylabel(ylabel)

        all_axes[0].legend(fontsize=7, loc='best')
        fig.suptitle(f'{ylabel} at Fixed Offsets — Chunk 0', fontsize=13, fontweight='bold')
        fname = f"offsets_{metric.replace('mean_', '')}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# =============================================================================
# Plot 4: Flip rate + group sizes per removal layer
# =============================================================================

def plot_flip_summary(rows, output_dir):
    """Bar chart of flip rate and group sizes per removal layer."""
    removal_layers = sorted(set(r['removed_from_layer'] for r in rows))

    flip_counts = []
    total_counts = []
    correct_counts = []
    incorrect_counts = []

    for rem_l in removal_layers:
        all_row = [r for r in rows if r['removed_from_layer'] == rem_l
                   and r['group'] == 'all' and r['measured_at_layer'] == 0]
        flip_row = [r for r in rows if r['removed_from_layer'] == rem_l
                    and r['group'] == 'flipped' and r['measured_at_layer'] == 0]
        corr_row = [r for r in rows if r['removed_from_layer'] == rem_l
                    and r['group'] == 'baseline_correct' and r['measured_at_layer'] == 0]
        incorr_row = [r for r in rows if r['removed_from_layer'] == rem_l
                      and r['group'] == 'baseline_incorrect' and r['measured_at_layer'] == 0]

        total = all_row[0]['n_questions'] if all_row else 0
        flipped = flip_row[0]['n_questions'] if flip_row else 0
        correct = corr_row[0]['n_questions'] if corr_row else 0
        incorrect = incorr_row[0]['n_questions'] if incorr_row else 0

        total_counts.append(total)
        flip_counts.append(flipped)
        correct_counts.append(correct)
        incorrect_counts.append(incorrect)

    fig, ax1 = plt.subplots(figsize=(12, 4), constrained_layout=True)

    x = np.array(removal_layers)
    flip_rate = np.array(flip_counts) / np.array(total_counts) * 100

    ax1.bar(x, flip_rate, color='#E8A838', alpha=0.8, label='Flip rate (%)')
    ax1.set_xlabel('Removal Layer')
    ax1.set_ylabel('Flip Rate (%)', color='#E8A838')
    ax1.tick_params(axis='y', labelcolor='#E8A838')
    ax1.set_xlim(-0.5, 31.5)

    ax2 = ax1.twinx()
    ax2.plot(x, flip_counts, 'o-', color='#D94F4F', markersize=4, label='Flipped (n)')
    ax2.set_ylabel('Number of Flipped Questions', color='#D94F4F')
    ax2.tick_params(axis='y', labelcolor='#D94F4F')

    fig.suptitle('Answer Flip Rate by Removal Layer — Chunk 0', fontsize=13,
                 fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=9)

    fname = "flip_rate_by_layer.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 5: Flipped vs not-flipped comparison — key insight plot
# =============================================================================

def plot_flipped_vs_not(rows, output_dir):
    """Direct comparison: flipped vs not_flipped divergence at the final layer,
    as a function of removal layer."""
    groups = ['flipped', 'not_flipped']
    colors = {'flipped': '#E8A838', 'not_flipped': '#7BC47F'}

    for metric, ylabel in [
        ('mean_cos_sim', 'Cosine Similarity at Final Layer'),
        ('mean_rel_l2', 'Relative L2 at Final Layer'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        removal_layers = sorted(set(r['removed_from_layer'] for r in rows))

        for group in groups:
            x_vals = []
            y_vals = []
            y_err = []
            for rem_l in removal_layers:
                match = [r for r in rows
                         if r['removed_from_layer'] == rem_l
                         and r['measured_at_layer'] == 31
                         and r['group'] == group]
                if match:
                    x_vals.append(rem_l)
                    y_vals.append(match[0][metric])
                    y_err.append(match[0][metric.replace('mean_', 'std_')])

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_err, color=colors[group],
                            label=group, linewidth=1.5, marker='o', markersize=4,
                            capsize=2, alpha=0.8)

        ax.set_xlabel('Removal Layer')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, 31.5)
        ax.legend(fontsize=9)
        ax.set_title(f'{ylabel} — Flipped vs Not Flipped', fontsize=12, fontweight='bold')

        fname = f"flipped_vs_not_{metric.replace('mean_', '')}_final.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures/error_propagation_by_outcome/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv}...")
    rows = load_csv(args.csv)
    print(f"  {len(rows)} rows loaded")

    print("\nGenerating plots...")
    plot_heatmaps(rows, output_dir)
    plot_trajectories(rows, output_dir)
    plot_trajectories_all_only(rows, output_dir)
    plot_divergence_at_offsets(rows, output_dir)
    plot_flip_summary(rows, output_dir)
    plot_flipped_vs_not(rows, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
