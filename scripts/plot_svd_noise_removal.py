"""
Plot SVD Noise Removal experiment results (experiment 14).

Paper-ready figures for chunk-based SVD noise removal analysis:
1. Combined entropy + accuracy change by chunk (dual bar chart)
2. Entropy vs accuracy scatter (4-quadrant classification)
3. Classification summary (horizontal bar)
4. Cumulative noise removal (stacked subplots)
5. Energy profile with classification overlay

Auto-detects whether MCQ, QA, or both evaluation modes were used.

Usage:
    python scripts/plot_svd_noise_removal.py --results results/svd_noise_removal/.../results.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# Paper-ready style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})

# Consistent color palette
COLORS = {
    'true_noise': '#2ca02c',       # green
    'true_signal': '#1f77b4',      # blue
    'confident_wrong': '#d62728',  # red
    'uncertain_right': '#9467bd',  # purple
    'entropy': '#1f77b4',          # blue
    'accuracy': '#2ca02c',         # green
    'energy': '#ff7f0e',           # orange
    'neutral': '#7f7f7f',          # gray
}

CATEGORY_LABELS = {
    'true_noise': 'True Noise',
    'confident_wrong': 'Confident Wrong',
    'true_signal': 'True Signal',
    'uncertain_right': 'Uncertain Right',
}

CATEGORY_DESCRIPTIONS = {
    'true_noise': r'Entropy$\downarrow$ Accuracy$\uparrow$  (removing helps)',
    'confident_wrong': r'Entropy$\downarrow$ Accuracy$\downarrow$',
    'true_signal': r'Entropy$\uparrow$ Accuracy$\downarrow$  (removing hurts)',
    'uncertain_right': r'Entropy$\uparrow$ Accuracy$\uparrow$',
}


def load_results(results_path: str):
    """Load results from pickle file."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def detect_eval_modes(results: dict):
    """Detect which evaluation modes are present."""
    return 'baseline_mcq' in results, 'baseline_qa' in results


def _get_title_info(config):
    """Extract common title info."""
    model = config['model_name'].split('/')[-1]
    layer = config['target_layer']
    matrix = '+'.join(config['matrix_types'])
    chunk = config['chunk_size']
    return model, layer, matrix, chunk


def _classify_chunk(idx, rankings, ranking_prefix=""):
    """Return classification key for a chunk index."""
    for cat in ['true_noise', 'confident_wrong', 'true_signal', 'uncertain_right']:
        if idx in set(rankings.get(f'{ranking_prefix}{cat}', [])):
            return cat
    return None


def plot_chunk_profile(chunk_results, rankings, baseline, config, output_dir,
                       prefix="mcq_", mode_label="MCQ", ranking_prefix=""):
    """
    Plot 1: Combined entropy change + accuracy change by chunk with classification coloring.
    Two subplots stacked vertically sharing the x-axis.
    """
    model, layer, matrix, chunk_size = _get_title_info(config)

    chunks = [c['chunk_idx'] for c in chunk_results]
    entropy_changes = [c[f'{prefix}entropy_change'] for c in chunk_results]
    accuracy_changes = [c[f'{prefix}accuracy_change'] * 100 for c in chunk_results]
    energy = [c['avg_energy_removed'] * 100 for c in chunk_results]

    # Classify each chunk
    classifications = [_classify_chunk(c['chunk_idx'], rankings, ranking_prefix) for c in chunk_results]
    bar_colors = [COLORS.get(cl, COLORS['neutral']) for cl in classifications]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'hspace': 0.08})

    # Top: Entropy change
    ax1.bar(chunks, entropy_changes, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel(r'$\Delta$ Entropy')
    ax1.set_title(f'{model} Layer {layer} ({matrix}, chunk={chunk_size}) - {mode_label} Per-Chunk Analysis')

    # Bottom: Accuracy change
    ax2.bar(chunks, accuracy_changes, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel(r'$\Delta$ Accuracy (pp)')
    ax2.set_xlabel('Chunk Index (SVD component range)')

    # Shared legend placed outside the plot area (below title, above top subplot)
    legend_handles = [
        mpatches.Patch(color=COLORS['true_noise'], label=f'True Noise ({CATEGORY_DESCRIPTIONS["true_noise"]})'),
        mpatches.Patch(color=COLORS['confident_wrong'], label=f'Confident Wrong ({CATEGORY_DESCRIPTIONS["confident_wrong"]})'),
        mpatches.Patch(color=COLORS['true_signal'], label=f'True Signal ({CATEGORY_DESCRIPTIONS["true_signal"]})'),
        mpatches.Patch(color=COLORS['uncertain_right'], label=f'Uncertain Right ({CATEGORY_DESCRIPTIONS["uncertain_right"]})'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', fontsize=8, ncol=2,
               framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0.5, 0.89))

    # Subtle tick labels showing SVD ranges
    tick_step = max(1, len(chunks) // 10)
    ax2.set_xticks(chunks[::tick_step])
    ax2.set_xticklabels([f'{c}\n[{c*chunk_size}:{min((c+1)*chunk_size, config["total_components"])}]'
                         for c in chunks[::tick_step]], fontsize=8)

    fig.subplots_adjust(hspace=0.08, left=0.08, right=0.97, top=0.92, bottom=0.12)
    fname = f'{prefix}chunk_profile.png'
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / fname}")
    plt.close()


def plot_scatter(chunk_results, rankings, baseline, config, output_dir,
                 prefix="mcq_", mode_label="MCQ", ranking_prefix=""):
    """
    Plot 2: Entropy change vs accuracy change scatter.
    Points colored and sized by energy, with clear quadrant shading.
    """
    model, layer, matrix, chunk_size = _get_title_info(config)

    entropy_changes = [c[f'{prefix}entropy_change'] for c in chunk_results]
    accuracy_changes = [c[f'{prefix}accuracy_change'] * 100 for c in chunk_results]
    energy = [c['avg_energy_removed'] * 100 for c in chunk_results]
    classifications = [_classify_chunk(c['chunk_idx'], rankings, ranking_prefix) for c in chunk_results]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Shaded quadrants
    xlim_pad = max(abs(min(entropy_changes)), abs(max(entropy_changes))) * 1.3
    ylim_pad = max(abs(min(accuracy_changes)), abs(max(accuracy_changes))) * 1.3
    ax.axhspan(0, ylim_pad * 2, xmin=0, xmax=0.5, alpha=0.04, color=COLORS['true_noise'])    # top-left
    ax.axhspan(-ylim_pad * 2, 0, xmin=0, xmax=0.5, alpha=0.04, color=COLORS['confident_wrong'])  # bottom-left
    ax.axhspan(0, ylim_pad * 2, xmin=0.5, xmax=1, alpha=0.04, color=COLORS['uncertain_right'])  # top-right
    ax.axhspan(-ylim_pad * 2, 0, xmin=0.5, xmax=1, alpha=0.04, color=COLORS['true_signal'])  # bottom-right

    # Scatter points
    for i, c in enumerate(chunk_results):
        cl = classifications[i]
        color = COLORS.get(cl, COLORS['neutral'])
        size = 30 + energy[i] * 15  # bigger = more energy
        ax.scatter(entropy_changes[i], accuracy_changes[i], c=color, s=size,
                   alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
        ax.annotate(str(c['chunk_idx']), (entropy_changes[i], accuracy_changes[i]),
                    fontsize=7, ha='center', va='center', color='white', fontweight='bold', zorder=4)

    # Axis lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.6)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.6)

    # Quadrant labels
    pad = 0.03
    ax.text(pad, 1 - pad, 'Uncertain Right', transform=ax.transAxes, fontsize=9,
            va='top', ha='left', color=COLORS['uncertain_right'], fontstyle='italic')
    ax.text(1 - pad, 1 - pad, 'True Signal\n(removing hurts)', transform=ax.transAxes, fontsize=9,
            va='top', ha='right', color=COLORS['true_signal'], fontstyle='italic')
    ax.text(pad, pad, 'True Noise\n(removing helps)', transform=ax.transAxes, fontsize=9,
            va='bottom', ha='left', color=COLORS['true_noise'], fontstyle='italic')
    ax.text(1 - pad, pad, 'Confident Wrong', transform=ax.transAxes, fontsize=9,
            va='bottom', ha='right', color=COLORS['confident_wrong'], fontstyle='italic')

    baseline_ent = baseline.get('avg_entropy', 0)
    baseline_acc = baseline.get('accuracy', 0)

    ax.set_xlabel(r'$\Delta$ Entropy (when chunk removed)')
    ax.set_ylabel(r'$\Delta$ Accuracy (pp, when chunk removed)')
    ax.set_title(f'{model} Layer {layer} - {mode_label} Chunk Classification\n'
                 f'Baseline: H={baseline_ent:.4f}, Acc={baseline_acc*100:.1f}%  |  '
                 f'Point size $\\propto$ energy')
    ax.set_xlim(-xlim_pad, xlim_pad)
    ax.set_ylim(-ylim_pad, ylim_pad)

    plt.tight_layout()
    fname = f'{prefix}scatter.png'
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / fname}")
    plt.close()


def plot_classification_bar(rankings, config, output_dir,
                            prefix="mcq_", mode_label="MCQ", ranking_prefix=""):
    """
    Plot 3: Horizontal bar chart showing classification counts.
    Cleaner alternative to pie chart.
    """
    model, layer, matrix, chunk_size = _get_title_info(config)

    categories = ['true_noise', 'confident_wrong', 'true_signal', 'uncertain_right']
    counts = [len(rankings.get(f'{ranking_prefix}{cat}', [])) for cat in categories]
    total = sum(counts)
    # Labels with description on second line
    labels = [f'{CATEGORY_LABELS[cat]}' for cat in categories]
    colors = [COLORS[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(9, 4))

    y_pos = range(len(categories))
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.85, edgecolor='white', height=0.6)

    # Add count + percentage labels at end of bars
    for bar, count in zip(bars, counts):
        pct = count / total * 100 if total > 0 else 0
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.0f}%)', va='center', fontsize=10, fontweight='bold')

    # Y-axis: category name + description
    ax.set_yticks(y_pos)
    ylabels = []
    for cat in categories:
        desc = CATEGORY_DESCRIPTIONS[cat].replace(r'Entropy$\downarrow$', 'Entropy\u2193').replace(
            r'Entropy$\uparrow$', 'Entropy\u2191').replace(r'Accuracy$\uparrow$', 'Accuracy\u2191').replace(
            r'Accuracy$\downarrow$', 'Accuracy\u2193')
        ylabels.append(f'{CATEGORY_LABELS[cat]}\n{desc}')
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_xlabel('Number of Chunks')
    ax.set_title(f'{model} Layer {layer} - {mode_label} Classification ({total} chunks)')
    ax.set_xlim(0, max(counts) * 1.3 if max(counts) > 0 else 1)
    ax.invert_yaxis()

    sns.despine(left=True)
    plt.tight_layout()

    fname = f'{prefix}classification.png'
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / fname}")
    plt.close()


def plot_cumulative(cumulative_results, config, output_dir, has_mcq, has_qa):
    """
    Plot 4: Cumulative noise removal - two stacked subplots.
    Top: accuracy change as colored bars.
    Bottom: entropy change as a line with markers.
    """
    if not cumulative_results:
        print("No cumulative results to plot.")
        return

    model, layer, matrix, chunk_size = _get_title_info(config)

    steps = [cr['step'] + 1 for cr in cumulative_results]

    # Pick which mode
    if has_mcq and 'mcq_entropy_change' in cumulative_results[0]:
        ent_data = [cr['mcq_entropy_change'] for cr in cumulative_results]
        acc_data = [cr['mcq_accuracy_change'] * 100 for cr in cumulative_results]
        mode = "MCQ"
    elif has_qa and 'qa_entropy_change' in cumulative_results[0]:
        ent_data = [cr['qa_entropy_change'] for cr in cumulative_results]
        acc_data = [cr['qa_accuracy_change'] * 100 for cr in cumulative_results]
        mode = "QA"
    else:
        print("No entropy/accuracy data in cumulative results.")
        return

    fig, (ax_acc, ax_ent) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                          gridspec_kw={'hspace': 0.08})

    # --- Top subplot: Accuracy as bars ---
    acc_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in acc_data]
    ax_acc.bar(steps, acc_data, color=acc_colors, alpha=0.65, width=0.7, edgecolor='none')
    ax_acc.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_acc.set_ylabel(r'$\Delta$ Accuracy (pp)')
    ax_acc.set_title(f'{model} Layer {layer} ({matrix}) - {mode} Cumulative Noise Removal\n'
                     f'{len(cumulative_results)} noise chunks removed progressively')

    # Annotate peak accuracy
    best_acc_idx = int(np.argmax(acc_data))
    ax_acc.annotate(f'+{acc_data[best_acc_idx]:.1f}pp',
                    xy=(steps[best_acc_idx], acc_data[best_acc_idx]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=9, ha='center', fontweight='bold', color='#2ca02c',
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1))

    # --- Bottom subplot: Entropy as line ---
    ax_ent.plot(steps, ent_data, 's-', color='#1f77b4', linewidth=2.2,
                markersize=5, markerfacecolor='white', markeredgewidth=1.5,
                markeredgecolor='#1f77b4', zorder=3)
    ax_ent.fill_between(steps, ent_data, 0, alpha=0.15, color='#1f77b4')
    ax_ent.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_ent.set_ylabel(r'$\Delta$ Entropy')
    ax_ent.set_xlabel('Noise Chunks Removed (cumulative)')

    # Annotate lowest entropy
    best_ent_idx = int(np.argmin(ent_data))
    ax_ent.annotate(f'{ent_data[best_ent_idx]:.4f}',
                    xy=(steps[best_ent_idx], ent_data[best_ent_idx]),
                    xytext=(0, -14), textcoords='offset points',
                    fontsize=9, ha='center', fontweight='bold', color='#1f77b4',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1))

    fig.subplots_adjust(hspace=0.08, left=0.08, right=0.97, top=0.90, bottom=0.10)
    fig.savefig(output_dir / 'cumulative_noise_removal.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_noise_removal.png'}")
    plt.close()


def plot_energy_classification(chunk_results, rankings, config, output_dir, ranking_prefix=""):
    """
    Plot 5: Energy profile colored by classification.
    Shows where in the SVD spectrum the noise vs signal lives.
    """
    model, layer, matrix, chunk_size = _get_title_info(config)

    chunks = [c['chunk_idx'] for c in chunk_results]
    energy = [c['avg_energy_removed'] * 100 for c in chunk_results]
    classifications = [_classify_chunk(c['chunk_idx'], rankings, ranking_prefix) for c in chunk_results]
    bar_colors = [COLORS.get(cl, COLORS['neutral']) for cl in classifications]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(chunks, energy, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.3)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS[cat], label=CATEGORY_LABELS[cat])
        for cat in ['true_noise', 'confident_wrong', 'true_signal', 'uncertain_right']
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_xlabel('Chunk Index (low = top singular values, high = tail)')
    ax.set_ylabel('Energy in Chunk (%)')
    ax.set_title(f'{model} Layer {layer} ({matrix}) - SVD Energy Spectrum by Classification')

    sns.despine()
    plt.tight_layout()
    fig.savefig(output_dir / 'energy_classification.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'energy_classification.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot SVD Noise Removal experiment results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results_dir/figures/)")

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output) if args.output else results_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    config = results['config']
    chunk_results = results['chunk_results']
    rankings = results['rankings']
    cumulative_results = results.get('cumulative_results', [])

    has_mcq, has_qa = detect_eval_modes(results)
    print(f"Evaluation modes: MCQ={has_mcq}, QA={has_qa}")
    print(f"Chunks: {len(chunk_results)}, Cumulative steps: {len(cumulative_results)}")

    # MCQ plots
    if has_mcq:
        baseline_mcq = results['baseline_mcq']
        plot_chunk_profile(chunk_results, rankings, baseline_mcq, config, output_dir,
                           prefix="mcq_", mode_label="MCQ", ranking_prefix="")
        plot_scatter(chunk_results, rankings, baseline_mcq, config, output_dir,
                     prefix="mcq_", mode_label="MCQ", ranking_prefix="")
        plot_classification_bar(rankings, config, output_dir,
                                prefix="mcq_", mode_label="MCQ", ranking_prefix="")

    # QA plots
    if has_qa:
        baseline_qa = results['baseline_qa']
        plot_chunk_profile(chunk_results, rankings, baseline_qa, config, output_dir,
                           prefix="qa_", mode_label="QA", ranking_prefix="qa_")
        plot_scatter(chunk_results, rankings, baseline_qa, config, output_dir,
                     prefix="qa_", mode_label="QA", ranking_prefix="qa_")
        plot_classification_bar(rankings, config, output_dir,
                                prefix="qa_", mode_label="QA", ranking_prefix="qa_")

    # Cumulative noise removal
    plot_cumulative(cumulative_results, config, output_dir, has_mcq, has_qa)

    # Energy profile with classification (use MCQ rankings if available, else QA)
    rp = "" if has_mcq else "qa_"
    plot_energy_classification(chunk_results, rankings, config, output_dir, ranking_prefix=rp)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
