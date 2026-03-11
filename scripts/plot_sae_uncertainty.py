"""
Plot SAE uncertainty feature analysis (Experiment 28).

Usage:
    python scripts/plot_sae_uncertainty.py \
        --diff-csv results/sae_uncertainty/differential_features_Llama-3.1-8B.csv \
        --entropy-csv results/sae_uncertainty/entropy_features_Llama-3.1-8B.csv \
        --pkl results/sae_uncertainty/sae_uncertainty_Llama-3.1-8B.pkl \
        --output-dir figures/sae_uncertainty/
"""

import sys
sys.path.append('.')

import csv
import pickle
import numpy as np
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k in ('layer', 'rank', 'feature_idx'):
                r[k] = int(r[k])
            elif k != 'direction':
                r[k] = float(r[k])
    return rows


# =============================================================================
# Plot 1: Number of significant features per layer
# =============================================================================

def plot_sig_features_per_layer(diff_rows, entropy_rows, output_dir):
    layers = sorted(set(r['layer'] for r in diff_rows))

    diff_unc = []
    diff_conf = []
    ent_high = []
    ent_low = []

    for layer in layers:
        dr = [r for r in diff_rows if r['layer'] == layer and r['p_value'] < 0.05]
        diff_unc.append(len([r for r in dr if r['direction'] == 'uncertainty']))
        diff_conf.append(len([r for r in dr if r['direction'] == 'confidence']))

        er = [r for r in entropy_rows if r['layer'] == layer and r['p_value'] < 0.05]
        ent_high.append(len([r for r in er if r['direction'] == 'high_entropy']))
        ent_low.append(len([r for r in er if r['direction'] == 'low_entropy']))

    x = np.array(layers)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax1.bar(x - width/2, diff_unc, width, label='Uncertainty (more on incorrect)',
            color='#D94F4F', alpha=0.8)
    ax1.bar(x + width/2, diff_conf, width, label='Confidence (more on correct)',
            color='#4A90D9', alpha=0.8)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Significant Features (p<0.05)')
    ax1.set_title('Correct vs Incorrect Split')
    ax1.legend(fontsize=9)
    ax1.set_xticks(x)

    ax2.bar(x - width/2, ent_high, width, label='High entropy features',
            color='#E8A838', alpha=0.8)
    ax2.bar(x + width/2, ent_low, width, label='Low entropy features',
            color='#7BC47F', alpha=0.8)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Significant Features (p<0.05)')
    ax2.set_title('High vs Low Entropy Split')
    ax2.legend(fontsize=9)
    ax2.set_xticks(x)

    fig.suptitle('Differential SAE Features by Layer', fontsize=13, fontweight='bold')
    fname = 'sig_features_per_layer.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 2: Top feature effect sizes per layer
# =============================================================================

def plot_top_effect_sizes(diff_rows, output_dir):
    layers = sorted(set(r['layer'] for r in diff_rows))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for layer in layers:
        lr = [r for r in diff_rows if r['layer'] == layer and r['p_value'] < 0.05]
        lr.sort(key=lambda r: abs(r['effect_size']), reverse=True)
        top_n = lr[:10]

        for r in top_n:
            color = '#D94F4F' if r['direction'] == 'uncertainty' else '#4A90D9'
            alpha = min(1.0, 0.3 + 0.7 * (1 - r['p_value']))
            ax.scatter(layer, r['effect_size'], color=color, alpha=alpha, s=30, zorder=3)

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Effect Size (positive = more active on incorrect)')
    ax.set_title('Top 10 Significant Feature Effect Sizes per Layer', fontsize=12,
                 fontweight='bold')
    ax.set_xticks(layers)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D94F4F',
               markersize=8, label='Uncertainty feature'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90D9',
               markersize=8, label='Confidence feature'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fname = 'top_effect_sizes.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 3: Feature frequency comparison (top features)
# =============================================================================

def plot_freq_comparison(diff_rows, output_dir):
    layers = sorted(set(r['layer'] for r in diff_rows))

    # Select layers with strong signal
    selected = [l for l in layers if l >= 16]

    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4),
                             constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, layer in zip(axes, selected):
        lr = [r for r in diff_rows if r['layer'] == layer and r['p_value'] < 0.05]
        lr.sort(key=lambda r: abs(r['effect_size']), reverse=True)
        top = lr[:15]

        for r in top:
            color = '#D94F4F' if r['direction'] == 'uncertainty' else '#4A90D9'
            ax.scatter(r['correct_freq'], r['incorrect_freq'], color=color,
                       s=40, alpha=0.7, zorder=3)

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Frequency on Correct')
        ax.set_ylabel('Frequency on Incorrect')
        ax.set_title(f'Layer {layer}', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

    fig.suptitle('Feature Activation Frequency: Correct vs Incorrect',
                 fontsize=13, fontweight='bold')
    fname = 'freq_comparison.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 4: Overlap between correctness and entropy features
# =============================================================================

def plot_feature_overlap(diff_rows, entropy_rows, output_dir):
    layers = sorted(set(r['layer'] for r in diff_rows))

    diff_counts = []
    ent_counts = []
    overlap_counts = []

    for layer in layers:
        diff_sig = set(r['feature_idx'] for r in diff_rows
                       if r['layer'] == layer and r['p_value'] < 0.05)
        ent_sig = set(r['feature_idx'] for r in entropy_rows
                      if r['layer'] == layer and r['p_value'] < 0.05)
        overlap = diff_sig & ent_sig
        diff_counts.append(len(diff_sig))
        ent_counts.append(len(ent_sig))
        overlap_counts.append(len(overlap))

    x = np.array(layers)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    ax.bar(x - 0.25, diff_counts, 0.25, label='Correct/Incorrect sig.', color='#4A90D9',
           alpha=0.7)
    ax.bar(x, ent_counts, 0.25, label='High/Low Entropy sig.', color='#E8A838',
           alpha=0.7)
    ax.bar(x + 0.25, overlap_counts, 0.25, label='Overlap', color='#333333',
           alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Overlap: Correctness vs Entropy Splits', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(x)

    fname = 'feature_overlap.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 5: Per-question entropy vs number of active features
# =============================================================================

def plot_entropy_vs_active_features(pkl_data, output_dir):
    per_q = pkl_data['per_question_data']
    layers = sorted(pkl_data['config']['sae_layers'])

    selected = [0, 12, 20, 31]
    selected = [l for l in selected if l in layers]

    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4),
                             constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]

    for ax, layer in zip(axes, selected):
        entropies = []
        n_active = []
        colors = []
        for d in per_q:
            if layer in d['sae_features']:
                entropies.append(d['entropy'])
                n_active.append(d['sae_features'][layer]['n_active'])
                colors.append('#4A90D9' if d['correct'] else '#D94F4F')

        ax.scatter(entropies, n_active, c=colors, s=10, alpha=0.5)
        ax.set_xlabel('Entropy')
        ax.set_ylabel('# Active SAE Features')
        ax.set_title(f'Layer {layer}', fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90D9',
               markersize=6, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D94F4F',
               markersize=6, label='Incorrect'),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=8, loc='upper right')

    fig.suptitle('Entropy vs Active SAE Features per Question',
                 fontsize=13, fontweight='bold')
    fname = 'entropy_vs_active_features.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Plot 6: Max effect size trajectory across layers
# =============================================================================

def plot_max_effect_trajectory(diff_rows, entropy_rows, output_dir):
    layers = sorted(set(r['layer'] for r in diff_rows))

    # Max absolute effect size per layer for each analysis
    diff_max_unc = []
    diff_max_conf = []
    ent_max = []

    for layer in layers:
        dr = [r for r in diff_rows if r['layer'] == layer and r['p_value'] < 0.05]
        unc = [r['effect_size'] for r in dr if r['direction'] == 'uncertainty']
        conf = [abs(r['effect_size']) for r in dr if r['direction'] == 'confidence']
        diff_max_unc.append(max(unc) if unc else 0)
        diff_max_conf.append(max(conf) if conf else 0)

        er = [r for r in entropy_rows if r['layer'] == layer and r['p_value'] < 0.05]
        e_abs = [abs(r['effect_size']) for r in er]
        ent_max.append(max(e_abs) if e_abs else 0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(layers, diff_max_unc, 'o-', color='#D94F4F', label='Max uncertainty effect',
            linewidth=2, markersize=6)
    ax.plot(layers, diff_max_conf, 's-', color='#4A90D9', label='Max confidence effect',
            linewidth=2, markersize=6)
    ax.plot(layers, ent_max, '^-', color='#E8A838', label='Max entropy effect',
            linewidth=2, markersize=6)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Max |Effect Size|')
    ax.set_title('Peak Differential Feature Effect Size by Layer', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(layers)

    fname = 'max_effect_trajectory.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff-csv', type=str, required=True)
    parser.add_argument('--entropy-csv', type=str, required=True)
    parser.add_argument('--pkl', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures/sae_uncertainty/')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    diff_rows = load_csv(args.diff_csv)
    entropy_rows = load_csv(args.entropy_csv)
    with open(args.pkl, 'rb') as f:
        pkl_data = pickle.load(f)

    print(f"  Differential: {len(diff_rows)} rows")
    print(f"  Entropy: {len(entropy_rows)} rows")
    print(f"  Per-question: {len(pkl_data['per_question_data'])} questions")
    print(f"  Accuracy: {pkl_data['summary']['accuracy']*100:.1f}%")

    print("\nGenerating plots...")
    plot_sig_features_per_layer(diff_rows, entropy_rows, output_dir)
    plot_top_effect_sizes(diff_rows, output_dir)
    plot_freq_comparison(diff_rows, output_dir)
    plot_feature_overlap(diff_rows, entropy_rows, output_dir)
    plot_entropy_vs_active_features(pkl_data, output_dir)
    plot_max_effect_trajectory(diff_rows, entropy_rows, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
