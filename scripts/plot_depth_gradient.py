"""Plot SAE feature depth gradient: effect sizes and counts by layer."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures/')
    parser.add_argument('--prefix', type=str, default='depth_gradient')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(args.pickle, 'rb'))
    p = data['config']['p_threshold']
    layers = sorted(data['all_comparisons'].keys())

    # Collect stats per layer
    top_unc, top_inc, mean_unc, mean_inc = [], [], [], []
    n_pure_unc, n_pure_inc, n_both = [], [], []

    for l in layers:
        comp = data['all_comparisons'][l]

        sig_unc = {r['feature_idx'] for r in comp['pure_uncertainty']
                   if r['p_value'] < p and r['effect_size'] > 0}
        sig_inc = {r['feature_idx'] for r in comp['pure_incorrectness']
                   if r['p_value'] < p and r['effect_size'] > 0}

        n_pure_unc.append(len(sig_unc - sig_inc))
        n_pure_inc.append(len(sig_inc - sig_unc))
        n_both.append(len(sig_unc & sig_inc))

        unc_sig = [r for r in comp['pure_uncertainty']
                   if r['p_value'] < p and r['effect_size'] > 0]
        inc_sig = [r for r in comp['pure_incorrectness']
                   if r['p_value'] < p and r['effect_size'] > 0]

        unc_sig.sort(key=lambda x: x['effect_size'], reverse=True)
        inc_sig.sort(key=lambda x: x['effect_size'], reverse=True)

        top_unc.append(unc_sig[0]['effect_size'] if unc_sig else 0)
        top_inc.append(inc_sig[0]['effect_size'] if inc_sig else 0)
        mean_unc.append(np.mean([r['effect_size'] for r in unc_sig]) if unc_sig else 0)
        mean_inc.append(np.mean([r['effect_size'] for r in inc_sig]) if inc_sig else 0)

    x = np.arange(len(layers))
    layer_labels = [str(l) for l in layers]

    # --- Figure 1: Effect sizes by layer ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, top_unc, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='C vs A (uncertainty)')
    ax1.plot(x, top_inc, 's-', color='#3498db', linewidth=2, markersize=8, label='B vs A (incorrectness)')
    ax1.set_ylabel('Top Effect Size (Cohen\'s d)', fontsize=12)
    ax1.set_title('Peak Effect Size by Layer', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, mean_unc, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='C vs A (uncertainty)')
    ax2.plot(x, mean_inc, 's-', color='#3498db', linewidth=2, markersize=8, label='B vs A (incorrectness)')
    ax2.set_ylabel('Mean Effect Size (Cohen\'s d)', fontsize=12)
    ax2.set_title('Mean Effect Size of Significant Features by Layer', fontsize=14)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_effect_sizes.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_effect_sizes.png'}")
    plt.close()

    # --- Figure 2: Feature counts by category ---
    fig, ax = plt.subplots(figsize=(10, 5))

    width = 0.25
    ax.bar(x - width, n_pure_unc, width, color='#e74c3c', label='Pure uncertainty', alpha=0.85)
    ax.bar(x, n_both, width, color='#9b59b6', label='Both', alpha=0.85)
    ax.bar(x + width, n_pure_inc, width, color='#3498db', label='Pure incorrectness', alpha=0.85)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of Significant Features', fontsize=12)
    ax.set_title('Feature Classification by Layer (p < 0.05)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i in range(len(layers)):
        if n_pure_unc[i] > 0:
            ax.text(i - width, n_pure_unc[i] + 0.5, str(n_pure_unc[i]), ha='center', fontsize=9)
        if n_both[i] > 0:
            ax.text(i, n_both[i] + 0.5, str(n_both[i]), ha='center', fontsize=9)
        if n_pure_inc[i] > 0:
            ax.text(i + width, n_pure_inc[i] + 0.5, str(n_pure_inc[i]), ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f'{args.prefix}_feature_counts.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'{args.prefix}_feature_counts.png'}")
    plt.close()


if __name__ == '__main__':
    main()
