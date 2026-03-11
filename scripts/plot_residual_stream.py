"""
Plot Residual Stream Analysis results.

Generates layer-wise profiles showing how much MLP and Attention blocks
contribute to the residual stream at each layer.

Plots:
1. Cosine similarity profile (MLP vs Attention across layers)
2. Norm ratio profile (MLP vs Attention across layers)
3. Correct vs Incorrect comparison (cosine sim)
4. Correct vs Incorrect comparison (norm ratio)
5. Absolute norms across layers (residual, MLP, attention)
6. All-tokens vs Last-token comparison

Usage:
    python scripts/plot_residual_stream.py --results results/residual_stream/.../results.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path: str):
    """Load results from pickle file."""
    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_layer_profile(aggregated, block_key, metric_key, num_layers):
    """
    Extract a per-layer array of a given metric from aggregated results.

    Returns:
        means: np.array of shape [num_layers]
        stds: np.array of shape [num_layers]
    """
    means = []
    stds = []
    for layer_idx in range(num_layers):
        m = aggregated.get(layer_idx, {}).get(block_key, {}).get(metric_key, {})
        means.append(m.get('mean', np.nan))
        stds.append(m.get('std', np.nan))
    return np.array(means), np.array(stds)


def plot_residual_stream_results(results: dict, output_dir: Path):
    """Generate all residual stream analysis plots."""
    output_dir = Path(output_dir) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = results['config']
    aggregated_all = results['aggregated_all']
    aggregated_correct = results.get('aggregated_correct')
    aggregated_incorrect = results.get('aggregated_incorrect')
    summary = results['summary']

    model_name = config['model_name'].split('/')[-1]
    num_layers = config['num_layers']
    layers = np.arange(num_layers)

    # =========================================================================
    # Plot 1: Cosine Similarity - MLP vs Attention (last token)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    mlp_cos, mlp_cos_std = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'cosine_sim', num_layers)
    attn_cos, attn_cos_std = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'cosine_sim', num_layers)

    ax.plot(layers, mlp_cos, 'b-o', linewidth=2, markersize=4, label='MLP')
    ax.fill_between(layers, mlp_cos - mlp_cos_std, mlp_cos + mlp_cos_std,
                    alpha=0.15, color='blue')
    ax.plot(layers, attn_cos, 'r-s', linewidth=2, markersize=4, label='Attention')
    ax.fill_between(layers, attn_cos - attn_cos_std, attn_cos + attn_cos_std,
                    alpha=0.15, color='red')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity (x_before vs x_after)', fontsize=12)
    ax.set_title(f'{model_name}: Residual Stream Cosine Similarity (Last Token)\n'
                 f'cos(x_before, x_after) — closer to 1.0 = smaller directional change',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_similarity_profile.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cosine_similarity_profile.png'}")
    plt.close()

    # =========================================================================
    # Plot 2: Norm Ratio - MLP vs Attention (last token)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    mlp_nr, mlp_nr_std = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'norm_ratio', num_layers)
    attn_nr, attn_nr_std = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'norm_ratio', num_layers)

    ax.plot(layers, mlp_nr, 'b-o', linewidth=2, markersize=4, label='MLP')
    ax.fill_between(layers, mlp_nr - mlp_nr_std, mlp_nr + mlp_nr_std,
                    alpha=0.15, color='blue')
    ax.plot(layers, attn_nr, 'r-s', linewidth=2, markersize=4, label='Attention')
    ax.fill_between(layers, attn_nr - attn_nr_std, attn_nr + attn_nr_std,
                    alpha=0.15, color='red')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Norm Ratio (||block_output|| / ||residual||)', fontsize=12)
    ax.set_title(f'{model_name}: Block Output Relative Magnitude (Last Token)\n'
                 f'Higher = block contributes more relative to residual stream',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'norm_ratio_profile.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'norm_ratio_profile.png'}")
    plt.close()

    # =========================================================================
    # Plot 3: Correct vs Incorrect - Cosine Similarity (MLP, last token)
    # =========================================================================
    if aggregated_correct and aggregated_incorrect:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # MLP
        ax = axes[0]
        corr_cos, corr_std = extract_layer_profile(
            aggregated_correct, 'mlp_last_token', 'cosine_sim', num_layers)
        incorr_cos, incorr_std = extract_layer_profile(
            aggregated_incorrect, 'mlp_last_token', 'cosine_sim', num_layers)

        ax.plot(layers, corr_cos, 'g-o', linewidth=2, markersize=3,
                label=f'Correct (n={summary["num_correct"]})')
        ax.fill_between(layers, corr_cos - corr_std, corr_cos + corr_std,
                        alpha=0.15, color='green')
        ax.plot(layers, incorr_cos, 'r-s', linewidth=2, markersize=3,
                label=f'Incorrect (n={summary["num_incorrect"]})')
        ax.fill_between(layers, incorr_cos - incorr_std, incorr_cos + incorr_std,
                        alpha=0.15, color='red')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('MLP: Correct vs Incorrect', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_layers - 1)

        # Attention
        ax = axes[1]
        corr_cos, corr_std = extract_layer_profile(
            aggregated_correct, 'attn_last_token', 'cosine_sim', num_layers)
        incorr_cos, incorr_std = extract_layer_profile(
            aggregated_incorrect, 'attn_last_token', 'cosine_sim', num_layers)

        ax.plot(layers, corr_cos, 'g-o', linewidth=2, markersize=3,
                label=f'Correct (n={summary["num_correct"]})')
        ax.fill_between(layers, corr_cos - corr_std, corr_cos + corr_std,
                        alpha=0.15, color='green')
        ax.plot(layers, incorr_cos, 'r-s', linewidth=2, markersize=3,
                label=f'Incorrect (n={summary["num_incorrect"]})')
        ax.fill_between(layers, incorr_cos - incorr_std, incorr_cos + incorr_std,
                        alpha=0.15, color='red')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Attention: Correct vs Incorrect', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_layers - 1)

        fig.suptitle(f'{model_name}: Cosine Similarity — Correct vs Incorrect (Last Token)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'correct_vs_incorrect_cosine.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'correct_vs_incorrect_cosine.png'}")
        plt.close()

    # =========================================================================
    # Plot 4: Correct vs Incorrect - Norm Ratio (MLP, last token)
    # =========================================================================
    if aggregated_correct and aggregated_incorrect:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # MLP
        ax = axes[0]
        corr_nr, corr_std = extract_layer_profile(
            aggregated_correct, 'mlp_last_token', 'norm_ratio', num_layers)
        incorr_nr, incorr_std = extract_layer_profile(
            aggregated_incorrect, 'mlp_last_token', 'norm_ratio', num_layers)

        ax.plot(layers, corr_nr, 'g-o', linewidth=2, markersize=3,
                label=f'Correct (n={summary["num_correct"]})')
        ax.fill_between(layers, corr_nr - corr_std, corr_nr + corr_std,
                        alpha=0.15, color='green')
        ax.plot(layers, incorr_nr, 'r-s', linewidth=2, markersize=3,
                label=f'Incorrect (n={summary["num_incorrect"]})')
        ax.fill_between(layers, incorr_nr - incorr_std, incorr_nr + incorr_std,
                        alpha=0.15, color='red')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Norm Ratio', fontsize=12)
        ax.set_title('MLP: Correct vs Incorrect', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_layers - 1)

        # Attention
        ax = axes[1]
        corr_nr, corr_std = extract_layer_profile(
            aggregated_correct, 'attn_last_token', 'norm_ratio', num_layers)
        incorr_nr, incorr_std = extract_layer_profile(
            aggregated_incorrect, 'attn_last_token', 'norm_ratio', num_layers)

        ax.plot(layers, corr_nr, 'g-o', linewidth=2, markersize=3,
                label=f'Correct (n={summary["num_correct"]})')
        ax.fill_between(layers, corr_nr - corr_std, corr_nr + corr_std,
                        alpha=0.15, color='green')
        ax.plot(layers, incorr_nr, 'r-s', linewidth=2, markersize=3,
                label=f'Incorrect (n={summary["num_incorrect"]})')
        ax.fill_between(layers, incorr_nr - incorr_std, incorr_nr + incorr_std,
                        alpha=0.15, color='red')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Norm Ratio', fontsize=12)
        ax.set_title('Attention: Correct vs Incorrect', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, num_layers - 1)

        fig.suptitle(f'{model_name}: Norm Ratio — Correct vs Incorrect (Last Token)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'correct_vs_incorrect_norm_ratio.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'correct_vs_incorrect_norm_ratio.png'}")
        plt.close()

    # =========================================================================
    # Plot 5: Absolute Norms (residual, MLP output, attention output)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    mlp_bn, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'block_norm', num_layers)
    attn_bn, _ = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'block_norm', num_layers)
    resid_norm, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'residual_norm', num_layers)

    ax.plot(layers, resid_norm, 'k-^', linewidth=2, markersize=4, label='Residual Stream')
    ax.plot(layers, mlp_bn, 'b-o', linewidth=2, markersize=4, label='MLP Output')
    ax.plot(layers, attn_bn, 'r-s', linewidth=2, markersize=4, label='Attention Output')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title(f'{model_name}: Component Norms Across Layers (Last Token)\n'
                 f'Scale of residual stream vs block outputs',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'absolute_norms.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'absolute_norms.png'}")
    plt.close()

    # =========================================================================
    # Plot 6: Last Token vs All Tokens comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Cosine sim comparison
    ax = axes[0]
    mlp_last, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'cosine_sim', num_layers)
    mlp_all, _ = extract_layer_profile(
        aggregated_all, 'mlp_all_tokens', 'cosine_sim', num_layers)
    attn_last, _ = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'cosine_sim', num_layers)
    attn_all, _ = extract_layer_profile(
        aggregated_all, 'attn_all_tokens', 'cosine_sim', num_layers)

    ax.plot(layers, mlp_last, 'b-o', linewidth=2, markersize=3, label='MLP (last token)')
    ax.plot(layers, mlp_all, 'b--', linewidth=1.5, alpha=0.6, label='MLP (all tokens)')
    ax.plot(layers, attn_last, 'r-s', linewidth=2, markersize=3, label='Attn (last token)')
    ax.plot(layers, attn_all, 'r--', linewidth=1.5, alpha=0.6, label='Attn (all tokens)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    # Norm ratio comparison
    ax = axes[1]
    mlp_last, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'norm_ratio', num_layers)
    mlp_all, _ = extract_layer_profile(
        aggregated_all, 'mlp_all_tokens', 'norm_ratio', num_layers)
    attn_last, _ = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'norm_ratio', num_layers)
    attn_all, _ = extract_layer_profile(
        aggregated_all, 'attn_all_tokens', 'norm_ratio', num_layers)

    ax.plot(layers, mlp_last, 'b-o', linewidth=2, markersize=3, label='MLP (last token)')
    ax.plot(layers, mlp_all, 'b--', linewidth=1.5, alpha=0.6, label='MLP (all tokens)')
    ax.plot(layers, attn_last, 'r-s', linewidth=2, markersize=3, label='Attn (last token)')
    ax.plot(layers, attn_all, 'r--', linewidth=1.5, alpha=0.6, label='Attn (all tokens)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Norm Ratio', fontsize=12)
    ax.set_title('Norm Ratio', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    fig.suptitle(f'{model_name}: Last Token vs All Tokens Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'last_vs_all_tokens.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'last_vs_all_tokens.png'}")
    plt.close()

    # =========================================================================
    # Print text summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {summary['num_samples']} "
          f"(correct: {summary['num_correct']}, incorrect: {summary['num_incorrect']})")
    print(f"Accuracy: {summary['accuracy']*100:.1f}%")

    mlp_cos_last, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'cosine_sim', num_layers)
    attn_cos_last, _ = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'cosine_sim', num_layers)
    mlp_nr_last, _ = extract_layer_profile(
        aggregated_all, 'mlp_last_token', 'norm_ratio', num_layers)
    attn_nr_last, _ = extract_layer_profile(
        aggregated_all, 'attn_last_token', 'norm_ratio', num_layers)

    print(f"\nMLP cosine similarity (last token):  "
          f"min={np.nanmin(mlp_cos_last):.4f} (layer {np.nanargmin(mlp_cos_last)}), "
          f"max={np.nanmax(mlp_cos_last):.4f} (layer {np.nanargmax(mlp_cos_last)}), "
          f"final layer={mlp_cos_last[-1]:.4f}")
    print(f"Attn cosine similarity (last token): "
          f"min={np.nanmin(attn_cos_last):.4f} (layer {np.nanargmin(attn_cos_last)}), "
          f"max={np.nanmax(attn_cos_last):.4f} (layer {np.nanargmax(attn_cos_last)}), "
          f"final layer={attn_cos_last[-1]:.4f}")
    print(f"\nMLP norm ratio (last token):  "
          f"min={np.nanmin(mlp_nr_last):.4f} (layer {np.nanargmin(mlp_nr_last)}), "
          f"max={np.nanmax(mlp_nr_last):.4f} (layer {np.nanargmax(mlp_nr_last)}), "
          f"final layer={mlp_nr_last[-1]:.4f}")
    print(f"Attn norm ratio (last token): "
          f"min={np.nanmin(attn_nr_last):.4f} (layer {np.nanargmin(attn_nr_last)}), "
          f"max={np.nanmax(attn_nr_last):.4f} (layer {np.nanargmax(attn_nr_last)}), "
          f"final layer={attn_nr_last[-1]:.4f}")

    print(f"\nFigures saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Plot residual stream analysis results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as results)")

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output) if args.output else results_path.parent

    results = load_results(results_path)
    plot_residual_stream_results(results, output_dir)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
