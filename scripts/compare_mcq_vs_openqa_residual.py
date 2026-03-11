"""
Compare residual stream metrics between MCQ and Open-ended QA tasks.

This script loads results from both experiment 11 (MCQ) and experiment 12 (open QA)
and generates comparison plots to test the hypothesis that Layer 31 MLP's large
contribution is specifically for generation tasks.

Usage:
    python scripts/compare_mcq_vs_openqa_residual.py \
        --mcq results/residual_stream/Llama-2-7b-chat-hf_20260101_120000/results.pkl \
        --openqa results/residual_stream_open_qa/Llama-2-7b-chat-hf_nq_open_20260101_120000/results.pkl \
        --output figures/mcq_vs_openqa_comparison
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_layer_profile(aggregated, block_key, metric_key, num_layers):
    """Extract per-layer array of a given metric."""
    means = []
    stds = []
    for layer_idx in range(num_layers):
        m = aggregated.get(layer_idx, {}).get(block_key, {}).get(metric_key, {})
        means.append(m.get('mean', np.nan))
        stds.append(m.get('std', np.nan))
    return np.array(means), np.array(stds)


def plot_comparison(mcq_results, openqa_results, output_dir):
    """
    Generate comparison plots between MCQ and open-ended QA.

    Figures:
        1. MLP Cosine Similarity: MCQ vs Open QA (last token)
        2. MLP Norm Ratio: MCQ vs Open QA (last token)
        3. Attention Cosine Similarity: MCQ vs Open QA (last token)
        4. Attention Norm Ratio: MCQ vs Open QA (last token)
        5. Absolute Norms: MLP block output (MCQ vs Open QA)
        6. Absolute Norms: Residual stream before MLP (MCQ vs Open QA)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_layers_mcq = mcq_results['config']['num_layers']
    num_layers_openqa = openqa_results['config']['num_layers']

    if num_layers_mcq != num_layers_openqa:
        print(f"WARNING: Different number of layers (MCQ: {num_layers_mcq}, Open QA: {num_layers_openqa})")
        num_layers = min(num_layers_mcq, num_layers_openqa)
    else:
        num_layers = num_layers_mcq

    layers = np.arange(num_layers)

    agg_mcq = mcq_results['aggregated_all']
    agg_openqa = openqa_results['aggregated_all']

    # Extract metrics
    # MLP metrics
    mlp_cos_mcq, mlp_cos_mcq_std = extract_layer_profile(agg_mcq, 'mlp_last_token', 'cosine_sim', num_layers)
    mlp_cos_openqa, mlp_cos_openqa_std = extract_layer_profile(agg_openqa, 'mlp_last_token', 'cosine_sim', num_layers)

    mlp_norm_ratio_mcq, mlp_norm_ratio_mcq_std = extract_layer_profile(agg_mcq, 'mlp_last_token', 'norm_ratio', num_layers)
    mlp_norm_ratio_openqa, mlp_norm_ratio_openqa_std = extract_layer_profile(agg_openqa, 'mlp_last_token', 'norm_ratio', num_layers)

    mlp_block_norm_mcq, _ = extract_layer_profile(agg_mcq, 'mlp_last_token', 'block_norm', num_layers)
    mlp_block_norm_openqa, _ = extract_layer_profile(agg_openqa, 'mlp_last_token', 'block_norm', num_layers)

    mlp_resid_norm_mcq, _ = extract_layer_profile(agg_mcq, 'mlp_last_token', 'residual_norm', num_layers)
    mlp_resid_norm_openqa, _ = extract_layer_profile(agg_openqa, 'mlp_last_token', 'residual_norm', num_layers)

    # Attention metrics
    attn_cos_mcq, attn_cos_mcq_std = extract_layer_profile(agg_mcq, 'attn_last_token', 'cosine_sim', num_layers)
    attn_cos_openqa, attn_cos_openqa_std = extract_layer_profile(agg_openqa, 'attn_last_token', 'cosine_sim', num_layers)

    attn_norm_ratio_mcq, attn_norm_ratio_mcq_std = extract_layer_profile(agg_mcq, 'attn_last_token', 'norm_ratio', num_layers)
    attn_norm_ratio_openqa, attn_norm_ratio_openqa_std = extract_layer_profile(agg_openqa, 'attn_last_token', 'norm_ratio', num_layers)

    attn_block_norm_mcq, _ = extract_layer_profile(agg_mcq, 'attn_last_token', 'block_norm', num_layers)
    attn_block_norm_openqa, _ = extract_layer_profile(agg_openqa, 'attn_last_token', 'block_norm', num_layers)

    attn_resid_norm_mcq, _ = extract_layer_profile(agg_mcq, 'attn_last_token', 'residual_norm', num_layers)
    attn_resid_norm_openqa, _ = extract_layer_profile(agg_openqa, 'attn_last_token', 'residual_norm', num_layers)

    model_name_mcq = mcq_results['config']['model_name'].split('/')[-1]
    model_name_openqa = openqa_results['config']['model_name'].split('/')[-1]

    # ========== Figure 1: MLP Cosine Similarity ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, mlp_cos_mcq, 'o-', label='MCQ', color='blue', linewidth=2, markersize=4)
    plt.fill_between(layers, mlp_cos_mcq - mlp_cos_mcq_std, mlp_cos_mcq + mlp_cos_mcq_std,
                     alpha=0.2, color='blue')
    plt.plot(layers, mlp_cos_openqa, 's-', label='Open QA', color='red', linewidth=2, markersize=4)
    plt.fill_between(layers, mlp_cos_openqa - mlp_cos_openqa_std, mlp_cos_openqa + mlp_cos_openqa_std,
                     alpha=0.2, color='red')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Cosine Similarity (x_before vs x_after)', fontsize=12)
    plt.title(f'MLP Cosine Similarity: MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mlp_cosine_similarity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mlp_cosine_similarity_comparison.png'}")
    plt.close()

    # ========== Figure 2: MLP Norm Ratio ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, mlp_norm_ratio_mcq, 'o-', label='MCQ', color='blue', linewidth=2, markersize=4)
    plt.fill_between(layers, mlp_norm_ratio_mcq - mlp_norm_ratio_mcq_std,
                     mlp_norm_ratio_mcq + mlp_norm_ratio_mcq_std, alpha=0.2, color='blue')
    plt.plot(layers, mlp_norm_ratio_openqa, 's-', label='Open QA', color='red', linewidth=2, markersize=4)
    plt.fill_between(layers, mlp_norm_ratio_openqa - mlp_norm_ratio_openqa_std,
                     mlp_norm_ratio_openqa + mlp_norm_ratio_openqa_std, alpha=0.2, color='red')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Norm Ratio (||block_output|| / ||x_before||)', fontsize=12)
    plt.title(f'MLP Norm Ratio: MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mlp_norm_ratio_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mlp_norm_ratio_comparison.png'}")
    plt.close()

    # ========== Figure 3: Attention Cosine Similarity ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, attn_cos_mcq, 'o-', label='MCQ', color='blue', linewidth=2, markersize=4)
    plt.fill_between(layers, attn_cos_mcq - attn_cos_mcq_std, attn_cos_mcq + attn_cos_mcq_std,
                     alpha=0.2, color='blue')
    plt.plot(layers, attn_cos_openqa, 's-', label='Open QA', color='red', linewidth=2, markersize=4)
    plt.fill_between(layers, attn_cos_openqa - attn_cos_openqa_std, attn_cos_openqa + attn_cos_openqa_std,
                     alpha=0.2, color='red')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Cosine Similarity (x_before vs x_after)', fontsize=12)
    plt.title(f'Attention Cosine Similarity: MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attn_cosine_similarity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attn_cosine_similarity_comparison.png'}")
    plt.close()

    # ========== Figure 4: Attention Norm Ratio ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, attn_norm_ratio_mcq, 'o-', label='MCQ', color='blue', linewidth=2, markersize=4)
    plt.fill_between(layers, attn_norm_ratio_mcq - attn_norm_ratio_mcq_std,
                     attn_norm_ratio_mcq + attn_norm_ratio_mcq_std, alpha=0.2, color='blue')
    plt.plot(layers, attn_norm_ratio_openqa, 's-', label='Open QA', color='red', linewidth=2, markersize=4)
    plt.fill_between(layers, attn_norm_ratio_openqa - attn_norm_ratio_openqa_std,
                     attn_norm_ratio_openqa + attn_norm_ratio_openqa_std, alpha=0.2, color='red')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Norm Ratio (||block_output|| / ||x_before||)', fontsize=12)
    plt.title(f'Attention Norm Ratio: MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attn_norm_ratio_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attn_norm_ratio_comparison.png'}")
    plt.close()

    # ========== Figure 5: Absolute Norms - MLP Block Output ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, mlp_block_norm_mcq, 'o-', label='MCQ', color='blue', linewidth=2, markersize=4)
    plt.plot(layers, mlp_block_norm_openqa, 's-', label='Open QA', color='red', linewidth=2, markersize=4)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('||block_output||', fontsize=12)
    plt.title(f'MLP Block Output Norm: MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mlp_block_norm_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mlp_block_norm_comparison.png'}")
    plt.close()

    # ========== Figure 6: Absolute Norms - Residual Stream ==========
    plt.figure(figsize=(12, 6))
    plt.plot(layers, mlp_resid_norm_mcq, 'o-', label='MCQ (before MLP)', color='blue', linewidth=2, markersize=4)
    plt.plot(layers, mlp_resid_norm_openqa, 's-', label='Open QA (before MLP)', color='red', linewidth=2, markersize=4)

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('||x_before||', fontsize=12)
    plt.title(f'Residual Stream Norm (before MLP): MCQ vs Open QA\n{model_name_mcq}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_norm_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'residual_norm_comparison.png'}")
    plt.close()

    # ========== Summary Statistics ==========
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\nMCQ Results:")
    print(f"  Model: {mcq_results['config']['model_name']}")
    print(f"  Samples: {mcq_results['summary']['num_samples']}")
    print(f"  Accuracy: {mcq_results['summary']['accuracy']*100:.1f}%")

    print(f"\nOpen QA Results:")
    print(f"  Model: {openqa_results['config']['model_name']}")
    print(f"  Dataset: {openqa_results['config']['dataset_name']}")
    print(f"  Samples: {openqa_results['summary']['num_samples']}")
    print(f"  Accuracy: {openqa_results['summary']['accuracy']*100:.1f}%")

    # Focus on Layer 31 (or last layer)
    last_layer = num_layers - 1
    print(f"\n--- Layer {last_layer} MLP (Last Token) ---")
    print(f"{'Metric':<20} {'MCQ':>12} {'Open QA':>12} {'Difference':>12}")
    print("-" * 60)

    cos_diff = mlp_cos_openqa[last_layer] - mlp_cos_mcq[last_layer]
    print(f"{'Cosine Similarity':<20} {mlp_cos_mcq[last_layer]:>12.4f} {mlp_cos_openqa[last_layer]:>12.4f} {cos_diff:>12.4f}")

    norm_ratio_diff = mlp_norm_ratio_openqa[last_layer] - mlp_norm_ratio_mcq[last_layer]
    print(f"{'Norm Ratio':<20} {mlp_norm_ratio_mcq[last_layer]:>12.4f} {mlp_norm_ratio_openqa[last_layer]:>12.4f} {norm_ratio_diff:>12.4f}")

    block_norm_diff = mlp_block_norm_openqa[last_layer] - mlp_block_norm_mcq[last_layer]
    print(f"{'Block Norm':<20} {mlp_block_norm_mcq[last_layer]:>12.2f} {mlp_block_norm_openqa[last_layer]:>12.2f} {block_norm_diff:>12.2f}")

    resid_norm_diff = mlp_resid_norm_openqa[last_layer] - mlp_resid_norm_mcq[last_layer]
    print(f"{'Residual Norm':<20} {mlp_resid_norm_mcq[last_layer]:>12.2f} {mlp_resid_norm_openqa[last_layer]:>12.2f} {resid_norm_diff:>12.2f}")

    print("\n" + "=" * 70)
    print(f"All comparison figures saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare residual stream metrics between MCQ and open-ended QA"
    )
    parser.add_argument("--mcq", type=str, required=True,
                        help="Path to MCQ results.pkl (experiment 11)")
    parser.add_argument("--openqa", type=str, required=True,
                        help="Path to open QA results.pkl (experiment 12)")
    parser.add_argument("--output", type=str, default="figures/mcq_vs_openqa_comparison",
                        help="Output directory for comparison plots")

    args = parser.parse_args()

    print("Loading MCQ results...")
    mcq_results = load_results(args.mcq)

    print("Loading Open QA results...")
    openqa_results = load_results(args.openqa)

    print("\nGenerating comparison plots...")
    plot_comparison(mcq_results, openqa_results, args.output)


if __name__ == "__main__":
    main()
