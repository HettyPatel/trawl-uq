"""
Per-question layer sensitivity analysis.

For each question, computes which layers (and matrix types) cause it to flip
when chunks are removed. Builds a 500x32 sensitivity map and looks for:
1. Do questions cluster by which layers are sensitive?
2. Do correct vs wrong questions have different layer depth profiles?
3. Do MLP and attention sweeps show the same or different routing?
"""

import sys
sys.path.append('.')

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# Load per-question flip data from a sweep directory
# =============================================================================

def load_flip_sensitivity(sweep_dir, baseline_correct=None):
    """
    For each layer and each question, count how many chunk removals cause a flip
    (change in is_correct relative to baseline).

    Returns:
        sample_ids: list of 500 sample IDs (in order)
        baseline_correct: array (500,) bool — baseline correctness
        flip_matrix: array (500, 32) — number of flips per question per layer
        flip_fraction: array (500, 32) — fraction of chunks that cause a flip
        layer_indices: list of layer indices found
    """
    sweep_dir = Path(sweep_dir)
    pkls = sorted(sweep_dir.glob('*_layer*.pkl'))
    if not pkls:
        raise FileNotFoundError(f"No layer pickles in {sweep_dir}")

    # Get sample IDs from first pickle
    with open(pkls[0], 'rb') as f:
        first = pickle.load(f)
    sample_ids = [s['sample_id'] for s in first['chunk_results'][0]['mcq_per_sample']]
    n_questions = len(sample_ids)

    # Get baseline correctness — reconstruct from majority vote if not provided
    if baseline_correct is None:
        vote_correct = np.zeros(n_questions, dtype=int)
        vote_total = 0
        for pkl in pkls:
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            for cr in d['chunk_results']:
                for qi, s in enumerate(cr['mcq_per_sample']):
                    vote_correct[qi] += int(s['is_correct'])
                vote_total += 1
        baseline_correct = vote_correct > (vote_total / 2)

    # Build flip matrix: (n_questions, n_layers)
    layer_indices = []
    flip_counts = {}    # layer -> (n_questions,) int array
    num_chunks = {}     # layer -> int

    for pkl in pkls:
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        layer = d['config']['layer']
        layer_indices.append(layer)

        flips = np.zeros(n_questions, dtype=int)
        n_chunks = len(d['chunk_results'])
        for cr in d['chunk_results']:
            for qi, s in enumerate(cr['mcq_per_sample']):
                chunk_correct = s['is_correct']
                if chunk_correct != baseline_correct[qi]:
                    flips[qi] += 1
        flip_counts[layer] = flips
        num_chunks[layer] = n_chunks

    layer_indices = sorted(layer_indices)
    flip_matrix = np.stack([flip_counts[l] for l in layer_indices], axis=1)  # (500, 32)
    num_chunks_arr = np.array([num_chunks[l] for l in layer_indices])
    flip_fraction = flip_matrix / num_chunks_arr[np.newaxis, :]              # (500, 32)

    return sample_ids, baseline_correct, flip_matrix, flip_fraction, layer_indices


def load_baseline_correct(sweep_dir):
    """Load baseline correctness from baseline.pkl if available."""
    base_pkl = Path(sweep_dir) / 'baseline.pkl'
    if not base_pkl.exists():
        return None
    with open(base_pkl, 'rb') as f:
        b = pickle.load(f)
    if b.get('mcq_per_sample') is None:
        return None
    return np.array([s['is_correct'] for s in b['mcq_per_sample']])


# =============================================================================
# Analysis functions
# =============================================================================

def analyze_layer_depth_profile(flip_fraction, baseline_correct, layer_indices, label):
    """Compare layer sensitivity profiles for correct vs wrong questions."""
    correct = baseline_correct
    wrong = ~baseline_correct

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Correct: {correct.sum()}, Wrong: {wrong.sum()}")

    # Mean flip fraction per layer for each group
    mean_correct = flip_fraction[correct].mean(axis=0)
    mean_wrong = flip_fraction[wrong].mean(axis=0)

    print(f"\n  Per-layer mean flip fraction (correct vs wrong):")
    print(f"  {'Layer':>5}  {'Correct':>8}  {'Wrong':>8}  {'Diff':>8}")
    print(f"  {'-'*35}")
    for i, layer in enumerate(layer_indices):
        diff = mean_wrong[i] - mean_correct[i]
        marker = ' <--' if abs(diff) > 0.02 else ''
        print(f"  {layer:>5}  {mean_correct[i]:>8.3f}  {mean_wrong[i]:>8.3f}  {diff:>+8.3f}{marker}")

    # Overall: do wrong questions concentrate sensitivity in early, mid, or late layers?
    early = list(range(0, 11))
    mid = list(range(11, 22))
    late = list(range(22, 32))

    early_idx = [layer_indices.index(l) for l in early if l in layer_indices]
    mid_idx = [layer_indices.index(l) for l in mid if l in layer_indices]
    late_idx = [layer_indices.index(l) for l in late if l in layer_indices]

    print(f"\n  Layer region sensitivity (mean flip fraction):")
    print(f"  {'Region':>8}  {'Correct':>8}  {'Wrong':>8}  {'Diff':>8}")
    for name, idx in [('Early', early_idx), ('Mid', mid_idx), ('Late', late_idx)]:
        mc = flip_fraction[correct][:, idx].mean()
        mw = flip_fraction[wrong][:, idx].mean()
        print(f"  {name:>8}  {mc:>8.3f}  {mw:>8.3f}  {mw-mc:>+8.3f}")

    # Statistical test per layer
    if HAS_SCIPY:
        print(f"\n  Layers with significant correct vs wrong difference (Mann-Whitney, p<0.05):")
        found = False
        for i, layer in enumerate(layer_indices):
            stat, p = stats.mannwhitneyu(
                flip_fraction[correct, i],
                flip_fraction[wrong, i],
                alternative='two-sided'
            )
            if p < 0.05:
                direction = 'wrong>correct' if flip_fraction[wrong, i].mean() > flip_fraction[correct, i].mean() else 'correct>wrong'
                print(f"    Layer {layer:>2}: p={p:.4f}  {direction}")
                found = True
        if not found:
            print("    None found.")

    return mean_correct, mean_wrong


def analyze_routing_paths(flip_fraction, baseline_correct, layer_indices, label):
    """Find if questions cluster by which layers they're sensitive to."""
    print(f"\n  --- Routing path clusters ({label}) ---")

    # Total sensitivity per question
    total_sens = flip_fraction.sum(axis=1)
    print(f"  Total sensitivity — mean={total_sens.mean():.2f}, "
          f"median={np.median(total_sens):.2f}, "
          f"max={total_sens.max():.2f}")

    # Which layer is peak sensitivity for each question?
    peak_layer = np.array(layer_indices)[flip_fraction.argmax(axis=1)]
    print(f"\n  Peak sensitivity layer distribution:")
    for region, lo, hi in [('Early (0-10)', 0, 10), ('Mid (11-21)', 11, 21), ('Late (22-31)', 22, 31)]:
        count = ((peak_layer >= lo) & (peak_layer <= hi)).sum()
        print(f"    {region}: {count} questions ({count/len(peak_layer)*100:.1f}%)")

    # Correct vs wrong peak layer
    correct = baseline_correct
    wrong = ~baseline_correct
    peak_correct = peak_layer[correct]
    peak_wrong = peak_layer[wrong]
    print(f"\n  Mean peak layer — correct: {peak_correct.mean():.1f}, wrong: {peak_wrong.mean():.1f}")

    if HAS_SCIPY:
        stat, p = stats.mannwhitneyu(peak_correct, peak_wrong, alternative='two-sided')
        print(f"  Mann-Whitney peak layer test: p={p:.4f}")


def cross_sweep_agreement(flip_fraction_mlp, flip_fraction_attn, baseline_correct, layer_indices, label_mlp, label_attn):
    """Check if a question sensitive to a layer in MLP is also sensitive in attention."""
    print(f"\n  --- Cross-sweep routing agreement: {label_mlp} vs {label_attn} ---")

    # Correlate per-question total sensitivity between sweeps
    total_mlp = flip_fraction_mlp.sum(axis=1)
    total_attn = flip_fraction_attn.sum(axis=1)

    if HAS_SCIPY:
        r, p = stats.pearsonr(total_mlp, total_attn)
        print(f"  Total sensitivity correlation: r={r:.3f}, p={p:.4f}")

    # Per-layer correlation
    print(f"\n  Per-layer sensitivity correlation (MLP vs Attn):")
    print(f"  {'Layer':>5}  {'r':>7}  {'p':>8}")
    for i, layer in enumerate(layer_indices):
        if HAS_SCIPY:
            r, p = stats.pearsonr(flip_fraction_mlp[:, i], flip_fraction_attn[:, i])
            marker = ' *' if p < 0.05 else ''
            print(f"  {layer:>5}  {r:>7.3f}  {p:>8.4f}{marker}")


# =============================================================================
# Plotting
# =============================================================================

def plot_sensitivity_maps(flip_fraction, baseline_correct, layer_indices, label, output_dir):
    if not HAS_MPL:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = baseline_correct
    wrong = ~baseline_correct

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Sort questions by total sensitivity for better visualization
    order = np.argsort(flip_fraction.sum(axis=1))[::-1]
    flip_sorted = flip_fraction[order]
    correct_sorted = correct[order]

    # Full heatmap
    im = axes[0].imshow(flip_sorted.T, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest', vmin=0)
    axes[0].set_xlabel('Question (sorted by total sensitivity)', fontsize=10)
    axes[0].set_ylabel('Layer', fontsize=10)
    axes[0].set_title(f'{label}\nFlip Fraction Heatmap', fontsize=11)
    axes[0].set_yticks(range(len(layer_indices)))
    axes[0].set_yticklabels(layer_indices, fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.6).set_label('Flip fraction')

    # Mean per layer: correct vs wrong
    mean_c = flip_fraction[correct].mean(axis=0)
    mean_w = flip_fraction[wrong].mean(axis=0)
    axes[1].plot(mean_c, layer_indices, 'b-o', markersize=4, label=f'Correct (n={correct.sum()})')
    axes[1].plot(mean_w, layer_indices, 'r-o', markersize=4, label=f'Wrong (n={wrong.sum()})')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean flip fraction', fontsize=10)
    axes[1].set_ylabel('Layer', fontsize=10)
    axes[1].set_title('Layer Sensitivity by Correctness', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Peak layer distribution
    peak_layer_idx = flip_fraction.argmax(axis=1)
    peak_layer = np.array(layer_indices)[peak_layer_idx]
    bins = layer_indices + [layer_indices[-1] + 1]
    axes[2].hist(peak_layer[correct], bins=bins, alpha=0.6, color='blue',
                 label=f'Correct (n={correct.sum()})', density=True)
    axes[2].hist(peak_layer[wrong], bins=bins, alpha=0.6, color='red',
                 label=f'Wrong (n={wrong.sum()})', density=True)
    axes[2].set_xlabel('Peak sensitivity layer', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].set_title('Peak Sensitivity Layer Distribution', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Layer Routing Analysis — {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'layer_routing_{label.replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_cross_sweep(flip_fraction_mlp, flip_fraction_attn, baseline_correct,
                     layer_indices, label, output_dir):
    if not HAS_MPL:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = baseline_correct
    wrong = ~baseline_correct

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: total MLP vs total attn sensitivity per question
    total_mlp = flip_fraction_mlp.sum(axis=1)
    total_attn = flip_fraction_attn.sum(axis=1)
    colors = ['blue' if c else 'red' for c in correct]
    axes[0].scatter(total_mlp, total_attn, c=colors, alpha=0.4, s=15)
    axes[0].set_xlabel('Total MLP sensitivity (sum flip fraction)', fontsize=10)
    axes[0].set_ylabel('Total Attn sensitivity (sum flip fraction)', fontsize=10)
    axes[0].set_title('MLP vs Attention Routing\n(blue=correct, red=wrong)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    if HAS_SCIPY:
        r, p = stats.pearsonr(total_mlp, total_attn)
        axes[0].set_title(f'MLP vs Attention Routing\nr={r:.3f}, p={p:.4f}', fontsize=11)

    # Per-layer correlation
    if HAS_SCIPY:
        rs = [stats.pearsonr(flip_fraction_mlp[:, i], flip_fraction_attn[:, i])[0]
              for i in range(len(layer_indices))]
        axes[1].barh(layer_indices, rs, color=['steelblue' if r > 0 else 'tomato' for r in rs])
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_xlabel('Pearson r (MLP vs Attn sensitivity per layer)', fontsize=10)
        axes[1].set_ylabel('Layer', fontsize=10)
        axes[1].set_title('Per-Layer MLP↔Attn Correlation', fontsize=11)
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Cross-Sweep Routing: {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'cross_sweep_routing_{label.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    output_dir = Path('figures/layer_routing')
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Llama-2 MLP sweep ---
    mlp_dir = 'results/full_sweep/Llama-2-7b-chat-hf_paired_mlp_in+mlp_out_chunk100'
    print("Loading Llama-2 MLP sweep...")
    baseline_correct_l2 = load_baseline_correct(mlp_dir)  # None for Llama-2
    sample_ids, baseline_l2, flip_mlp, frac_mlp, layers = load_flip_sensitivity(
        mlp_dir, baseline_correct=baseline_correct_l2
    )
    print(f"  Loaded: {len(sample_ids)} questions, {len(layers)} layers")
    print(f"  Baseline accuracy: {baseline_l2.mean()*100:.1f}%")

    mean_c, mean_w = analyze_layer_depth_profile(frac_mlp, baseline_l2, layers, 'Llama-2 MLP (in+out)')
    analyze_routing_paths(frac_mlp, baseline_l2, layers, 'Llama-2 MLP')
    plot_sensitivity_maps(frac_mlp, baseline_l2, layers, 'Llama-2 MLP', output_dir)

    # --- Llama-2 VO sweep ---
    vo_dir = 'results/full_sweep/Llama-2-7b-chat-hf_paired_attn_v+attn_o_chunk100_eval_set_mcq_arc_challenge_500'
    print("\nLoading Llama-2 VO sweep...")
    _, _, flip_vo, frac_vo, _ = load_flip_sensitivity(vo_dir, baseline_correct=baseline_l2)
    analyze_layer_depth_profile(frac_vo, baseline_l2, layers, 'Llama-2 VO (attn_v+attn_o)')
    analyze_routing_paths(frac_vo, baseline_l2, layers, 'Llama-2 VO')
    plot_sensitivity_maps(frac_vo, baseline_l2, layers, 'Llama-2 VO', output_dir)

    # --- Llama-2 QK sweep ---
    qk_dir = 'results/full_sweep/Llama-2-7b-chat-hf_paired_attn_q+attn_k_chunk100_eval_set_mcq_arc_challenge_500'
    print("\nLoading Llama-2 QK sweep...")
    _, _, flip_qk, frac_qk, _ = load_flip_sensitivity(qk_dir, baseline_correct=baseline_l2)
    analyze_layer_depth_profile(frac_qk, baseline_l2, layers, 'Llama-2 QK (attn_q+attn_k)')
    analyze_routing_paths(frac_qk, baseline_l2, layers, 'Llama-2 QK')
    plot_sensitivity_maps(frac_qk, baseline_l2, layers, 'Llama-2 QK', output_dir)

    # --- Cross-sweep comparisons ---
    print("\n" + "="*60)
    print("CROSS-SWEEP COMPARISONS")
    cross_sweep_agreement(frac_mlp, frac_vo, baseline_l2, layers, 'MLP', 'VO')
    cross_sweep_agreement(frac_mlp, frac_qk, baseline_l2, layers, 'MLP', 'QK')
    cross_sweep_agreement(frac_vo, frac_qk, baseline_l2, layers, 'VO', 'QK')
    plot_cross_sweep(frac_mlp, frac_vo, baseline_l2, layers, 'MLP_vs_VO', output_dir)
    plot_cross_sweep(frac_mlp, frac_qk, baseline_l2, layers, 'MLP_vs_QK', output_dir)

    # --- Llama-3 MLP sweep ---
    mlp3_dir = 'results/full_sweep/Meta-Llama-3-8B-Instruct_paired_mlp_in+mlp_out_chunk100_eval_set_mcq_arc_challenge_500'
    print("\nLoading Llama-3 MLP sweep...")
    baseline_correct_l3 = load_baseline_correct(mlp3_dir)
    _, baseline_l3, flip_mlp3, frac_mlp3, layers3 = load_flip_sensitivity(
        mlp3_dir, baseline_correct=baseline_correct_l3
    )
    print(f"  Baseline accuracy: {baseline_l3.mean()*100:.1f}%")
    analyze_layer_depth_profile(frac_mlp3, baseline_l3, layers3, 'Llama-3 MLP (in+out)')
    analyze_routing_paths(frac_mlp3, baseline_l3, layers3, 'Llama-3 MLP')
    plot_sensitivity_maps(frac_mlp3, baseline_l3, layers3, 'Llama-3 MLP', output_dir)

    print(f"\n\nAll figures saved to {output_dir}/")
