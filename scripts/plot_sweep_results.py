"""
Plot results from the full layer sweep (experiments 15/18).

Generates:
  1. Classification heatmap
  2. Acc + Entropy dual heatmap (percentile-clipped, --exclude-layers for colormap)
  3. Noise fraction by layer (bar + line overlay, dual y-axis for counts)
  4. Noise map: accuracy heatmap masked to noise-only chunks
  5. Entropy vs Accuracy scatter
  6. Layer profile: stacked area
  7. Cumulative final effect
  8. Chunk index profile: noise fraction by spectral region

Supports --compare-dir for side-by-side two-model comparison.

Usage:
    python scripts/plot_sweep_results.py --results-dir results/full_sweep/... --output-dir figures/...
    python scripts/plot_sweep_results.py --results-dir ... --exclude-layers 0,1
    python scripts/plot_sweep_results.py --results-dir ... --compare-dir results/full_sweep/other_model/...
"""

import sys
sys.path.append('.')

import pickle
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
# Colors — semantic
# =============================================================================
CLASS_COLORS = {
    'true_noise':       '#D94F4F',  # red — safe to remove
    'true_signal':      '#4A90D9',  # blue — important
    'confident_wrong':  '#B0B0B0',  # gray — ambiguous
    'uncertain_right':  '#7BC47F',  # green — mildly good
    'critical_signal':  '#2D2D2D',  # black — model breaks
}

CLASS_LABELS = {
    'true_noise':       'Noise (ent↓ acc↑) — safe to drop',
    'true_signal':      'Signal (ent↑ acc↓) — important',
    'confident_wrong':  'Confident Wrong (ent↓ acc↓)',
    'uncertain_right':  'Uncertain Right (ent↑ acc↑)',
    'critical_signal':  'Critical (NaN — model breaks)',
}

def _cc(name):
    return CLASS_COLORS[name]


# =============================================================================
# Data loading
# =============================================================================

def load_results(results_dir: Path, model_short: str = None):
    results_dir = Path(results_dir)
    if model_short is None:
        # Auto-detect model_short from pickle filenames (more reliable than dir name)
        layer_pkls = sorted(results_dir.glob("*_layer*.pkl"))
        if layer_pkls:
            # Extract prefix before "_layer" from first pickle filename
            fname = layer_pkls[0].stem  # e.g. "Llama-2-7b-chat-hf_layer0_mlp_in+mlp_out"
            model_short = fname.split('_layer')[0]
        else:
            dir_name = results_dir.name
            model_short = dir_name.rsplit('_chunk', 1)[0]

    results = {}
    for pkl_file in sorted(results_dir.glob(f"{model_short}_layer*.pkl")):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Normalize Open QA keys → MCQ key names
        if data['config'].get('eval_type') == 'open_qa':
            s = data['summary']
            if 'sum_logprob_change' in s and 'sum_entropy_change' not in s:
                s['sum_entropy_change'] = s['sum_logprob_change']
            bl = data.get('baseline_qa', {})
            if bl and 'baseline_mcq' not in data:
                data['baseline_mcq'] = {
                    'avg_entropy': bl.get('mean_neg_logprob', 0),
                    'avg_nll': bl.get('mean_neg_logprob', 0),
                    'accuracy': bl.get('accuracy', 0),
                }
            for c in data.get('chunk_results', []):
                if 'qa_logprob_change' in c and 'mcq_entropy_change' not in c:
                    c['mcq_entropy_change'] = c['qa_logprob_change']
                if 'qa_accuracy_change' in c and 'mcq_accuracy_change' not in c:
                    c['mcq_accuracy_change'] = c['qa_accuracy_change']
                if 'qa_accuracy' in c and 'mcq_accuracy' not in c:
                    c['mcq_accuracy'] = c['qa_accuracy']
                if 'qa_mean_neg_logprob' in c and 'mcq_entropy' not in c:
                    c['mcq_entropy'] = c['qa_mean_neg_logprob']
            for cr in data.get('cumulative_results', []):
                if 'qa_logprob_change' in cr and 'mcq_entropy_change' not in cr:
                    cr['mcq_entropy_change'] = cr['qa_logprob_change']
                if 'qa_accuracy_change' in cr and 'mcq_accuracy_change' not in cr:
                    cr['mcq_accuracy_change'] = cr['qa_accuracy_change']
                if 'qa_accuracy' in cr and 'mcq_accuracy' not in cr:
                    cr['mcq_accuracy'] = cr['qa_accuracy']
                if 'qa_mean_neg_logprob' in cr and 'mcq_entropy' not in cr:
                    cr['mcq_entropy'] = cr['qa_mean_neg_logprob']

        layer = data['config']['layer']
        matrix_type = data['config']['matrix_type']
        results[(layer, matrix_type)] = data

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results, model_short


# =============================================================================
# Data extraction helpers
# =============================================================================

def get_classification_matrix(results, layers, matrix_type):
    """0=noise, 1=signal, 2=conf_wrong, 3=unc_right, 4=critical, -1=missing"""
    max_chunks = 0
    for layer in layers:
        key = (layer, matrix_type)
        if key in results:
            max_chunks = max(max_chunks, results[key]['config']['num_chunks'])
    if max_chunks == 0:
        return None, []

    matrix = np.full((len(layers), max_chunks), -1, dtype=int)
    for i, layer in enumerate(layers):
        key = (layer, matrix_type)
        if key not in results:
            continue
        cl = results[key]['classification']
        for ci in cl['true_noise']:       matrix[i, ci] = 0
        for ci in cl['true_signal']:      matrix[i, ci] = 1
        for ci in cl['confident_wrong']:  matrix[i, ci] = 2
        for ci in cl['uncertain_right']:  matrix[i, ci] = 3
        for ci in cl.get('critical_signal', []): matrix[i, ci] = 4
    return matrix, layers


def get_entropy_matrix(results, layers, matrix_type):
    max_chunks = 0
    for layer in layers:
        key = (layer, matrix_type)
        if key in results:
            max_chunks = max(max_chunks, results[key]['config']['num_chunks'])
    if max_chunks == 0:
        return None
    matrix = np.full((len(layers), max_chunks), np.nan)
    for i, layer in enumerate(layers):
        key = (layer, matrix_type)
        if key not in results:
            continue
        for c in results[key]['chunk_results']:
            matrix[i, c['chunk_idx']] = c['mcq_entropy_change']
    return matrix


def get_accuracy_matrix(results, layers, matrix_type):
    max_chunks = 0
    for layer in layers:
        key = (layer, matrix_type)
        if key in results:
            max_chunks = max(max_chunks, results[key]['config']['num_chunks'])
    if max_chunks == 0:
        return None
    matrix = np.full((len(layers), max_chunks), np.nan)
    for i, layer in enumerate(layers):
        key = (layer, matrix_type)
        if key not in results:
            continue
        for c in results[key]['chunk_results']:
            matrix[i, c['chunk_idx']] = c['mcq_accuracy_change']
    return matrix


def print_summary_table(results, layers, matrix_types):
    print(f"\n{'Layer':>5} {'Matrix':>18} {'Noise':>6} {'Signal':>7} {'CfWrng':>7} {'UcRght':>7} "
          f"{'Crit':>5} {'NoiseFr':>8} {'SumEnt':>9} {'SumAcc':>9}")
    print("-" * 100)
    for layer in layers:
        for mt in matrix_types:
            key = (layer, mt)
            if key not in results:
                continue
            s = results[key]['summary']
            num_crit = s.get('num_critical_signal', 0)
            print(f"{layer:>5} {mt:>18} {s['num_true_noise']:>6} {s['num_true_signal']:>7} "
                  f"{s['num_confident_wrong']:>7} {s['num_uncertain_right']:>7} "
                  f"{num_crit:>5} "
                  f"{s['noise_fraction']:>7.1%} {s['sum_entropy_change']:>+9.4f} "
                  f"{s['sum_accuracy_change']*100:>+8.1f}pp")


# =============================================================================
# Plot 1: Classification heatmap (full + zoomed)
# =============================================================================

def plot_classification_heatmap(results, layers, matrix_type, output_dir, model_short):
    if not HAS_MPL:
        return

    class_matrix, valid_layers = get_classification_matrix(results, layers, matrix_type)
    if class_matrix is None:
        return

    cmap = mcolors.ListedColormap([
        '#F0F0F0', _cc('true_noise'), _cc('true_signal'),
        _cc('confident_wrong'), _cc('uncertain_right'), _cc('critical_signal'),
    ])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    legend_elements = [
        Patch(facecolor=_cc('true_noise'), label=CLASS_LABELS['true_noise']),
        Patch(facecolor=_cc('true_signal'), label=CLASS_LABELS['true_signal']),
        Patch(facecolor=_cc('confident_wrong'), label=CLASS_LABELS['confident_wrong']),
        Patch(facecolor=_cc('uncertain_right'), label=CLASS_LABELS['uncertain_right']),
        Patch(facecolor=_cc('critical_signal'), label=CLASS_LABELS['critical_signal']),
    ]

    fig, ax = plt.subplots(figsize=(14, max(6, len(valid_layers) * 0.3)))
    ax.imshow(class_matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xlabel('Chunk Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'{model_short} — {matrix_type} Classification\n'
                 f'({class_matrix.shape[1]} chunks per layer)', fontsize=14)
    ax.set_yticks(range(len(valid_layers)))
    ax.set_yticklabels(valid_layers)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f'classification_heatmap_{matrix_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: classification_heatmap_{matrix_type}.png")


# =============================================================================
# Plot 2: Acc + Entropy dual heatmap (with exclude-layers for colormap)
# =============================================================================

def plot_acc_entropy_dual_heatmap(results, layers, matrix_type, output_dir, model_short,
                                  exclude_layers=None, fixed_acc_range=None, fixed_ent_range=None):
    if not HAS_MPL:
        return

    acc_matrix = get_accuracy_matrix(results, layers, matrix_type)
    ent_matrix = get_entropy_matrix(results, layers, matrix_type)
    if acc_matrix is None or ent_matrix is None:
        return

    acc_pp = acc_matrix * 100
    exclude_set = set(exclude_layers or [])

    # Compute color limits from NON-excluded layers only
    include_mask = np.array([layer not in exclude_set for layer in layers])

    def _get_sym_lim(matrix, mask, floor):
        """Symmetric color limit from included rows using p2/p98."""
        included = matrix[mask]
        flat = included[np.isfinite(included)]
        if len(flat) == 0:
            return floor
        lo, hi = np.percentile(flat, 2), np.percentile(flat, 98)
        lim = max(abs(lo), abs(hi))
        return max(lim, floor)

    acc_lim = fixed_acc_range if fixed_acc_range is not None else _get_sym_lim(acc_pp, include_mask, 1.0)
    ent_lim = fixed_ent_range if fixed_ent_range is not None else _get_sym_lim(ent_matrix, include_mask, 0.01)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, max(10, len(layers) * 0.55)),
                                    sharex=True, gridspec_kw={'hspace': 0.15})

    im1 = ax1.imshow(acc_pp, aspect='auto', cmap='RdYlGn', vmin=-acc_lim, vmax=acc_lim,
                      interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Accuracy Change (pp)', fontsize=9)
    ax1.set_ylabel('Layer', fontsize=11)
    excl_note = f', colormap from layers excl. {sorted(exclude_set)}' if exclude_set else ''
    ax1.set_title(f'{model_short} — {matrix_type}\nAccuracy Change{excl_note}', fontsize=12)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers)

    im2 = ax2.imshow(ent_matrix, aspect='auto', cmap='RdBu_r', vmin=-ent_lim, vmax=ent_lim,
                      interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Entropy Change', fontsize=9)
    ax2.set_xlabel('Chunk Index', fontsize=11)
    ax2.set_ylabel('Layer', fontsize=11)
    ax2.set_title(f'Entropy Change', fontsize=12)
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels(layers)

    plt.tight_layout()
    plt.savefig(output_dir / f'acc_entropy_dual_heatmap_{matrix_type}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: acc_entropy_dual_heatmap_{matrix_type}.png")


# =============================================================================
# Plot 3: Noise fraction by layer (bar + line + dual y-axis)
# =============================================================================

def plot_noise_fraction_by_layer(results, layers, matrix_types, output_dir, model_short):
    if not HAS_MPL:
        return

    for mt in matrix_types:
        fractions = []
        counts = []
        totals = []
        valid_layers = []
        for layer in layers:
            key = (layer, mt)
            if key in results:
                s = results[key]['summary']
                fractions.append(s['noise_fraction'])
                counts.append(s['num_true_noise'])
                total = (s['num_true_noise'] + s['num_true_signal'] +
                         s['num_confident_wrong'] + s['num_uncertain_right'] +
                         s.get('num_critical_signal', 0))
                totals.append(total)
                valid_layers.append(layer)

        if not valid_layers:
            continue

        fig, ax1 = plt.subplots(figsize=(13, 5))

        # Bars: noise fraction %
        bar_colors = ['#D94F4F' if f > 0.25 else '#E8A0A0' if f > 0.1 else '#D0D0D0'
                      for f in fractions]
        bars = ax1.bar(valid_layers, [f * 100 for f in fractions], color=bar_colors,
                       width=0.7, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Noise Fraction (%)', fontsize=12, color='#D94F4F')
        ax1.set_xticks(valid_layers)
        ax1.set_ylim(0, max(f * 100 for f in fractions) * 1.25 + 5)
        ax1.axhline(y=25, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.grid(True, alpha=0.12, axis='y')
        ax1.tick_params(axis='y', labelcolor='#D94F4F')

        # Line overlay connecting the fraction values
        ax1.plot(valid_layers, [f * 100 for f in fractions], 'o-', color='#8B0000',
                 markersize=4, linewidth=1.2, alpha=0.7, zorder=5)

        # Second y-axis: absolute noise count
        ax2 = ax1.twinx()
        ax2.plot(valid_layers, counts, 's--', color='#4A90D9', markersize=4,
                 linewidth=1.0, alpha=0.7, label='Noise count')
        ax2.set_ylabel('Noise Chunk Count', fontsize=12, color='#4A90D9')
        ax2.tick_params(axis='y', labelcolor='#4A90D9')

        # Labels on bars: count/total
        for bar, count, total in zip(bars, counts, totals):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{count}/{total}', ha='center', va='bottom', fontsize=6, color='#555')

        ax1.set_title(f'{model_short} — {mt}\nNoise Fraction by Layer', fontsize=13)
        fig.tight_layout()
        plt.savefig(output_dir / f'noise_fraction_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: noise_fraction_{mt}.png")


# =============================================================================
# Plot 4: Noise map — accuracy heatmap masked to noise-only chunks
# =============================================================================

def plot_noise_map(results, layers, matrix_type, output_dir, model_short):
    """The paper's central figure: accuracy effect of each chunk, but only noise
    chunks are shown in color. Everything else is grayed out."""
    if not HAS_MPL:
        return

    acc_matrix = get_accuracy_matrix(results, layers, matrix_type)
    class_matrix, _ = get_classification_matrix(results, layers, matrix_type)
    if acc_matrix is None or class_matrix is None:
        return

    acc_pp = acc_matrix * 100
    noise_mask = (class_matrix == 0)  # True where noise

    fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.3)))

    # Background: gray for non-noise, lighter gray for missing
    bg_colors = np.full(class_matrix.shape + (4,), 0.0)  # RGBA
    for i in range(class_matrix.shape[0]):
        for j in range(class_matrix.shape[1]):
            if class_matrix[i, j] == -1:
                bg_colors[i, j] = [0.95, 0.95, 0.95, 1.0]
            elif class_matrix[i, j] == 0:
                bg_colors[i, j] = [0, 0, 0, 0]  # transparent — will be overlaid
            else:
                bg_colors[i, j] = [0.85, 0.85, 0.85, 1.0]

    ax.imshow(bg_colors, aspect='auto', interpolation='nearest')

    # Overlay: noise chunks colored by accuracy change
    noise_acc = np.where(noise_mask, acc_pp, np.nan)
    valid_noise = noise_acc[np.isfinite(noise_acc)]
    if len(valid_noise) > 0:
        lo, hi = np.percentile(valid_noise, 5), np.percentile(valid_noise, 95)
        vmax = max(abs(lo), abs(hi))
        vmax = max(vmax, 0.5)
    else:
        vmax = 5.0

    im = ax.imshow(noise_acc, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                    interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy Change (pp) — noise chunks only', fontsize=10)

    ax.set_xlabel('Chunk Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'{model_short} — {matrix_type}\nNoise Map: accuracy effect of noise chunks '
                 f'(gray = non-noise)', fontsize=13)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    plt.tight_layout()
    plt.savefig(output_dir / f'noise_map_{matrix_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: noise_map_{matrix_type}.png")


# =============================================================================
# Plot 5: Entropy vs Accuracy scatter
# =============================================================================

def plot_entropy_vs_accuracy_scatter(results, layers, matrix_types, output_dir, model_short):
    if not HAS_MPL:
        return

    for mt in matrix_types:
        ent_changes = []
        acc_changes = []
        layer_ids = []

        for layer in layers:
            key = (layer, mt)
            if key not in results:
                continue
            for c in results[key]['chunk_results']:
                if np.isnan(c['mcq_entropy_change']):
                    continue
                ent_changes.append(c['mcq_entropy_change'])
                acc_changes.append(c['mcq_accuracy_change'] * 100)
                layer_ids.append(layer)

        if not ent_changes:
            continue

        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(ent_changes, acc_changes, c=layer_ids, cmap='viridis',
                        alpha=0.45, s=18, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.8).set_label('Layer', fontsize=10)

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Entropy Change', fontsize=12)
        ax.set_ylabel('Accuracy Change (pp)', fontsize=12)
        ax.set_title(f'{model_short} — {mt}\nPer-Chunk Entropy vs Accuracy', fontsize=13)

        ax.text(0.02, 0.98, 'NOISE\n(ent↓ acc↑)', transform=ax.transAxes,
                fontsize=9, va='top', ha='left', color=_cc('true_noise'), fontweight='bold')
        ax.text(0.98, 0.02, 'SIGNAL\n(ent↑ acc↓)', transform=ax.transAxes,
                fontsize=9, va='bottom', ha='right', color=_cc('true_signal'), fontweight='bold')
        ax.text(0.02, 0.02, 'Conf. Wrong\n(ent↓ acc↓)', transform=ax.transAxes,
                fontsize=9, va='bottom', ha='left', color='#888')
        ax.text(0.98, 0.98, 'Unc. Right\n(ent↑ acc↑)', transform=ax.transAxes,
                fontsize=9, va='top', ha='right', color=_cc('uncertain_right'))
        ax.grid(True, alpha=0.12)

        plt.tight_layout()
        plt.savefig(output_dir / f'entropy_vs_accuracy_scatter_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: entropy_vs_accuracy_scatter_{mt}.png")


# =============================================================================
# Plot 6: Layer profile (stacked area)
# =============================================================================

def plot_layer_profile(results, layers, matrix_types, output_dir, model_short):
    if not HAS_MPL:
        return

    for mt in matrix_types:
        noise_frac, signal_frac, conf_frac, unc_frac = [], [], [], []
        valid_layers = []

        for layer in layers:
            key = (layer, mt)
            if key not in results:
                continue
            s = results[key]['summary']
            total = (s['num_true_noise'] + s['num_true_signal'] +
                     s['num_confident_wrong'] + s['num_uncertain_right'] +
                     s.get('num_critical_signal', 0))
            if total == 0:
                continue
            noise_frac.append(s['num_true_noise'] / total)
            signal_frac.append(s['num_true_signal'] / total)
            conf_frac.append(s['num_confident_wrong'] / total)
            unc_frac.append(s['num_uncertain_right'] / total)
            valid_layers.append(layer)

        if not valid_layers:
            continue

        noise_frac = np.array(noise_frac)
        signal_frac = np.array(signal_frac)
        conf_frac = np.array(conf_frac)
        unc_frac = np.array(unc_frac)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(valid_layers, 0, noise_frac,
                        color=_cc('true_noise'), alpha=0.85, label=CLASS_LABELS['true_noise'])
        ax.fill_between(valid_layers, noise_frac, noise_frac + unc_frac,
                        color=_cc('uncertain_right'), alpha=0.85, label=CLASS_LABELS['uncertain_right'])
        ax.fill_between(valid_layers, noise_frac + unc_frac,
                        noise_frac + unc_frac + conf_frac,
                        color=_cc('confident_wrong'), alpha=0.85, label=CLASS_LABELS['confident_wrong'])
        ax.fill_between(valid_layers, noise_frac + unc_frac + conf_frac, 1.0,
                        color=_cc('true_signal'), alpha=0.85, label=CLASS_LABELS['true_signal'])

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Fraction', fontsize=11)
        ax.set_title(f'{model_short} — {mt}\nClassification Distribution by Layer', fontsize=13)
        ax.set_ylim(0, 1)
        ax.set_xlim(valid_layers[0], valid_layers[-1])
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.12, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'layer_profile_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: layer_profile_{mt}.png")


# =============================================================================
# Plot 7: Cumulative final effect
# =============================================================================

def plot_cumulative_final_effect(results, layers, matrix_types, output_dir, model_short):
    if not HAS_MPL:
        return

    for mt in matrix_types:
        final_acc, final_ent, n_removed, valid_layers = [], [], [], []
        for layer in layers:
            key = (layer, mt)
            if key not in results:
                continue
            cum = results[key].get('cumulative_results', [])
            if not cum:
                continue
            last = cum[-1]
            final_acc.append(last['mcq_accuracy_change'] * 100)
            final_ent.append(last['mcq_entropy_change'])
            n_removed.append(last['num_chunks_removed'])
            valid_layers.append(layer)

        if not valid_layers:
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        acc_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in final_acc]
        bars1 = ax1.bar(valid_layers, final_acc, color=acc_colors, alpha=0.8, width=0.7)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_ylabel('Accuracy Change (pp)', fontsize=11)
        ax1.set_title(f'{model_short} — {mt}\nEffect of Removing ALL Noise Chunks', fontsize=13)
        ax1.grid(True, alpha=0.2, axis='y')
        for bar, n, v in zip(bars1, n_removed, final_acc):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.1 if v >= 0 else -0.3),
                     f'n={n}', ha='center', va='bottom' if v >= 0 else 'top',
                     fontsize=7, color='gray')

        ent_colors = ['#2ca02c' if v < 0 else '#d62728' for v in final_ent]
        ax2.bar(valid_layers, final_ent, color=ent_colors, alpha=0.8, width=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_ylabel('Entropy Change', fontsize=11)
        ax2.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'cumulative_final_effect_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: cumulative_final_effect_{mt}.png")


# =============================================================================
# Plot 8b: Cumulative noise removal trajectory (line plot per layer)
# =============================================================================

def plot_cumulative_trajectory(results, layers, matrix_types, output_dir, model_short):
    if not HAS_MPL:
        return

    for mt in matrix_types:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
        fig.subplots_adjust(right=0.82)

        all_acc_lines = []
        all_ent_lines = []
        layer_labels = []

        for layer in layers:
            key = (layer, mt)
            if key not in results:
                continue
            cum = results[key].get('cumulative_results', [])
            if not cum:
                continue
            steps = [0] + [c['num_chunks_removed'] for c in cum]
            acc   = [0] + [c['mcq_accuracy_change'] * 100 for c in cum]
            ent   = [0] + [c['mcq_entropy_change'] for c in cum]
            all_acc_lines.append((steps, acc, layer))
            all_ent_lines.append((steps, ent, layer))
            layer_labels.append(layer)

        if not all_acc_lines:
            continue

        # Color layers by index — early=blue, late=red
        cmap = plt.cm.coolwarm
        max_layer = max(layer_labels)

        for steps, acc, layer in all_acc_lines:
            color = cmap(layer / max_layer)
            lw = 2.0 if len(steps) > 6 else 1.0  # thicker for layers with more steps
            ax1.plot(steps, acc, color=color, linewidth=lw, alpha=0.7, marker='o', markersize=3)
            ax1.text(steps[-1], acc[-1], f'{layer}', fontsize=6, color=color, va='center')

        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_ylabel('Accuracy Change (pp)', fontsize=11)
        ax1.set_title(f'{model_short} — {mt}\nCumulative Noise Removal Trajectory', fontsize=13)
        ax1.grid(True, alpha=0.2)

        for steps, ent, layer in all_ent_lines:
            color = cmap(layer / max_layer)
            lw = 2.0 if len(steps) > 6 else 1.0
            ax2.plot(steps, ent, color=color, linewidth=lw, alpha=0.7, marker='o', markersize=3)
            ax2.text(steps[-1], ent[-1], f'{layer}', fontsize=6, color=color, va='center')

        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Number of noise chunks removed', fontsize=11)
        ax2.set_ylabel('Entropy Change', fontsize=11)
        ax2.grid(True, alpha=0.2)

        # Colorbar outside the plot area
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_layer))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        fig.colorbar(sm, cax=cbar_ax).set_label('Layer', fontsize=10)
        fname = output_dir / f'cumulative_trajectory_{mt}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: cumulative_trajectory_{mt}.png")


# =============================================================================
# Plot 8: Chunk index profile — noise fraction by spectral region
# =============================================================================

def plot_chunk_index_profile(results, layers, matrix_types, output_dir, model_short):
    """For each chunk index, compute the fraction of layers where that chunk is noise.
    Shows whether noise concentrates in specific spectral regions.
    """
    if not HAS_MPL:
        return

    for mt in matrix_types:
        class_matrix, valid_layers = get_classification_matrix(results, layers, mt)
        if class_matrix is None:
            continue

        n_layers, n_chunks = class_matrix.shape

        # Per-chunk: fraction of layers where this chunk is noise
        chunk_noise_frac = []
        chunk_signal_frac = []
        for ci in range(n_chunks):
            col = class_matrix[:, ci]
            valid = col[col >= 0]  # exclude missing
            if len(valid) == 0:
                chunk_noise_frac.append(0)
                chunk_signal_frac.append(0)
            else:
                chunk_noise_frac.append(np.sum(valid == 0) / len(valid))
                chunk_signal_frac.append(np.sum(valid == 1) / len(valid))

        chunk_noise_frac = np.array(chunk_noise_frac)
        chunk_signal_frac = np.array(chunk_signal_frac)

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.bar(range(n_chunks), chunk_noise_frac * 100, color=_cc('true_noise'),
               alpha=0.8, width=0.8, label='Noise fraction')
        ax.bar(range(n_chunks), -chunk_signal_frac * 100, color=_cc('true_signal'),
               alpha=0.8, width=0.8, label='Signal fraction (inverted)')

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Chunk Index (low = top singular values, high = tail)', fontsize=12)
        ax.set_ylabel('% of layers classified as noise / signal', fontsize=12)
        ax.set_title(f'{model_short} — {mt}\nSpectral Profile: noise vs signal by chunk position', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.12, axis='y')

        # Annotate spectral regions
        if n_chunks > 10:
            ax.axvspan(0, min(5, n_chunks), alpha=0.05, color='blue')
            ax.axvspan(max(0, n_chunks - 10), n_chunks, alpha=0.05, color='red')
            ax.text(2, ax.get_ylim()[1] * 0.9, 'top SVs\n(high energy)', fontsize=8,
                    ha='center', color='#4A90D9', alpha=0.7)
            ax.text(n_chunks - 5, ax.get_ylim()[1] * 0.9, 'tail SVs\n(low energy)', fontsize=8,
                    ha='center', color='#D94F4F', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_dir / f'chunk_index_profile_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: chunk_index_profile_{mt}.png")


# =============================================================================
# Cross-model comparison plots
# =============================================================================

def plot_comparison(results1, model1, results2, model2, layers, matrix_types, output_dir):
    """Side-by-side noise fraction and classification profile for two models."""
    if not HAS_MPL:
        return

    for mt in matrix_types:
        # Noise fraction comparison
        frac1, frac2 = [], []
        valid_layers = []
        for layer in layers:
            k1, k2 = (layer, mt), (layer, mt)
            if k1 in results1 and k2 in results2:
                frac1.append(results1[k1]['summary']['noise_fraction'] * 100)
                frac2.append(results2[k2]['summary']['noise_fraction'] * 100)
                valid_layers.append(layer)

        if not valid_layers:
            continue

        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.array(valid_layers)
        w = 0.35
        ax.bar(x - w/2, frac1, w, color='#D94F4F', alpha=0.7, label=model1)
        ax.bar(x + w/2, frac2, w, color='#4A90D9', alpha=0.7, label=model2)
        ax.plot(x - w/2, frac1, 'o-', color='#8B0000', markersize=3, linewidth=0.8, alpha=0.6)
        ax.plot(x + w/2, frac2, 's-', color='#1B4F8B', markersize=3, linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Noise Fraction (%)', fontsize=12)
        ax.set_title(f'Noise Fraction Comparison — {mt}', fontsize=13)
        ax.set_xticks(valid_layers)
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.12, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / f'compare_noise_fraction_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: compare_noise_fraction_{mt}.png")

        # Layer profile comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)

        for ax, res, mname in [(ax1, results1, model1), (ax2, results2, model2)]:
            n_f, s_f, c_f, u_f, vl = [], [], [], [], []
            for layer in layers:
                key = (layer, mt)
                if key not in res:
                    continue
                s = res[key]['summary']
                total = (s['num_true_noise'] + s['num_true_signal'] +
                         s['num_confident_wrong'] + s['num_uncertain_right'] +
                         s.get('num_critical_signal', 0))
                if total == 0:
                    continue
                n_f.append(s['num_true_noise'] / total)
                s_f.append(s['num_true_signal'] / total)
                c_f.append(s['num_confident_wrong'] / total)
                u_f.append(s['num_uncertain_right'] / total)
                vl.append(layer)

            if not vl:
                continue
            n_f, s_f, c_f, u_f = [np.array(x) for x in [n_f, s_f, c_f, u_f]]

            ax.fill_between(vl, 0, n_f, color=_cc('true_noise'), alpha=0.85)
            ax.fill_between(vl, n_f, n_f + u_f, color=_cc('uncertain_right'), alpha=0.85)
            ax.fill_between(vl, n_f + u_f, n_f + u_f + c_f, color=_cc('confident_wrong'), alpha=0.85)
            ax.fill_between(vl, n_f + u_f + c_f, 1.0, color=_cc('true_signal'), alpha=0.85)
            ax.set_title(mname, fontsize=12)
            ax.set_xlabel('Layer', fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_xlim(vl[0], vl[-1])
            ax.grid(True, alpha=0.12, axis='y')

        ax1.set_ylabel('Fraction', fontsize=11)

        # Shared legend
        legend_elements = [
            Patch(facecolor=_cc('true_noise'), label='Noise'),
            Patch(facecolor=_cc('uncertain_right'), label='Unc. Right'),
            Patch(facecolor=_cc('confident_wrong'), label='Conf. Wrong'),
            Patch(facecolor=_cc('true_signal'), label='Signal'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'Classification Distribution — {mt}', fontsize=13, y=1.06)
        plt.tight_layout()
        plt.savefig(output_dir / f'compare_layer_profile_{mt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: compare_layer_profile_{mt}.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot full sweep results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--matrices", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--model-short", type=str, default=None)
    parser.add_argument("--exclude-layers", type=str, default=None,
                        help="Comma-separated layers to exclude from colormap range (e.g. 0,1)")
    parser.add_argument("--acc-range", type=float, default=None,
                        help="Fixed symmetric accuracy range in pp (e.g. 2.5 means -2.5 to +2.5)")
    parser.add_argument("--ent-range", type=float, default=None,
                        help="Fixed symmetric entropy range (e.g. 0.035 means -0.035 to +0.035)")
    parser.add_argument("--compare-dir", type=str, default=None,
                        help="Second results dir for cross-model comparison")
    parser.add_argument("--compare-model-short", type=str, default=None,
                        help="Model short name for comparison model")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results, model_short = load_results(results_dir, model_short=args.model_short)
    if not results:
        print("No results found!")
        return

    all_layers = sorted(set(k[0] for k in results.keys()))
    all_matrices = sorted(set(k[1] for k in results.keys()))

    if args.layers:
        layers = set()
        for part in args.layers.split(','):
            part = part.strip()
            if '-' in part:
                s, e = part.split('-')
                layers.update(range(int(s), int(e) + 1))
            else:
                layers.add(int(part))
        layers = sorted(layers)
    else:
        layers = all_layers

    matrix_types = args.matrices if args.matrices else all_matrices

    exclude_layers = []
    if args.exclude_layers:
        exclude_layers = [int(x.strip()) for x in args.exclude_layers.split(',')]

    print(f"Model: {model_short}")
    print(f"Layers: {layers}")
    print(f"Matrices: {matrix_types}")
    if exclude_layers:
        print(f"Exclude from colormap: {exclude_layers}")

    output_dir = Path(args.output_dir) if args.output_dir else Path('figures/sweep')
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary_table(results, layers, matrix_types)

    if not HAS_MPL:
        print("\nSkipping plots (matplotlib not available)")
        return

    print(f"\nGenerating plots to {output_dir}...")

    for mt in matrix_types:
        plot_classification_heatmap(results, layers, mt, output_dir, model_short)
        plot_acc_entropy_dual_heatmap(results, layers, mt, output_dir, model_short,
                                      exclude_layers=exclude_layers,
                                      fixed_acc_range=args.acc_range,
                                      fixed_ent_range=args.ent_range)
        plot_noise_map(results, layers, mt, output_dir, model_short)

    plot_noise_fraction_by_layer(results, layers, matrix_types, output_dir, model_short)
    plot_entropy_vs_accuracy_scatter(results, layers, matrix_types, output_dir, model_short)
    plot_layer_profile(results, layers, matrix_types, output_dir, model_short)
    plot_cumulative_final_effect(results, layers, matrix_types, output_dir, model_short)
    plot_cumulative_trajectory(results, layers, matrix_types, output_dir, model_short)
    plot_chunk_index_profile(results, layers, matrix_types, output_dir, model_short)

    # Cross-model comparison
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        results2, model2 = load_results(compare_dir, model_short=args.compare_model_short)
        if results2:
            print(f"\nGenerating comparison plots ({model_short} vs {model2})...")
            plot_comparison(results, model_short, results2, model2, layers, matrix_types, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
