"""
Plot SAE suppression-by-category results (Experiment 29b).

Usage:
    python scripts/plot_sae_suppression.py \
        --arc-csv results/sae_suppression_by_category_1094/suppression_by_category.csv \
        --mmlu-csv results/sae_suppression_by_category_mmlu14k/suppression_by_category.csv \
        --output-dir figures/sae_suppression/

    # Single dataset:
    python scripts/plot_sae_suppression.py \
        --arc-csv results/sae_suppression_by_category_1094/suppression_by_category.csv \
        --output-dir figures/sae_suppression/
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour / style constants ──────────────────────────────────────────────────
CAT_COLORS = {
    'pure_uncertainty':  '#2196F3',   # blue
    'pure_incorrectness': '#F44336',  # red
    'both':              '#9C27B0',   # purple
}
CAT_LABELS = {
    'pure_uncertainty':   'Certainty-associated',
    'pure_incorrectness': 'Correctness-associated',
    'both':               'Both',
}
LAYER_ORDER = [20, 24, 28, 31, 'all']


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('n_features', 'n_flipped_to_correct', 'n_flipped_to_incorrect'):
            if r[k]:
                r[k] = int(float(r[k]))
        for k in ('accuracy', 'acc_delta', 'mean_entropy'):
            if r[k]:
                r[k] = float(r[k])
        if r['layer'] not in ('', 'all'):
            try:
                r['layer'] = int(r['layer'])
            except ValueError:
                pass
    return rows


def baseline_acc(rows):
    for r in rows:
        if r['condition'] == 'baseline':
            return r['accuracy']
    return None


def per_layer_rows(rows, category):
    """Return per-layer rows for a given category (exclude 'all' aggregate)."""
    return [r for r in rows
            if r['category'] == category and r['layer'] != 'all']


def aggregate_row(rows, category):
    for r in rows:
        if r['category'] == category and r['layer'] == 'all':
            return r
    return None


# ── Plot 1: acc_delta by layer, grouped by category ──────────────────────────

def plot_delta_by_layer(rows, baseline, title, output_path):
    """Bar chart: acc_delta per layer, one group of bars per category."""
    layers = [l for l in LAYER_ORDER if l != 'all']
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']

    x = np.arange(len(layers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cat in enumerate(categories):
        deltas = []
        for layer in layers:
            match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
            deltas.append(match[0]['acc_delta'] if match else 0.0)
        bars = ax.bar(x + i * width, deltas, width,
                      label=CAT_LABELS[cat], color=CAT_COLORS[cat], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy delta (pp)')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Plot 2: aggregate delta comparison (ARC vs MMLU) ─────────────────────────

def plot_aggregate_comparison(arc_rows, mmlu_rows, output_path):
    """Grouped bar chart comparing aggregate acc_delta across datasets."""
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    datasets = []
    if arc_rows:
        datasets.append(('ARC-1094', arc_rows))
    if mmlu_rows:
        datasets.append(('MMLU-14K', mmlu_rows))

    x = np.arange(len(categories))
    width = 0.35
    dataset_colors = ['#455A64', '#FF7043']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, rows) in enumerate(datasets):
        deltas = []
        for cat in categories:
            r = aggregate_row(rows, cat)
            deltas.append(r['acc_delta'] if r else 0.0)
        ax.bar(x + i * width, deltas, width, label=label,
               color=dataset_colors[i], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Feature category')
    ax.set_ylabel('Accuracy delta (pp)')
    ax.set_title('Suppression effect — aggregate (all layers)')
    ax.set_xticks(x + width / 2 if len(datasets) == 2 else x)
    ax.set_xticklabels([CAT_LABELS[c] for c in categories], rotation=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Plot 3: flipped questions (correct↔incorrect) ────────────────────────────

def plot_flipped(rows, title, output_path):
    """Stacked bar: questions flipped to correct vs incorrect per condition."""
    plot_rows = [r for r in rows if r['condition'] != 'baseline'
                 and r['layer'] != 'all']

    labels = [f"L{r['layer']}\n{CAT_LABELS.get(r['category'], r['category'])}"
              for r in plot_rows]
    to_correct   = [r['n_flipped_to_correct']   for r in plot_rows]
    to_incorrect = [r['n_flipped_to_incorrect'] for r in plot_rows]

    x = np.arange(len(plot_rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, to_correct,   label='Flipped → correct',   color='#4CAF50', alpha=0.85)
    ax.bar(x, [-v for v in to_incorrect], label='Flipped → incorrect', color='#E53935', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Questions flipped')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Plot 4: entropy change after suppression ──────────────────────────────────

def plot_entropy_change(rows, baseline_entropy, title, output_path):
    """Show mean entropy before (baseline) vs after suppression per category."""
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    agg = {cat: aggregate_row(rows, cat) for cat in categories}

    x = np.arange(len(categories))
    width = 0.4
    entropies = [agg[cat]['mean_entropy'] if agg[cat] else baseline_entropy
                 for cat in categories]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, entropies, width,
                  color=[CAT_COLORS[c] for c in categories], alpha=0.85)
    ax.axhline(baseline_entropy, color='black', linewidth=1.2,
               linestyle='--', label=f'Baseline entropy ({baseline_entropy:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in categories], rotation=10)
    ax.set_ylabel('Mean entropy after suppression')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Plot 5: heatmap — acc_delta by layer × category ──────────────────────────

def plot_heatmap(rows, baseline, title, output_path):
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    layers = [l for l in LAYER_ORDER if l != 'all']

    matrix = np.zeros((len(categories), len(layers)))
    for i, cat in enumerate(categories):
        for j, layer in enumerate(layers):
            match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
            if match:
                matrix[i, j] = match[0]['acc_delta']

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.abs(matrix).max() or 1
    im = ax.imshow(matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([CAT_LABELS[c] for c in categories])
    ax.set_xlabel('Layer')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Accuracy delta (pp)')

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(layers)):
            ax.text(j, i, f'{matrix[i, j]:+.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix[i, j]) > vmax * 0.5 else 'black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _layer_bars(ax, rows, categories, layers, value_fn, baseline_line=None):
    x = np.arange(len(layers))
    width = 0.25
    for i, cat in enumerate(categories):
        values = []
        for layer in layers:
            match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
            values.append(value_fn(match[0]) if match else 0.0)
        ax.bar(x + (i - 1) * width, values, width,
               label=CAT_LABELS[cat], color=CAT_COLORS[cat], alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    if baseline_line is not None:
        ax.axhline(baseline_line, color='grey', linewidth=0.8,
                   linestyle=':', label=f'Baseline ({baseline_line:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel('Layer')
    ax.legend()


def plot_acc_by_layer(rows, title_prefix, output_path, exclude=None, highlight=None):
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    layers = [l for l in LAYER_ORDER if l != 'all' and l not in (exclude or set())]
    fig, ax = plt.subplots(figsize=(10, 5))
    _layer_bars(ax, rows, categories, layers, lambda r: r['acc_delta'])
    for hl in (highlight or set()):
        if hl in layers:
            ax.axvline(layers.index(hl), color='orange', linewidth=1.5,
                       linestyle='--', alpha=0.7, label=f'Layer {hl}')
    ax.set_ylabel('Accuracy delta (pp)')
    ax.set_title(f'{title_prefix} — Accuracy change after suppression')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_entropy_by_layer(rows, baseline_entropy, title_prefix, output_path, exclude=None, highlight=None):
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    layers = [l for l in LAYER_ORDER if l != 'all' and l not in (exclude or set())]
    fig, ax = plt.subplots(figsize=(10, 5))
    _layer_bars(ax, rows, categories, layers,
                lambda r: r['mean_entropy'] - baseline_entropy)
    for hl in (highlight or set()):
        if hl in layers:
            ax.axvline(layers.index(hl), color='orange', linewidth=1.5,
                       linestyle='--', alpha=0.7, label=f'Layer {hl}')
    ax.set_ylabel('Entropy change')
    ax.set_title(f'{title_prefix} — Entropy change after suppression')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_line_by_layer(rows, baseline_entropy, title_prefix, output_acc, output_ent, exclude=None):
    """
    Line plots: one line per category across layers.
    Separate figures for accuracy delta and entropy change.
    """
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    layers = [l for l in LAYER_ORDER if l != 'all' and l not in (exclude or set())]
    layer_labels = [str(l) for l in layers]
    markers = ['o', 's', '^']

    # Accuracy delta
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat, marker in zip(categories, markers):
        values = []
        for layer in layers:
            match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
            values.append(match[0]['acc_delta'] if match else 0.0)
        ax.plot(layer_labels, values, marker=marker, label=CAT_LABELS[cat],
                color=CAT_COLORS[cat], linewidth=2, markersize=7)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy delta (pp)')
    ax.set_title(f'{title_prefix} — Accuracy change after suppression')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_acc, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_acc}")

    # Entropy change
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat, marker in zip(categories, markers):
        values = []
        for layer in layers:
            match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
            values.append((match[0]['mean_entropy'] - baseline_entropy) if match else 0.0)
        ax.plot(layer_labels, values, marker=marker, label=CAT_LABELS[cat],
                color=CAT_COLORS[cat], linewidth=2, markersize=7)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entropy change')
    ax.set_title(f'{title_prefix} — Entropy change after suppression')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_ent, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_ent}")


def plot_selected_layers(rows, baseline_acc, baseline_entropy, title_prefix, layers, output_acc, output_ent):
    """
    Two figures (accuracy, entropy), each with 3 bars per selected layer showing
    absolute values with a baseline reference line.
    """
    categories = ['pure_uncertainty', 'pure_incorrectness', 'both']
    x = np.arange(len(categories))
    width = 0.5

    for metric, ylabel, value_fn, baseline_val, output_path in [
        ('Accuracy after suppression', 'Accuracy',
         lambda r: r['accuracy'], baseline_acc, output_acc),
        ('Entropy after suppression', 'Mean entropy',
         lambda r: r['mean_entropy'], baseline_entropy, output_ent),
    ]:
        for layer in layers:
            fig, ax = plt.subplots(figsize=(7, 5))
            values = []
            for cat in categories:
                match = [r for r in rows if r['category'] == cat and r['layer'] == layer]
                values.append(value_fn(match[0]) if match else baseline_val)
            ax.bar(x, values, width,
                   color=[CAT_COLORS[c] for c in categories], alpha=0.85,
                   label=[CAT_LABELS[c] for c in categories])
            ax.axhline(baseline_val, color='black', linewidth=1.5,
                       linestyle='--', label=f'Baseline ({baseline_val:.3f})')
            ax.set_xticks(x)
            ax.set_xticklabels([CAT_LABELS[c] for c in categories], rotation=10)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_prefix} — {metric} (Layer {layer})')
            ax.legend()
            # Zoom y-axis around the values so differences are visible
            all_vals = values + [baseline_val]
            margin = max(abs(max(all_vals) - min(all_vals)) * 0.5, 0.01)
            ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        type=str, required=True,  help='Suppression CSV file')
    parser.add_argument('--layer',      type=int, default=None,   help='Single layer to plot (omit for all layers)')
    parser.add_argument('--output-dir', type=str, default='figures/sae_suppression')
    parser.add_argument('--label',      type=str, default='',     help='Dataset label for titles')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    bl_acc     = baseline_acc(rows)
    bl_entropy = next(r['mean_entropy'] for r in rows if r['condition'] == 'baseline')
    label = args.label or Path(args.csv).stem

    if args.layer is not None:
        layers = [args.layer]
    else:
        numeric = sorted({r['layer'] for r in rows
                          if r['layer'] != 'all' and r['condition'] != 'baseline'})
        layers = numeric + ['all']

    for layer in layers:
        plot_selected_layers(
            rows, bl_acc, bl_entropy,
            title_prefix=label,
            layers=[layer],
            output_acc=out / f'acc_L{layer}.png',
            output_ent=out / f'entropy_L{layer}.png')

    print(f"\nSaved to {out}/")


if __name__ == '__main__':
    main()
