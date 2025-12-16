# Scripts

## analysis_report.py

Analyzes component search experiment results to identify noise vs. signal components. Works with both Tucker and CP decomposition results.

### Usage

```bash
# Single layer analysis (auto-detects Tucker or CP from results)
python scripts/analysis_report.py --results path/to/results.pkl

# Compare two layers
python scripts/analysis_report.py --results path/to/layer30/results.pkl path/to/layer31/results.pkl

# Custom output directory
python scripts/analysis_report.py --results path/to/results.pkl --output results/my_analysis

# Custom layer names
python scripts/analysis_report.py --results results1.pkl results2.pkl --layers "Layer 30" "Layer 31"
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--results` | Yes | Path(s) to results.pkl file(s). Provide 1 or 2 paths. |
| `--output` | No | Output directory (default: `results/analysis/component_search_{tucker\|cp}`) |
| `--layers` | No | Custom layer names for labeling (e.g., `"Layer 30" "Layer 31"`) |

### Examples

```bash
# Analyze Tucker results from layer 30
python scripts/analysis_report.py \
    --results results/component_search/coqa_20251213_142049/results.pkl

# Compare Tucker results from two layers
python scripts/analysis_report.py \
    --results results/component_search/coqa_20251213_142049/results.pkl \
             results/component_search/coqa_20251213_142059/results.pkl \
    --layers "Layer 30" "Layer 31"

# Analyze CP results
python scripts/analysis_report.py \
    --results results/component_search_cp/coqa_20251215_100000/results.pkl
```

### Output

The script generates:

| File | Description |
|------|-------------|
| `uncertainty_change_by_component.png` | Bar chart showing uncertainty change when each component is removed |
| `layer_comparison.png` | Side-by-side comparison (only when 2 layers provided) |
| `uncertainty_vs_blockiness.png` | Scatter plot of uncertainty vs blockiness changes |
| `change_distributions.png` | Histograms of uncertainty and blockiness changes |
| `analysis_report.txt` | Text report with component categorization and findings |

### Interpretation

- **Noise components** (green bars, uncertainty change < 0): Removing them reduces uncertainty. Safe to compress.
- **Signal components** (orange bars, uncertainty change > 0): Removing them increases uncertainty. Important to keep.
