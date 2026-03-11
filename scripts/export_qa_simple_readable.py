"""
Export Open QA Results in Simple Readable Format

Creates a CSV with one row per sample showing question, gold answer,
and responses at key compression levels.

Usage:
    python scripts/export_qa_simple_readable.py --results results/open_qa_svd/.../results.pkl
"""

import pickle
import argparse
import csv
from pathlib import Path


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def export_simple_readable(results_data, output_path):
    """
    Export simple readable format with key compression levels.

    Columns:
    - sample_id
    - question
    - gold_answer
    - baseline_response
    - baseline_correct (✓/✗)
    - response_at_75%
    - correct_at_75% (✓/✗)
    - response_at_90%
    - correct_at_90% (✓/✗)
    - response_at_95%
    - correct_at_95% (✓/✗)
    - response_at_99.5%
    - correct_at_99.5% (✓/✗)
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    # Get compression levels
    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    # Find specific compression levels
    compression_levels = {}
    target_pcts = [75.0, 90.0, 95.0, 99.5]

    for key, result in items_sorted:
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        if red_pct in target_pcts:
            compression_levels[red_pct] = result

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        'ID',
        'Question',
        'Gold Answer',
        'Baseline',
        '75%',
        '90%',
        '95%',
        '99.5%'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, baseline_sample in enumerate(baseline_samples):
            row = [
                i,
                baseline_sample['question'],
                baseline_sample['gold_answers'][0],  # First gold answer
                baseline_sample['generated_text']
            ]

            # Add responses at each compression level
            for pct in target_pcts:
                if pct in compression_levels:
                    result = compression_levels[pct]
                    sample = result['results'][i]
                    row.append(sample['generated_text'])
                else:
                    row.append('N/A')

            writer.writerow(row)

    print(f"Simple readable CSV exported to: {output_path}")
    print(f"Samples: {len(baseline_samples)}")
    print(f"Columns: {len(header)} (ID + Question + Gold Answer + Baseline + {len(target_pcts)} compression levels)")


def export_all_levels_readable(results_data, output_path):
    """
    Export ALL compression levels in readable format.

    One row per sample with all compression levels.
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    # Build header dynamically
    header = ['ID', 'Question', 'Gold Answer', 'Baseline']

    for key, result in items_sorted:
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        k = result['k']
        header.append(f'{red_pct:.1f}% (k={k})')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, baseline_sample in enumerate(baseline_samples):
            row = [
                i,
                baseline_sample['question'],
                baseline_sample['gold_answers'][0],
                baseline_sample['generated_text']
            ]

            for key, result in items_sorted:
                sample = result['results'][i]
                row.append(sample['generated_text'])

            writer.writerow(row)

    print(f"All-levels readable CSV exported to: {output_path}")
    print(f"Samples: {len(baseline_samples)}")
    print(f"Compression levels: {len(items_sorted)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export QA results in simple readable format"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as results)")
    parser.add_argument("--all-levels", action="store_true",
                        help="Export all compression levels (not just key ones)")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results_data = load_results(args.results)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        results_path = Path(args.results)
        output_dir = results_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    config = results_data['config']
    print(f"\nModel: {config['model_name']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Samples: {config['num_samples']}")

    print("\n" + "=" * 70)

    if args.all_levels:
        print("Exporting ALL compression levels...")
        export_all_levels_readable(results_data, output_dir / 'readable_all_levels.csv')
    else:
        print("Exporting key compression levels (75%, 90%, 95%, 99.5%)...")
        export_simple_readable(results_data, output_dir / 'readable_key_levels.csv')

    print("=" * 70)


if __name__ == "__main__":
    main()
