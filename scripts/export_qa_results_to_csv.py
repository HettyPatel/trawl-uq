"""
Export Open QA SVD Results to CSV for Manual Analysis

Creates a detailed CSV file with all responses across compression levels
for easy analysis in Excel/spreadsheet tools.

Usage:
    python scripts/export_qa_results_to_csv.py --results results/open_qa_svd/.../results.pkl
"""

import pickle
import argparse
import csv
from pathlib import Path


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def export_to_csv(results_data, output_path):
    """
    Export results to CSV with one row per question.

    Columns:
    - sample_id
    - question
    - gold_answers
    - baseline_response
    - baseline_correct
    - baseline_length
    - [for each compression level]:
        - {reduction_pct}%_response
        - {reduction_pct}%_correct
        - {reduction_pct}%_length
        - {reduction_pct}%_change_type (improved/degraded/unchanged)
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    # Get all compression levels sorted
    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    # Build header
    header = [
        'sample_id',
        'question',
        'gold_answers',
        'baseline_response',
        'baseline_correct',
        'baseline_length'
    ]

    # Add columns for each compression level
    for key, result in items_sorted:
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        k = result['k']
        prefix = f"{red_pct:.1f}%_k{k}"
        header.extend([
            f"{prefix}_response",
            f"{prefix}_correct",
            f"{prefix}_length",
            f"{prefix}_change"
        ])

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write each sample
        for i, baseline_sample in enumerate(baseline_samples):
            row = [
                i,
                baseline_sample['question'],
                ' | '.join(baseline_sample['gold_answers']),
                baseline_sample['generated_text'],
                baseline_sample['is_correct'],
                baseline_sample['generated_length']
            ]

            # Add compressed results
            for key, result in items_sorted:
                compressed_sample = result['results'][i]

                # Determine change type
                if baseline_sample['is_correct'] == compressed_sample['is_correct']:
                    change_type = 'unchanged'
                elif compressed_sample['is_correct']:
                    change_type = 'improved'
                else:
                    change_type = 'degraded'

                row.extend([
                    compressed_sample['generated_text'],
                    compressed_sample['is_correct'],
                    compressed_sample['generated_length'],
                    change_type
                ])

            writer.writerow(row)

    print(f"CSV exported to: {output_path}")
    print(f"Total samples: {len(baseline_samples)}")
    print(f"Compression levels: {len(items_sorted)}")


def export_summary_csv(results_data, output_path):
    """
    Export summary statistics to CSV.

    One row per compression level with aggregate metrics.
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'reduction_pct',
            'k',
            'accuracy',
            'delta_vs_baseline',
            'avg_length',
            'energy_retention',
            'num_improved',
            'num_degraded',
            'num_unchanged',
            'net_change'
        ])

        # Baseline row
        baseline_acc = baseline['accuracy']
        baseline_avg_len = sum(r['generated_length'] for r in baseline['results']) / len(baseline['results'])

        writer.writerow([
            0.0,
            baseline['k'],
            baseline_acc,
            0.0,
            baseline_avg_len,
            baseline['energy_retention'],
            0,
            0,
            len(baseline['results']),
            0
        ])

        # Compressed levels
        baseline_results = baseline['results']

        for key, result in items_sorted:
            red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
            accuracy = result['accuracy']
            delta = accuracy - baseline_acc

            compressed_results = result['results']
            avg_len = sum(r['generated_length'] for r in compressed_results) / len(compressed_results)

            # Count changes
            improved = 0
            degraded = 0
            unchanged = 0

            for b, c in zip(baseline_results, compressed_results):
                if b['is_correct'] == c['is_correct']:
                    unchanged += 1
                elif c['is_correct']:
                    improved += 1
                else:
                    degraded += 1

            net_change = improved - degraded

            writer.writerow([
                red_pct,
                result['k'],
                accuracy,
                delta,
                avg_len,
                result['energy_retention'],
                improved,
                degraded,
                unchanged,
                net_change
            ])

    print(f"Summary CSV exported to: {output_path}")


def export_changes_only_csv(results_data, output_path):
    """
    Export only samples that changed at any compression level.

    Easier to focus on what actually changed.
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    # Identify samples that changed at ANY compression level
    changed_samples = set()

    for i, baseline_sample in enumerate(baseline_samples):
        for key, result in items_sorted:
            compressed_sample = result['results'][i]
            if baseline_sample['is_correct'] != compressed_sample['is_correct']:
                changed_samples.add(i)
                break

    print(f"Found {len(changed_samples)} samples with correctness changes")

    # Build header
    header = [
        'sample_id',
        'question',
        'gold_answers',
        'baseline_response',
        'baseline_correct',
        'first_change_at',  # First compression level where it changed
        'change_type'  # improved or degraded
    ]

    # Add columns for each compression level
    for key, result in items_sorted:
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        k = result['k']
        prefix = f"{red_pct:.1f}%_k{k}"
        header.extend([
            f"{prefix}_response",
            f"{prefix}_correct"
        ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write only changed samples
        for i in sorted(changed_samples):
            baseline_sample = baseline_samples[i]

            # Find first change
            first_change_pct = None
            change_type = None

            for key, result in items_sorted:
                compressed_sample = result['results'][i]
                if baseline_sample['is_correct'] != compressed_sample['is_correct']:
                    red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
                    first_change_pct = f"{red_pct:.1f}%"
                    change_type = 'improved' if compressed_sample['is_correct'] else 'degraded'
                    break

            row = [
                i,
                baseline_sample['question'],
                ' | '.join(baseline_sample['gold_answers']),
                baseline_sample['generated_text'],
                baseline_sample['is_correct'],
                first_change_pct,
                change_type
            ]

            # Add compressed results
            for key, result in items_sorted:
                compressed_sample = result['results'][i]
                row.extend([
                    compressed_sample['generated_text'],
                    compressed_sample['is_correct']
                ])

            writer.writerow(row)

    print(f"Changes-only CSV exported to: {output_path}")
    print(f"Samples with changes: {len(changed_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export open QA SVD results to CSV files"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as results)")

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
    print(f"Layer: {config['target_layer']}")
    print(f"Matrix: {config['matrix_type']}")

    # Export all three CSV types
    print("\n" + "=" * 70)
    print("EXPORTING CSV FILES")
    print("=" * 70)

    print("\n1. Full results (all samples, all compression levels)...")
    export_to_csv(results_data, output_dir / 'results_full.csv')

    print("\n2. Summary statistics...")
    export_summary_csv(results_data, output_dir / 'results_summary.csv')

    print("\n3. Changes only (samples with correctness changes)...")
    export_changes_only_csv(results_data, output_dir / 'results_changes_only.csv')

    print("\n" + "=" * 70)
    print(f"All CSV files saved to: {output_dir}")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  - results_full.csv: Complete dataset with all responses")
    print(f"  - results_summary.csv: Aggregate statistics per compression level")
    print(f"  - results_changes_only.csv: Only samples that changed correctness")


if __name__ == "__main__":
    main()
