"""
Analyze Open-Ended QA SVD Truncation Results

Compares generation quality across different compression levels and identifies
which questions are affected by Layer 31 MLP compression.

Usage:
    python scripts/analyze_open_qa_svd.py --results results/open_qa_svd/.../results.pkl
    python scripts/analyze_open_qa_svd.py --results results/open_qa_svd/.../results.pkl --show-examples
"""

import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def analyze_compression_effects(results_data):
    """
    Analyze how SVD compression affects generation quality.

    Returns detailed statistics and examples of changes.
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']

    print("=" * 80)
    print("COMPRESSION EFFECTS ANALYSIS")
    print("=" * 80)

    # Overall accuracy comparison
    print("\n1. ACCURACY ACROSS COMPRESSION LEVELS")
    print("-" * 80)
    print(f"{'Reduction %':>12} {'k':>6} {'Accuracy':>10} {'Δ vs Baseline':>15} {'Energy %':>10}")
    print("-" * 80)

    baseline_acc = baseline['accuracy']
    print(f"{'Baseline':>12} {baseline['k']:>6} {baseline_acc*100:>9.1f}% {'---':>15} {baseline['energy_retention']*100:>9.1f}%")

    # Sort by reduction percentage
    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    for key, result in items_sorted:
        acc = result['accuracy']
        delta = (acc - baseline_acc) * 100
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
        k = result['k']
        energy = result['energy_retention']

        print(f"{red_pct:>11.1f}% {k:>6} {acc*100:>9.1f}% {delta:>+14.1f}pp {energy*100:>9.1f}%")

    # Find questions that changed
    print("\n2. ANSWER CHANGES BY COMPRESSION LEVEL")
    print("-" * 80)

    baseline_results = baseline['results']
    num_samples = len(baseline_results)

    change_summary = {}

    for key, result in items_sorted:
        compressed_results = result['results']
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0

        # Count changes
        became_correct = 0
        became_incorrect = 0
        stayed_same = 0

        changes = []

        for b, c in zip(baseline_results, compressed_results):
            if b['is_correct'] == c['is_correct']:
                stayed_same += 1
            elif c['is_correct']:  # Baseline wrong, compressed right
                became_correct += 1
                changes.append({
                    'type': 'improved',
                    'question': b['question'],
                    'gold': b['gold_answers'][0],
                    'baseline_gen': b['generated_text'],
                    'compressed_gen': c['generated_text']
                })
            else:  # Baseline right, compressed wrong
                became_incorrect += 1
                changes.append({
                    'type': 'degraded',
                    'question': b['question'],
                    'gold': b['gold_answers'][0],
                    'baseline_gen': b['generated_text'],
                    'compressed_gen': c['generated_text']
                })

        change_summary[red_pct] = {
            'became_correct': became_correct,
            'became_incorrect': became_incorrect,
            'stayed_same': stayed_same,
            'changes': changes,
            'k': result['k']
        }

        print(f"\nReduction {red_pct:.1f}% (k={result['k']}):")
        print(f"  Improved:  {became_correct:>3} questions ({became_correct/num_samples*100:.1f}%)")
        print(f"  Degraded:  {became_incorrect:>3} questions ({became_incorrect/num_samples*100:.1f}%)")
        print(f"  Unchanged: {stayed_same:>3} questions ({stayed_same/num_samples*100:.1f}%)")
        print(f"  Net change: {became_correct - became_incorrect:+d} questions")

    return change_summary


def show_example_changes(change_summary, max_examples=3):
    """Show example questions that changed with compression."""
    print("\n3. EXAMPLE ANSWER CHANGES")
    print("=" * 80)

    for red_pct, data in sorted(change_summary.items()):
        changes = data['changes']

        if not changes:
            continue

        print(f"\n--- Reduction {red_pct:.1f}% (k={data['k']}) ---")

        # Show improvements
        improvements = [c for c in changes if c['type'] == 'improved']
        if improvements:
            print(f"\nImproved ({len(improvements)} total), showing up to {max_examples}:")
            for i, change in enumerate(improvements[:max_examples]):
                print(f"\n  Example {i+1}:")
                print(f"    Q: {change['question']}")
                print(f"    Gold: {change['gold']}")
                print(f"    Baseline:   {change['baseline_gen']} ❌")
                print(f"    Compressed: {change['compressed_gen']} ✅")

        # Show degradations
        degradations = [c for c in changes if c['type'] == 'degraded']
        if degradations:
            print(f"\nDegraded ({len(degradations)} total), showing up to {max_examples}:")
            for i, change in enumerate(degradations[:max_examples]):
                print(f"\n  Example {i+1}:")
                print(f"    Q: {change['question']}")
                print(f"    Gold: {change['gold']}")
                print(f"    Baseline:   {change['baseline_gen']} ✅")
                print(f"    Compressed: {change['compressed_gen']} ❌")


def analyze_generation_length(results_data):
    """Analyze how compression affects generation length."""
    all_results = results_data['all_results']
    baseline = all_results['baseline']

    print("\n4. GENERATION LENGTH ANALYSIS")
    print("=" * 80)

    baseline_results = baseline['results']
    baseline_lengths = [r['generated_length'] for r in baseline_results]
    baseline_avg = np.mean(baseline_lengths)

    print(f"Baseline average length: {baseline_avg:.1f} tokens")
    print(f"\nLength changes by compression level:")
    print(f"{'Reduction %':>12} {'k':>6} {'Avg Length':>12} {'Δ vs Baseline':>15}")
    print("-" * 80)

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    for key, result in items_sorted:
        compressed_results = result['results']
        compressed_lengths = [r['generated_length'] for r in compressed_results]
        compressed_avg = np.mean(compressed_lengths)

        delta = compressed_avg - baseline_avg
        red_pct = result['reduction_pct'] if result['reduction_pct'] else 0

        print(f"{red_pct:>11.1f}% {result['k']:>6} {compressed_avg:>11.1f} {delta:>+14.1f}")


def find_optimal_compression(results_data):
    """Find the compression level with best accuracy."""
    all_results = results_data['all_results']
    baseline_acc = all_results['baseline']['accuracy']

    print("\n5. OPTIMAL COMPRESSION LEVEL")
    print("=" * 80)

    best_acc = baseline_acc
    best_reduction = 0
    best_k = all_results['baseline']['k']

    items = [(k, v) for k, v in all_results.items() if k != 'baseline']

    for key, result in items:
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            best_reduction = result['reduction_pct'] if result['reduction_pct'] else 0
            best_k = result['k']

    improvement = (best_acc - baseline_acc) * 100

    print(f"Best accuracy: {best_acc*100:.1f}%")
    print(f"Achieved at: {best_reduction:.1f}% reduction (k={best_k})")
    print(f"Improvement: {improvement:+.1f}pp over baseline")

    if best_reduction >= 90:
        print(f"\n✅ Layer 31 MLP can be compressed by {best_reduction:.1f}% with accuracy improvement!")
        print(f"   This confirms redundancy for open-ended generation tasks.")
    else:
        print(f"\n⚠️  Best compression is only {best_reduction:.1f}%")
        print(f"   Layer 31 may be more important for generation than MCQ.")


def compare_with_mcq(openqa_results_path, mcq_results_path):
    """Compare open QA results with MCQ results."""
    print("\n6. COMPARISON WITH MCQ RESULTS")
    print("=" * 80)

    try:
        with open(mcq_results_path, 'rb') as f:
            mcq_data = pickle.load(f)

        openqa_data = load_results(openqa_results_path)

        # Compare baseline
        mcq_baseline = mcq_data['all_results']['baseline']
        openqa_baseline = openqa_data['all_results']['baseline']

        print(f"Baseline Accuracy:")
        print(f"  MCQ:     {mcq_baseline['accuracy']*100:.1f}%")
        print(f"  Open QA: {openqa_baseline['accuracy']*100:.1f}%")

        # Compare at key compression levels
        print(f"\nAccuracy at Key Compression Levels:")
        print(f"{'Reduction %':>12} {'MCQ':>10} {'Open QA':>10} {'Difference':>12}")
        print("-" * 80)

        for red_pct in [90, 95, 99]:
            # Find matching entries
            mcq_key = f"k{red_pct}_red{red_pct}" if f"k{red_pct}_red{red_pct}" in mcq_data['all_results'] else None
            openqa_key = None

            # Find by reduction percentage
            for k, v in mcq_data['all_results'].items():
                if k != 'baseline' and v.get('reduction_pct') == red_pct:
                    mcq_key = k
                    break

            for k, v in openqa_data['all_results'].items():
                if k != 'baseline' and v.get('reduction_pct') == red_pct:
                    openqa_key = k
                    break

            if mcq_key and openqa_key:
                mcq_acc = mcq_data['all_results'][mcq_key]['accuracy']
                openqa_acc = openqa_data['all_results'][openqa_key]['accuracy']
                diff = (openqa_acc - mcq_acc) * 100

                print(f"{red_pct:>11.1f}% {mcq_acc*100:>9.1f}% {openqa_acc*100:>9.1f}% {diff:>+11.1f}pp")

        print("\nConclusion:")
        print("If both MCQ and Open QA show flat/improving accuracy at high compression,")
        print("this strongly confirms Layer 31 MLP redundancy across task types.")

    except FileNotFoundError:
        print("MCQ results file not found. Skipping comparison.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze open-ended QA SVD truncation results"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--show-examples", action="store_true",
                        help="Show example answer changes")
    parser.add_argument("--max-examples", type=int, default=3,
                        help="Max examples to show per compression level")
    parser.add_argument("--compare-mcq", type=str,
                        help="Path to MCQ results.pkl for comparison")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OPEN-ENDED QA SVD TRUNCATION ANALYSIS")
    print("=" * 80)

    # Load results
    print(f"\nLoading results from: {args.results}")
    results_data = load_results(args.results)

    config = results_data['config']
    print(f"\nExperiment Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Dataset: {config['dataset_name']}")
    print(f"  Samples: {config['num_samples']}")
    print(f"  Target Layer: {config['target_layer']}")
    print(f"  Matrix Type: {config['matrix_type']}")
    print(f"  Mode: {config['mode']}")

    # Run analyses
    change_summary = analyze_compression_effects(results_data)

    if args.show_examples:
        show_example_changes(change_summary, max_examples=args.max_examples)

    analyze_generation_length(results_data)
    find_optimal_compression(results_data)

    if args.compare_mcq:
        compare_with_mcq(args.results, args.compare_mcq)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
