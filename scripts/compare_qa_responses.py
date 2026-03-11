"""
Compare Generated Responses Across Compression Levels

Analyzes how actual generated text changes with SVD compression,
showing side-by-side comparisons of responses.

Usage:
    python scripts/compare_qa_responses.py --results results/open_qa_svd/.../results.pkl
    python scripts/compare_qa_responses.py --results results/open_qa_svd/.../results.pkl --sample-ids 0 1 2 3 4
    python scripts/compare_qa_responses.py --results results/open_qa_svd/.../results.pkl --show-all-changed
"""

import pickle
import argparse
from pathlib import Path


def load_results(pkl_path):
    """Load results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compare_specific_samples(results_data, sample_ids=None, max_samples=10):
    """
    Show how responses change for specific samples across all compression levels.

    Args:
        results_data: Loaded results dictionary
        sample_ids: List of sample indices to show (None = first max_samples)
        max_samples: Maximum samples to show if sample_ids not specified
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    if sample_ids is None:
        sample_ids = list(range(min(max_samples, len(baseline_samples))))

    # Get all compression levels sorted
    items = [(k, v) for k, v in all_results.items() if k != 'baseline']
    items_sorted = sorted(items, key=lambda x: x[1]['reduction_pct'] if x[1]['reduction_pct'] else 0)

    print("=" * 100)
    print("RESPONSE COMPARISON ACROSS COMPRESSION LEVELS")
    print("=" * 100)

    for sample_id in sample_ids:
        baseline_sample = baseline_samples[sample_id]

        print(f"\n{'='*100}")
        print(f"SAMPLE {sample_id}")
        print(f"{'='*100}")
        print(f"Question: {baseline_sample['question']}")
        print(f"Gold Answer: {baseline_sample['gold_answers'][0]}")
        print(f"\n{'-'*100}")

        # Show baseline
        print(f"\nBASELINE (k=4096):")
        print(f"  Generated: {baseline_sample['generated_text']}")
        print(f"  Correct: {baseline_sample['is_correct']} ✅" if baseline_sample['is_correct'] else f"  Correct: {baseline_sample['is_correct']} ❌")
        print(f"  Length: {baseline_sample['generated_length']} tokens")

        # Show compressed versions
        for key, result in items_sorted:
            compressed_sample = result['results'][sample_id]
            red_pct = result['reduction_pct'] if result['reduction_pct'] else 0
            k = result['k']

            # Check if response changed
            response_changed = compressed_sample['generated_text'] != baseline_sample['generated_text']
            correctness_changed = compressed_sample['is_correct'] != baseline_sample['is_correct']

            if response_changed or correctness_changed:
                print(f"\n{red_pct:.1f}% REDUCTION (k={k}):")
                print(f"  Generated: {compressed_sample['generated_text']}")

                if correctness_changed:
                    if compressed_sample['is_correct']:
                        print(f"  Correct: {compressed_sample['is_correct']} ✅ (IMPROVED from baseline)")
                    else:
                        print(f"  Correct: {compressed_sample['is_correct']} ❌ (DEGRADED from baseline)")
                else:
                    print(f"  Correct: {compressed_sample['is_correct']} ✅" if compressed_sample['is_correct'] else f"  Correct: {compressed_sample['is_correct']} ❌")

                print(f"  Length: {compressed_sample['generated_length']} tokens")

                if response_changed:
                    print(f"  [RESPONSE CHANGED]")


def show_changed_responses(results_data, reduction_pct, max_examples=20):
    """
    Show all responses that changed at a specific compression level.

    Args:
        results_data: Loaded results dictionary
        reduction_pct: Which compression level to analyze
        max_examples: Maximum number of examples to show
    """
    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    # Find the matching compression level
    target_result = None
    for key, result in all_results.items():
        if key != 'baseline' and result.get('reduction_pct') == reduction_pct:
            target_result = result
            break

    if target_result is None:
        print(f"ERROR: No results found for {reduction_pct}% reduction")
        return

    compressed_samples = target_result['results']
    k = target_result['k']

    print("=" * 100)
    print(f"ALL CHANGED RESPONSES AT {reduction_pct}% REDUCTION (k={k})")
    print("=" * 100)

    # Collect all changes
    changes = {
        'improved': [],
        'degraded': [],
        'response_changed_only': []
    }

    for i, (baseline_s, compressed_s) in enumerate(zip(baseline_samples, compressed_samples)):
        response_changed = baseline_s['generated_text'] != compressed_s['generated_text']
        correctness_changed = baseline_s['is_correct'] != compressed_s['is_correct']

        if correctness_changed:
            if compressed_s['is_correct']:
                changes['improved'].append((i, baseline_s, compressed_s))
            else:
                changes['degraded'].append((i, baseline_s, compressed_s))
        elif response_changed:
            changes['response_changed_only'].append((i, baseline_s, compressed_s))

    # Show improvements
    print(f"\n{'='*100}")
    print(f"IMPROVED ANSWERS ({len(changes['improved'])} total)")
    print(f"{'='*100}")

    for idx, (i, baseline_s, compressed_s) in enumerate(changes['improved'][:max_examples]):
        print(f"\nExample {idx+1} (Sample {i}):")
        print(f"  Q: {baseline_s['question']}")
        print(f"  Gold: {baseline_s['gold_answers'][0]}")
        print(f"  Baseline:   {baseline_s['generated_text']} ❌")
        print(f"  Compressed: {compressed_s['generated_text']} ✅")

    if len(changes['improved']) > max_examples:
        print(f"\n... and {len(changes['improved']) - max_examples} more improved answers")

    # Show degradations
    print(f"\n{'='*100}")
    print(f"DEGRADED ANSWERS ({len(changes['degraded'])} total)")
    print(f"{'='*100}")

    for idx, (i, baseline_s, compressed_s) in enumerate(changes['degraded'][:max_examples]):
        print(f"\nExample {idx+1} (Sample {i}):")
        print(f"  Q: {baseline_s['question']}")
        print(f"  Gold: {baseline_s['gold_answers'][0]}")
        print(f"  Baseline:   {baseline_s['generated_text']} ✅")
        print(f"  Compressed: {compressed_s['generated_text']} ❌")

    if len(changes['degraded']) > max_examples:
        print(f"\n... and {len(changes['degraded']) - max_examples} more degraded answers")

    # Show response changes without correctness change
    print(f"\n{'='*100}")
    print(f"RESPONSE CHANGED (correctness unchanged) ({len(changes['response_changed_only'])} total)")
    print(f"{'='*100}")

    for idx, (i, baseline_s, compressed_s) in enumerate(changes['response_changed_only'][:max_examples]):
        status = "✅" if baseline_s['is_correct'] else "❌"
        print(f"\nExample {idx+1} (Sample {i}):")
        print(f"  Q: {baseline_s['question']}")
        print(f"  Gold: {baseline_s['gold_answers'][0]}")
        print(f"  Baseline:   {baseline_s['generated_text']} {status}")
        print(f"  Compressed: {compressed_s['generated_text']} {status}")

    if len(changes['response_changed_only']) > max_examples:
        print(f"\n... and {len(changes['response_changed_only']) - max_examples} more changed responses")


def analyze_response_quality(results_data, reduction_pcts=None):
    """
    Analyze qualitative aspects of generated responses across compression levels.

    Args:
        results_data: Loaded results dictionary
        reduction_pcts: Which compression levels to analyze (None = key levels)
    """
    if reduction_pcts is None:
        reduction_pcts = [10, 50, 75, 90, 95, 99]

    all_results = results_data['all_results']
    baseline = all_results['baseline']
    baseline_samples = baseline['results']

    print("=" * 100)
    print("RESPONSE QUALITY ANALYSIS")
    print("=" * 100)

    for red_pct in reduction_pcts:
        # Find matching result
        target_result = None
        for key, result in all_results.items():
            if key != 'baseline' and result.get('reduction_pct') == red_pct:
                target_result = result
                break

        if target_result is None:
            continue

        compressed_samples = target_result['results']
        k = target_result['k']

        print(f"\n{'-'*100}")
        print(f"Reduction: {red_pct}% (k={k})")
        print(f"{'-'*100}")

        # Count different types of changes
        total_changes = 0
        shorter_responses = 0
        longer_responses = 0
        same_length = 0
        hallucinations = 0  # wrong factual info

        for baseline_s, compressed_s in zip(baseline_samples, compressed_samples):
            if baseline_s['generated_text'] != compressed_s['generated_text']:
                total_changes += 1

                if compressed_s['generated_length'] < baseline_s['generated_length']:
                    shorter_responses += 1
                elif compressed_s['generated_length'] > baseline_s['generated_length']:
                    longer_responses += 1
                else:
                    same_length += 1

                # Check for potential hallucination (wrong answer when baseline was right)
                if baseline_s['is_correct'] and not compressed_s['is_correct']:
                    hallucinations += 1

        print(f"  Total responses changed: {total_changes}/{len(baseline_samples)} ({total_changes/len(baseline_samples)*100:.1f}%)")
        print(f"  Shorter responses: {shorter_responses}")
        print(f"  Longer responses: {longer_responses}")
        print(f"  Same length but different: {same_length}")
        print(f"  Potential hallucinations: {hallucinations}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated responses across compression levels"
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results.pkl file")
    parser.add_argument("--sample-ids", type=int, nargs="+",
                        help="Specific sample IDs to compare")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Max samples to show if sample-ids not specified")
    parser.add_argument("--show-changed-at", type=float,
                        help="Show all changed responses at specific reduction % (e.g., 90)")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Max examples to show for changed responses")
    parser.add_argument("--analyze-quality", action="store_true",
                        help="Analyze response quality metrics")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results_data = load_results(args.results)

    config = results_data['config']
    print(f"\nModel: {config['model_name']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Samples: {config['num_samples']}")
    print(f"Layer: {config['target_layer']}")
    print(f"Matrix: {config['matrix_type']}")

    # Run requested analysis
    if args.show_changed_at is not None:
        show_changed_responses(results_data, args.show_changed_at, max_examples=args.max_examples)
    elif args.analyze_quality:
        analyze_response_quality(results_data)
    else:
        compare_specific_samples(results_data, sample_ids=args.sample_ids, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
