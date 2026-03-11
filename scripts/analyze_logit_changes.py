"""
Deep analysis of exp 23 logit change results.

Analyzes:
1. Per-layer token shift profiles (early vs mid vs late layers)
2. Answer token specificity (correct answer vs all A/B/C/D)
3. Token categories (answer letters, reasoning, punctuation, gibberish)
4. Question fragility (which questions are most affected)
5. Comparison: mlp_out only vs paired

Usage:
    python scripts/analyze_logit_changes.py results/logit_change/logit_change_*.pkl
"""

import sys
sys.path.append('.')

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import re


# =============================================================================
# Token categorization
# =============================================================================

ANSWER_LETTERS = {'A', 'B', 'C', 'D', ' A', ' B', ' C', ' D'}
ANSWER_BIGRAMS = {'AB', 'AC', 'AD', 'BA', 'BC', 'BD', 'CA', 'CB', 'CD',
                  'DA', 'DB', 'DC', 'AA', 'BB', 'CC', 'DD'}
REASONING_WORDS = {'correct', 'Correct', 'CORRECT', 'answer', 'Answer', 'ANSWER',
                   'right', 'Right', 'wrong', 'Wrong', 'true', 'True', 'false', 'False',
                   'yes', 'Yes', 'no', 'No', 'both', 'Both', 'all', 'All', 'ALL',
                   'none', 'None', 'NONE', 'neither', 'Neither',
                   'because', 'Because', 'therefore', 'Therefore', 'since', 'Since',
                   'however', 'However', 'but', 'But', 'not', 'Not'}
FORMAT_WORDS = {'Option', 'option', 'Options', 'options', 'Question', 'question',
                'choice', 'Choice', ':', '.', ',', '(', ')', '[', ']', '\n', '**', '--'}


def categorize_token(token_str):
    """Categorize a token into semantic groups."""
    s = token_str.strip()
    if s in ANSWER_LETTERS or token_str in ANSWER_LETTERS:
        return 'answer_letter'
    if s in ANSWER_BIGRAMS:
        return 'answer_bigram'
    if s in REASONING_WORDS or token_str in REASONING_WORDS:
        return 'reasoning'
    if s in FORMAT_WORDS or token_str in FORMAT_WORDS:
        return 'format'
    # Check if it's mostly non-ASCII (multilingual/special)
    if s and sum(1 for c in s if ord(c) > 127) > len(s) / 2:
        return 'non_ascii'
    # Check if it looks like gibberish (very short subword pieces)
    if len(s) <= 2 and s.isalpha():
        return 'subword'
    return 'other'


# =============================================================================
# Analysis functions
# =============================================================================

def analyze_per_layer_profiles(data):
    """Show which token categories shift most at each layer depth."""
    print("\n" + "=" * 80)
    print("PER-LAYER TOKEN SHIFT PROFILES (important chunks only)")
    print("=" * 80)

    layer_groups = {
        'Early (0-4)': list(range(0, 5)),
        'Mid (5-10)': list(range(5, 11)),
        'Late (11-15)': list(range(11, 16)),
    }

    for group_name, group_layers in layer_groups.items():
        inc_cats = defaultdict(lambda: {'count': 0, 'total_diff': 0.0})
        dec_cats = defaultdict(lambda: {'count': 0, 'total_diff': 0.0})

        for layer_idx in group_layers:
            if layer_idx not in data['per_layer']:
                continue
            for chunk in data['per_layer'][layer_idx]['chunks_tested']:
                if chunk['importance'] != 'important':
                    continue
                for t in chunk['aggregate_top_increases'][:20]:
                    cat = categorize_token(t['token_str'])
                    inc_cats[cat]['count'] += t['count']
                    inc_cats[cat]['total_diff'] += t['mean_logit_diff'] * t['count']
                for t in chunk['aggregate_top_decreases'][:20]:
                    cat = categorize_token(t['token_str'])
                    dec_cats[cat]['count'] += t['count']
                    dec_cats[cat]['total_diff'] += t['mean_logit_diff'] * t['count']

        print(f"\n--- {group_name} ---")
        print("  INCREASED by removal:")
        for cat, info in sorted(inc_cats.items(), key=lambda x: -x[1]['count']):
            mean = info['total_diff'] / max(info['count'], 1)
            print(f"    {cat:<20} count={info['count']:>5}  mean_diff={mean:+.3f}")
        print("  DECREASED by removal:")
        for cat, info in sorted(dec_cats.items(), key=lambda x: -x[1]['count']):
            mean = info['total_diff'] / max(info['count'], 1)
            print(f"    {cat:<20} count={info['count']:>5}  mean_diff={mean:+.3f}")


def analyze_answer_specificity(data):
    """Check if removal specifically suppresses the CORRECT answer token."""
    print("\n" + "=" * 80)
    print("ANSWER TOKEN SPECIFICITY")
    print("=" * 80)
    print("Does chunk removal specifically suppress the correct answer letter?")

    # For each chunk removal, look at how each answer letter's probability changes
    imp_correct_changes = []
    imp_incorrect_changes = []
    unimp_correct_changes = []
    unimp_incorrect_changes = []

    for layer_idx, layer_data in data['per_layer'].items():
        for chunk in layer_data['chunks_tested']:
            for q in chunk['per_question']:
                bl_probs = q['answer_probs_baseline']
                md_probs = q['answer_probs_modified']

                # Find correct letter from baseline
                # We need to figure out which letter was correct
                correct_letter = None
                for letter in ['A', 'B', 'C', 'D']:
                    # The one that was predicted correctly in baseline
                    pass

                # Use flipped info instead
                for letter in ['A', 'B', 'C', 'D']:
                    change = md_probs.get(letter, 0) - bl_probs.get(letter, 0)

                    if letter == q.get('baseline_predicted', ''):
                        # This is what the model was predicting
                        if chunk['importance'] == 'important':
                            imp_correct_changes.append(change)
                        else:
                            unimp_correct_changes.append(change)
                    else:
                        if chunk['importance'] == 'important':
                            imp_incorrect_changes.append(change)
                        else:
                            unimp_incorrect_changes.append(change)

    print(f"\nImportant chunks:")
    print(f"  Predicted letter prob change: {np.mean(imp_correct_changes):+.4f} "
          f"(n={len(imp_correct_changes)})")
    print(f"  Other letters prob change:    {np.mean(imp_incorrect_changes):+.4f} "
          f"(n={len(imp_incorrect_changes)})")
    print(f"  → Predicted letter drops {abs(np.mean(imp_correct_changes))/abs(np.mean(imp_incorrect_changes)):.1f}x "
          f"more than others" if np.mean(imp_incorrect_changes) != 0 else "")

    if unimp_correct_changes:
        print(f"\nUnimportant chunks:")
        print(f"  Predicted letter prob change: {np.mean(unimp_correct_changes):+.4f} "
              f"(n={len(unimp_correct_changes)})")
        print(f"  Other letters prob change:    {np.mean(unimp_incorrect_changes):+.4f} "
              f"(n={len(unimp_incorrect_changes)})")


def analyze_question_fragility(data):
    """Which questions are most affected by chunk removal?"""
    print("\n" + "=" * 80)
    print("QUESTION FRAGILITY ANALYSIS")
    print("=" * 80)

    # Track per-question flip counts and KL
    question_stats = defaultdict(lambda: {
        'flip_count': 0, 'total_kl': 0.0, 'n_tests': 0,
        'baseline_correct': None, 'sample_id': None
    })

    for layer_idx, layer_data in data['per_layer'].items():
        for chunk in layer_data['chunks_tested']:
            if chunk['importance'] != 'important':
                continue
            for q in chunk['per_question']:
                sid = q['sample_id']
                question_stats[sid]['sample_id'] = sid
                question_stats[sid]['n_tests'] += 1
                question_stats[sid]['total_kl'] += q['kl_divergence']
                question_stats[sid]['baseline_correct'] = q['baseline_correct']
                if q['flipped']:
                    question_stats[sid]['flip_count'] += 1

    stats_list = list(question_stats.values())
    stats_list.sort(key=lambda x: x['flip_count'], reverse=True)

    # Summary
    flip_counts = [s['flip_count'] for s in stats_list]
    print(f"\nTotal questions: {len(stats_list)}")
    print(f"Questions with >=1 flip:  {sum(1 for f in flip_counts if f > 0)} "
          f"({sum(1 for f in flip_counts if f > 0)/len(stats_list)*100:.0f}%)")
    print(f"Questions with >=5 flips: {sum(1 for f in flip_counts if f >= 5)} "
          f"({sum(1 for f in flip_counts if f >= 5)/len(stats_list)*100:.0f}%)")
    print(f"Questions with 0 flips:   {sum(1 for f in flip_counts if f == 0)} "
          f"({sum(1 for f in flip_counts if f == 0)/len(stats_list)*100:.0f}%)")

    print(f"\nFlip count distribution:")
    from collections import Counter
    flip_dist = Counter(flip_counts)
    for n_flips in sorted(flip_dist.keys()):
        bar = '#' * flip_dist[n_flips]
        print(f"  {n_flips:>3} flips: {flip_dist[n_flips]:>3} questions  {bar}")

    # Fragile questions (baseline correct but many flips)
    fragile = [s for s in stats_list if s['baseline_correct'] and s['flip_count'] >= 5]
    if fragile:
        print(f"\nMost FRAGILE questions (baseline correct, >=5 flips from important chunk removal):")
        for s in fragile[:10]:
            mean_kl = s['total_kl'] / max(s['n_tests'], 1)
            print(f"  {s['sample_id'][:60]:<60} flips={s['flip_count']:>2}  mean_KL={mean_kl:.4f}")

    # Robust correct
    robust = [s for s in stats_list if s['baseline_correct'] and s['flip_count'] == 0]
    print(f"\nRobust questions (baseline correct, 0 flips): {len(robust)}")

    # Baseline wrong but got fixed by removal
    fixed = [s for s in stats_list if not s['baseline_correct'] and s['flip_count'] > 0]
    print(f"Baseline WRONG, flipped to correct by removal: {len(fixed)}")


def analyze_kl_by_layer_and_chunk_position(data):
    """Show how KL varies by layer depth and chunk position (SV index)."""
    print("\n" + "=" * 80)
    print("KL DIVERGENCE HEATMAP (Layer × Chunk Position)")
    print("=" * 80)

    print(f"\n{'Layer':>5} | ", end='')
    # Get all chunk indices
    all_chunks = set()
    for layer_data in data['per_layer'].values():
        for chunk in layer_data['chunks_tested']:
            all_chunks.add(chunk['chunk_idx'])
    chunk_list = sorted(all_chunks)
    for ci in chunk_list:
        print(f"ch{ci:>2}", end='  ')
    print()
    print('-' * (8 + 6 * len(chunk_list)))

    for layer_idx in sorted(data['per_layer'].keys()):
        print(f"{layer_idx:>5} | ", end='')
        chunk_kls = {c['chunk_idx']: c['mean_kl_div'] for c in data['per_layer'][layer_idx]['chunks_tested']}
        for ci in chunk_list:
            if ci in chunk_kls:
                kl = chunk_kls[ci]
                if np.isnan(kl):
                    print(f"  NaN", end=' ')
                elif kl > 0.1:
                    print(f"{kl:5.3f}", end=' ')
                elif kl > 0.01:
                    print(f"{kl:5.3f}", end=' ')
                else:
                    print(f"{kl:5.4f}"[:5], end=' ')
            else:
                print(f"    -", end=' ')
        print()


def compare_paired_vs_single(single_path, paired_path):
    """Compare mlp_out-only vs paired removal."""
    print("\n" + "=" * 80)
    print("COMPARISON: mlp_out only vs paired (mlp_in+mlp_out)")
    print("=" * 80)

    with open(single_path, 'rb') as f:
        single = pickle.load(f)
    with open(paired_path, 'rb') as f:
        paired = pickle.load(f)

    print(f"\n{'Metric':<30} {'mlp_out':>12} {'paired':>12} {'ratio':>8}")
    print("-" * 65)

    for label, d in [('mlp_out', single), ('paired', paired)]:
        globals()[f'{label}_imp_kl'] = []
        globals()[f'{label}_unimp_kl'] = []
        for layer_data in d['per_layer'].values():
            for chunk in layer_data['chunks_tested']:
                if np.isnan(chunk['mean_kl_div']):
                    continue
                if chunk['importance'] == 'important':
                    globals()[f'{label}_imp_kl'].append(chunk['mean_kl_div'])
                else:
                    globals()[f'{label}_unimp_kl'].append(chunk['mean_kl_div'])

    s_imp = np.mean(mlp_out_imp_kl)
    p_imp = np.mean(paired_imp_kl)
    print(f"{'Important mean KL':<30} {s_imp:>12.4f} {p_imp:>12.4f} {p_imp/s_imp:>7.1f}x")

    s_unimp = np.mean(mlp_out_unimp_kl)
    p_unimp = np.mean(paired_unimp_kl)
    print(f"{'Unimportant mean KL':<30} {s_unimp:>12.4f} {p_unimp:>12.4f} {p_unimp/s_unimp:>7.1f}x")

    print(f"{'KL ratio (imp/unimp)':<30} {s_imp/s_unimp:>12.1f}x {p_imp/p_unimp:>12.1f}x")

    # Per-layer chunk 0 comparison
    print(f"\nChunk 0 KL by layer:")
    print(f"{'Layer':>5} {'mlp_out':>10} {'paired':>10} {'ratio':>8}")
    print("-" * 38)
    for layer_idx in sorted(single['per_layer'].keys()):
        s_kl = None
        p_kl = None
        for c in single['per_layer'][layer_idx]['chunks_tested']:
            if c['chunk_idx'] == 0:
                s_kl = c['mean_kl_div']
        for c in paired['per_layer'][layer_idx]['chunks_tested']:
            if c['chunk_idx'] == 0:
                p_kl = c['mean_kl_div']
        if s_kl is not None and p_kl is not None:
            s_str = 'NaN' if np.isnan(s_kl) else '%.4f' % s_kl
            p_str = 'NaN' if np.isnan(p_kl) else '%.4f' % p_kl
            if not np.isnan(s_kl) and not np.isnan(p_kl) and s_kl > 0:
                r_str = '%.1fx' % (p_kl / s_kl)
            else:
                r_str = '-'
            print(f"{layer_idx:>5} {s_str:>10} {p_str:>10} {r_str:>8}")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_logit_changes.py <result.pkl> [paired.pkl]")
        sys.exit(1)

    pkl_path = sys.argv[1]
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Config: {data['config']['num_questions']} questions, "
          f"layers {data['config']['layers']}, "
          f"paired={data['config']['paired']}")

    # Run all analyses
    analyze_kl_by_layer_and_chunk_position(data)
    analyze_per_layer_profiles(data)
    analyze_answer_specificity(data)
    analyze_question_fragility(data)

    # If paired result also provided, compare
    if len(sys.argv) >= 3:
        compare_paired_vs_single(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
