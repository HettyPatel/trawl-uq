"""
SAE Feature Quadrant Analysis: Disentangling Uncertainty from Incorrectness

Uses the 2x2 matrix of (correct/incorrect) x (confident/uncertain) to identify:
  - Pure uncertainty features: fire on uncertain predictions regardless of correctness
  - Pure incorrectness features: fire on incorrect predictions regardless of confidence
  - Combination features: fire only when both uncertain AND incorrect

Quadrants (using median entropy as threshold):
  Group A: Correct + Confident (low entropy)
  Group B: Incorrect + Confident (low entropy)  -- confidently wrong
  Group C: Correct + Uncertain (high entropy)    -- lucky guess / hedging
  Group D: Incorrect + Uncertain (high entropy)  -- uncertain and wrong

Comparisons:
  1. Uncertainty features: (C+D) vs (A+B) -- high vs low entropy, ignoring correctness
  2. Incorrectness features: (B+D) vs (A+C) -- incorrect vs correct, ignoring entropy
  3. Pure uncertainty: C vs A (both correct, but different entropy)
  4. Pure incorrectness: B vs A (both confident, but different correctness)
  5. Interaction: features unique to D (uncertain AND wrong)

Usage:
    python scripts/analyze_sae_quadrant.py \
        --pickle results/sae_uncertainty/sae_uncertainty_Llama-3.1-8B.pkl \
        --output-dir results/sae_quadrant/
"""

import pickle
import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats
import argparse


def load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def build_activation_matrix(questions, layer_idx, active_features, feat_to_col):
    """Build a dense activation matrix for a set of questions."""
    n_active = len(active_features)
    acts = np.zeros((len(questions), n_active))
    for i, d in enumerate(questions):
        if layer_idx in d['sae_features']:
            sf = d['sae_features'][layer_idx]
            for idx, val in zip(sf['indices'], sf['values']):
                if idx in feat_to_col:
                    acts[i, feat_to_col[idx]] = val
    return acts


def differential_analysis(group1_acts, group2_acts, active_features, min_freq=0.01):
    """
    Compare two groups of activations. Positive effect = more active in group2.
    Returns list of feature results sorted by |effect_size|.
    """
    results = []
    for col_idx, feat_idx in enumerate(active_features):
        g1 = group1_acts[:, col_idx]
        g2 = group2_acts[:, col_idx]

        g1_freq = (g1 > 0).mean()
        g2_freq = (g2 > 0).mean()

        if g1_freq < min_freq and g2_freq < min_freq:
            continue

        g1_mean = g1.mean()
        g2_mean = g2.mean()

        try:
            _, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        except ValueError:
            p_value = 1.0

        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        effect_size = (g2_mean - g1_mean) / pooled_std if pooled_std > 0 else 0.0

        results.append({
            'feature_idx': feat_idx,
            'group1_mean': float(g1_mean),
            'group2_mean': float(g2_mean),
            'group1_freq': float(g1_freq),
            'group2_freq': float(g2_freq),
            'effect_size': float(effect_size),
            'p_value': float(p_value),
        })

    results.sort(key=lambda x: abs(x['effect_size']), reverse=True)
    return results


def classify_feature(feat_idx, pure_unc_set, pure_inc_set):
    """Classify a feature based on which comparisons it appears in."""
    in_unc = feat_idx in pure_unc_set
    in_inc = feat_idx in pure_inc_set
    if in_unc and in_inc:
        return 'both'
    elif in_unc:
        return 'pure_uncertainty'
    elif in_inc:
        return 'pure_incorrectness'
    else:
        return 'neither'


def main():
    parser = argparse.ArgumentParser(description="SAE Quadrant Analysis")
    parser.add_argument('--pickle', type=str,
                        default='results/sae_uncertainty/sae_uncertainty_Llama-3.1-8B.pkl')
    parser.add_argument('--output-dir', type=str, default='results/sae_quadrant/')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top features to report per comparison')
    parser.add_argument('--p-threshold', type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading results...")
    data = load_results(args.pickle)
    per_question = data['per_question_data']
    sae_layers = data['config']['sae_layers']
    n_total = len(per_question)

    # Compute median entropy for splitting
    all_entropies = [d['entropy'] for d in per_question]
    median_entropy = np.median(all_entropies)
    print(f"Total questions: {n_total}")
    print(f"Median entropy: {median_entropy:.4f}")

    # Build the 4 quadrants
    group_A = []  # correct + confident (low entropy)
    group_B = []  # incorrect + confident
    group_C = []  # correct + uncertain (high entropy)
    group_D = []  # incorrect + uncertain

    for d in per_question:
        is_correct = d['correct']
        is_uncertain = d['entropy'] >= median_entropy

        if is_correct and not is_uncertain:
            group_A.append(d)
        elif not is_correct and not is_uncertain:
            group_B.append(d)
        elif is_correct and is_uncertain:
            group_C.append(d)
        else:  # incorrect and uncertain
            group_D.append(d)

    print(f"\nQuadrant sizes:")
    print(f"  A (correct + confident):   {len(group_A)}")
    print(f"  B (incorrect + confident): {len(group_B)}")
    print(f"  C (correct + uncertain):   {len(group_C)}")
    print(f"  D (incorrect + uncertain): {len(group_D)}")

    # Verify
    assert len(group_A) + len(group_B) + len(group_C) + len(group_D) == n_total

    # Accuracy within quadrants
    n_correct = len(group_A) + len(group_C)
    n_incorrect = len(group_B) + len(group_D)
    print(f"\nOverall accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
    print(f"  Confident half: {len(group_A)}/{len(group_A)+len(group_B)} "
          f"({100*len(group_A)/(len(group_A)+len(group_B)):.1f}%)")
    print(f"  Uncertain half: {len(group_C)}/{len(group_C)+len(group_D)} "
          f"({100*len(group_C)/(len(group_C)+len(group_D)):.1f}%)")

    # Per-layer analysis
    all_comparisons = {}
    summary_rows = []

    for layer_idx in sae_layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        # Find all active features across all questions
        active_features = set()
        for d in per_question:
            if layer_idx in d['sae_features']:
                active_features.update(d['sae_features'][layer_idx]['indices'].tolist())
        active_features = sorted(active_features)
        if not active_features:
            print("  No active features, skipping.")
            continue

        feat_to_col = {f: i for i, f in enumerate(active_features)}

        # Build activation matrices for each group
        acts_A = build_activation_matrix(group_A, layer_idx, active_features, feat_to_col)
        acts_B = build_activation_matrix(group_B, layer_idx, active_features, feat_to_col)
        acts_C = build_activation_matrix(group_C, layer_idx, active_features, feat_to_col)
        acts_D = build_activation_matrix(group_D, layer_idx, active_features, feat_to_col)

        # Comparison 1: Original (incorrect vs correct) - for reference
        acts_correct = np.vstack([acts_A, acts_C])
        acts_incorrect = np.vstack([acts_B, acts_D])
        original = differential_analysis(acts_correct, acts_incorrect, active_features)

        # Comparison 2: Original entropy (high vs low) - for reference
        acts_low_ent = np.vstack([acts_A, acts_B])
        acts_high_ent = np.vstack([acts_C, acts_D])
        original_entropy = differential_analysis(acts_low_ent, acts_high_ent, active_features)

        # Comparison 3: PURE UNCERTAINTY — C vs A
        # Both correct, but C is uncertain and A is confident
        # Positive effect = more active when uncertain (controlling for correctness)
        pure_uncertainty = differential_analysis(acts_A, acts_C, active_features)

        # Comparison 4: PURE INCORRECTNESS — B vs A
        # Both confident, but B is incorrect and A is correct
        # Positive effect = more active when incorrect (controlling for confidence)
        pure_incorrectness = differential_analysis(acts_A, acts_B, active_features)

        # Comparison 5: INTERACTION — D vs A
        # D is uncertain AND incorrect, A is confident AND correct
        # Features here could be either or both
        interaction = differential_analysis(acts_A, acts_D, active_features)

        # Identify significant features in each comparison
        sig_original = {r['feature_idx'] for r in original
                       if r['p_value'] < args.p_threshold and r['effect_size'] > 0}
        sig_entropy = {r['feature_idx'] for r in original_entropy
                      if r['p_value'] < args.p_threshold and r['effect_size'] > 0}
        sig_pure_unc = {r['feature_idx'] for r in pure_uncertainty
                       if r['p_value'] < args.p_threshold and r['effect_size'] > 0}
        sig_pure_inc = {r['feature_idx'] for r in pure_incorrectness
                       if r['p_value'] < args.p_threshold and r['effect_size'] > 0}
        sig_interaction = {r['feature_idx'] for r in interaction
                          if r['p_value'] < args.p_threshold and r['effect_size'] > 0}

        # Classification
        pure_unc_only = sig_pure_unc - sig_pure_inc
        pure_inc_only = sig_pure_inc - sig_pure_unc
        both = sig_pure_unc & sig_pure_inc
        original_overlap = sig_original & sig_entropy

        print(f"  Original (incorrect vs correct): {len(sig_original)} sig features")
        print(f"  Original (high vs low entropy):  {len(sig_entropy)} sig features")
        print(f"  Original overlap:                {len(original_overlap)} "
              f"({100*len(original_overlap)/max(len(sig_original),1):.0f}% of correctness features)")
        print(f"")
        print(f"  PURE uncertainty (C vs A):        {len(sig_pure_unc)} sig features")
        print(f"  PURE incorrectness (B vs A):      {len(sig_pure_inc)} sig features")
        print(f"  Interaction (D vs A):             {len(sig_interaction)} sig features")
        print(f"")
        print(f"  Pure uncertainty ONLY:            {len(pure_unc_only)}")
        print(f"  Pure incorrectness ONLY:          {len(pure_inc_only)}")
        print(f"  Both (confounded):                {len(both)}")

        # Top pure uncertainty features
        if pure_uncertainty:
            top_unc = [r for r in pure_uncertainty if r['feature_idx'] in sig_pure_unc][:5]
            if top_unc:
                print(f"\n  Top PURE uncertainty features (C vs A):")
                for r in top_unc:
                    label = classify_feature(r['feature_idx'], sig_pure_unc, sig_pure_inc)
                    print(f"    #{r['feature_idx']:>6d}  effect={r['effect_size']:+.3f}  "
                          f"p={r['p_value']:.4f}  freq A={r['group1_freq']:.3f} C={r['group2_freq']:.3f}  "
                          f"[{label}]")

        # Top pure incorrectness features
        if pure_incorrectness:
            top_inc = [r for r in pure_incorrectness if r['feature_idx'] in sig_pure_inc][:5]
            if top_inc:
                print(f"\n  Top PURE incorrectness features (B vs A):")
                for r in top_inc:
                    label = classify_feature(r['feature_idx'], sig_pure_unc, sig_pure_inc)
                    print(f"    #{r['feature_idx']:>6d}  effect={r['effect_size']:+.3f}  "
                          f"p={r['p_value']:.4f}  freq A={r['group1_freq']:.3f} B={r['group2_freq']:.3f}  "
                          f"[{label}]")

        all_comparisons[layer_idx] = {
            'original': original,
            'original_entropy': original_entropy,
            'pure_uncertainty': pure_uncertainty,
            'pure_incorrectness': pure_incorrectness,
            'interaction': interaction,
            'sig_counts': {
                'original_incorrectness': len(sig_original),
                'original_entropy': len(sig_entropy),
                'original_overlap': len(original_overlap),
                'pure_uncertainty': len(sig_pure_unc),
                'pure_incorrectness': len(sig_pure_inc),
                'pure_uncertainty_only': len(pure_unc_only),
                'pure_incorrectness_only': len(pure_inc_only),
                'both_confounded': len(both),
                'interaction': len(sig_interaction),
            }
        }

        summary_rows.append({
            'layer': layer_idx,
            **all_comparisons[layer_idx]['sig_counts'],
        })

    # Save summary CSV
    summary_file = output_dir / 'quadrant_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'layer', 'original_incorrectness', 'original_entropy', 'original_overlap',
            'pure_uncertainty', 'pure_incorrectness',
            'pure_uncertainty_only', 'pure_incorrectness_only', 'both_confounded',
            'interaction',
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {summary_file}")

    # Save detailed per-layer CSVs for pure uncertainty and pure incorrectness
    for layer_idx in sae_layers:
        if layer_idx not in all_comparisons:
            continue

        for comp_name in ['pure_uncertainty', 'pure_incorrectness', 'interaction']:
            results = all_comparisons[layer_idx][comp_name]
            if not results:
                continue

            comp_file = output_dir / f'L{layer_idx}_{comp_name}.csv'
            with open(comp_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'feature_idx', 'effect_size', 'p_value',
                    'group1_mean', 'group2_mean', 'group1_freq', 'group2_freq',
                ])
                writer.writeheader()
                for r in results[:args.top_k]:
                    writer.writerow(r)

    # Save full results pickle
    pkl_file = output_dir / 'quadrant_analysis.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'config': {
                'source_pickle': args.pickle,
                'median_entropy': float(median_entropy),
                'p_threshold': args.p_threshold,
                'quadrant_sizes': {
                    'A_correct_confident': len(group_A),
                    'B_incorrect_confident': len(group_B),
                    'C_correct_uncertain': len(group_C),
                    'D_incorrect_uncertain': len(group_D),
                },
            },
            'all_comparisons': all_comparisons,
        }, f)
    print(f"Saved: {pkl_file}")

    # Print final summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Feature counts by type per layer")
    print(f"{'='*80}")
    print(f"{'Layer':>6} {'Orig(Inc)':>10} {'Orig(Ent)':>10} {'Overlap':>8} "
          f"{'PureUnc':>8} {'PureInc':>8} {'UncOnly':>8} {'IncOnly':>8} {'Both':>6}")
    print(f"{'-'*80}")
    for row in summary_rows:
        print(f"{row['layer']:>6} {row['original_incorrectness']:>10} "
              f"{row['original_entropy']:>10} {row['original_overlap']:>8} "
              f"{row['pure_uncertainty']:>8} {row['pure_incorrectness']:>8} "
              f"{row['pure_uncertainty_only']:>8} {row['pure_incorrectness_only']:>8} "
              f"{row['both_confounded']:>6}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
