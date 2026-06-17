"""
score_dedup.py — Turn a hand-labeled pairs.csv into a fair SIFT-vs-AKAZE accuracy comparison.

Run compare_dedup.py with --dump-pairs pairs.csv first, then open pairs.csv and fill
the 'true_dup' column by eye: 1 if the two crops are the SAME physical object, 0 if not.
Judge by what the crop SHOWS, not by object number (numbering is not consistent across
frames). You only need to label pairs you can confidently judge; leave others blank and
they'll be skipped.

Then:
    python score_dedup.py --pairs pairs.csv

WHAT IT DOES
  - reads sift_matches / akaze_matches / true_dup
  - for EACH method independently, sweeps the match-count threshold and finds the one
    that maximizes F1 against your labels (because the two methods live on different
    match-count scales, a single shared threshold would be unfair)
  - reports precision / recall / F1 at each method's best threshold, side by side

This is the comparison that supports a real claim like:
  "At its best threshold AKAZE matches SIFT's F1 on duplicate detection while running
   ~20x faster" — IF that's what the numbers show. If AKAZE is worse, it will show that too.
"""

import argparse
import pandas as pd
import numpy as np


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def best_threshold(scores, labels):
    """Sweep integer thresholds; return (best_thr, p, r, f1) maximizing F1.
    A pair is predicted duplicate if score >= thr."""
    best = (0, 0.0, 0.0, -1.0)
    candidates = sorted(set(int(s) for s in scores)) or [0]
    # also try one above the max so 'flag nothing' is considered
    for thr in range(min(candidates), max(candidates) + 2):
        tp = fp = fn = 0
        for s, y in zip(scores, labels):
            pred = s >= thr
            if pred and y == 1: tp += 1
            elif pred and y == 0: fp += 1
            elif (not pred) and y == 1: fn += 1
        p, r, f = prf(tp, fp, fn)
        if f > best[3]:
            best = (thr, p, r, f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    df = df[pd.to_numeric(df["true_dup"], errors="coerce").notna()].copy()
    if df.empty:
        print("No labeled rows. Fill the 'true_dup' column (1 = same object, 0 = not) and rerun.")
        return
    df["true_dup"] = df["true_dup"].astype(int)

    n = len(df)
    n_pos = int(df["true_dup"].sum())
    print(f"Labeled pairs: {n}  (duplicates: {n_pos}, non-duplicates: {n - n_pos})")
    if n_pos == 0:
        print("No positive (duplicate) labels — can't compute recall. Label some true duplicates.")
        return

    print("=" * 60)
    for method, col in [("SIFT", "sift_matches"), ("AKAZE", "akaze_matches")]:
        thr, p, r, f = best_threshold(df[col].tolist(), df["true_dup"].tolist())
        print(f"\n{method}  (best threshold: >= {thr} matches)")
        print(f"  precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")

    print("\n" + "=" * 60)
    print("Notes:")
    print("- Each method is tuned to ITS OWN best threshold — that's the fair comparison,")
    print("  since SIFT and AKAZE match counts are on different scales.")
    print("- Pair these accuracy numbers with the runtime numbers from compare_dedup.py")
    print("  for the full picture (speed vs. accuracy tradeoff).")
    print("- Small labeled sets give noisy estimates; label more pairs for confidence.")


if __name__ == "__main__":
    main()