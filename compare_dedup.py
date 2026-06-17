"""
compare_dedup.py — Compare SIFT vs AKAZE for duplicate-object removal.

Mirrors the logic in Ray's routes/remove_overlap.py (GPS KDTree pre-filter,
kNN ratio test, ">= MATCH_THRESHOLD good matches => same object") but runs BOTH
SIFT and AKAZE over the same object crops so you can compare them directly.

WHAT IT MEASURES
  Rigorous (no ground truth needed):
    - feature extraction time per method (over all crops)
    - matching time per method (over all GPS-near pairs)
    - total dedup runtime per method
    - avg keypoints per crop
    - avg good matches per evaluated pair
    - number of pairs each method flags as duplicates
  Descriptive (no ground truth needed):
    - agreement: on how many pairs do SIFT and AKAZE give the SAME verdict
    - disagreements: listed so you can eyeball who's right

  NOT measured here (needs a labeled duplicate set):
    - true precision/recall / accuracy. Agreement != correctness. To claim one
      method is more ACCURATE, you need ground-truth "same object: yes/no" labels
      for a set of pairs. See --dump-pairs to help build that.

USAGE
    python compare_dedup.py --images eval_images --pred detections.csv
    python compare_dedup.py --images eval_images --pred detections.csv --threshold 50 --radius 10
    python compare_dedup.py --images eval_images --pred detections.csv --dump-pairs pairs.csv

NOTES
  - Crops are taken from the detection boxes in detections.csv (full-res x1,y1,x2,y2).
  - SIFT uses float descriptors + FLANN(KDTree) + L2, like Ray's code.
  - AKAZE uses binary descriptors + BFMatcher + Hamming (the correct matcher for
    binary descriptors). The match COUNT scale differs between methods, so the same
    numeric threshold is not directly comparable -- see the note printed at the end.
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.spatial import KDTree


# ----- matching, mirroring Ray's ratio test -----
RATIO = 0.7


def good_match_count(des0, des1, matcher, norm_is_binary):
    """Count good matches between two descriptor sets using the kNN ratio test."""
    if des0 is None or des1 is None:
        return 0
    if len(des0) < 2 or len(des1) < 2:
        return 0
    try:
        matches = matcher.knnMatch(des0, des1, k=2)
    except cv2.error:
        return 0
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good += 1
    return good


def build_matchers():
    # SIFT: float descriptors -> FLANN with KDTree (same as Ray's code)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),   # FLANN_INDEX_KDTREE
        dict(checks=50),
    )
    # AKAZE: binary descriptors -> brute force Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    return flann, bf


def load_crops(pred_path, image_dir):
    """Return list of dicts: {name, number, crop(grayscale np), lon, lat}."""
    df = pd.read_csv(pred_path)
    df = df[pd.to_numeric(df["x1"], errors="coerce").notna()].copy()
    for c in ["x1", "y1", "x2", "y2", "longitude", "latitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x1", "y1", "x2", "y2", "longitude", "latitude"])

    crops = []
    cache = {}
    for _, r in df.iterrows():
        name = str(r["image_name"])
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            continue
        if name not in cache:
            cache[name] = Image.open(path).convert("RGB")
        im = cache[name]
        box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        crop = np.array(im.crop(box).convert("L"))  # grayscale for features
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        crops.append({
            "name": name,
            "number": r["number"],
            "crop": crop,
            "lon": float(r["longitude"]),
            "lat": float(r["latitude"]),
        })
    return crops


def extract_features(crops, detector):
    """Detect+compute for every crop; return (list_of_des, list_of_kp_counts, total_time)."""
    des_list, kp_counts = [], []
    t0 = time.perf_counter()
    for c in crops:
        kp, des = detector.detectAndCompute(c["crop"], None)
        des_list.append(des)
        kp_counts.append(0 if kp is None else len(kp))
    return des_list, kp_counts, time.perf_counter() - t0


def gps_pairs(crops, radius_m):
    """Yield index pairs (i, j) that are within radius_m and from different images."""
    R = 6371e3
    radius_deg = radius_m / R * (180.0 / np.pi)
    tree = KDTree([(c["lon"], c["lat"]) for c in crops])
    seen = set()
    for i, c in enumerate(crops):
        for j in tree.query_ball_point([c["lon"], c["lat"]], radius_deg):
            if j == i:
                continue
            if crops[i]["name"] == crops[j]["name"]:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            yield key


def run_method(name, detector, matcher, crops, pairs, threshold):
    des_list, kp_counts, feat_t = extract_features(crops, detector)
    match_t = 0.0
    verdicts = {}
    match_counts = []
    for (i, j) in pairs:
        t0 = time.perf_counter()
        gm = good_match_count(des_list[i], des_list[j], matcher, True)
        match_t += time.perf_counter() - t0
        verdicts[(i, j)] = gm >= threshold
        match_counts.append(gm)
    return {
        "name": name,
        "feat_time": feat_t,
        "match_time": match_t,
        "total_time": feat_t + match_t,
        "avg_kp": float(np.mean(kp_counts)) if kp_counts else 0.0,
        "avg_matches": float(np.mean(match_counts)) if match_counts else 0.0,
        "n_flagged": sum(verdicts.values()),
        "verdicts": verdicts,
        "match_counts": match_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="eval_images")
    ap.add_argument("--pred", default="detections.csv")
    ap.add_argument("--threshold", type=int, default=50,
                    help="good-match count to call a pair a duplicate (Ray uses 50)")
    ap.add_argument("--radius", type=float, default=10.0, help="GPS pre-filter radius (meters)")
    ap.add_argument("--dump-pairs", default=None,
                    help="optional CSV: every evaluated pair with both methods' match counts "
                         "(use this to hand-label ground truth later)")
    args = ap.parse_args()

    print("Loading crops from detections + images...")
    crops = load_crops(args.pred, args.images)
    print(f"  {len(crops)} object crops loaded")
    if len(crops) < 2:
        print("Need at least 2 crops to compare. Check that detections.csv has boxes "
              "and the images are in the folder.")
        return

    pairs = list(gps_pairs(crops, args.radius))
    print(f"  {len(pairs)} GPS-near cross-image pairs to evaluate (radius {args.radius} m)")
    if not pairs:
        print("No GPS-near pairs found. With no overlapping locations there are no duplicates "
              "to detect -- try a larger --radius or an image set with repeated objects.")
        return

    flann, bf = build_matchers()
    sift = cv2.SIFT_create()
    akaze = cv2.AKAZE_create()

    print("\nRunning SIFT...")
    s = run_method("SIFT", sift, flann, crops, pairs, args.threshold)
    print("Running AKAZE...")
    a = run_method("AKAZE", akaze, bf, crops, pairs, args.threshold)

    # ----- report -----
    def row(label, sv, av, fmt="{:.3f}"):
        print(f"  {label:<24}{fmt.format(sv):>14}{fmt.format(av):>14}")

    print("\n" + "=" * 54)
    print(f"  {'metric':<24}{'SIFT':>14}{'AKAZE':>14}")
    print("=" * 54)
    row("feature time (s)", s["feat_time"], a["feat_time"])
    row("match time (s)", s["match_time"], a["match_time"])
    row("total time (s)", s["total_time"], a["total_time"])
    row("avg keypoints/crop", s["avg_kp"], a["avg_kp"], "{:.1f}")
    row("avg good matches/pair", s["avg_matches"], a["avg_matches"], "{:.1f}")
    print(f"  {'pairs flagged dup':<24}{s['n_flagged']:>14}{a['n_flagged']:>14}")
    if a["total_time"] > 0:
        print(f"\n  AKAZE is {s['total_time']/a['total_time']:.2f}x SIFT's total runtime "
              f"(>1 means AKAZE faster)")

    # agreement
    agree = sum(1 for p in pairs if s["verdicts"][p] == a["verdicts"][p])
    print(f"\n  Agreement on verdicts: {agree}/{len(pairs)} pairs "
          f"({100*agree/len(pairs):.1f}%)")
    disagree = [p for p in pairs if s["verdicts"][p] != a["verdicts"][p]]
    if disagree:
        print(f"  Disagreements ({len(disagree)}): SIFT vs AKAZE differ here -- "
              f"eyeball these to see who's right:")
        for (i, j) in disagree[:15]:
            print(f"    {crops[i]['name']}#{crops[i]['number']} <-> "
                  f"{crops[j]['name']}#{crops[j]['number']}: "
                  f"SIFT={'DUP' if s['verdicts'][(i,j)] else 'no'} "
                  f"AKAZE={'DUP' if a['verdicts'][(i,j)] else 'no'}")

    print("\n  NOTE: SIFT and AKAZE match COUNTS are on different scales, so the same")
    print("  --threshold is not directly comparable. Agreement is descriptive, not")
    print("  accuracy. For true precision/recall, hand-label the pairs (use --dump-pairs).")

    if args.dump_pairs:
        rows = []
        for k, (i, j) in enumerate(pairs):
            rows.append({
                "img_a": crops[i]["name"], "obj_a": crops[i]["number"],
                "img_b": crops[j]["name"], "obj_b": crops[j]["number"],
                "sift_matches": s["match_counts"][k],
                "akaze_matches": a["match_counts"][k],
                "sift_dup": int(s["verdicts"][(i, j)]),
                "akaze_dup": int(a["verdicts"][(i, j)]),
                "true_dup": "",  # <-- you fill this in by hand: 1 = same object, 0 = not
            })
        pd.DataFrame(rows).to_csv(args.dump_pairs, index=False)
        print(f"\n  Wrote {len(rows)} pairs to {args.dump_pairs} -- fill the 'true_dup' "
              f"column by eye to build ground truth.")


if __name__ == "__main__":
    main()