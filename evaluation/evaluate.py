

import argparse
import sys
import pandas as pd
import numpy as np
from collections import defaultdict


# ========================= EDITABLE CONFIG =========================
# Ground-truth label  ->  model class (CLIP prompt term).
# Set a value to None to EXCLUDE that ground-truth label from F1 scoring.
LABEL_MAP = {
    "plastic":   "plastic",
    "metal":     "metal",
    "wood":      "wood",
    "trap":      "cage",     
    "others":    None,       
    "non-trash": None,       
}

# The model's full class vocabulary (CLIP prompts in app.py predict_class()).
MODEL_CLASSES = ["wood", "cage", "fishing gear", "nature", "plastic", "metal", "wheel"]
# ===================================================================


def iou(boxA, boxB):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(path, include_nontrash):
    df = pd.read_csv(path, index_col=0)
    # Convert (x, y, w, h) -> (x1, y1, x2, y2)
    df = df.copy()
    df["x1"] = df["bbox_x"]
    df["y1"] = df["bbox_y"]
    df["x2"] = df["bbox_x"] + df["bbox_width"]
    df["y2"] = df["bbox_y"] + df["bbox_height"]
    if not include_nontrash:
        df = df[df["label_name"] != "non-trash"]
    return df


def load_predictions(path):
    df = pd.read_csv(path)
    # Drop rows with no detection (image had NA boxes)
    df = df[pd.to_numeric(df["x1"], errors="coerce").notna()].copy()
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def evaluate(pred_path, gt_path, iou_threshold, include_nontrash):
    gt = load_ground_truth(gt_path, include_nontrash)
    pred = load_predictions(pred_path)

    # Only score images present in BOTH files
    pred_imgs = set(pred["image_name"].unique())
    gt_imgs = set(gt["image_name"].unique())
    eval_imgs = sorted(pred_imgs & gt_imgs)

    if not eval_imgs:
        print("ERROR: No images appear in both the prediction and ground-truth files.")
        print(f"  Prediction images: {sorted(pred_imgs)[:10]}...")
        print(f"  (Run the model on images that exist in the annotation file.)")
        sys.exit(1)

    print("=" * 70)
    print(f"Evaluating {len(eval_imgs)} image(s) present in both files")
    print(f"IoU match threshold: {iou_threshold}   |   include non-trash: {include_nontrash}")
    print("=" * 70)

    all_ious = []          # for Mean IoU (detection)
    y_true, y_pred = [], []  # for F1 (classification), matched pairs only
    n_gt = n_pred = n_matched = 0

    for img in eval_imgs:
        g = gt[gt["image_name"] == img]
        p = pred[pred["image_name"] == img]
        n_gt += len(g)
        n_pred += len(p)

        pred_boxes = p[["x1", "y1", "x2", "y2"]].values.tolist()
        pred_types = p["type"].tolist()

        for _, grow in g.iterrows():
            gbox = [grow["x1"], grow["y1"], grow["x2"], grow["y2"]]
            gt_label = grow["label_name"]

            # Best-overlapping prediction for this GT box
            best_iou, best_j = 0.0, -1
            for j, pbox in enumerate(pred_boxes):
                v = iou(gbox, pbox)
                if v > best_iou:
                    best_iou, best_j = v, j

            all_ious.append(best_iou)  # detection: how well was this GT object localized

            # Classification: only if matched above threshold AND label is mapped
            mapped = LABEL_MAP.get(gt_label, None)
            if best_iou >= iou_threshold and best_j >= 0 and mapped is not None:
                n_matched += 1
                y_true.append(mapped)
                y_pred.append(pred_types[best_j])

    # DETECTION: Mean IoU 
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print("\n--- DETECTION ---")
    print(f"Ground-truth objects: {n_gt}   Predicted objects: {n_pred}")
    print(f"Mean IoU (avg best-overlap per GT object): {mean_iou:.4f}")
    print(f"   (paper reported 0.69)")

    # CLASSIFICATION: F1
    print("\n--- CLASSIFICATION (matched objects only) ---")
    print(f"Matched objects used for F1: {n_matched}")
    if n_matched == 0:
        print("No matched objects above IoU threshold — cannot compute F1.")
        print("Try lowering --iou-threshold, or check the label mapping.")
        return

    try:
        from sklearn.metrics import (f1_score, precision_score, recall_score,
                                      classification_report, confusion_matrix)
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}   Micro F1: {micro_f1:.4f}")
    print(f"   (paper reported 0.74 F1)")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix — shows e.g. metal predicted as wood
    labels_present = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    print("Confusion matrix (rows = true, cols = predicted):")
    header = "          " + "  ".join(f"{l[:6]:>6}" for l in labels_present)
    print(header)
    for i, lab in enumerate(labels_present):
        row = "  ".join(f"{cm[i][j]:>6}" for j in range(len(labels_present)))
        print(f"{lab[:8]:>8}  {row}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate marine-debris pipeline.")
    ap.add_argument("--pred", default="detections.csv", help="Predictions CSV")
    ap.add_argument("--gt", default="annotations_2023_23_01.csv", help="Ground-truth CSV")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="Min IoU to count a prediction as matching a GT object (default 0.5)")
    ap.add_argument("--include-nontrash", action="store_true",
                    help="Include non-trash GT boxes in scoring (default: exclude)")
    args = ap.parse_args()
    evaluate(args.pred, args.gt, args.iou_threshold, args.include_nontrash)


if __name__ == "__main__":
    main()