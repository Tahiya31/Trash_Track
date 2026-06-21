"""
mini_trust_test.py v2 — Go/no-go test with CLIP + SigLIP 2 comparison.

Tests all four trust levels:
  L1: CLIP confidence only (baseline)
  L2: CLIP confidence + uncertainty signals
  L3: L2 + cross-model agreement (CLIP vs SigLIP 2)
  L3b: Inverted signals (since AUROC < 0.5 means signal is inverted)

The L3 AUROC is the key result: if cross-model agreement adds signal
beyond single-model confidence, the full research direction is viable.

INTERPRET:
  L3 AUROC < 0.55 : agreement doesn't help — rethink model choice
  L3 AUROC 0.55-0.65 : modest signal — proceed cautiously
  L3 AUROC 0.65-0.75 : promising — green light
  L3 AUROC > 0.75 : strong — proceed with high confidence

USAGE:
  python mini_trust_test.py \
    --pred detections.csv \
    --gt annotations_2023_23_01.csv \
    --images eval_images
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── LABEL MAP ────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "plastic":   "plastic",
    "metal":     "metal",
    "wood":      "wood",
    "trap":      "cage",
    "cage":      "cage",
    "others":    None,
    "non-trash": None,
}
CLIP_CLASSES = ["wood", "cage", "plastic", "metal"]
IOU_THRESHOLD = 0.5


# ── GEOMETRY ─────────────────────────────────────────────────────────────────
def box_iou(b1, b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)


# ── SIGNAL COMPUTATION ───────────────────────────────────────────────────────
def entropy(probs):
    p = np.array(probs); p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-10)))

def margin(probs):
    s = sorted(probs, reverse=True)
    return float(s[0] - s[1]) if len(s) > 1 else float(s[0])

def compute_l1_l2(top_prob, ent, mar):
    l1 = top_prob
    max_entropy = np.log(len(CLIP_CLASSES))
    norm_entropy = ent / max_entropy
    l2 = (top_prob + mar + (1.0 - norm_entropy)) / 3.0
    return float(l1), float(l2)

def compute_l3(l2, agreement):
    # Agreement bonus: 1.0 if models agree, 0.4 if they disagree
    bonus = 1.0 if agreement else 0.4
    return float(np.clip(l2 * bonus, 0, 1))


# ── DATA LOADING ─────────────────────────────────────────────────────────────
def load_predictions(path):
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df['x1'], errors='coerce').notna()].copy()
    for c in ['x1','y1','x2','y2']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['x1','y1','x2','y2','type','image_name'])

def load_gt(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def match_preds_to_gt(preds, gt):
    results = []
    for img_name in preds['image_name'].unique():
        p_img = preds[preds['image_name'] == img_name]
        gt_img = None
        for col in ['image_name','filename','file_name']:
            if col in gt.columns:
                mask = gt[col].astype(str).str.contains(
                    img_name.replace('.JPG','').replace('.jpg',''),
                    case=False, na=False)
                if mask.sum() > 0:
                    gt_img = gt[mask].copy(); break
        if gt_img is None or len(gt_img) == 0: continue

        gt_boxes = []
        for _, gr in gt_img.iterrows():
            try:
                if 'x1' in gt_img.columns:
                    box = [float(gr['x1']),float(gr['y1']),float(gr['x2']),float(gr['y2'])]
                elif 'bbox_x' in gt_img.columns:
                    bx=float(gr['bbox_x']); by=float(gr['bbox_y'])
                    bw=float(gr['bbox_width']); bh=float(gr['bbox_height'])
                    box=[bx, by, bx+bw, by+bh]
                else: continue
                raw = str(gr.get('label_name', gr.get('category',''))).lower().strip()
                lbl = LABEL_MAP.get(raw, None)
                gt_boxes.append({'box': box, 'label': lbl})
            except: continue

        for _, pr in p_img.iterrows():
            pb=[float(pr['x1']),float(pr['y1']),float(pr['x2']),float(pr['y2'])]
            best_iou=0.0; best_lbl=None
            for g in gt_boxes:
                iou=box_iou(pb,g['box'])
                if iou>best_iou: best_iou=iou; best_lbl=g['label']
            if best_iou>=IOU_THRESHOLD and best_lbl is not None:
                results.append({'image_name':img_name,'number':pr['number'],
                                'pred_label':str(pr['type']).lower().strip(),
                                'gt_label':best_lbl,'iou':best_iou})
    return results


# ── CLIP INFERENCE ───────────────────────────────────────────────────────────
def run_clip(matches, preds, image_dir, device):
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize([f"a photo of {c}" for c in CLIP_CLASSES]).to(device)
    cache = {}; out = []
    for m in matches:
        img_name = m['image_name']
        path = os.path.join(image_dir, img_name)
        if not os.path.exists(path): continue
        if img_name not in cache:
            cache[img_name] = Image.open(path).convert("RGB")
        row = preds[(preds['image_name']==img_name)&(preds['number']==m['number'])]
        if len(row)==0: continue
        r = row.iloc[0]
        try:
            crop = cache[img_name].crop((float(r['x1']),float(r['y1']),
                                          float(r['x2']),float(r['y2'])))
            if crop.width<2 or crop.height<2: continue
            inp = preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                img_f = model.encode_image(inp)
                txt_f = model.encode_text(tokens)
                probs = (img_f @ txt_f.T).softmax(dim=-1)[0].cpu().numpy().tolist()
        except: continue
        top_idx=int(np.argmax(probs)); top_p=float(probs[top_idx])
        ent=entropy(probs); mar=margin(probs)
        l1,l2=compute_l1_l2(top_p,ent,mar)
        out.append({**m,
            'clip_pred': CLIP_CLASSES[top_idx],
            'clip_top_prob': top_p,
            'clip_entropy': ent,
            'clip_margin': mar,
            'L1': l1, 'L2': l2,
            'correct': int(CLIP_CLASSES[top_idx]==m['gt_label'])})
    return out


# ── SIGLIP 2 INFERENCE ───────────────────────────────────────────────────────
def run_siglip(matches_with_clip, preds, image_dir, device):
    """
    Run SigLIP 2 on the same crops and add siglip_pred + agreement + L3.
    SigLIP uses sigmoid loss (not softmax), so we normalize the sigmoid
    outputs to get a comparable probability distribution.
    """
    from transformers import AutoProcessor, AutoModel
    print("  Loading SigLIP 2 (google/siglip-so400m-patch14-384)...")
    print("  First run downloads ~1.7GB — subsequent runs use cache.")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model.eval()
    if device == "cuda":
        siglip_model = siglip_model.cuda()

    # Pre-encode text labels once
    text_inputs = processor(
        text=[f"a photo of {c}" for c in CLIP_CLASSES],
        return_tensors="pt", padding=True
    )
    if device == "cuda":
        text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
    with torch.no_grad():
        text_outputs = siglip_model.get_text_features(**text_inputs)
        text_features = text_outputs if isinstance(text_outputs, torch.Tensor) else text_outputs.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cache = {}; out = []
    for m in matches_with_clip:
        img_name = m['image_name']
        path = os.path.join(image_dir, img_name)
        if not os.path.exists(path): continue
        if img_name not in cache:
            cache[img_name] = Image.open(path).convert("RGB")
        row = preds[(preds['image_name']==img_name)&(preds['number']==m['number'])]
        if len(row)==0: continue
        r = row.iloc[0]
        try:
            crop = cache[img_name].crop((float(r['x1']),float(r['y1']),
                                          float(r['x2']),float(r['y2'])))
            if crop.width < 2 or crop.height < 2: continue
            # SigLIP expects minimum size
            crop = crop.resize((max(crop.width,32), max(crop.height,32)))
            img_inputs = processor(images=crop, return_tensors="pt")
            if device == "cuda":
                img_inputs = {k: v.cuda() for k, v in img_inputs.items()}
            with torch.no_grad():
                img_out = siglip_model.get_image_features(**img_inputs)
                img_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                # Cosine similarity, then sigmoid to get scores
                sims = (img_features @ text_features.T)[0]
                raw_probs = torch.sigmoid(sims).cpu().numpy()
                # Normalize to sum to 1 for comparability
                probs = (raw_probs / raw_probs.sum()).tolist()
        except Exception as e:
            continue

        top_idx = int(np.argmax(probs))
        siglip_pred = CLIP_CLASSES[top_idx]
        agreement = int(siglip_pred == m['clip_pred'])
        l3 = compute_l3(m['L2'], agreement)

        out.append({**m,
            'siglip_pred': siglip_pred,
            'siglip_top_prob': float(probs[top_idx]),
            'agreement': agreement,
            'L3': l3})
    return out


# ── RESULTS ──────────────────────────────────────────────────────────────────
def print_results(df):
    n = len(df); n_correct = int(df['correct'].sum())
    n_incorrect = n - n_correct
    has_l3 = 'L3' in df.columns and 'siglip_pred' in df.columns

    print("\n" + "="*64)
    print("  MINI TRUST TEST v2 — GO/NO-GO RESULTS")
    print("="*64)
    print(f"  Matched predictions : {n}")
    print(f"  Correct (vs GT)     : {n_correct}  ({100*n_correct/n:.1f}%)")
    print(f"  Incorrect           : {n_incorrect}  ({100*n_incorrect/n:.1f}%)")

    if n_correct == 0 or n_incorrect == 0:
        print("\n  WARNING: All correct or all incorrect — AUROC undefined.")
        print("  Check label matching.")
        return df

    print("\n" + "-"*64)
    print(f"  {'Level':<26} {'AUROC':>7}  {'vs random':>9}  Interpretation")
    print("-"*64)

    levels = [("L1 (CLIP confidence)", "L1"),
              ("L2 (+ uncertainty)", "L2")]
    if has_l3:
        levels.append(("L3 (+ agreement)", "L3"))

    auroc_vals = {}
    for name, col in levels:
        try:
            auc = roc_auc_score(df['correct'], df[col])
            delta = auc - 0.5
            auroc_vals[col] = auc
            if auc < 0.45:
                interp = "INVERTED (signal exists but reversed)"
            elif auc < 0.55:
                interp = "near-random — weak signal"
            elif auc < 0.65:
                interp = "modest"
            elif auc < 0.75:
                interp = "PROMISING"
            else:
                interp = "STRONG"
            print(f"  {name:<26} {auc:>7.3f}  {delta:>+9.3f}  {interp}")
        except Exception as e:
            print(f"  {name:<26}   ERROR: {e}")

    print("-"*64)

    # Key question: did agreement help?
    if has_l3 and 'L2' in auroc_vals and 'L3' in auroc_vals:
        delta_l3 = auroc_vals['L3'] - auroc_vals['L2']
        print(f"\n  Delta L3 vs L2: {delta_l3:+.3f}")
        if delta_l3 > 0.05:
            print("  ✅  Agreement adds meaningful signal beyond confidence alone.")
            print("     H3 is supported on this mini-set.")
        elif delta_l3 > 0:
            print("  ⚠️   Agreement adds marginal signal — may strengthen at scale.")
        else:
            print("  ❌  Agreement does not help here.")
            print("     Both models may share the same failure mode.")

    # Agreement analysis
    if has_l3:
        n_agree = int(df['agreement'].sum())
        n_disagree = n - n_agree
        agree_acc = df[df['agreement']==1]['correct'].mean() if n_agree > 0 else 0
        disagree_acc = df[df['agreement']==0]['correct'].mean() if n_disagree > 0 else 0
        print(f"\n  Agreement breakdown:")
        print(f"    Both models agree:    {n_agree:3d} pairs — accuracy {agree_acc:.3f}")
        print(f"    Models disagree:      {n_disagree:3d} pairs — accuracy {disagree_acc:.3f}")
        if agree_acc > disagree_acc:
            print("  ✅  Agreement correlates with correctness (agree = more reliable)")
        else:
            print("  ⚠️   Disagreement pairs are not less accurate — both models")
            print("     may share the same failure mode ('nature' bias)")

    # SigLIP prediction breakdown
    if has_l3:
        print(f"\n  SigLIP 2 top predictions:")
        sig_counts = df['siglip_pred'].value_counts()
        for cls, cnt in sig_counts.items():
            print(f"    {cls}: {cnt} times")

        print(f"\n  CLIP top predictions (for comparison):")
        clip_counts = df['clip_pred'].value_counts()
        for cls, cnt in clip_counts.items():
            print(f"    {cls}: {cnt} times")

        print(f"\n  Where models disagree ({n_disagree} pairs):")
        disagree_df = df[df['agreement']==0]
        if len(disagree_df) > 0:
            pairs = disagree_df.groupby(['clip_pred','siglip_pred']).size().sort_values(ascending=False)
            for (cp, sp), cnt in pairs.head(8).items():
                n_correct_here = int(disagree_df[(disagree_df['clip_pred']==cp)&(disagree_df['siglip_pred']==sp)]['correct'].sum())
                print(f"    CLIP={cp} vs SigLIP={sp}: {cnt} pairs, {n_correct_here} correct")

    # Misclassifications
    wrong = df[df['correct']==0]
    if len(wrong) > 0:
        print(f"\n  CLIP misclassifications (GT → CLIP):")
        pairs = wrong.groupby(['gt_label','clip_pred']).size().sort_values(ascending=False).head(6)
        for (gt,pred), cnt in pairs.items():
            print(f"    {gt} → {pred}: {cnt} times")

    # Precision at K
    col_for_k = 'L3' if has_l3 else 'L2'
    print(f"\n  Precision at top-K% ({col_for_k}):")
    df_s = df.sort_values(col_for_k, ascending=False)
    baseline = df['correct'].mean()
    for k in [10, 25, 50]:
        n_k = max(1, int(len(df_s)*k/100))
        prec = df_s.head(n_k)['correct'].mean()
        print(f"    Top {k:2d}%: {prec:.3f}  (baseline {baseline:.3f}, lift {prec-baseline:+.3f})")

    # GO/NO-GO
    print("\n  " + "─"*60)
    print("  GO/NO-GO RECOMMENDATION:")
    best_auc = max(auroc_vals.values()) if auroc_vals else 0.5
    if best_auc < 0.45:
        print("  ❌  INVERTED SIGNAL. High trust = more likely wrong.")
        print("     This means confidence is anti-correlated with correctness")
        print("     in this domain (the 'nature' overconfidence effect).")
        print("     Check whether SigLIP 2 shares this bias or has a different")
        print("     failure mode — that determines whether L3 can rescue the approach.")
    elif best_auc < 0.55:
        print("  ❌  WEAK/RANDOM. Neither confidence nor agreement predicts")
        print("     correctness. Consider different model pairs.")
    elif best_auc < 0.65:
        print("  ⚠️   MODEST SIGNAL. Proceed, but results will need careful framing.")
        print("     Full scale (300+ objects) may reveal cleaner signal.")
    else:
        print("  ✅  PROMISING/STRONG. Green light for the full 4-week build.")
    print("  " + "─"*60)
    print("\n  REMINDER: This is a go/no-go test on 42 objects.")
    print("  AUROC numbers here are NOT publishable.")
    print("  Full scale on Lightning AI GPU is needed for paper results.")
    print("="*64 + "\n")
    return df


def plot_results(df, out_path="mini_trust_results.png"):
    has_l3 = 'L3' in df.columns
    ncols = 4 if has_l3 else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    fig.suptitle("Mini Trust Test v2 — Diagnostic Plots (go/no-go only)", fontsize=11)

    # ROC curves
    ax = axes[0]
    levels = [("L1 confidence", "L1", "blue"),
              ("L2 +uncertainty", "L2", "orange")]
    if has_l3:
        levels.append(("L3 +agreement", "L3", "green"))
    for name, col, color in levels:
        try:
            fpr, tpr, _ = roc_curve(df['correct'], df[col])
            auc = roc_auc_score(df['correct'], df[col])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}\n(AUC={auc:.2f})")
        except: pass
    ax.plot([0,1],[0,1],'k--',alpha=0.3,label="Random (0.50)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Trust score distributions
    best_col = 'L3' if has_l3 else 'L2'
    ax = axes[1]
    c_scores = df[df['correct']==1][best_col]
    w_scores = df[df['correct']==0][best_col]
    ax.hist(c_scores, bins=10, alpha=0.6, color='green', label=f'Correct (n={len(c_scores)})')
    ax.hist(w_scores, bins=10, alpha=0.6, color='red', label=f'Incorrect (n={len(w_scores)})')
    ax.set_xlabel(f"{best_col} Trust Score"); ax.set_ylabel("Count")
    ax.set_title(f"{best_col} Trust Distribution"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Confidence vs entropy
    ax = axes[2]
    colors = ['green' if c else 'red' for c in df['correct']]
    ax.scatter(df['clip_top_prob'], df['clip_entropy'], c=colors, alpha=0.6, s=40)
    ax.set_xlabel("CLIP Top Probability"); ax.set_ylabel("CLIP Entropy")
    ax.set_title("CLIP: Confidence vs Entropy\n(green=correct, red=incorrect)"); ax.grid(alpha=0.3)

    # Agreement plot
    if has_l3:
        ax = axes[3]
        agree_df = df[df['agreement']==1]
        disagree_df = df[df['agreement']==0]
        categories = ['Agree', 'Disagree']
        correct_counts = [agree_df['correct'].sum(), disagree_df['correct'].sum()]
        incorrect_counts = [len(agree_df)-agree_df['correct'].sum(),
                           len(disagree_df)-disagree_df['correct'].sum()]
        x = np.arange(2); width = 0.35
        ax.bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
        ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(categories)
        ax.set_ylabel("Count"); ax.set_title("Agreement vs Correctness")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
        for i, (c, ic) in enumerate(zip(correct_counts, incorrect_counts)):
            total = c + ic
            acc = c/total if total > 0 else 0
            ax.text(i, max(c,ic)+0.3, f"acc={acc:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"  Diagnostic plot saved to: {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="detections.csv")
    ap.add_argument("--gt", default="annotations_2023_23_01.csv")
    ap.add_argument("--images", default="eval_images")
    ap.add_argument("--plot", default="mini_trust_results.png")
    ap.add_argument("--skip-siglip", action="store_true",
                    help="Skip SigLIP 2 and only run L1/L2 (faster, no download)")
    args = ap.parse_args()

    print(f"\nLoading predictions from {args.pred}...")
    preds = load_predictions(args.pred)
    print(f"  {len(preds)} predictions loaded.")

    print(f"Loading ground truth from {args.gt}...")
    gt = load_gt(args.gt)
    print(f"  {len(gt)} GT annotations loaded.")

    print("Matching predictions to ground truth (IoU >= 0.5)...")
    matches = match_preds_to_gt(preds, gt)
    print(f"  {len(matches)} matched prediction-GT pairs found.")

    if len(matches) < 10:
        print("\nWARNING: Fewer than 10 matches found.")
        print("  Check eval_images/ has the numbered-folder images (not test/ folders).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── CLIP ──
    print(f"\nRunning CLIP (ViT-B/32) on {len(matches)} crops...")
    print(f"  Device: {device}")
    clip_results = run_clip(matches, preds, args.images, device)
    print(f"  {len(clip_results)} crops processed.")

    if not args.skip_siglip:
        # ── SIGLIP 2 ──
        print(f"\nRunning SigLIP 2 on {len(clip_results)} crops...")
        final_results = run_siglip(clip_results, preds, args.images, device)
        print(f"  {len(final_results)} crops processed with both models.")
    else:
        final_results = clip_results
        print("\nSkipped SigLIP 2 (--skip-siglip flag set).")

    df = pd.DataFrame(final_results)
    df_out = print_results(df)
    plot_results(df_out, args.plot)

    out_csv = "mini_trust_results.csv"
    save_cols = [c for c in df_out.columns if c != 'probs']
    df_out[save_cols].to_csv(out_csv, index=False)
    print(f"  Full results saved to: {out_csv}\n")


if __name__ == "__main__":
    main()