"""
export_crops.py — Save each detected object as a small labeled thumbnail.

Why: labeling duplicate pairs by eye is hard when each image holds dozens of objects
and you only have an object NUMBER. This renders every detection box as its own little
image named <image>_obj<number>.png, so when pairs.csv says "DJI_0631 #2 <-> DJI_0632 #3"
you just open the two thumbnails and compare them directly.

USAGE
    python export_crops.py --images eval_images --pred detections.csv --out crops
    # then open the crops/ folder and eyeball pairs

OPTIONAL — make a side-by-side sheet for exactly the pairs you want to judge:
    python export_crops.py --images eval_images --pred detections.csv --out crops \
        --pairs pairs.csv --min-sift 10
    # writes crops/_pairs_to_check.html : open in a browser, see each candidate pair
    # side by side with its match counts, decide same-object or not at a glance.
"""

import argparse
import os
import base64
import io
import pandas as pd
from PIL import Image


def load_pred(pred_path):
    df = pd.read_csv(pred_path)
    df = df[pd.to_numeric(df["x1"], errors="coerce").notna()].copy()
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x1", "y1", "x2", "y2"])
    return df


def crop_key(name, number):
    base = os.path.splitext(str(name))[0]
    return f"{base}_obj{number}"


def export_crops(df, image_dir, out_dir, pad=12, thumb=220):
    os.makedirs(out_dir, exist_ok=True)
    cache = {}
    saved = {}
    for _, r in df.iterrows():
        name = str(r["image_name"])
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            continue
        if name not in cache:
            cache[name] = Image.open(path).convert("RGB")
        im = cache[name]
        W, H = im.size
        x1 = max(0, int(r["x1"]) - pad); y1 = max(0, int(r["y1"]) - pad)
        x2 = min(W, int(r["x2"]) + pad); y2 = min(H, int(r["y2"]) + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = im.crop((x1, y1, x2, y2))
        crop.thumbnail((thumb, thumb))
        key = crop_key(name, r["number"])
        fp = os.path.join(out_dir, key + ".png")
        crop.save(fp)
        saved[key] = fp
    return saved


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_pairs_sheet(pairs_path, saved, out_dir, min_sift):
    pd_pairs = pd.read_csv(pairs_path)
    rows = pd_pairs[pd_pairs["sift_matches"] >= min_sift].sort_values(
        "sift_matches", ascending=False
    )
    cells = []
    for _, r in rows.iterrows():
        ka = crop_key(r["img_a"], r["obj_a"])
        kb = crop_key(r["img_b"], r["obj_b"])
        ia = f'<img src="data:image/png;base64,{b64(saved[ka])}">' if ka in saved else "(missing)"
        ib = f'<img src="data:image/png;base64,{b64(saved[kb])}">' if kb in saved else "(missing)"
        cells.append(f"""
        <div class="pair">
          <div class="imgs">{ia}{ib}</div>
          <div class="meta">{r['img_a']} #{r['obj_a']} &harr; {r['img_b']} #{r['obj_b']}<br>
            SIFT {r['sift_matches']} &middot; AKAZE {r['akaze_matches']}</div>
        </div>""")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
      body {{ font-family: sans-serif; background:#0c2c38; color:#eaf6f5; padding:20px; }}
      h1 {{ font-size:20px; }}
      .pair {{ display:inline-block; vertical-align:top; margin:10px; padding:10px;
               background:#08242e; border:1px solid #18424f; border-radius:10px; }}
      .imgs img {{ height:180px; border:1px solid #18424f; border-radius:6px; margin:2px; }}
      .meta {{ font-size:13px; color:#8fb3bb; margin-top:6px; text-align:center; }}
    </style></head><body>
    <h1>Candidate duplicate pairs (SIFT &ge; {min_sift}) — same object or not?</h1>
    <p style="color:#8fb3bb">Left vs right: if they show the SAME physical object, mark true_dup=1 in pairs.csv; else 0.</p>
    {''.join(cells)}
    </body></html>"""
    fp = os.path.join(out_dir, "_pairs_to_check.html")
    with open(fp, "w") as f:
        f.write(html)
    return fp, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="eval_images")
    ap.add_argument("--pred", default="detections.csv")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--pairs", default=None, help="optional pairs.csv to build a side-by-side sheet")
    ap.add_argument("--min-sift", type=int, default=10, help="only show pairs with >= this many SIFT matches")
    ap.add_argument("--pad", type=int, default=12, help="pixels of context around each box")
    args = ap.parse_args()

    df = load_pred(args.pred)
    saved = export_crops(df, args.images, args.out, pad=args.pad)
    print(f"Saved {len(saved)} crops to {args.out}/")
    print("  Open the folder and compare crops named <image>_obj<number>.png")

    if args.pairs:
        fp, n = build_pairs_sheet(args.pairs, saved, args.out, args.min_sift)
        print(f"\nWrote side-by-side sheet for {n} candidate pairs (SIFT >= {args.min_sift}):")
        print(f"  {fp}")
        print("  Open it in a browser, decide same-object or not for each, then fill")
        print("  true_dup in pairs.csv accordingly (everything not shown is almost")
        print("  certainly true_dup=0).")


if __name__ == "__main__":
    main()