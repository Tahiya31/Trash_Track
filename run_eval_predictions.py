import os, glob, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import supervision as sv
from script import delete_big, delete_rock, delete_overlap, delete_box, get_lat_lon, get_altitude, get_exif
from GroundingDINO.groundingdino.util.inference import Model

# ---- config ----
IMAGE_DIR = "eval_images"          # folder with your 35 test images
OUTPUT_CSV = "detections.csv"
RESIZE_LONG_SIDE = None            # set to None to disable resizing (slower, full accuracy)
# -----------------

current_dir = os.getcwd()
gd = Model(current_dir + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
           current_dir + "/weights/groundingdino_swint_ogc.pth", device="cpu")
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
PROMPTS = ['wood','cage','fishing gear','nature','plastic','metal','wheel']
CLASSES = ['all trashes', 'rocks']

def classify(crop):
    inp = proc(text=PROMPTS, images=crop, return_tensors="pt", padding=True)
    return PROMPTS[torch.argmax(clip(**inp).logits_per_image.softmax(dim=1))]

rows = []
for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))):
    name = os.path.basename(path)
    pil = Image.open(path).convert("RGB")
    orig_w, orig_h = pil.size
    scale = 1.0
    work = pil
    if RESIZE_LONG_SIDE and max(orig_w, orig_h) > RESIZE_LONG_SIDE:
        scale = RESIZE_LONG_SIDE / max(orig_w, orig_h)
        work = pil.resize((int(orig_w*scale), int(orig_h*scale)))
    im = np.asarray(work)

    # same two-pass detection as app.py
    d1 = gd.predict_with_classes(image=im, classes=CLASSES, box_threshold=0.3, text_threshold=0.25)
    d2 = gd.predict_with_classes(image=im, classes=CLASSES, box_threshold=0.15, text_threshold=0.10)
    d1 = delete_rock(delete_big(d1, im)); d2 = delete_rock(delete_big(d2, im))
    xyxy = np.vstack((d1.xyxy, d2.xyxy))
    conf = np.concatenate((d1.confidence, d2.confidence))
    cid  = np.concatenate((d1.class_id, d2.class_id))
    det = sv.Detections(xyxy, d1.mask, conf, cid, d1.tracker_id)
    det = delete_box(delete_overlap(det))

    try:    lon, lat = get_lat_lon(get_exif(path)); alt = get_altitude(path)
    except: lon = lat = alt = "NA"

    for i in range(len(det.xyxy)):
        box = det.xyxy[i] / scale          # <-- scale boxes BACK to full resolution
        crop = np.array(pil.crop(box))     # crop from ORIGINAL image
        rows.append([box[0], box[1], box[2], box[3], lon, lat, alt, name, i, classify(crop)])
    print(f"done {name}: {len(det.xyxy)} objects")

pd.DataFrame(rows, columns=['x1','y1','x2','y2','longitude','latitude','altitude','image_name','number','type']).to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(rows)} detections to {OUTPUT_CSV}")