import numpy as np
import torch
import timm
from PIL import Image
import cv2
import io
import base64
import os
import joblib

# SAM 2 & DINOv3 Globals
sam2_predictor = None
dinov3_model = None
dinov3_transform = None
svm_classifier = None
device = "cuda" if torch.cuda.is_available() else "cpu"

WORK_LONG_SIDE = 1024
TILE_OVERLAP_FRAC = 0.15
DEDUP_IOU_THRESHOLD = 0.5
MIN_AREA_FRAC = 0.0003
MAX_AREA_FRAC = 0.5
THRESHOLD = 0.8  # Updated production threshold as requested

def mask_to_tight_box(seg):
    ys, xs = np.where(seg)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def get_tile_bounds(img_w, img_h, overlap_frac=TILE_OVERLAP_FRAC):
    half_w, half_h = img_w // 2, img_h // 2
    margin_w = int(half_w * overlap_frac)
    margin_h = int(half_h * overlap_frac)

    tiles = [
        (0, 0, half_w + margin_w, half_h + margin_h),
        (half_w - margin_w, 0, img_w, half_h + margin_h),
        (0, half_h - margin_h, half_w + margin_w, img_h),
        (half_w - margin_w, half_h - margin_h, img_w, img_h),
    ]
    clipped = []
    for x1, y1, x2, y2 in tiles:
        clipped.append((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return clipped

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

def dedup_nms(detections, iou_threshold=DEDUP_IOU_THRESHOLD):
    detections = sorted(detections, key=lambda d: d["box_conf"], reverse=True)
    kept = []
    for d in detections:
        box = [d["x1"], d["y1"], d["x2"], d["y2"]]
        if any(iou(box, [k["x1"], k["y1"], k["x2"], k["y2"]]) > iou_threshold for k in kept):
            continue
        kept.append(d)
    return kept

def init_models():
    """Load production DINOv3 timm model, SAM2 predictor, and whitened SVM filter."""
    global sam2_predictor, dinov3_model, dinov3_transform, svm_classifier, device
    print(f"Loading Production ML Models on device: {device}...")

    # 1. Load SAM 2 Image Predictor
    print("Loading SAM 2 Image Predictor...")
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
        sam2_predictor.model.to(device)
    except Exception as e:
        print(f"SAM 2 Predictor init failed: {e}")

    # 2. Load DINOv3 backbone via timm (exact match to training script)
    print("Loading DINOv3 timm backbone (vit_large_patch16_dinov3.sat493m)...")
    try:
        dinov3_model = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0)
        dinov3_model = dinov3_model.to(device).eval()
        config = timm.data.resolve_model_data_config(dinov3_model)
        dinov3_transform = timm.data.create_transform(**config, is_training=False)
    except Exception as e:
        print(f"DINOv3 timm init failed: {e}")

    # 3. Load Whitened SVM Filter
    print("Loading Whitened SVM Filter...")
    svm_path = "S2C_pipeline_results/master_debris_filter_whitened.pkl"
    if os.path.exists(svm_path):
        try:
            svm_classifier = joblib.load(svm_path)
            print("Whitened SVM Filter loaded successfully!")
        except Exception as e:
            print(f"WARNING: Could not load SVM filter: {e}")
    else:
        print(f"WARNING: SVM filter not found at {svm_path}.")

    print("Model initialization complete.")

def extract_features_batched(model, transform, crop_list, device, batch_size=64):
    features = []
    with torch.no_grad():
        for i in range(0, len(crop_list), batch_size):
            batch_crops = crop_list[i:i + batch_size]
            batch_tensors = torch.stack([transform(crop) for crop in batch_crops]).to(device)
            feats = model(batch_tensors).cpu().numpy()
            features.extend(feats)
    return np.array(features)

def run_full_pipeline(image_numpy):
    """
    Takes RGB numpy array, runs SAM2 + Timm DINOv3 + Whitened SVM with threshold 0.8.
    """
    global sam2_predictor, dinov3_model, dinov3_transform, svm_classifier, device
    
    # Ensure array is writable to prevent PyTorch tensor warnings/errors
    if not image_numpy.flags.writeable:
        image_numpy = image_numpy.copy()
    
    pil_img = Image.fromarray(image_numpy)
    orig_w, orig_h = pil_img.size
    full_img_area = orig_w * orig_h
    
    print(f"\n[Pipeline] Processing image of size {orig_w}x{orig_h}...")
    image_detections = []
    
    if sam2_predictor:
        try:
            sam2_predictor.set_image(image_numpy)
        except Exception as e:
            print(f"[Pipeline Error] SAM2 set_image failed: {e}")
            return []

        tile_bounds = get_tile_bounds(orig_w, orig_h)
        print(f"[Pipeline] Generated {len(tile_bounds)} tiles for processing.")
        
        for idx, (tx1, ty1, tx2, ty2) in enumerate(tile_bounds):
            tile_w, tile_h = tx2 - tx1, ty2 - ty1
            if tile_w <= 10 or tile_h <= 10: continue
            
            grid_step = 128
            boxes = []
            for gx in range(0, tile_w, grid_step):
                for gy in range(0, tile_h, grid_step):
                    bx1 = gx + tx1
                    by1 = gy + ty1
                    bx2 = min(tx2, bx1 + 128)
                    by2 = min(ty2, by1 + 128)
                    if bx2 - bx1 > 20 and by2 - by1 > 20:
                        boxes.append([bx1, by1, bx2, by2])
            
            if not boxes: continue
            
            try:
                chunk_boxes = np.array(boxes)
                chunk_masks, chunk_scores, _ = sam2_predictor.predict(box=chunk_boxes, multimask_output=False)
                
                for j, box in enumerate(chunk_boxes):
                    mask = chunk_masks[j][0]
                    area = mask.sum()
                    area_frac = area / full_img_area
                    
                    if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
                        continue
                        
                    box_tight = mask_to_tight_box(mask)
                    if box_tight is None: continue
                    
                    x1, y1, x2, y2 = box_tight
                    image_detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "box_conf": float(chunk_scores[j] if j < len(chunk_scores) else 0.9)
                    })
            except Exception as e:
                print(f"[Pipeline Warning] Tile {idx} prediction exception: {e}")
                continue
                
        candidate_boxes = dedup_nms(image_detections)
        print(f"[Pipeline] SAM 2 proposed {len(candidate_boxes)} candidate boxes after NMS.")
    else:
        print("[Pipeline Warning] SAM2 predictor not loaded! Using fallback box.")
        candidate_boxes = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "box_conf": 0.99}]

    valid_detections = []
    crops_to_extract = []
    row_meta = []
    
    for d in candidate_boxes:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        if x2 - x1 <= 5 or y2 - y1 <= 5: continue
        
        try:
            crop_np = image_numpy[y1:y2, x1:x2]
            pil_crop = Image.fromarray(crop_np.astype(np.uint8))
            pil_crop.thumbnail((224, 224))
            crops_to_extract.append(pil_crop)
            row_meta.append(d)
        except Exception:
            continue
            
    print(f"[Pipeline] Extracted {len(crops_to_extract)} valid crops for DINOv3 + SVM evaluation.")

    if crops_to_extract and dinov3_model and svm_classifier and dinov3_transform:
        features = extract_features_batched(dinov3_model, dinov3_transform, crops_to_extract, device)
        scores = svm_classifier.decision_function(features)
        
        survived_count = 0
        for i, score in enumerate(scores):
            # Threshold check (0.8)
            if score > THRESHOLD:
                survived_count += 1
                d = row_meta[i]
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                crop_pil = crops_to_extract[i]
                
                buffered = io.BytesIO()
                crop_pil.save(buffered, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                valid_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': "pending_cluster",
                    'crop_base64': img_b64,
                    'embedding': features[i].tolist()
                })
        print(f"[Pipeline] SVM Filtering complete. Survived threshold ({THRESHOLD}): {survived_count} items.")
                
    return valid_detections