import threading, uuid
import numpy as np
from PIL import Image
import insightface
import json

# ---------- InsightFace analyzer (shared) ----------
_face_analyser = None
_face_analyser_lock = threading.Lock()

def get_roop_face_analyser():
    global _face_analyser
    with _face_analyser_lock:
        if _face_analyser is None:
            _face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyser


def get_roop_faces(image_np):
    """
    Return list of dicts with index and bbox (LTRB).
    LTRB = (left, top, right, bottom)
    """
    analyser = get_roop_face_analyser()
    faces = analyser.get(image_np)
    results = []
    for idx, face in enumerate(faces):
        # face.bbox: [left, top, right, bottom]
        l, t, r, b = [int(v) for v in face.bbox]
        face_img = image_np[t:b, l:r]
        results.append({
            "index": idx,
            "bbox": (l, t, r, b),   # <-- canonical: LTRB
            "face_img": face_img
        })
    return results


def bbox_iou(a_ltrb, b_ltrb):
    """
    IoU for LTRB boxes: (left, top, right, bottom)
    """
    al, at, ar, ab = map(int, a_ltrb)
    bl, bt, br, bb = map(int, b_ltrb)
    ix1, iy1 = max(al, bl), max(at, bt)
    ix2, iy2 = min(ar, br), min(ab, bb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(ar - al, 0) * max(ab - at, 0)
    b_area = max(br - bl, 0) * max(bb - bt, 0)
    union = max(a_area + b_area - inter, 1e-6)
    return inter / union


def match_face_by_iou(intended_bbox_ltrb, faces_now):
    """
    faces_now entries must have "bbox" in LTRB.
    """
    best, best_iou = None, -1.0
    for f in faces_now:
        iou = bbox_iou(intended_bbox_ltrb, f["bbox"])
        if iou > best_iou:
            best, best_iou = f, iou
    return best


def expand_bbox_ltrb(bbox, margin, img_w, img_h):
    """
    Expand an LTRB box by `margin` percent (e.g., 0.25) and clamp to image.
    """
    l, t, r, b = map(int, bbox)
    w, h = (r - l), (b - t)
    cx, cy = l + w / 2.0, t + h / 2.0
    nw, nh = w * (1 + margin), h * (1 + margin)
    nl = max(0, int(cx - nw / 2.0))
    nt = max(0, int(cy - nh / 2.0))
    nr = min(img_w, int(cx + nw / 2.0))
    nb = min(img_h, int(cy + nh / 2.0))
    return (nl, nt, nr, nb)


def map_bbox_to_roop_index(image_path, intended_bbox_ltrb, iou_threshold=0.05):
    """
    Given an intended bbox (LTRB), find the Roop detection index with max IoU on that image.
    """
    image_np = np.array(Image.open(image_path).convert("RGB"))
    roop_faces = get_roop_faces(image_np)

    best_idx, best_iou = None, -1.0
    for face in roop_faces:
        iou = bbox_iou(intended_bbox_ltrb, face["bbox"])  # LTRB vs LTRB
        if iou > best_iou:
            best_iou = iou
            best_idx = face["index"]

    if best_iou < iou_threshold:
        print(f"[WARN] map_bbox_to_roop_index: best_iou={best_iou:.3f} < {iou_threshold}")
        return None

    print(f"[DEBUG] map_bbox_to_roop_index: best_idx={best_idx}, IoU={best_iou:.3f}")
    return int(best_idx)


# (Optional) keep your mse/pick_changed_bbox helpers if you use them elsewhere;
# if they accept TRBL, convert them to LTRB for consistency too.

def crop_source_face_to_temp(source_path: str, chosen_index: int = 0, margin: float = 0.35) -> str:
    """
    Detect faces in the source image, pick `chosen_index`, expand its LTRB bbox by `margin`,
    crop to /tmp, and return that temp path. Falls back to original source if something fails.
    Ensures crop is large enough for Roop to detect the face.
    All boxes are LTRB.
    """
    try:
        src_img = Image.open(source_path).convert("RGB")
        src_np = np.array(src_img)

        faces = get_roop_faces(src_np)  # [{"index", "bbox" (LTRB), "face_img"}]
        if not faces:
            print("[WARN] crop_source_face_to_temp: no faces found in source — using full source.")
            return source_path

        if chosen_index < 0 or chosen_index >= len(faces):
            print(f"[WARN] crop_source_face_to_temp: index {chosen_index} out of range — using 0.")
            chosen_index = 0

        bbox_ltrb = faces[chosen_index]["bbox"]  # (l, t, r, b)

        # Ensure at least 0.5 margin for Roop
        safe_margin = max(margin, 0.5)
        nl, nt, nr, nb = expand_bbox_ltrb(
            bbox_ltrb, margin=safe_margin,
            img_w=src_img.width, img_h=src_img.height
        )

        if nl >= nr or nt >= nb:
            print("[WARN] crop_source_face_to_temp: invalid expanded box — using full source.")
            return source_path

        # Enforce minimum crop size
        MIN_SIZE = 256
        crop_w, crop_h = nr - nl, nb - nt
        if crop_w < MIN_SIZE or crop_h < MIN_SIZE:
            print(f"[INFO] crop_source_face_to_temp: expanding crop to min {MIN_SIZE}px for Roop.")
            cx, cy = nl + crop_w // 2, nt + crop_h // 2
            half_w, half_h = max(MIN_SIZE // 2, crop_w // 2), max(MIN_SIZE // 2, crop_h // 2)
            nl = max(0, cx - half_w)
            nr = min(src_img.width, cx + half_w)
            nt = max(0, cy - half_h)
            nb = min(src_img.height, cy + half_h)

        cropped = src_img.crop((nl, nt, nr, nb))
        tmp = f"/tmp/source_face_{uuid.uuid4().hex}.png"
        cropped.save(tmp)
        print(f"[DEBUG] Temp source face saved -> {tmp} (size: {cropped.size})")
        return tmp

    except Exception as e:
        print(f"[WARN] crop_source_face_to_temp failed: {e}; using full source.")
        return source_path
    
def parse_indices(raw: str) -> list[int]:
    """
    Accepts '3,17' or '[3, 17]' and returns [3, 17].
    Returns [] on any parse error.
    """
    if not raw:
        return []
    s = raw.strip()
    try:
        if s.startswith('['):
            arr = json.loads(s)
            return [int(x) for x in arr]
        return [int(x) for x in s.split(',') if x.strip()]
    except Exception:
        return []


