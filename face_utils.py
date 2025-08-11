import threading, uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import insightface

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
            _face_analyser.prepare(ctx_id=0)
    return _face_analyser


def get_roop_faces(image_np):
    """Return list of dicts with index, coords [t,r,b,l], bbox (np array), and cropped face."""
    analyser = get_roop_face_analyser()
    faces = analyser.get(image_np)
    results = []
    for idx, face in enumerate(faces):
        # face.bbox: [left, top, right, bottom]
        left, top, right, bottom = face.bbox.astype(int)
        face_img = image_np[top:bottom, left:right]
        results.append({
            "index": idx,
            "coords": [int(top), int(right), int(bottom), int(left)],
            "bbox": face.bbox.astype(int),
            "face_img": face_img
        })
    return results


def crop_source_face_to_temp(source_path: str, chosen_index: int, margin: float = 0.35) -> str:
    """
    Crop just the selected face from the source image (with a margin) so Roop
    uses the intended face. Returns the path to a temp PNG; falls back to original path.
    """
    try:
        src_img = Image.open(source_path).convert("RGB")
        src_np = np.array(src_img)
        faces = get_roop_faces(src_np)
        if not faces or chosen_index < 0 or chosen_index >= len(faces):
            print(f"[WARN] crop_source_face_to_temp: index {chosen_index} not found — using full source.")
            return source_path

        top, right, bottom, left = faces[chosen_index]["coords"]
        h, w = src_np.shape[:2]
        bw, bh = (right - left), (bottom - top)
        cx, cy = left + bw / 2.0, top + bh / 2.0

        new_w, new_h = int(bw * (1 + margin)), int(bh * (1 + margin))
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(w, int(cx + new_w / 2))
        y2 = min(h, int(cy + new_h / 2))
        if x2 <= x1 or y2 <= y1:
            return source_path

        cropped = src_img.crop((x1, y1, x2, y2))
        tmp = f"/tmp/source_face_{uuid.uuid4().hex}.png"
        cropped.save(tmp)
        print(f"[DEBUG] Temp source face saved -> {tmp}")
        return tmp
    except Exception as e:
        print(f"[WARN] crop_source_face_to_temp failed: {e}; using full source.")
        return source_path
    
def bbox_iou(a, b):
    # (top, right, bottom, left)
    at, ar, ab, al = map(int, a)
    bt, br, bb, bl = map(int, b)
    ix1, iy1 = max(al, bl), max(at, bt)
    ix2, iy2 = min(ar, br), min(ab, bb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(ar - al, 0) * max(ab - at, 0)
    b_area = max(br - bl, 0) * max(bb - bt, 0)
    union = max(a_area + b_area - inter, 1e-6)
    return inter / union

def match_face_by_iou(intended_bbox, faces_now):
    # faces_now entries have "coords" in (top,right,bottom,left)
    best, best_iou = None, -1.0
    for f in faces_now:
        iou = bbox_iou(intended_bbox, f["coords"])
        if iou > best_iou:
            best, best_iou = f, iou
    return best

def parse_indices(s: str) -> list[int]:
    if not s:
        return []
    s = s.strip()
    if s.startswith('['):
        try:
            return [int(x) for x in json.loads(s)]
        except Exception:
            return []
    return [int(x) for x in s.split(',') if x.strip().isdigit()]

def crop_source_face_to_temp(source_path: str, chosen_index: int = 0, margin: float = 0.35) -> str:
    """Crop the chosen selfie face with a margin and save to /tmp."""
    src_np = np.array(Image.open(source_path).convert("RGB"))
    faces = get_roop_faces(src_np)
    if not faces:
        return source_path
    if chosen_index >= len(faces):
        chosen_index = 0
    top, right, bottom, left = [int(v) for v in faces[chosen_index]["coords"]]
    w, h = right - left, bottom - top
    cx, cy = left + w // 2, top + h // 2
    ml, mt = int(w * (1 + margin) / 2), int(h * (1 + margin) / 2)
    L = max(0, cx - ml); T = max(0, cy - mt)
    R = min(src_np.shape[1], cx + ml); B = min(src_np.shape[0], cy + mt)
    crop = Image.fromarray(src_np).crop((L, T, R, B))
    out = f"/tmp/source_face_{uuid.uuid4().hex}.png"
    crop.save(out)
    print(f"[DEBUG] Temp source face saved -> {out}")
    return out

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    diff = a - b
    return float((diff * diff).mean())

def pick_changed_bbox(target_img: Image.Image,
                       roop_img: Image.Image,
                       bboxes_trbl: list[list[int]]) -> tuple[int, list[int]]:
    """
    Among all target bboxes, find which one changed the most between
    target_img and roop_img by MSE. Returns (index_in_list, bbox).
    """
    tgt = np.array(target_img.convert("RGB"))
    out = np.array(roop_img.convert("RGB"))
    best_i, best_bbox, best_score = -1, None, -1.0

    for i, (top, right, bottom, left) in enumerate(bboxes_trbl):
        # clamp
        top = max(0, min(top, tgt.shape[0])); bottom = max(0, min(bottom, tgt.shape[0]))
        left = max(0, min(left, tgt.shape[1])); right = max(0, min(right, tgt.shape[1]))
        if top >= bottom or left >= right:
            continue

        tgt_crop = tgt[top:bottom, left:right]
        out_crop = out[top:bottom, left:right]
        if tgt_crop.size == 0 or out_crop.size == 0:
            continue

        # If shapes differ (can happen rarely), resize roop crop to tgt crop
        if out_crop.shape != tgt_crop.shape:
            rh, rw = tgt_crop.shape[0], tgt_crop.shape[1]
            out_crop_img = Image.fromarray(out_crop).resize((rw, rh), Image.BILINEAR)
            out_crop = np.array(out_crop_img)

        score = mse(tgt_crop, out_crop)
        if score > best_score:
            best_i, best_bbox, best_score = i, [top, right, bottom, left], score

    return best_i, best_bbox


