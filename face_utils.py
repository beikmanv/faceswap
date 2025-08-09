import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import face_recognition
import insightface
import threading
from itertools import chain

def mask_and_extract_face(image_path, face_index=0):
    """Extract full face region using InsightFace bounding box."""
    from PIL import ImageDraw, ImageFilter

    image = Image.open(image_path).convert("RGBA")
    image_np = np.array(image.convert("RGB"))

    faces = get_roop_faces(image_np)

    if not faces or face_index >= len(faces):
        raise ValueError("Face index out of range or no faces detected.")

    face = faces[face_index]
    top, right, bottom, left = face["coords"]

    # Create a mask the same size as the image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, right, bottom], fill=255)

    # Optionally feather the mask edges
    mask = mask.filter(ImageFilter.GaussianBlur(3))

    # Apply alpha mask to image
    r, g, b, _ = image.split()
    masked_face = Image.merge("RGBA", (r, g, b, mask))

    # Crop to bounding box
    cropped = masked_face.crop((left, top, right, bottom))

    return cropped, (left, top, right, bottom)

def paste_masked_face(target_path, masked_face_img, bbox, output_path):
    base_img = Image.open(target_path).convert("RGBA")

    # Resize bbox to match new face size
    x, y, x2, y2 = bbox
    face_w, face_h = masked_face_img.size
    paste_box = (x, y, x + face_w, y + face_h)

    # Paste using alpha channel
    base_img.paste(masked_face_img, (x, y), masked_face_img)

    base_img.convert("RGB").save(output_path)

def crop_exact_region(image_path, coords):
    """Crop any specific rectangular region."""
    top, right, bottom, left = coords
    img = Image.open(image_path).convert("RGB")
    return img.crop((left, top, right, bottom))

def create_masked_target(image_path, selected_coords, padding=60):
    img = face_recognition.load_image_file(image_path)

    if not selected_coords:
        raise ValueError("Selected face coordinates not provided.")

    top, right, bottom, left = selected_coords

    # Apply padding
    top_pad = max(0, top - padding)
    bottom_pad = min(img.shape[0], bottom + padding)
    left_pad = max(0, left - padding)
    right_pad = min(img.shape[1], right + padding)

    # Create black background
    black_img = img.copy()
    black_img[:, :] = 0  # make all pixels black

    # Paste selected face region
    black_img[top_pad:bottom_pad, left_pad:right_pad] = img[top_pad:bottom_pad, left_pad:right_pad]

    return Image.fromarray(black_img), (top_pad, right_pad, bottom_pad, left_pad)

_face_analyser = None
_face_analyser_lock = threading.Lock()

def get_roop_face_analyser():
    global _face_analyser
    with _face_analyser_lock:
        if _face_analyser is None:
            _face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            _face_analyser.prepare(ctx_id=0)
    return _face_analyser

def get_roop_faces(image_np):
    analyser = get_roop_face_analyser()
    faces = analyser.get(image_np)
    results = []

    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int)  # [left, top, right, bottom]
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        face_img = image_np[top:bottom, left:right]
        results.append({
            "index": idx,
            "coords": [top, right, bottom, left],
            "bbox": bbox,
            "face_img": face_img
        })
    return results

def crop_source_face_to_temp(source_path: str, chosen_index: int, margin: float = 0.3) -> str:
    """
    Crops the chosen face from the source image with a margin, saves a temp PNG, and returns its path.
    If detection fails or index invalid, returns the original source_path.
    """
    try:
        src_img = Image.open(source_path).convert("RGB")
        src_np = np.array(src_img)
        faces = get_roop_faces(src_np)
        if not faces or chosen_index >= len(faces) or chosen_index < 0:
            print(f"[WARN] Source face index {chosen_index} not found; using full source.")
            return source_path

        top, right, bottom, left = faces[chosen_index]["coords"]
        h, w = src_np.shape[:2]

        # expand bbox by margin
        bw, bh = (right - left), (bottom - top)
        cx, cy = left + bw / 2.0, top + bh / 2.0
        new_w, new_h = int(bw * (1 + margin)), int(bh * (1 + margin))
        nx1 = max(0, int(cx - new_w / 2))
        ny1 = max(0, int(cy - new_h / 2))
        nx2 = min(w, int(cx + new_w / 2))
        ny2 = min(h, int(cy + new_h / 2))

        # safety: ensure at least some size
        if nx2 <= nx1 or ny2 <= ny1:
            print("[WARN] Invalid crop after margin; using full source.")
            return source_path

        cropped = src_img.crop((nx1, ny1, nx2, ny2))
        temp_src_path = f"/tmp/source_face_{uuid.uuid4().hex}.png"
        cropped.save(temp_src_path)
        print(f"[DEBUG] Temp source face saved: {temp_src_path} (index={chosen_index})")
        return temp_src_path
    except Exception as e:
        print(f"[WARN] Could not crop source face: {e}; using full source.")
        return source_path
    
    # --- helpers for parsing indices from form fields ---
def _parse_indices(s: str, default=None):
    """
    Accepts:
      - JSON list string: "[0,2,3]"
      - comma list: "0,2,3"
      - single number: "1"
      - empty/None -> default
    Returns a list[int].
    """
    if s is None or s == "":
        return default if default is not None else []
    s = s.strip()
    try:
        # try JSON first
        import json
        v = json.loads(s)
        if isinstance(v, list):
            return [int(x) for x in v]
        return [int(v)]
    except Exception:
        # fallback: comma-separated
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [int(p) for p in parts] if parts else (default if default is not None else [])


