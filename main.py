from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, io, json, uuid, base64, numpy as np
from PIL import Image
import face_recognition

from swap import swap_faces
from upscale import upscale_image
from face_utils import (
    get_roop_faces,
    crop_source_face_to_temp,
    bbox_iou,
    match_face_by_iou,
    map_bbox_to_roop_index,
    expand_bbox_ltrb
)

app = FastAPI()

@app.middleware("http")
async def add_cache_headers(request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API contract + infra =====
import asyncio, time
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# Concurrency cap (prevent stampedes)
MAX_CONCURRENCY = 2  # tune 1..2 as you like
SEM = asyncio.Semaphore(MAX_CONCURRENCY)

# In-memory job store (pretend async)
# job_id -> {"status": "queued|processing|completed|failed", "download_url": str|None, "error": str|None, "ts": float}
JOBS: dict[str, dict] = {}

# Validation
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

def _job_ok(job_id: str, download_url: str | None = None):
    JOBS[job_id] = {"job_id": job_id, "status": "completed", "download_url": download_url, "error": None, "ts": time.time()}
    return JOBS[job_id]

def _job_fail(job_id: str, msg: str, http=500):
    JOBS[job_id] = {"job_id": job_id, "status": "failed", "download_url": None, "error": msg, "ts": time.time()}
    return JSONResponse(status_code=http, content=JOBS[job_id])

def _bad_client(msg: str):
    # 4xx helper
    return JSONResponse(status_code=400, content={"success": False, "error": msg})

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif", "application/octet-stream"}

def _mime_ok(mime: str | None):
    # Accept common image types and octet-stream (CDNs often send images that way).
    # We'll still *verify* bytes with Pillow below.
    if not mime:
        return True
    mime = mime.split(";")[0].strip().lower()
    return mime in ALLOWED_IMAGE_MIMES



OUTPUT_DIR = "static/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")




# --- fetch_image: proxy the external URL so the browser avoids CORS and we prefer jpeg/png ---
from fastapi import Query, HTTPException
from fastapi.responses import Response, JSONResponse
from urllib.parse import urlparse
import requests

@app.get("/fetch_image")
def fetch_image(
    url: str = Query(..., description="Direct image URL"),
    debug: int = Query(0, description="Return debug info on failure (1)"),
):
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed")

    try:
        r = requests.get(
            url,
            timeout=20,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://faceswap.tilda.ws/",
                "Accept": "image/jpeg,image/png;q=0.9,image/*;q=0.5,*/*;q=0.1",
            },
        )
        r.raise_for_status()
        content = r.content
        ct_header = (r.headers.get("Content-Type") or "").lower().split(";")[0].strip()

        head = content[:16]
        if head.startswith(b"\xff\xd8\xff"): mime = "image/jpeg"
        elif head.startswith(b"\x89PNG\r\n\x1a\n"): mime = "image/png"
        elif head.startswith(b"GIF87a") or head.startswith(b"GIF89a"): mime = "image/gif"
        elif len(content) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP": mime = "image/webp"
        elif ct_header.startswith("image/"): mime = ct_header
        else:
            if debug:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": r.status_code,
                        "content_type": ct_header,
                        "len": len(content),
                        "head_hex": content[:64].hex(),
                        "headers": dict(r.headers),
                    },
                )
            raise HTTPException(status_code=400, detail=f"URL did not return an image (content-type={ct_header})")

        return Response(content=content, media_type=mime, headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not fetch image: {e}"})

@app.post("/swap")
async def swap_api(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile | None = File(None),
    selected_source_face_index: int = Form(0),
    selected_face_index: int = Form(0),
    selected_face_bbox: str = Form("[]"),
    selected_faces: str = Form("[]"),
    selected_target_face_indices: str = Form("[]"),
    selected_face_bboxes: str = Form("[]"),
    target_url: str = Form(""),
):
    """
    Contract: returns {success, job_id, status, download_url?, error?}
    Today it's synchronous under a semaphore, but we reply as if async with status='completed'.
    """
    job_id = str(uuid.uuid4())
    # Queue semantics (pretend): record 'processing' early so /status works while the request is in-flight
    JOBS[job_id] = {"job_id": job_id, "status": "processing", "download_url": None, "error": None, "ts": time.time()}

    # Basic client validation first (return 4xx on bad input)
    if not source:
        return _bad_client("Missing 'source' file.")
    if not target and not (target_url or "").strip():
        return _bad_client("Provide 'target' file or 'target_url'.")
    if not _mime_ok(source.content_type):
        return _bad_client(f"Unsupported source mime {source.content_type}.")
    if target and not _mime_ok(target.content_type):
        return _bad_client(f"Unsupported target mime {target.content_type}.")

    # Size caps (read bytes once)
    source_bytes = await source.read()
    if not source_bytes:
        return _bad_client("Source is empty.")
    if len(source_bytes) > MAX_IMAGE_BYTES:
        return _bad_client("Source is too large.")

    target_bytes = b""
    if target is not None:
        tb = await target.read()
        if not tb:
            return _bad_client("Target is empty.")
        if len(tb) > MAX_IMAGE_BYTES:
            return _bad_client("Target is too large.")
        target_bytes = tb

    # File locations
    OUTPUT_DIR = "static/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
    target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"

    # Validate as images (Pillow verify)
    try:
        Image.open(io.BytesIO(source_bytes)).verify()
    except Exception:
        return _bad_client("Source is not a valid image.")
    with open(source_path, "wb") as f:
        f.write(source_bytes)

    # Target: prefer uploaded file; otherwise download from your existing proxy flow
    if target_bytes:
        try:
            Image.open(io.BytesIO(target_bytes)).verify()
        except Exception:
            return _bad_client("Target is not a valid image.")
        with open(target_path, "wb") as f:
            f.write(target_bytes)
    else:
        # Use your existing server-side fetch endpoint logic
        # We call internal function by making an HTTP request to /fetch_image, but since this is the same service
        # just reuse the python 'requests' you already have below in your file.
        import requests as _req
        fetch = _req.get(
            f"{request.base_url}fetch_image",
            params={"url": target_url, "debug": "0"},
            headers={"Accept": "image/*"},
            timeout=20,
        )
        if not fetch.ok:
            return _bad_client("Could not download target_url.")
        with open(target_path, "wb") as f:
            f.write(fetch.content)

    # Parse UI selection payloads (your helpers already exist)
    def _parse_bbox(s: str):
        try:
            arr = json.loads(s) if (s or "").strip().startswith("[") else []
            return tuple(int(v) for v in arr) if arr else None
        except Exception:
            return None

    fe_bboxes_ltrb = []
    try:
        arr = json.loads(selected_face_bboxes) if (selected_face_bboxes or "").strip().startswith("[") else []
        fe_bboxes_ltrb = [(bb[3], bb[0], bb[1], bb[2]) for bb in arr] if arr else []
    except Exception:
        fe_bboxes_ltrb = []

    # Do the actual work under a small concurrency gate
    try:
        async with SEM:
            # 1) Crop chosen selfie face (keeps memory lean)
            temp_source = crop_source_face_to_temp(
                source_path=source_path,
                chosen_index=int(selected_source_face_index),
                margin=0.6
            )

            # 2) Detect target faces (for bbox mapping) using your existing util
            canvas = Image.open(target_path).convert("RGBA")
            detected = get_roop_faces(np.array(canvas.convert("RGB")))
            if not detected:
                return _bad_client("No faces in target.")

            idx2bbox_ltrb = {int(d["index"]): tuple(int(v) for v in d["bbox"]) for d in detected}

            # Which faces to act on? Keep your current logic: if multiple chosen via chips, iterate them, else single.
            try:
                arr = json.loads(selected_target_face_indices) if (selected_target_face_indices or "").strip().startswith("[") else []
                target_indices = [int(x) for x in arr] if arr else [int(selected_face_index)]
            except Exception:
                target_indices = [int(selected_face_index)]

            # Composite
            for i, tgt_idx in enumerate(target_indices):
                intended_bbox = fe_bboxes_ltrb[i] if i < len(fe_bboxes_ltrb) and fe_bboxes_ltrb[i] else idx2bbox_ltrb.get(int(tgt_idx))
                if not intended_bbox:
                    continue

                # Map to roop-face-index
                roop_idx = map_bbox_to_roop_index(target_path, intended_bbox)
                # 2.1 swap (your wrapper already normalizes outputs)
                tmp_out = f"{OUTPUT_DIR}/{job_id}_swap_{i}.jpg"
                swap_out = swap_faces(
                    source_path=temp_source,
                    target_path=target_path,
                    output_path=tmp_out,
                    selected_face_index=roop_idx,
                    selected_face_bbox=None
                )

                # paste back (and upscale patch for nicer quality)
                l, t, r, b = intended_bbox
                nl, nt, nr, nb = max(0, l), max(0, t), min(canvas.width, r), min(canvas.height, b)
                target_w, target_h = (nr - nl), (nb - nt)

                # upscale full swapped result then crop patch (simple + fast; keeps code stable)
                upscaled_path = upscale_image(swap_out)
                up = Image.open(upscaled_path).convert("RGBA")
                if up.size != canvas.size:
                    up = up.resize(canvas.size, Image.LANCZOS)
                patch = up.crop((nl, nt, nr, nb)).convert("RGBA")

                region = canvas.crop((nl, nt, nr, nb)).convert("RGBA")
                if region.size != patch.size:
                    patch = patch.resize(region.size, Image.LANCZOS)
                blended = Image.alpha_composite(region, patch)
                canvas.paste(blended, (nl, nt))

            final_path = f"{OUTPUT_DIR}/{job_id}_multi_swap_final.jpg"
            canvas.convert("RGB").save(final_path, format="JPEG", quality=95, subsampling=0)

    except Exception as e:
        # Mark job failed and return 5xx (internal error)
        return _job_fail(job_id, str(e), http=500)

    # Success: freeze API shape
    job = _job_ok(job_id, download_url=f"/static/output/{os.path.basename(final_path)}")
    return JSONResponse({"success": True, **job})




@app.post("/detect_faces_roop")
async def detect_faces_roop(image: UploadFile = File(...)):
    contents = await image.read()
    img_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    results = []
    for face in get_roop_faces(img_np):
        l, t, r, b = face["bbox"]  # LTRB
        pil_img = Image.fromarray(face["face_img"])
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        thumb = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append({
            "index": int(face["index"]),
            # Only for frontend compatibility:
            "coords": [int(t), int(r), int(b), int(l)],  # TRBL for display only
            "bbox": [int(l), int(t), int(r), int(b)],    # LTRB (new field, if you want to use it)
            "thumbnail": f"data:image/jpeg;base64,{thumb}"
        })

    return JSONResponse({"faces": results})


@app.post("/detect_faces_roop_url")
async def detect_faces_roop_url(url: str = Form(...)):
    try:
        r = requests.get(
            url,
            timeout=20,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://faceswap.tilda.ws/",
                "Accept": "image/jpeg,image/png;q=0.9,image/*;q=0.5,*/*;q=0.1",
            },
        )
        r.raise_for_status()
        img_np = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not fetch/decode image: {e}"})

    results = []
    for face in get_roop_faces(img_np):
        l, t, r, b = face["bbox"]  # LTRB
        pil_img = Image.fromarray(face["face_img"])
        buf = io.BytesIO(); pil_img.save(buf, format="JPEG")
        thumb = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append({
            "index": int(face["index"]),
            "coords": [int(t), int(r), int(b), int(l)],  # TRBL for FE
            "bbox":   [int(l), int(t), int(r), int(b)],  # LTRB
            "thumbnail": f"data:image/jpeg;base64,{thumb}",
        })
    return JSONResponse({"faces": results})


@app.post("/upscale")
async def upscale_only_api(request: Request, image: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "processing", "download_url": None, "error": None, "ts": time.time()}

    if not image:
        return _bad_client("Missing 'image' file.")
    if not _mime_ok(image.content_type):
        return _bad_client(f"Unsupported mime {image.content_type}.")

    raw = await image.read()
    if not raw:
        return _bad_client("Empty image.")
    if len(raw) > MAX_IMAGE_BYTES:
        return _bad_client("Image too large.")

    OUTPUT_DIR = "static/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        async with SEM:
            inp = f"{OUTPUT_DIR}/{job_id}_upscale_input.png"
            with open(inp, "wb") as f:
                f.write(raw)
            out = upscale_image(inp)
    except Exception as e:
        return _job_fail(job_id, str(e), http=500)

    job = _job_ok(job_id, download_url=f"/static/output/{os.path.basename(out)}")
    return JSONResponse({"success": True, **job})

    
@app.get("/ping")
def ping():
    return {"ok": True}
    
@app.get("/healthz")
def healthz():
    # Lightweight; do not load heavy models here.
    return {"ok": True}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"job_id": job_id, "status": "failed", "error": "unknown job_id"})
    # Shape: {job_id, status, download_url?, error?}
    return JSONResponse(job)

