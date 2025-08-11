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
    parse_indices
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

OUTPUT_DIR = "static/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/swap")
async def swap_faces_api(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    selected_source_face_index: int = Form(0),
    selected_target_face_indices: str = Form("[]"),  # "[3,17]" or "3,17"
):
 
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap job_id={job_id}")
    print(f"[DEBUG] raw selected_target_face_indices: {selected_target_face_indices!r}")

    def parse_indices(raw: str):
        if not raw:
            return []
        try:
            s = raw.strip()
            return json.loads(s) if s.startswith("[") else [int(x) for x in s.split(",") if x.strip()]
        except Exception as e:
            print("[WARN] parse_indices failed:", e)
            return []

    # Parse once, use everywhere
    face_indices = parse_indices(selected_target_face_indices)
    if not face_indices:
        face_indices = [0]
    print(f"[INFO] Will swap target indices: {face_indices}")


    try:
        # 1) Save uploads
        source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
        target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
        with open(source_path, "wb") as f: f.write(await source.read())
        with open(target_path, "wb") as f: f.write(await target.read())
        print(f"[DEBUG] Saved source -> {source_path}")
        print(f"[DEBUG] Saved target -> {target_path}")

        # 2) Validate source has a face
        if not face_recognition.face_locations(face_recognition.load_image_file(source_path)):
            return JSONResponse(status_code=400, content={"error": "No face in source."})

        # 3) Parse which targets to swap
        target_indices = parse_indices(selected_target_face_indices)
        if not target_indices:
            target_indices = [0]
        print(f"[INFO] Will swap target indices: {target_indices}")

        # 4) Detect faces in the original target
        target_np = np.array(Image.open(target_path).convert("RGB"))
        detected = get_roop_faces(target_np)
        if not detected:
            return JSONResponse(status_code=400, content={"error": "No faces in target."})
        print(f"[INFO] Detected {len(detected)} faces in target.")
        idx2bbox = {int(d["index"]): [int(c) for c in d["coords"]] for d in detected}

        # 5) Crop the chosen selfie face
        temp_source = crop_source_face_to_temp(
            source_path=source_path,
            chosen_index=int(selected_source_face_index),
            margin=0.35
        )

        # 6) Prepare final canvas
        canvas = Image.open(target_path).convert("RGBA")

        for i, tgt_idx in enumerate(target_indices):
            if tgt_idx not in idx2bbox:
                print(f"[WARN] Target index {tgt_idx} not found; skipping.")
                continue

            intended_bbox = idx2bbox[tgt_idx]
            roop_out = f"{OUTPUT_DIR}/{job_id}_roop_{i}.png"
            print(f"[DEBUG] Roop swapping target index {tgt_idx} -> {roop_out}")

            # Run Roop
            swap_faces(
                source_path=temp_source,
                target_path=target_path,
                output_path=roop_out,
                selected_face_index=int(tgt_idx),
            )

            # Match swapped face by IoU
            swapped_img = Image.open(roop_out).convert("RGBA")
            faces_now = get_roop_faces(np.array(swapped_img.convert("RGB")))
            matched = match_face_by_iou(intended_bbox, faces_now) if faces_now else None
            if matched is None:
                print(f"[WARN] Could not match swapped face for idx {tgt_idx}; skipping.")
                continue

            # Crop with margin
            mt, mr, mb, ml = map(int, matched["coords"])
            margin = 0.25
            w, h = mr - ml, mb - mt
            cx, cy = ml + w / 2.0, mt + h / 2.0
            nl = max(0, int(cx - w * (1 + margin) / 2))
            nt = max(0, int(cy - h * (1 + margin) / 2))
            nr = min(swapped_img.width,  int(cx + w * (1 + margin) / 2))
            nb = min(swapped_img.height, int(cy + h * (1 + margin) / 2))

            if nl >= nr or nt >= nb:
                print(f"[WARN] Invalid crop for idx {tgt_idx}; skipping.")
                continue

            # Upscale + paste
            face_crop = swapped_img.crop((nl, nt, nr, nb))
            tmp = f"/tmp/{uuid.uuid4().hex}.png"
            face_crop.save(tmp)
            upscaled_path = upscale_image(tmp)
            upscaled = Image.open(upscaled_path).convert("RGBA")
            final_face = upscaled.resize((nr - nl, nb - nt), Image.LANCZOS)
            canvas.paste(final_face, (nl, nt), final_face)

        # 7) Save final with high quality
        final_path = f"{OUTPUT_DIR}/{job_id}_multi_swap_final.jpg"
        canvas.convert("RGB").save(final_path, format="JPEG", quality=95, subsampling=0)
        print(f"[✅] Multi-face swap saved at: {final_path}")

        return JSONResponse({
            "success": True,
            "download_url": f"/static/output/{os.path.basename(final_path)}"
        })

    except Exception as e:
        print(f"[❌ ERROR] Swap failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/detect_faces_roop")
async def detect_faces_roop(image: UploadFile = File(...)):
    contents = await image.read()
    img_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    results = []
    for face in get_roop_faces(img_np):
        pil_img = Image.fromarray(face["face_img"])
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        thumb = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append({
            "index": int(face["index"]),
            "coords": [int(c) for c in face["coords"]],
            "thumbnail": f"data:image/jpeg;base64,{thumb}"
        })

    return JSONResponse({"faces": results})



@app.post("/upscale")
async def upscale_only_api(request: Request, image: UploadFile = File(...)):
    try:
        job_id = str(uuid.uuid4())
        inp = f"{OUTPUT_DIR}/{job_id}_upscale_input.png"
        with open(inp, "wb") as f:
            f.write(await image.read())
        out = upscale_image(inp)
        return JSONResponse({
            "success": True,
            "download_url": f"/static/output/{os.path.basename(out)}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
