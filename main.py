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

OUTPUT_DIR = "static/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/swap")
async def swap_faces_api(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    selected_source_face_index: int = Form(0),
    selected_target_face_indices: str = Form("[]"),
    selected_face_bboxes: str = Form("[]"),   # TRBL from FE
):
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap job_id={job_id}")
    print(f"[DEBUG] raw selected_target_face_indices: {selected_target_face_indices!r}")

    # use local parser; remove parse_indices from your imports
    def parse_indices(raw: str):
        if not raw:
            return []
        try:
            s = raw.strip()
            return json.loads(s) if s.startswith("[") else [int(x) for x in s.split(",") if x.strip()]
        except Exception as e:
            print("[WARN] parse_indices failed:", e)
            return []

    target_indices = parse_indices(selected_target_face_indices) or [0]
    print(f"[INFO] Will swap target indices: {target_indices}")

    # Parse FE bboxes (TRBL list aligned to indices), convert to LTRB
    def parse_bboxes(raw: str):
        try:
            arr = json.loads(raw) if raw and raw.strip().startswith('[') else []
            # each arr[i] is [t, r, b, l] -> (l, t, r, b)
            return [(bb[3], bb[0], bb[1], bb[2]) for bb in arr]
        except Exception:
            return []

    fe_bboxes_ltrb = parse_bboxes(selected_face_bboxes)  # may be []

    print(f"[DEBUG] FE bboxes (LTRB): {fe_bboxes_ltrb}")
    print(f"[DEBUG] target_indices: {target_indices}")

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

        # 3) Detect faces on ORIGINAL target (fallback + later matching)
        target_np = np.array(Image.open(target_path).convert("RGB"))
        detected = get_roop_faces(target_np)  # { index, bbox (LTRB), face_img }
        if not detected:
            return JSONResponse(status_code=400, content={"error": "No faces in target."})
        print(f"[INFO] Detected {len(detected)} faces in target.")
        idx2bbox_ltrb = {int(d["index"]): tuple(int(v) for v in d["bbox"]) for d in detected}

        # 4) Crop chosen selfie face
        temp_source = crop_source_face_to_temp(
            source_path=source_path,
            chosen_index=int(selected_source_face_index),
            margin=0.35
        )

        # 5) Composite canvas
        canvas = Image.open(target_path).convert("RGBA")

        # 6) For each selected face: choose intended bbox, map to Roop index, swap, upscale, paste
        # build once, before the loop
        idx2bbox_ltrb = {int(d["index"]): tuple(int(v) for v in d["bbox"]) for d in detected}

        for i, tgt_idx in enumerate(target_indices):
            # 1) choose intended bbox (prefer FE by position)
            if i < len(fe_bboxes_ltrb) and fe_bboxes_ltrb[i]:
                intended_bbox = tuple(map(int, fe_bboxes_ltrb[i]))
            else:
                intended_bbox = idx2bbox_ltrb.get(int(tgt_idx))
            if intended_bbox is None:
                print(f"[WARN] No bbox for UI idx {tgt_idx}; skipping.")
                continue

            print(f"[DEBUG] intended_bbox(LTRB)={intended_bbox}")

            # 2) map bbox -> roop index
            roop_idx = map_bbox_to_roop_index(target_path, intended_bbox)
            print(f"[DEBUG] resolved roop_idx={roop_idx} for UI idx {tgt_idx}")
            if roop_idx is None:
                print(f"[WARN] No match for intended bbox {intended_bbox}; skipping.")
                continue

            # 3) swap
            roop_out = f"{OUTPUT_DIR}/{job_id}_roop_{i}.png"
            swap_faces(
                source_path=temp_source,
                target_path=target_path,
                output_path=roop_out,
                selected_face_index=roop_idx
            )

            # 4) find same region on swapped image (LTRB everywhere)
            swapped_img = Image.open(roop_out).convert("RGBA")
            faces_now = get_roop_faces(np.array(swapped_img.convert("RGB")))
            matched = match_face_by_iou(intended_bbox, faces_now) if faces_now else None
            if not matched:
                print(f"[WARN] Could not match swapped face for idx {tgt_idx}; skipping.")
                continue
            print(f"[LOOP] matched_bbox={matched['bbox']} iou={bbox_iou(intended_bbox, matched['bbox']):.3f}")

            # 1) crop region (add a little margin if you like)
            nl, nt, nr, nb = expand_bbox_ltrb(
                matched["bbox"], margin=0.25,
                img_w=swapped_img.width, img_h=swapped_img.height
            )
            if nl >= nr or nt >= nb:
                print(f"[WARN] Invalid crop for idx {tgt_idx}; skipping.")
                continue

            target_w, target_h = nr - nl, nb - nt
            print(f"[LOOP] crop_box={(nl, nt, nr, nb)} size={(target_w, target_h)}")

            # 2) save crop and upscale
            tmp = f"/tmp/{uuid.uuid4().hex}.png"
            swapped_img.crop((nl, nt, nr, nb)).convert("RGBA").save(tmp)
            upscaled_path = upscale_image(tmp)

            # 3) ensure mode/size, then blend OVER the canvas patch
            up = Image.open(upscaled_path).convert("RGBA")
            if up.size != (target_w, target_h):
                up = up.resize((target_w, target_h), Image.LANCZOS)
            print(f"[DEBUG] upscaled size={up.size}, region={(target_w, target_h)}")

            region = canvas.crop((nl, nt, nr, nb)).convert("RGBA")
            if region.size != up.size:
                # ultra safety (shouldn’t happen after the resize above)
                up = up.resize(region.size, Image.LANCZOS)

            blended = Image.alpha_composite(region, up)  # sizes must match
            canvas.paste(blended, (nl, nt))              # no mask -> no mismatch


        # 7) Save final
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
