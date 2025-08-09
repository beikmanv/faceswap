from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import face_recognition
from PIL import Image, UnidentifiedImageError, ImageFilter
import io
import base64
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from swap import swap_faces
from upscale import upscale_image
from face_utils import mask_and_extract_face, paste_masked_face, create_masked_target, get_roop_faces, crop_source_face_to_temp, _parse_indices
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# Load face parsing model
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
segformer = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
segformer.to("cpu")  # or "cuda" if available

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
    selected_face_index: int = Form(0),                 # backward-compat (single target face)
    selected_source_face_index: int = Form(0),          # selfie face to use
    selected_target_face_indices: str = Form("[]"),     # JSON or CSV list, e.g. "[0,2]" or "0,2"
):
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap called with job_id={job_id}")
    try:
        # === Save uploaded images ===
        source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
        target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
        with open(source_path, "wb") as f:
            f.write(await source.read())
        with open(target_path, "wb") as f:
            f.write(await target.read())

        # === Validate source face exists ===
        img = face_recognition.load_image_file(source_path)
        if not face_recognition.face_locations(img):
            return JSONResponse(status_code=400, content={"error": "No face in source."})

        # === Parse target indices (fallback to old single index) ===
        target_indices = _parse_indices(selected_target_face_indices, default=None)
        if not target_indices:
            target_indices = [int(selected_face_index)]

        # === Make a temp source that contains ONLY the selected selfie face ===
        temp_source_path = crop_source_face_to_temp(
            source_path=source_path,
            chosen_index=int(selected_source_face_index),
            margin=0.35,  # tweak 0.25–0.5
        )

        # === Accumulating canvas (start from original target) ===
        target_base_img = Image.open(target_path).convert("RGBA")

        # === Process each requested target face ===
        for i, tgt_idx in enumerate(target_indices):
            roop_output_path = f"{OUTPUT_DIR}/{job_id}_roop_swapped_{i}.png"

            # 1) Run Roop for THIS target face index using the cropped source face
            print(f"[DEBUG] Running Roop for target face index {tgt_idx} ...")
            # IMPORTANT: your swap_faces wrapper must NOT pass --source-face-index
            swap_faces(
                source_path=temp_source_path,
                target_path=target_path,      # always swap against the original target
                output_path=roop_output_path,
                target_face_index=int(tgt_idx)
            )

            # 2) Load Roop output and locate the same face index region
            swapped_img = Image.open(roop_output_path).convert("RGB")
            faces = get_roop_faces(np.array(swapped_img))
            if int(tgt_idx) >= len(faces):
                print(f"[WARN] Target face index {tgt_idx} not found in Roop output; skipping.")
                continue

            face_data = faces[int(tgt_idx)]
            top, right, bottom, left = face_data["coords"]
            cropped_face = swapped_img.crop((left, top, right, bottom)).convert("RGB")

            # 3) SegFormer face mask (optionally include hair/neck)
            face_for_masking = cropped_face.resize((256, 256), Image.LANCZOS)
            inputs = processor(images=face_for_masking, return_tensors="pt").to(segformer.device)
            outputs = segformer(**inputs)
            logits = outputs.logits
            logits = F.interpolate(logits, size=cropped_face.size[::-1], mode="bilinear", align_corners=False)
            labels = logits.argmax(dim=1)[0].cpu().numpy()

            # skin, brows, eyes, nose, lips, hair, neck (tweak as you prefer)
            FACE_LABELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,17]
            mask_array = np.isin(labels, FACE_LABELS).astype(np.uint8) * 255

            # 4) Grow the mask a bit (optional)
            kernel = np.ones((10, 10), np.uint8)
            mask_dilated = cv2.dilate(mask_array, kernel, iterations=2)

            # 5) Alpha feathering via distance transform (smoother than blur)
            dist = cv2.distanceTransform(mask_dilated, cv2.DIST_L2, 5)
            dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
            FEATHER_RADIUS = 20.0
            alpha = np.clip(dist_norm / (FEATHER_RADIUS / 255.0), 0, 1)
            alpha_mask = (alpha * 255).astype(np.uint8)

            mask_img = Image.fromarray(alpha_mask, mode="L")
            cropped_rgba = cropped_face.convert("RGBA")
            cropped_rgba.putalpha(mask_img)

            # 6) Upscale the masked face only
            temp_face_path = f"/tmp/cropped_face_{uuid.uuid4().hex}.png"
            cropped_rgba.save(temp_face_path)
            upscaled_face_path = upscale_image(temp_face_path)
            upscaled_face = Image.open(upscaled_face_path).convert("RGBA")

            # 7) Resize to original bbox and paste with alpha onto the accumulating canvas
            width, height = right - left, bottom - top
            resized_face = upscaled_face.resize((width, height), Image.LANCZOS)
            target_base_img.paste(resized_face, (left, top), resized_face)

        # === Save final composited image ===
        final_path = f"{OUTPUT_DIR}/{job_id}_clean_swap_final.jpg"
        target_base_img.convert("RGB").save(final_path)

        print(f"[✅] Final clean multi-swap saved at: {final_path}")
        return JSONResponse(content={
            "success": True,
            "download_url": f"/static/output/{os.path.basename(final_path)}"
        })

    except Exception as e:
        print(f"[❌ ERROR] Swap failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Swap failed: {str(e)}"})


@app.post("/detect_faces_roop")
async def detect_faces_roop(image: UploadFile = File(...)):
    contents = await image.read()
    img_np = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    results = []
    for face in get_roop_faces(img_np):
        pil_img = Image.fromarray(face["face_img"])
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        thumbnail = base64.b64encode(buffer.getvalue()).decode("utf-8")
        results.append({
            "index": int(face["index"]),
            "coords": [int(c) for c in face["coords"]],
            "thumbnail": f"data:image/jpeg;base64,{thumbnail}"
        })

    return JSONResponse(content={"faces": results})


@app.post("/upscale")
async def upscale_only_api(request: Request, image: UploadFile = File(...)):
    try:
        job_id = str(uuid.uuid4())
        print("[DEBUG] Upscale request received. Job ID:", job_id)

        input_path = f"{OUTPUT_DIR}/{job_id}_upscale_input.png"
        with open(input_path, "wb") as f:
            f.write(await image.read())
        print("[DEBUG] Image saved to:", input_path)

        output_path = upscale_image(input_path)
        print("[DEBUG] Upscaled image saved to:", output_path)

        full_url = request.base_url._url.rstrip("/") + f"/static/output/{os.path.basename(output_path)}"
        print("✅ Upscale done:", full_url)

        return JSONResponse(content={
            "success": True,
            "download_url": f"/static/output/{os.path.basename(output_path)}"
        })

    except Exception as e:
        print("[ERROR] Upscale failed:", str(e))
        return JSONResponse(status_code=500, content={"error": f"Upscale failed: {str(e)}"})

