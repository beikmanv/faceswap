from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import face_recognition
from PIL import Image, UnidentifiedImageError
import io
import base64
import json
import numpy as np
from fastapi.responses import Response

from swap import swap_faces
from upscale import upscale_image
from face_utils import mask_and_extract_face, paste_masked_face, create_masked_target, get_roop_faces

import torch
import torch.nn.functional as F
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
    selected_face_index: int = Form(0),
):
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap called with job_id={job_id}")
    print(f"[DEBUG] Received source file: {source.filename}")
    print(f"[DEBUG] Received target file: {target.filename}")
    print(f"[DEBUG] selected_face_index: {selected_face_index}")

    try:
        # === Save uploaded files ===
        source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
        target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
        swapped_path = f"{OUTPUT_DIR}/{job_id}_swapped.png"
        final_path = f"{OUTPUT_DIR}/{job_id}_swapped_upscaled.png"

        with open(source_path, "wb") as f:
            f.write(await source.read())
        print(f"[DEBUG] Saved source to: {source_path}")

        with open(target_path, "wb") as f:
            f.write(await target.read())
        print(f"[DEBUG] Saved target to: {target_path}")

        # === Validate source face ===
        print(f"[DEBUG] Checking face in source...")
        img = face_recognition.load_image_file(source_path)
        if not face_recognition.face_locations(img):
            print("[ERROR] No face found in source.")
            return JSONResponse(status_code=400, content={"error": "No face in source."})

        # === Face swap ===
        print("[DEBUG] Performing face swap...")
        swap_faces(source_path, target_path, swapped_path, selected_face_index)
        print("[INFO] Roop face swap succeeded.")

        # === Extract face from swapped image ===
        swapped_img_np = np.array(Image.open(swapped_path).convert("RGB"))
        faces = get_roop_faces(swapped_img_np)

        if selected_face_index >= len(faces):
            return JSONResponse(status_code=400, content={"error": "Invalid selected_face_index."})

        face_data = faces[selected_face_index]
        top, right, bottom, left = face_data["coords"]
        cropped_face_np = face_data["face_img"]
        cropped_face = Image.fromarray(cropped_face_np).convert("RGB")
        temp_cropped_path = f"/tmp/cropped_{uuid.uuid4().hex}.png"

        # === Apply SegFormer face mask ===
        inputs = processor(images=cropped_face, return_tensors="pt").to(segformer.device)
        outputs = segformer(**inputs)
        logits = outputs.logits  # shape: (1, 19, 128, 128)
        logits = F.interpolate(
            logits, size=cropped_face.size[::-1], mode="bilinear", align_corners=False
        )
        labels = logits.argmax(dim=1)[0].cpu().numpy()

        FACE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # skin, brows, eyes, nose, lips, etc.
        mask_array = np.isin(labels, FACE_LABELS).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_array, mode="L")

        cropped_face.putalpha(mask_img)
        cropped_face.save(temp_cropped_path)
        masked_face = cropped_face

        print(f"[DEBUG] Masked + cropped face saved at: {temp_cropped_path}")

        # === Upscaling ===
        print("[DEBUG] Upscaling...")
        upscaled_path = upscale_image(temp_cropped_path)
        upscaled_face = Image.open(upscaled_path).convert("RGBA")
        print(f"[DEBUG] Upscaled face image: {upscaled_path}")

        # === Resize + Paste ===
        width, height = right - left, bottom - top
        resized_upscaled_face = upscaled_face.resize((width, height), Image.LANCZOS)
        swapped_img = Image.open(swapped_path).convert("RGBA")
        swapped_img.paste(resized_upscaled_face, (left, top), resized_upscaled_face)
        swapped_img.convert("RGB").save(final_path)

        print(f"[INFO] Final image saved at: {final_path}")

        return JSONResponse(content={
            "success": True,
            "download_url": f"/static/output/{os.path.basename(final_path)}"
        })

    except Exception as e:
        print(f"[ERROR] Swap+Upscale failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Swap+Upscale failed: {str(e)}"})




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

