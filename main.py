from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
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
from fastapi.responses import Response
import cv2

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
    try:
        # === Save uploaded images ===
        source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
        target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
        roop_output_path = f"{OUTPUT_DIR}/{job_id}_roop_swapped.png"
        final_path = f"{OUTPUT_DIR}/{job_id}_clean_swap_final.jpg"

        with open(source_path, "wb") as f:
            f.write(await source.read())
        with open(target_path, "wb") as f:
            f.write(await target.read())

        # === Validate source face exists ===
        img = face_recognition.load_image_file(source_path)
        if not face_recognition.face_locations(img):
            return JSONResponse(status_code=400, content={"error": "No face in source."})

        # === Run Roop Face Swap ===
        print("[DEBUG] Running Roop...")
        swap_faces(source_path, target_path, roop_output_path, selected_face_index)
        print("[INFO] Roop face swap succeeded.")

        # === Load Roop output and extract face region ===
        swapped_img = Image.open(roop_output_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGBA")

        # === Detect face in Roop output ===
        faces = get_roop_faces(np.array(swapped_img))
        if selected_face_index >= len(faces):
            return JSONResponse(status_code=400, content={"error": "Invalid selected_face_index."})

        face_data = faces[selected_face_index]
        top, right, bottom, left = face_data["coords"]
        cropped_face = swapped_img.crop((left, top, right, bottom)).convert("RGB")

        # === Generate SegFormer mask ===
        face_for_masking = cropped_face.resize((256, 256), Image.LANCZOS)
        # Resize before parsing for better lip/eye/nose detection
        face_for_masking = cropped_face.resize((256, 256), Image.LANCZOS)
        inputs = processor(images=face_for_masking, return_tensors="pt").to(segformer.device)
        outputs = segformer(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=cropped_face.size[::-1], mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()

        FACE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16,]  # face parts
        mask_array = np.isin(labels, FACE_LABELS).astype(np.uint8) * 255
        
        # === Optional: Expand + smooth the mask ===
        kernel = np.ones((10, 10), np.uint8)
        mask_array_dilated = cv2.dilate(mask_array, kernel, iterations=3)
        mask_img = Image.fromarray(mask_array_dilated, mode="L").filter(ImageFilter.GaussianBlur(radius=3))

        # === Apply mask to cropped face ===
        cropped_rgba = cropped_face.convert("RGBA")
        cropped_rgba.putalpha(mask_img)

        # === Save cropped face to temp file
        temp_face_path = f"/tmp/cropped_face_{uuid.uuid4().hex}.png"
        cropped_rgba.save(temp_face_path)

        # === Upscale the face only
        upscaled_face_path = upscale_image(temp_face_path)
        upscaled_face = Image.open(upscaled_face_path).convert("RGBA")

        # === Resize back to original bounding box size
        orig_width, orig_height = right - left, bottom - top
        resized_face = upscaled_face.resize((orig_width, orig_height), Image.LANCZOS)

        # === Paste masked face onto original target ===
        target_img.paste(resized_face, (left, top), resized_face)
        target_img.convert("RGB").save(final_path)

        print(f"[✅] Final clean swap saved at: {final_path}")
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

