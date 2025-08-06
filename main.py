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
    selected_face_bbox: str = Form(None),
):
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap called with job_id={job_id}")
    print(f"[DEBUG] Received source file: {source.filename}")
    print(f"[DEBUG] Received target file: {target.filename}")
    print(f"[DEBUG] selected_face_index: {selected_face_index}")
    print(f"[DEBUG] selected_face_bbox (raw): {selected_face_bbox}")

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

        # === Check and parse bbox ===
        if not selected_face_bbox:
            print("[ERROR] No selected_face_bbox provided.")
            return JSONResponse(status_code=400, content={"error": "No selected_face_bbox provided."})

        try:
            coords = json.loads(selected_face_bbox)
            top, right, bottom, left = coords
            print(f"[DEBUG] Parsed bbox: {coords}")
        except Exception as e:
            print(f"[ERROR] Failed to parse selected_face_bbox: {e}")
            return JSONResponse(status_code=400, content={"error": "Invalid selected_face_bbox format."})

        # === Clamp crop coordinates ===
        swapped_img = Image.open(swapped_path).convert("RGBA")
        img_width, img_height = swapped_img.size
        top = max(0, min(top, img_height))
        bottom = max(0, min(bottom, img_height))
        left = max(0, min(left, img_width))
        right = max(0, min(right, img_width))

        print(f"[DEBUG] Clamped bbox: top={top}, bottom={bottom}, left={left}, right={right}")

        if top >= bottom or left >= right:
            print("[ERROR] Invalid crop dimensions.")
            return JSONResponse(status_code=400, content={"error": "Invalid crop coordinates after clamping."})

        # === Crop and upscale ===
        cropped_face = swapped_img.crop((left, top, right, bottom))
        temp_cropped_path = f"/tmp/cropped_{uuid.uuid4().hex}.png"
        cropped_face.save(temp_cropped_path)
        print(f"[DEBUG] Cropped face saved at: {temp_cropped_path}")

        print("[DEBUG] Upscaling...")
        upscaled_path = upscale_image(temp_cropped_path)
        upscaled_face = Image.open(upscaled_path).convert("RGBA")
        print(f"[DEBUG] Upscaled face image: {upscaled_path}")

        # === Resize and paste ===
        width, height = right - left, bottom - top
        resized_upscaled_face = upscaled_face.resize((width, height), Image.LANCZOS)
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

