from fastapi import FastAPI, UploadFile, File, Form, Request
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

from swap import swap_faces
from upscale import upscale_image
from face_utils import mask_and_extract_face, paste_masked_face, create_masked_target, get_roop_faces

app = FastAPI()

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
    selected_face_index: int = Form(0),  # default to 0 for safety
):
    job_id = str(uuid.uuid4())
    print(f"\n[DEBUG] /swap called with job_id={job_id}")

    try:
        print(f"[DEBUG] Request form fields: {await request.form()}")
        print(f"[DEBUG] Source filename: {source.filename}")
        print(f"[DEBUG] Target filename: {target.filename}")
        print(f"[DEBUG] Selected face index: {selected_face_index}")
    except Exception as e:
        print(f"[ERROR] Reading request form failed: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid form data"})

    try:
        source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
        target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
        swapped_face_path = f"{OUTPUT_DIR}/{job_id}_swapped_face.png"

        # Save files
        with open(source_path, "wb") as f:
            f.write(await source.read())
        with open(target_path, "wb") as f:
            f.write(await target.read())

        # Check face in source
        img = face_recognition.load_image_file(source_path)
        face_locs = face_recognition.face_locations(img)
        print(f"[DEBUG] Source face count: {len(face_locs)}")
        if not face_locs:
            return JSONResponse(status_code=400, content={"error": "No face in source."})

        # Swap
        swap_faces(source_path, target_path, swapped_face_path, selected_face_index)

        return JSONResponse(content={
            "success": True,
            "download_url": f"/static/output/{os.path.basename(swapped_face_path)}"
        })

    except Exception as e:
        print(f"[ERROR] Swap failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/detect_faces")
async def detect_faces_api(image: UploadFile = File(...)):
    contents = await image.read()
    img = face_recognition.load_image_file(io.BytesIO(contents))
    face_locations = face_recognition.face_locations(img)

    if not face_locations:
        return JSONResponse(content={"faces": []})

    results = []
    for idx, (top, right, bottom, left) in enumerate(face_locations):
        if bottom <= top or right <= left:
            continue
        face_img = img[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        results.append({
            "index": idx,
            "thumbnail": f"data:image/jpeg;base64,{encoded}",
            "coords": [top, right, bottom, left]
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

