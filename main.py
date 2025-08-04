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

from swap import swap_faces
from upscale import upscale_image
from face_utils import extract_face_region, paste_upscaled_face, extract_face_by_index, crop_exact_region, blur_all_but_selected_face

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
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    selected_face_coords: str = Form(...),  # <-- updated here
):
    job_id = str(uuid.uuid4())
    source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
    target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
    cropped_face_path = f"{OUTPUT_DIR}/{job_id}_target_face_crop.jpg"
    swapped_face_path = f"{OUTPUT_DIR}/{job_id}_swapped_face.png"
    final_output_path = f"{OUTPUT_DIR}/{job_id}_final.png"

    # Save uploads
    with open(source_path, "wb") as f:
        f.write(await source.read())
    with open(target_path, "wb") as f:
        f.write(await target.read())

    # Validate image
    try:
        Image.open(source_path).verify()
        Image.open(target_path).verify()
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded."})

    # Parse coordinates from form field
    try:
        selected_coords = tuple(json.loads(selected_face_coords))  # [top, right, bottom, left]
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid face coordinates format."})

    # Blur all other faces and get padded coords
    try:
        masked_img, padded_coords = blur_all_but_selected_face(target_path, selected_coords)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    masked_img.save(cropped_face_path)  # Used as Roop target input

    # Swap face
    swap_faces(source_path, cropped_face_path, swapped_face_path)

    print("🧠 Padded Coords:", padded_coords)
    print("📏 Image size:", Image.open(swapped_face_path).size)

    # Step 1: Crop swapped face using padded coords
    swapped_face_img = crop_exact_region(swapped_face_path, padded_coords)

    if swapped_face_img.width == 0 or swapped_face_img.height == 0:
        return JSONResponse(status_code=500, content={"error": "❌ Cropped swapped face is empty. Face detection or Roop may have failed."})

    print("🔍 Cropped face size:", swapped_face_img.size)

    # Step 2: Save that face and upscale
    swapped_face_out_path = f"{OUTPUT_DIR}/{job_id}_swapped_face_out.png"
    swapped_face_img.save(swapped_face_out_path)
    upscaled_face_path = upscale_image(swapped_face_out_path)

    # Step 3: Paste upscaled face back to target
    paste_upscaled_face(target_path, upscaled_face_path, padded_coords, final_output_path)

    return JSONResponse(content={
        "success": True,
        "download_url": f"/static/output/{os.path.basename(final_output_path)}"
    })

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

