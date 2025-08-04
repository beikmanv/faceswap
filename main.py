from fastapi import FastAPI, UploadFile, File, Form
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
from face_utils import mask_and_extract_face, paste_masked_face, create_masked_target

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
    selected_face_coords: str = Form(...),
):
    job_id = str(uuid.uuid4())
    source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
    target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
    cropped_face_path = f"{OUTPUT_DIR}/{job_id}_target_face_crop.jpg"
    swapped_face_path = f"{OUTPUT_DIR}/{job_id}_swapped_face.png"
    swapped_face_masked_out = f"{OUTPUT_DIR}/{job_id}_swapped_face_out.png"
    final_output_path = f"{OUTPUT_DIR}/{job_id}_final.png"

    # Save input files
    with open(source_path, "wb") as f:
        f.write(await source.read())
    with open(target_path, "wb") as f:
        f.write(await target.read())

    try:
        Image.open(source_path).verify()
        Image.open(target_path).verify()
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded."})

    try:
        selected_coords = tuple(json.loads(selected_face_coords))  # [top, right, bottom, left]
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid face coordinates format."})

    # Mask everything except the selected face
    try:
        masked_img, _ = create_masked_target(target_path, selected_coords)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    masked_img.save(cropped_face_path)

    # Run face swap
    swap_faces(source_path, cropped_face_path, swapped_face_path)

    # Extract the swapped face using facial landmarks mask
    try:
        masked_face_img, face_bbox = mask_and_extract_face(swapped_face_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"❌ Face masking failed: {str(e)}"})

    masked_face_img.save(swapped_face_masked_out)

    # Upscale the masked face
    upscaled_face_path = upscale_image(swapped_face_masked_out)
    upscaled_face_img = Image.open(upscaled_face_path).convert("RGBA")

    # Paste final upscaled face on original target
    paste_masked_face(target_path, upscaled_face_img, face_bbox, final_output_path)

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
