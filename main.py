from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

from swap import swap_faces
from upscale import upscale_image
from face_utils import extract_face_region, paste_upscaled_face

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ or set to ["https://faceswap.tilda.ws"]
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
):
    job_id = str(uuid.uuid4())
    source_path = f"{OUTPUT_DIR}/{job_id}_source.jpg"
    target_path = f"{OUTPUT_DIR}/{job_id}_target.jpg"
    swapped_path = f"{OUTPUT_DIR}/{job_id}_swapped.png"

    with open(source_path, "wb") as f:
        f.write(await source.read())
    with open(target_path, "wb") as f:
        f.write(await target.read())

    # Step 1: Run Roop face swap
    swap_faces(source_path, target_path, swapped_path)
    print("✅ FaceSwap done:", swapped_path)

    # Step 2: Extract face from swapped result
    face_img, coords = extract_face_region(swapped_path)
    cropped_face_path = f"{OUTPUT_DIR}/{job_id}_face_crop.png"
    face_img.save(cropped_face_path)

    # Step 3: Upscale the cropped face
    upscaled_face_path = upscale_image(cropped_face_path)

    # Step 4: Paste it back into full image
    final_output_path = f"{OUTPUT_DIR}/{job_id}_final.png"
    paste_upscaled_face(swapped_path, upscaled_face_path, coords, final_output_path)

    print("✅ Final image with upscaled face:", final_output_path)

    return JSONResponse(content={
        "success": True,
        "download_url": f"/static/output/{os.path.basename(final_output_path)}"
    })

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

