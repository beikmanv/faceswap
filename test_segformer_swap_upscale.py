import os
import uuid
import subprocess
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from swap import swap_faces  # Your existing function

# === CONFIG
ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
REAL_ESRGAN_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/Real-ESRGAN/inference_realesrgan.py"

# === Load SegFormer
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
segformer = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
segformer.to("cpu")  # Use "cuda" if available

# === Input paths
source_path = "static/output/source.jpg"
target_path = "static/output/target.jpg"
swapped_path = "static/output/swapped.png"
final_path = "static/output/final_upscaled.png"


# === Perform face swap
swap_faces(source_path, target_path, swapped_path, selected_face_index=1)
print("✅ Face swapped.")

# === Load swapped image
swapped_img = Image.open(swapped_path).convert("RGB")

# === SegFormer: get face mask
inputs = processor(images=swapped_img, return_tensors="pt").to(segformer.device)
outputs = segformer(**inputs)
logits = outputs.logits
logits = F.interpolate(logits, size=swapped_img.size[::-1], mode="bilinear", align_corners=False)
labels = logits.argmax(dim=1)[0].cpu().numpy()

# === Build binary mask (face parts)
FACE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
mask_array = np.isin(labels, FACE_LABELS).astype(np.uint8) * 255
mask = Image.fromarray(mask_array, mode="L")

# === Get bounding box
bbox = mask.getbbox()
if not bbox:
    raise RuntimeError("No face mask detected.")
left, top, right, bottom = bbox

# === Crop and apply alpha mask
cropped_face = swapped_img.crop(bbox).convert("RGBA")
cropped_mask = mask.crop(bbox)
cropped_face.putalpha(cropped_mask)

# === Save cropped face
tmp_id = uuid.uuid4().hex
rgb_path = f"/tmp/rgb_{tmp_id}.png"
cropped_face.save(rgb_path)
print("✅ Cropped face saved.")

# === Upscale face with Real-ESRGAN
def upscale_image(image_path: str) -> str:
    from glob import glob
    import time

    resized_input = image_path
    out_dir = os.path.dirname(resized_input)

    # Expected name (may not be created!)
    expected_output = image_path.replace(".png", "_upscaled.png")

    # Run upscaler
    cmd = [
        ROOP_PYTHON,
        REAL_ESRGAN_SCRIPT,
        "-n", "RealESRGAN_x2plus",
        "-i", resized_input,
        "-o", out_dir,
        "--fp32",
        "--tile", "512",
        "--outscale", "4",
        "--face_enhance",
        "--suffix", "_upscaled"
    ]

    print("[INFO] Running upscaler...")
    subprocess.run(cmd, check=True)

    # If expected file doesn't exist, try fallback match
    if os.path.exists(expected_output):
        return expected_output

    print("[WARNING] Expected output not found:", expected_output)
    print("[INFO] Scanning directory for fallback match...")

    basename = os.path.basename(resized_input).replace(".png", "")
    candidates = glob(f"{out_dir}/{basename}*upscaled*.png")

    if candidates:
        print("[INFO] Found fallback match:", candidates[0])
        return candidates[0]

    raise FileNotFoundError(f"[ERROR] Upscaled file not found: {expected_output}")


upscaled_path = upscale_image(rgb_path)
print("✅ Upscaled image saved:", upscaled_path)

# === Paste back to swapped image
upscaled_face = Image.open(upscaled_path).convert("RGBA")
paste_x = left + (right - left)//2 - upscaled_face.width//2
paste_y = top + (bottom - top)//2 - upscaled_face.height//2

result_img = Image.open(swapped_path).convert("RGBA")
result_img.paste(upscaled_face, (paste_x, paste_y), upscaled_face)
result_img.convert("RGB").save(final_path)
print("🎉 Final image saved at:", final_path)
