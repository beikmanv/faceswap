import os, subprocess, glob
from PIL import Image

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
REAL_ESRGAN_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/Real-ESRGAN/inference_realesrgan.py"

def resize_if_needed(image_path: str, max_size: int = 1600) -> str:
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) <= max_size:
        return image_path
    if w > h:
        nw, nh = max_size, int(max_size / w * h)
    else:
        nh, nw = max_size, int(max_size / h * w)
    img = img.resize((nw, nh), Image.LANCZOS)
    root, ext = os.path.splitext(image_path)
    resized_path = f"{root}_resized{ext}"          # <-- keep original extension
    img.save(resized_path)
    return resized_path

def upscale_image(image_path: str) -> str:
    inp = resize_if_needed(image_path)
    in_dir = os.path.dirname(inp)
    base  = os.path.splitext(os.path.basename(inp))[0]

    cmd = [
        ROOP_PYTHON, REAL_ESRGAN_SCRIPT,
        "-n", "RealESRGAN_x2plus",
        "-i", inp,
        "-o", in_dir,
        "--fp32", "--tile", "512",
        "--outscale", "4", "--face_enhance",
    ]
    subprocess.run(cmd, check=True)

    # Real-ESRGAN writes <base>_out.(png|jpg|webp|…)
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        cand = os.path.join(in_dir, f"{base}_out{ext}")
        if os.path.exists(cand):
            return cand

    # Fallback: grab whatever matches
    matches = sorted(glob.glob(os.path.join(in_dir, f"{base}_out*")))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Upscaler output not found for base '{base}_out*' in {in_dir}")
