import subprocess
import os
from PIL import Image

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
REAL_ESRGAN_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/Real-ESRGAN/inference_realesrgan.py"


def resize_if_needed(image_path: str, max_size: int = 1600) -> str:
    img = Image.open(image_path)
    width, height = img.size

    if max(width, height) <= max_size:
        return image_path  # ✅ No resizing needed

    # 🧮 Resize while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    resized_path = image_path.replace(".png", "_resized.png")
    resized_img.save(resized_path)

    print(f"[INFO] Resized image from ({width}x{height}) to ({new_width}x{new_height}), saved as {resized_path}")
    return resized_path


def upscale_image(image_path: str) -> str:
    # ✅ Resize first
    resized_input = resize_if_needed(image_path)

    # 🧠 Output file path
    out_path = resized_input.replace(".png", "_out.png")

    # 💡 You can dynamically set tile_size later
    tile_size = 512

    cmd = [
        ROOP_PYTHON,
        REAL_ESRGAN_SCRIPT,
        "-n", "RealESRGAN_x2plus",
        "-i", resized_input,
        "-o", os.path.dirname(resized_input),
        "--fp32",
        "--tile", str(tile_size),
        "--outscale", "4",
        "--face_enhance"
    ]

    print(f"[INFO] Running upscaler with tile size: {tile_size}")
    subprocess.run(cmd, check=True)

    return out_path
