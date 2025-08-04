import subprocess
import os

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
ROOP_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/run.py"

def swap_faces(source_path, target_path, output_path, selected_face_index=0):
    print("[INFO] Swapping using Roop:")
    print(f"       source: {source_path}")
    print(f"       target: {target_path}")
    print(f"       output: {output_path}")
    print(f"       selected_face_index: {selected_face_index}")

    cmd = [
        ROOP_PYTHON, ROOP_SCRIPT,
        "--source", source_path,
        "--target", target_path,
        "--output", output_path,
        "--target-face-index", str(selected_face_index)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Roop failed:")
        print(result.stderr)
        raise RuntimeError(f"Roop face swap failed:\n{result.stderr}")

    print("[INFO] Roop face swap succeeded.")
