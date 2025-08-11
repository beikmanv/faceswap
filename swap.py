# swap.py
import subprocess

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
ROOP_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/run.py"

def swap_faces(source_path, target_path, output_path,
              selected_face_index=None, selected_face_bbox=None):
    print("[INFO] Swapping using Roop:")
    print(f"       source: {source_path}")
    print(f"       target: {target_path}")
    print(f"       output: {output_path}")
    print(f"       selected_face_index: {selected_face_index}")
    print(f"       selected_face_bbox:  {selected_face_bbox}")

    cmd = [ROOP_PYTHON, ROOP_SCRIPT,
           "--source", source_path,
           "--target", target_path,
           "--output", output_path]

    if selected_face_bbox is not None:
        # LTRB tuple -> "l,t,r,b"
        cmd += ["--target-face-bbox", ",".join(map(str, selected_face_bbox))]
    elif selected_face_index is not None:
        cmd += ["--target-face-index", str(selected_face_index)]
    else:
        cmd += ["--target-face-index", "0"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Roop failed:\n", result.stderr)
        raise RuntimeError(f"Roop face swap failed:\n{result.stderr}")
    print("[INFO] Roop face swap succeeded.")
