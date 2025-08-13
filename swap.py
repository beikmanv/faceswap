# swap.py
import subprocess, os

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
ROOP_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/run.py"

def swap_faces(source_path, target_path, output_path,
               selected_face_index=None, selected_face_bbox=None) -> str:
    src = os.path.abspath(source_path)
    tgt = os.path.abspath(target_path)
    out = os.path.abspath(output_path)

    print("[INFO] Swapping using Roop:")
    print(f"       source: {src}")
    print(f"       target: {tgt}")
    print(f"       output: {out}")
    print(f"       selected_face_index: {selected_face_index}")
    print(f"       selected_face_bbox:  {selected_face_bbox}")

    cmd = [ROOP_PYTHON, ROOP_SCRIPT, "--source", src, "--target", tgt, "--output", out]
    if selected_face_bbox is not None:
        cmd += ["--target-face-bbox", ",".join(map(str, selected_face_bbox))]
    elif selected_face_index is not None:
        cmd += ["--target-face-index", str(selected_face_index)]
    else:
        cmd += ["--target-face-index", "0"]

    res = subprocess.run(cmd, cwd=os.path.dirname(ROOP_SCRIPT),
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Roop failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")

    # Normalize output path if Roop changed extension
    if not os.path.exists(out):
        tgt_ext = os.path.splitext(tgt)[1].lower() or ".jpg"
        candidates = [
            os.path.splitext(out)[0] + tgt_ext,
            os.path.splitext(out)[0] + ".jpg",
            os.path.splitext(out)[0] + ".png",
        ]
        for c in candidates:
            if os.path.exists(c):
                out = c
                break
        else:
            raise FileNotFoundError(
                "Roop returned 0 but output not found.\n"
                f"Expected: {os.path.abspath(output_path)}\nTried: {candidates}\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

    print("[INFO] Roop face swap succeeded ->", out)
    return out
