import subprocess, os

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"
ROOP_SCRIPT = "/Users/beikmanv/northcoders/faceswap/roop/run.py"

def swap_faces(source_path, target_path, output_path,
               selected_face_index=None, selected_face_bbox=None) -> str:
    # absolute paths + ensure dir
    src = os.path.abspath(source_path)
    tgt = os.path.abspath(target_path)
    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)

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

    # Roop may change the extension to match target; accept that
    if not os.path.exists(out):
        tgt_ext = os.path.splitext(tgt)[1].lower() or ".jpg"
        alt = os.path.splitext(out)[0] + tgt_ext
        if os.path.exists(alt):
            out = alt
        else:
            raise FileNotFoundError(
                "Roop returned 0 but output not found.\n"
                f"Expected: {out}\nTried: {alt}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

    print("[INFO] Roop face swap succeeded ->", out)
    return out
