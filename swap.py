import subprocess

ROOP_PYTHON = "/Users/beikmanv/northcoders/faceswap/roop/venv/bin/python"

def swap_faces(source_path, target_path, output_path):
    cmd = [
        ROOP_PYTHON,
        "/Users/beikmanv/northcoders/faceswap/roop/run.py",
        "--source", source_path,
        "--target", target_path,
        "--output", output_path
    ]
    subprocess.run(cmd, check=True)
