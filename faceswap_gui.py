import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import threading
import shutil
import os
import webbrowser
from pathlib import Path
from datetime import datetime
import sys

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Setup paths ---
DOWNLOADS_DIR = str(Path.home() / "Downloads")
INPUT_DIR = os.path.join(DOWNLOADS_DIR, "faceswap_input")
OUTPUT_DIR = os.path.join(DOWNLOADS_DIR, "faceswap_output")
SOURCE_IMG = os.path.join(INPUT_DIR, "source.jpg")
TARGET_IMG = os.path.join(INPUT_DIR, "target.jpg")
OUTPUT_IMG = os.path.join(OUTPUT_DIR, f"output_{timestamp}.png")


os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GUI setup ---
root = tk.Tk()
root.title("Faceswap")
root.geometry("1000x1300")

# Variables for images and previews
generated_img_var = tk.StringVar()
selfie_var = tk.StringVar()
preview_target = None
preview_source = None

def open_generator_site(url):
    webbrowser.open(url)

image_tools = {
    "Google ImageFX": "https://labs.google/fx",
    "Craiyon (DALL-E mini)": "https://www.craiyon.com",
    "Bing Image Creator": "https://www.bing.com/images/create",
    "Pixray": "https://pixray.gob.io",
    "Artbreeder": "https://www.artbreeder.com",
    "NightCafe": "https://creator.nightcafe.studio",
    "DeepAI": "https://deepai.org/machine-learning-model/text2img",
    "Playground AI": "https://playgroundai.com",
    "Lexica": "https://lexica.art",
    "Ideaogram": "https://ideogram.ai",
    "Leonardo AI": "https://app.leonardo.ai",
    "Aux Machina": "https://www.auxmachina.com",
    "Krea AI": "https://www.krea.ai",
    "Imagine": "https://imagine.art",
}

tk.Label(root, text="Choose a free AI image generator:").pack(pady=(10, 0))

# Create a frame to hold the grid of buttons
generator_frame = tk.Frame(root)
generator_frame.pack(pady=(5, 10))

# Add buttons in a 2-column grid
for index, (name, link) in enumerate(image_tools.items()):
    row = index // 2
    col = index % 2
    btn = tk.Button(generator_frame, text=name, width=40, command=lambda l=link: open_generator_site(l))
    btn.grid(row=row, column=col, padx=5, pady=5)

def show_preview(image_path, preview_label):
    try:
        img = Image.open(image_path)
        img.thumbnail((120, 120))
        photo = ImageTk.PhotoImage(img)
        preview_label.config(image=photo)
        preview_label.image = photo  # prevent garbage collection
    except Exception as e:
        preview_label.config(image="", text="Preview failed")

def choose_generated_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            shutil.copy(file_path, TARGET_IMG)
            if os.path.getsize(TARGET_IMG) == 0:
                raise ValueError("Target image is empty.")
            generated_img_var.set(file_path)
            print("[INFO] Target image copied to:", TARGET_IMG)
            show_preview(TARGET_IMG, target_preview)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy target image: {e}")

def choose_selfie():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            shutil.copy(file_path, SOURCE_IMG)
            if os.path.getsize(SOURCE_IMG) == 0:
                raise ValueError("Source image is empty.")
            selfie_var.set(file_path)
            print("[INFO] Source image copied to:", SOURCE_IMG)
            show_preview(SOURCE_IMG, source_preview)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy source image: {e}")

def upscale_output_image(input_path):
    upscaled_output_dir = OUTPUT_DIR
    try:
        subprocess.run([
            sys.executable,  # Uses the same Python as the GUI
            os.path.join(os.path.dirname(__file__), "Real-ESRGAN", "inference_realesrgan.py"),
            "-n", "RealESRGAN_x4plus",
            "-i", input_path,
            "-o", upscaled_output_dir,
            "--fp32",
            "--tile", "128"
        ], check=True)
        print("[INFO] Output image upscaled successfully.")
        upscaled_file = os.path.join(upscaled_output_dir, f"{Path(input_path).stem}_out.png")
        messagebox.showinfo("Upscaled", f"Upscaled image saved as:\n{upscaled_file}")
    except Exception as e:
        print("[ERROR] Upscaler failed:", e)
        messagebox.showerror("Upscaling Failed", f"Error: {e}")

def restart_app():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def handle_manual_upscale():
    latest_output = max(
        (f for f in os.listdir(OUTPUT_DIR) if f.startswith("output_") and f.endswith(".png")),
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
        default=None
    )
    if latest_output:
        image_path = os.path.join(OUTPUT_DIR, latest_output)
        upscale_output_image(image_path)
    else:
        messagebox.showwarning("No Output Found", "Run a face swap first before upscaling.")

def run_roop_faceswap():
    def task():
        if not os.path.exists(SOURCE_IMG) or not os.path.exists(TARGET_IMG):
            messagebox.showwarning("Missing file", "Both images must be uploaded first.")
            return

        print("[INFO] Running Roop...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_img = os.path.join(OUTPUT_DIR, f"output_{timestamp}.png")
        cmd = [
            sys.executable,  # 👈 ensures subprocess uses the same Python env as the GUI
            "/Users/beikmanv/northcoders/faceswap/roop/run.py",
            "--source", SOURCE_IMG,
            "--target", TARGET_IMG,
            "--output", output_img
        ]

        try:
            subprocess.run(cmd, check=True)
            print("[INFO] Face swap complete.")

            messagebox.showinfo("Done", f"Face swap complete.\nSaved as:\n{output_img}")
            quit_button.config(state=tk.NORMAL)
        except subprocess.CalledProcessError as e:
            print("[ERROR] Roop failed:", e)
            messagebox.showerror("Error", "Face swap failed. See terminal for details.")

    threading.Thread(target=task).start()

# --- GUI Layout ---
tk.Label(root, text="Upload the generated image:").pack(pady=(15, 0))
tk.Entry(root, textvariable=generated_img_var, width=60).pack()
tk.Button(root, text="Choose File", command=choose_generated_image).pack(pady=5)
target_preview = tk.Label(root, text="No target preview", width=120, height=120)
target_preview.pack()

tk.Label(root, text="Upload your selfie:").pack(pady=(15, 0))
tk.Entry(root, textvariable=selfie_var, width=60).pack()
tk.Button(root, text="Choose File", command=choose_selfie).pack(pady=5)
source_preview = tk.Label(root, text="No selfie preview", width=120, height=120)
source_preview.pack()

tk.Button(root, text="Swap Faces", command=run_roop_faceswap, bg="green", fg="black", height=2, width=20).pack(pady=20)
tk.Button(root, text="Upscale", command=handle_manual_upscale, bg="blue", fg="black", height=2, width=20).pack(pady=10)

tk.Button(root, text="Restart App", command=restart_app).pack(pady=10)
quit_button = tk.Button(root, text="Quit", command=root.quit, state=tk.DISABLED)
quit_button.pack(pady=10)

root.mainloop()
