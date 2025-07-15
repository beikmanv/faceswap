import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

# ---- Roop Config ----
ROOP_SCRIPT = "run.py"  # path to your Roop runner
OUTPUT_PATH = "final_result.jpg"

# ---- GUI App ----
def browse_file(label, var):
    file_path = filedialog.askopenfilename()
    if file_path:
        var.set(file_path)
        label.config(text=os.path.basename(file_path))

def run_faceswap():
    if not img_path.get() or not face_path.get():
        messagebox.showerror("Missing Files", "Please upload both the generated image and your face photo.")
        return

    try:
        subprocess.run([
            "python", ROOP_SCRIPT,
            "--source_image", face_path.get(),
            "--target_image", img_path.get(),
            "--output", OUTPUT_PATH
        ], check=True)
        messagebox.showinfo("Done", f"Face swapped! Saved to {OUTPUT_PATH}")
        os.system(f"open {OUTPUT_PATH}")  # macOS; replace with "start" on Windows
    except Exception as e:
        messagebox.showerror("Error", str(e))

# App window
root = tk.Tk()
root.title("AI Portrait Swapper")

tk.Label(root, text="Enter your creative prompt:").pack()
prompt_entry = tk.Entry(root, width=60)
prompt_entry.pack(pady=5)

def open_links():
    prompt = prompt_entry.get()
    if not prompt:
        messagebox.showerror("Missing Prompt", "Please enter a prompt first.")
        return
    # Open multiple sites
    os.system(f'open "https://labs.google/fx"')
    os.system(f'open "https://www.ideogram.ai"')
    os.system(f'open "https://www.mage.space"')

tk.Button(root, text="Generate Images (Open Sites)", command=open_links).pack(pady=5)

img_path = tk.StringVar()
face_path = tk.StringVar()

tk.Label(root, text="Upload Generated Image:").pack()
tk.Button(root, text="Choose Image", command=lambda: browse_file(img_label, img_path)).pack()
img_label = tk.Label(root, text="No file selected")
img_label.pack()

tk.Label(root, text="Upload Your Selfie:").pack()
tk.Button(root, text="Choose Selfie", command=lambda: browse_file(face_label, face_path)).pack()
face_label = tk.Label(root, text="No file selected")
face_label.pack()

tk.Button(root, text="Run Face Swap", command=run_faceswap, bg="green", fg="white").pack(pady=10)

root.mainloop()

