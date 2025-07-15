import tkinter as tk

print("✅ Running test_gui.py...")

root = tk.Tk()
root.title("Test GUI Window")
root.geometry("300x150")
tk.Label(root, text="✅ If you see this, Tkinter works!").pack(pady=40)

root.mainloop()

