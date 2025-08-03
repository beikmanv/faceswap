# face_utils.py

import face_recognition
from PIL import Image
import os

def extract_face_region(image_path, padding=40):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        raise Exception("❌ No face found in swapped image.")

    top, right, bottom, left = face_locations[0]

    top = max(0, top - padding)
    bottom = min(image.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(image.shape[1], right + padding)

    cropped_face = image[top:bottom, left:right]
    face_img = Image.fromarray(cropped_face)
    return face_img, (top, right, bottom, left)


def paste_upscaled_face(original_path, upscaled_face_path, face_coords, output_path):
    original_img = Image.open(original_path).convert("RGB")
    upscaled_face = Image.open(upscaled_face_path).convert("RGB")

    top, right, bottom, left = face_coords
    width, height = right - left, bottom - top
    upscaled_resized = upscaled_face.resize((width, height), Image.LANCZOS)

    original_img.paste(upscaled_resized, (left, top))
    original_img.save(output_path)
