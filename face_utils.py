# face_utils.py

import os
from PIL import Image, ImageDraw, ImageFilter
import face_recognition
import numpy as np  # Ensure this is at the top of your file if not already


def paste_upscaled_face(original_path, upscaled_face_path, face_coords, output_path):
    original_img = Image.open(original_path).convert("RGB")
    upscaled_face = Image.open(upscaled_face_path).convert("RGB")

    top, right, bottom, left = face_coords
    width, height = right - left, bottom - top
    upscaled_resized = upscaled_face.resize((width, height), Image.LANCZOS)

    original_img.paste(upscaled_resized, (left, top))
    original_img.save(output_path)

def crop_face_by_index(image_path, face_index):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)

    if not face_locations or face_index >= len(face_locations):
        raise ValueError("Face index is out of bounds.")

    top, right, bottom, left = face_locations[face_index]
    face_img = img[top:bottom, left:right]
    pil_img = Image.fromarray(face_img)

    return pil_img

def extract_face_by_index(image_path, index):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)

    if index >= len(face_locations):
        return None, None

    top, right, bottom, left = face_locations[index]
    pil_img = Image.fromarray(img)
    face_img = pil_img.crop((left, top, right, bottom))
    return face_img, (top, right, bottom, left)

def extract_face_region(image_path, padding=80):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        raise Exception("❌ No face found in swapped image.")

    top, right, bottom, left = face_locations[0]
    face_coords = (top, right, bottom, left)  # unpadded coords

    # Padding only for better extraction
    top_pad = max(0, top - padding)
    bottom_pad = min(image.shape[0], bottom + padding)
    left_pad = max(0, left - padding)
    right_pad = min(image.shape[1], right + padding)

    cropped_face = image[top_pad:bottom_pad, left_pad:right_pad]
    face_img = Image.fromarray(cropped_face)

    return face_img, face_coords

def blur_all_but_selected_face(image_path, selected_coords, padding=60):
    img = face_recognition.load_image_file(image_path)

    if not selected_coords:
        raise ValueError("Selected face coordinates not provided.")

    top, right, bottom, left = selected_coords

    # Apply padding
    top_pad = max(0, top - padding)
    bottom_pad = min(img.shape[0], bottom + padding)
    left_pad = max(0, left - padding)
    right_pad = min(img.shape[1], right + padding)

    # Create black background
    black_img = img.copy()
    black_img[:, :] = 0  # fill entire image with black

    # Copy selected face region from original
    black_img[top_pad:bottom_pad, left_pad:right_pad] = img[top_pad:bottom_pad, left_pad:right_pad]

    return Image.fromarray(black_img), (top_pad, right_pad, bottom_pad, left_pad)

def crop_exact_region(image_path, coords):
    top, right, bottom, left = coords
    img = Image.open(image_path).convert("RGB")
    return img.crop((left, top, right, bottom))


