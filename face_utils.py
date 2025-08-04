import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import face_recognition

def mask_and_extract_face(image_path, face_index=0):
    """Create a smooth RGBA mask of the face using facial landmarks."""
    image = face_recognition.load_image_file(image_path)
    landmarks_list = face_recognition.face_landmarks(image)

    if not landmarks_list or face_index >= len(landmarks_list):
        raise ValueError("No valid face detected.")

    landmarks = landmarks_list[face_index]
    face_outline = landmarks["chin"]

    # Create binary mask using chin outline
    mask_img = Image.new("L", (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(face_outline, outline=255, fill=255)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(3))  # Feather edges

    # Convert original to RGBA and apply mask as alpha
    pil_img = Image.fromarray(image).convert("RGBA")
    r, g, b, _ = pil_img.split()
    final_img = Image.merge("RGBA", (r, g, b, mask_img))

    # Crop tightly to masked face region
    bbox = mask_img.getbbox()
    cropped_face = final_img.crop(bbox)

    return cropped_face, bbox


def paste_masked_face(target_path, masked_face_img, bbox, output_path):
    base_img = Image.open(target_path).convert("RGBA")

    # Resize bbox to match new face size
    x, y, x2, y2 = bbox
    face_w, face_h = masked_face_img.size
    paste_box = (x, y, x + face_w, y + face_h)

    # Paste using alpha channel
    base_img.paste(masked_face_img, (x, y), masked_face_img)

    base_img.convert("RGB").save(output_path)

def crop_exact_region(image_path, coords):
    """Crop any specific rectangular region."""
    top, right, bottom, left = coords
    img = Image.open(image_path).convert("RGB")
    return img.crop((left, top, right, bottom))

def create_masked_target(image_path, selected_coords, padding=60):
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
    black_img[:, :] = 0  # make all pixels black

    # Paste selected face region
    black_img[top_pad:bottom_pad, left_pad:right_pad] = img[top_pad:bottom_pad, left_pad:right_pad]

    return Image.fromarray(black_img), (top_pad, right_pad, bottom_pad, left_pad)

