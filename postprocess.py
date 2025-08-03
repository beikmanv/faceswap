import cv2
import numpy as np

def create_face_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.5))  # elliptical mask
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def detect_face_center_mediapipe(image_path):
    import cv2
    import mediapipe as mp

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detections:
        raise ValueError("No face detected")

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    center_x = int((bbox.xmin + bbox.width / 2) * w)
    center_y = int((bbox.ymin + bbox.height / 2) * h)
    return (center_x, center_y)

def smart_blend_face(base_img, face_img, mask, center, scale=1.05, mode='mixed'):
    bh, bw = base_img.shape[:2]
    fh, fw = face_img.shape[:2]

    # Scale face image
    new_w, new_h = int(fw * scale), int(fh * scale)

    # Ensure scaled face fits inside base image
    if new_w > bw or new_h > bh:
        scale_w = bw / fw
        scale_h = bh / fh
        scale = min(scale_w, scale_h) * 0.9  # Slightly smaller to avoid edges
        new_w, new_h = int(fw * scale), int(fh * scale)

    face_img_scaled = cv2.resize(face_img, (new_w, new_h))
    mask_scaled = cv2.resize(mask, (new_w, new_h))

    if len(mask_scaled.shape) == 2:
        mask_scaled = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)

    # Adjust center after scaling
    offset_x = (new_w - fw) // 2
    offset_y = (new_h - fh) // 2
    new_center = (center[0] - offset_x, center[1] - offset_y)

    # Clamp center to stay within bounds
    min_x = new_w // 2
    min_y = new_h // 2
    max_x = bw - new_w // 2
    max_y = bh - new_h // 2
    new_center = (
        max(min_x, min(max_x, new_center[0])),
        max(min_y, min(max_y, new_center[1]))
    )

    # Debug logging
    print(f"[DEBUG] Face size: {new_w}x{new_h}, Base size: {bw}x{bh}, Center: {new_center}")

    try:
        mode_flag = cv2.MIXED_CLONE if mode == 'mixed' else cv2.NORMAL_CLONE
        result = cv2.seamlessClone(face_img_scaled, base_img, mask_scaled, new_center, mode_flag)
        return result
    except cv2.error as e:
        raise RuntimeError(f"[ERROR] seamlessClone failed: {str(e)}\n"
                           f"Image sizes: face {new_w}x{new_h}, base {bw}x{bh}, center {new_center}")
