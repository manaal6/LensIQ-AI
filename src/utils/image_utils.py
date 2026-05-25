"""
Image Utilities — Resize, draw bounding boxes, decode uploads.
Decoupled from detector internals for clean architecture.
"""

import cv2
import hashlib
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Distinct colors for up to 21 VOC classes (BGR)
_BOX_COLORS = [
    (75, 25, 230),   (75, 180, 60),   (200, 130, 0),   (48, 130, 245),
    (180, 30, 145),  (240, 240, 70),  (230, 50, 240),  (60, 245, 210),
    (40, 110, 170),  (200, 250, 255), (0, 0, 128),     (195, 255, 170),
    (128, 0, 0),     (128, 128, 0),   (0, 128, 128),   (128, 0, 128),
    (255, 190, 230), (40, 150, 255),  (128, 128, 128), (0, 255, 0),
    (255, 0, 0),
]


def resize_for_inference(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Resize an image so its width does not exceed `max_width`.
    Maintains aspect ratio. Returns a copy (original is untouched).

    Args:
        image: BGR numpy array.
        max_width: Maximum width in pixels.

    Returns:
        Resized image (or original if already small enough).
    """
    h, w = image.shape[:2]

    if w <= max_width:
        return image.copy()

    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info("Resized image from %dx%d to %dx%d for inference", w, h, new_w, new_h)
    return resized


def draw_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw labeled bounding boxes on a copy of the image.

    Each detection dict must contain:
        {"label": str, "confidence": float, "box": [x, y, w, h]}

    Args:
        image: BGR numpy array.
        detections: List of detection dictionaries.

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()

    for idx, det in enumerate(detections):
        label = det["label"]
        confidence = det["confidence"]
        x, y, w, h = det["box"]

        color = _BOX_COLORS[idx % len(_BOX_COLORS)]

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label text
        text = f"{label}: {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )

        # Draw label background
        cv2.rectangle(
            annotated,
            (x, y - th - baseline - 6),
            (x + tw + 4, y),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated, text, (x + 2, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

    return annotated


def decode_upload(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Safely decode uploaded image bytes into a BGR numpy array.

    Args:
        file_bytes: Raw bytes from file upload.

    Returns:
        BGR image array, or None if decoding fails.
    """
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("cv2.imdecode returned None — invalid image data")
            return None

        if image.size == 0:
            logger.warning("Decoded image has zero size")
            return None

        logger.info("Decoded upload: %dx%d (%d bytes)",
                     image.shape[1], image.shape[0], len(file_bytes))
        return image

    except Exception as e:
        logger.error("Failed to decode uploaded image: %s", e)
        return None


def compute_image_hash(image: np.ndarray) -> str:
    """
    Compute a fast hash for an image (used for result caching).
    Uses a downsampled thumbnail for speed.

    Args:
        image: BGR numpy array.

    Returns:
        Hex digest string.
    """
    thumb = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return hashlib.md5(thumb.tobytes()).hexdigest()
