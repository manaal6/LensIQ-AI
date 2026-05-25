"""
Object detection module.

Primary path:
  - YOLOv8 via Ultralytics when the package/model is available.

Fallback paths:
  - Local MobileNet-SSD Caffe model stored in models/.
  - OpenCV Haar face detector to recover person detections when the
    object model is unavailable or misses obvious people.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_backend = None
_model_loaded = False

_MODEL_DIR = Path("models")
_PROTOTXT = _MODEL_DIR / "MobileNetSSD_deploy.prototxt"
_CAFFEMODEL = _MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

_MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]


def get_model():
    """Return a cached detector, preferring YOLO and falling back locally."""
    global _model, _model_backend, _model_loaded

    if _model_loaded and _model is not None:
        return _model

    try:
        from ultralytics import YOLO

        logger.info("Loading YOLOv8n model...")
        _model = YOLO("yolov8n.pt")
        _model_backend = "yolo"
        _model_loaded = True
        logger.info("YOLOv8n model loaded successfully")
        return _model
    except Exception as exc:
        logger.warning("YOLOv8 unavailable; trying local MobileNet-SSD: %s", exc)

    if not _PROTOTXT.exists() or not _CAFFEMODEL.exists():
        raise RuntimeError("No detector is available. Install ultralytics or add MobileNet-SSD files.")

    try:
        logger.info("Loading MobileNet-SSD model from %s", _MODEL_DIR.resolve())
        _model = cv2.dnn.readNetFromCaffe(str(_PROTOTXT), str(_CAFFEMODEL))
        _model_backend = "mobilenet"
        _model_loaded = True
        logger.info("MobileNet-SSD model loaded successfully")
        return _model
    except Exception as exc:
        logger.error("Failed to load MobileNet-SSD model: %s", exc)
        raise RuntimeError(f"Model load failed: {exc}") from exc


def is_model_loaded() -> bool:
    """Check if any detection model is currently loaded."""
    return _model_loaded and _model is not None


def get_model_backend() -> str:
    """Return the active detector backend name for logging/debug output."""
    return _model_backend or "none"


def detect_objects(
    image: np.ndarray,
    confidence_threshold: float = 0.4,
) -> List[Dict]:
    """
    Detect objects in a BGR image.

    Returns:
        [{"label": str, "confidence": float, "box": [x, y, w, h]}, ...]
    """
    try:
        model = get_model()
    except Exception as exc:
        logger.error("Cannot run detection; model unavailable: %s", exc)
        model = None

    detections: List[Dict] = []

    if model is not None and _model_backend == "yolo":
        detections = _detect_with_yolo(model, image, confidence_threshold)
    elif model is not None and _model_backend == "mobilenet":
        detections = _detect_with_mobilenet(model, image, confidence_threshold)

    labels = {d["label"] for d in detections}
    if "person" not in labels:
        face_people = _detect_people_from_faces(image)
        detections.extend(face_people)

    detections = _dedupe_detections(detections)
    logger.info(
        "Detection complete via %s: %d objects found",
        get_model_backend(),
        len(detections),
    )
    return detections


def _detect_with_yolo(model, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
    h, w = image.shape[:2]
    detections: List[Dict] = []

    try:
        results = model(image, conf=confidence_threshold, verbose=False)
    except Exception as exc:
        logger.error("YOLOv8 inference failed: %s", exc)
        return detections

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            label = result.names.get(class_id, f"class_{class_id}")
            detections.append(_make_detection(label, confidence, x1, y1, x2, y2, w, h))

    return detections


def _detect_with_mobilenet(model, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
    h, w = image.shape[:2]
    detections: List[Dict] = []

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5,
    )
    model.setInput(blob)

    try:
        output = model.forward()
    except Exception as exc:
        logger.error("MobileNet-SSD inference failed: %s", exc)
        return detections

    for i in range(output.shape[2]):
        confidence = float(output[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(output[0, 0, i, 1])
        if class_id <= 0 or class_id >= len(_MOBILENET_CLASSES):
            continue

        label = _MOBILENET_CLASSES[class_id]
        x1, y1, x2, y2 = (output[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        detections.append(_make_detection(label, confidence, x1, y1, x2, y2, w, h))

    return detections


def _detect_people_from_faces(image: np.ndarray) -> List[Dict]:
    """
    Estimate person boxes from face detections.

    This is intentionally conservative: only upper-frame face candidates are
    used, and overlapping face boxes are merged before estimating body boxes.
    It helps fashion/lifestyle photos where full-body detectors may miss people.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ]

    face_boxes: List[Tuple[int, int, int, int]] = []
    min_face = max(48, int(min(w, h) * 0.055))

    for name in cascade_names:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(min_face, min_face),
        )

        for x, y, fw, fh in faces:
            if y > h * 0.38 or fw > w * 0.25 or fh > h * 0.25:
                continue
            if not _looks_like_face_region(image, int(x), int(y), int(fw), int(fh)):
                continue
            face_boxes.append((int(x), int(y), int(fw), int(fh)))

    merged_faces = _merge_boxes(face_boxes, iou_threshold=0.2)
    people: List[Dict] = []

    for x, y, fw, fh in merged_faces:
        cx = x + fw / 2
        top = y - 0.9 * fh
        person_h = 9.5 * fh
        person_w = 4.0 * fw
        x1 = int(cx - person_w / 2)
        y1 = int(top)
        x2 = int(cx + person_w / 2)
        y2 = int(top + person_h)
        people.append(_make_detection("person", 0.55, x1, y1, x2, y2, w, h))

    if people:
        logger.info("Face fallback estimated %d person detection(s)", len(people))

    return people


def _looks_like_face_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Reject common false positives from clothing logos and upholstery."""
    image_h, image_w = image.shape[:2]
    pad_x = int(w * 0.12)
    pad_y = int(h * 0.12)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + w + pad_x)
    y2 = min(image_h, y + h + pad_y)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([35, 133, 77], dtype=np.uint8)
    upper = np.array([245, 173, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    skin_ratio = float(cv2.countNonZero(skin_mask)) / float(skin_mask.size)

    return 0.08 <= skin_ratio <= 0.75


def _make_detection(
    label: str,
    confidence: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_w: int,
    image_h: int,
) -> Dict:
    x1 = max(0, min(int(x1), image_w - 1))
    y1 = max(0, min(int(y1), image_h - 1))
    x2 = max(x1 + 1, min(int(x2), image_w))
    y2 = max(y1 + 1, min(int(y2), image_h))

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "box": [x1, y1, x2 - x1, y2 - y1],
    }


def _merge_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []

    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou_xywh(box, kept) < iou_threshold for kept in merged):
            merged.append(box)

    return merged


def _dedupe_detections(detections: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []

    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            if det["label"] == existing["label"] and _iou_xywh(det["box"], existing["box"]) > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    return kept


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
