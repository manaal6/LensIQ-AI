"""
Vision Pipeline — Main orchestration engine.

Runs the full analysis pipeline:
  1. Resize image for inference
  2. Object detection (MobileNet-SSD)
  3. OCR (pytesseract)
  4. Scene summary (vision agent)
  5. Return unified JSON result

Production features:
  • Structured output format
  • Processing time measurement
  • Full error handling with safe fallback
  • Console + file logging
"""

import time
import logging
import logging.handlers
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.core.detector import detect_objects
from src.core.ocr import extract_text
from src.core.agent import generate_scene_summary
from src.utils.image_utils import resize_for_inference

logger = logging.getLogger(__name__)

# ── Logging configuration ─────────────────────────────────────────────
_logging_configured = False


def configure_logging() -> None:
    """
    Configure the application logging system.
    Logs to both console (INFO) and file (DEBUG).
    Safe to call multiple times — only configures once.
    """
    global _logging_configured
    if _logging_configured:
        return

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))

    # File handler (DEBUG level, rotating 5MB × 3 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    ))

    root.addHandler(console)
    root.addHandler(file_handler)

    _logging_configured = True
    logger.info("Logging configured — file: %s", log_file)


def _safe_fallback(message: str, elapsed_ms: int = 0) -> Dict[str, Any]:
    """Return a safe failure response."""
    return {
        "status": "failed",
        "message": message,
        "scene_summary": "",
        "objects": [],
        "text": "",
        "processing_time_ms": elapsed_ms,
    }


def process_image(
    image: np.ndarray,
    confidence_threshold: float = 0.4,
    max_inference_width: int = 640,
) -> Dict[str, Any]:
    """
    Run the full vision analysis pipeline on an image.

    Args:
        image: BGR numpy array (original resolution).
        confidence_threshold: Minimum detection confidence.
        max_inference_width: Max width for inference resizing.

    Returns:
        Structured result dict:
        {
            "status": "success" | "failed",
            "scene_summary": str,
            "objects": [{"label", "confidence", "box"}, ...],
            "text": str,
            "processing_time_ms": int,
            "message": str  (only on failure)
        }
    """
    start = time.perf_counter()

    # ── Validate input ────────────────────────────────────────────────
    if image is None or image.size == 0:
        logger.error("process_image received invalid image (None or empty)")
        return _safe_fallback("Invalid image input")

    logger.info(
        "Pipeline started — image: %dx%d, threshold: %.2f",
        image.shape[1], image.shape[0], confidence_threshold,
    )

    # ── Step 1: Resize for inference ──────────────────────────────────
    try:
        inference_img = resize_for_inference(image, max_width=max_inference_width)
        logger.info("Step 1/4: Image resized for inference")
    except Exception as e:
        logger.error("Resize failed: %s", e)
        inference_img = image  # fallback to original

    # ── Step 2: Object detection ──────────────────────────────────────
    try:
        # Run detection on the original image so returned boxes line up with
        # the uploaded preview and annotated output.
        objects = detect_objects(image, confidence_threshold=confidence_threshold)
        logger.info("Step 2/4: Detection complete — %d objects", len(objects))
    except Exception as e:
        logger.error("Detection step failed: %s", e)
        objects = []

    # ── Step 3: OCR ───────────────────────────────────────────────────
    try:
        # OCR benefits from original resolution, especially for logos and
        # small printed text.
        ocr_text = extract_text(image)
        logger.info("Step 3/4: OCR complete — %d chars", len(ocr_text))
    except Exception as e:
        logger.error("OCR step failed: %s", e)
        ocr_text = ""

    # ── Step 4: Scene summary ─────────────────────────────────────────
    try:
        scene_summary = generate_scene_summary(objects, ocr_text)
        logger.info("Step 4/4: Scene summary generated")
    except Exception as e:
        logger.error("Scene summary failed: %s", e)
        scene_summary = "Unable to generate scene summary."

    # ── Build result ──────────────────────────────────────────────────
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    result = {
        "status": "success",
        "scene_summary": scene_summary,
        "objects": objects,
        "text": ocr_text,
        "processing_time_ms": elapsed_ms,
    }

    logger.info("Pipeline complete in %d ms", elapsed_ms)
    return result
