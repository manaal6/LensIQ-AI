"""
OCR module: text extraction via pytesseract.

The app is often used on natural photos, so this module favors reliable text
over long noisy strings. It tries a few OCR page segmentation modes, keeps
high-confidence words, and returns an empty string when Tesseract is mostly
reading image texture as text.
"""

import logging
import os
import re
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

_tesseract_available = False

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    _tesseract_available = True
    logger.info("pytesseract configured with: %s", _TESSERACT_CMD)
except ImportError:
    logger.warning("pytesseract is not installed; OCR will be unavailable")


def is_ocr_available() -> bool:
    """Check if the OCR engine is available."""
    return _tesseract_available and os.path.exists(_TESSERACT_CMD)


def preprocess_for_ocr(image: np.ndarray) -> List[np.ndarray]:
    """Build OCR variants for natural images and printed text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_width = 1100
    if gray.shape[1] < max_width:
        scale = max_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.bilateralFilter(gray, 7, 50, 50)

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        7,
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return [denoised, adaptive, otsu]


def extract_text(image: np.ndarray) -> str:
    """
    Extract reliable text from a BGR image.

    Returns an empty string when OCR confidence is too low. That is better for
    summaries than presenting random visual texture as readable text.
    """
    if not _tesseract_available:
        logger.warning("OCR skipped; pytesseract not available")
        return ""

    logger.info("Running OCR on %dx%d image", image.shape[1], image.shape[0])

    try:
        candidates: List[str] = []
        for processed in preprocess_for_ocr(image):
            for psm in (6, 11):
                text = _extract_confident_words(processed, psm)
                if _looks_reliable(text):
                    candidates.append(text)

        if not candidates:
            logger.info("OCR complete: no reliable text found")
            return ""

        result = max(candidates, key=lambda item: (len(item.split()), len(item)))
        logger.info("OCR complete: extracted %d reliable characters", len(result))
        return result

    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return ""


def _extract_confident_words(image: np.ndarray, psm: int) -> str:
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
        timeout=6,
    )

    words: List[str] = []
    for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue

        text = _clean_word(raw_text)
        if not text or confidence < 72:
            continue
        if len(text) == 1 and not text.isdigit():
            continue
        if not re.search(r"[A-Za-z0-9]", text):
            continue

        words.append(text)

    return " ".join(words)


def _clean_word(word: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", word.strip())


def _looks_reliable(text: str) -> bool:
    if not text:
        return False

    words = text.split()
    if not words:
        return False

    if _noise_score(text) > 0.22:
        return False

    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    long_words = [w for w in alpha_words if len(w) >= 4]
    dictionaryish_words = [w for w in alpha_words if re.search(r"[aeiouAEIOU]", w) and not re.search(r"(.)\1{2,}", w)]

    if len(words) == 1:
        return bool(long_words)

    return len(long_words) >= 1 and len(dictionaryish_words) >= 2


def _noise_score(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 1.0

    noisy = sum(1 for ch in chars if not ch.isalnum() and ch not in "-&'")
    short_words = sum(1 for word in text.split() if len(word) <= 2 and not word.isdigit())
    return (noisy / len(chars)) + min(0.2, short_words / max(20, len(text.split())) * 0.2)
