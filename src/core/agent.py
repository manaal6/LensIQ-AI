"""
Vision Reasoning Agent — Scene summarization and Q&A.

Rule-based approach (no external APIs):
  • Generates human-readable scene summaries from detections + OCR
  • Answers user questions about processed images
"""

import logging
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)


def generate_scene_summary(objects: List[Dict], ocr_text: str) -> str:
    """
    Generate a human-readable scene summary from detection and OCR results.

    Args:
        objects: List of detection dicts [{"label", "confidence", "box"}, ...]
        ocr_text: Extracted text from OCR.

    Returns:
        Scene summary string (1–3 sentences).
    """
    parts = []

    if objects:
        # Count objects by label
        label_counts = Counter(obj["label"] for obj in objects)
        total = len(objects)

        # Build object description
        obj_parts = []
        for label, count in label_counts.most_common():
            if count == 1:
                obj_parts.append(f"a {label}")
            elif label == "person":
                obj_parts.append(f"{count} people")
            else:
                obj_parts.append(f"{count} {label}s")

        if len(obj_parts) == 1:
            obj_desc = obj_parts[0]
        elif len(obj_parts) == 2:
            obj_desc = f"{obj_parts[0]} and {obj_parts[1]}"
        else:
            obj_desc = ", ".join(obj_parts[:-1]) + f", and {obj_parts[-1]}"

        # Compute average confidence
        avg_conf = sum(o["confidence"] for o in objects) / total
        conf_desc = "high" if avg_conf >= 0.7 else "moderate" if avg_conf >= 0.5 else "low"

        parts.append(
            f"The scene contains {obj_desc} "
            f"(detected with {conf_desc} confidence)."
        )

        # Spatial hints
        _add_spatial_hints(objects, parts)
    else:
        parts.append("No recognizable objects were detected in the image.")

    # OCR text
    if ocr_text and len(ocr_text.strip()) > 2:
        clean_text = ocr_text.strip()
        if len(clean_text) > 120:
            preview = clean_text[:120].rsplit(" ", 1)[0] + "…"
        else:
            preview = clean_text
        parts.append(f'Text visible in the image: "{preview}"')
    else:
        parts.append("No readable text was found in the image.")

    summary = " ".join(parts)
    logger.info("Generated scene summary (%d chars)", len(summary))
    return summary


def _add_spatial_hints(objects: List[Dict], parts: List[str]) -> None:
    """Add basic spatial relationship descriptions."""
    if len(objects) < 2:
        return

    # Check for objects occupying large portions of the image
    labels_by_area = []
    for obj in objects:
        x, y, w, h = obj["box"]
        area = w * h
        labels_by_area.append((obj["label"], area))

    labels_by_area.sort(key=lambda t: t[1], reverse=True)

    if len(labels_by_area) >= 2:
        dominant = labels_by_area[0][0]
        if dominant == "person" and sum(1 for obj in objects if obj["label"] == "person") > 1:
            parts.append("People are the most prominent subjects in the frame.")
        else:
            parts.append(f"The {dominant} is the most prominent object in the frame.")


def ask_about_image(question: str, result: Dict[str, Any]) -> str:
    """
    Answer a user question about a processed image using the structured result.

    Args:
        question: The user's natural-language question.
        result: The full pipeline result dict (status, objects, text, scene_summary).

    Returns:
        Answer string.
    """
    if not result or result.get("status") != "success":
        return "I don't have any analysis results for this image yet. Please run the analysis first."

    q = question.lower().strip()
    objects = result.get("objects", [])
    ocr_text = result.get("text", "")
    summary = result.get("scene_summary", "")

    # ── Question routing ──────────────────────────────────────────────

    # "What do you see?" / general scene questions
    if any(kw in q for kw in ["what do you see", "describe", "what is", "what's in", "summary", "scene"]):
        return summary if summary else "I couldn't generate a summary for this image."

    # Object counting
    if any(kw in q for kw in ["how many", "count", "number of"]):
        if not objects:
            return "No objects were detected in this image."

        label_counts = Counter(obj["label"] for obj in objects)

        # Check if asking about a specific object
        for label, count in label_counts.items():
            if label in q:
                return f"I detected {count} {label}{'s' if count > 1 else ''} in the image."

        # General count
        total = len(objects)
        breakdown = ", ".join(f"{c} {l}{'s' if c > 1 else ''}" for l, c in label_counts.most_common())
        return f"I detected {total} object{'s' if total > 1 else ''} total: {breakdown}."

    # Text/OCR questions
    if any(kw in q for kw in ["text", "read", "written", "says", "words", "ocr"]):
        if ocr_text:
            return f'The text found in the image reads:\n"{ocr_text}"'
        return "No readable text was found in this image."

    # Object presence ("is there a...", "do you see a...")
    if any(kw in q for kw in ["is there", "do you see", "can you find", "any", "detect"]):
        if not objects:
            return "No objects were detected in this image."

        labels = set(obj["label"] for obj in objects)
        for label in labels:
            if label in q:
                count = sum(1 for o in objects if o["label"] == label)
                return f"Yes, I detected {count} {label}{'s' if count > 1 else ''} in the image."

        return f"The detected objects are: {', '.join(sorted(labels))}. I didn't find what you're specifically asking about."

    # Confidence questions
    if any(kw in q for kw in ["confidence", "sure", "certain", "accurate"]):
        if not objects:
            return "No objects were detected, so there are no confidence scores to report."

        lines = [f"  • {o['label']}: {o['confidence']:.1%}" for o in objects]
        return "Detection confidence scores:\n" + "\n".join(lines)

    # Processing time
    if any(kw in q for kw in ["time", "fast", "speed", "long", "duration"]):
        ms = result.get("processing_time_ms", 0)
        return f"The analysis took {ms:,} milliseconds ({ms / 1000:.1f} seconds)."

    # Fallback — return summary
    if summary:
        return f"Based on my analysis: {summary}"

    return "I'm not sure how to answer that question. Try asking about objects, text, or the overall scene."
