"""
I/O Utilities — Save/load JSON and images to disk.
Production-grade with error handling and logging.
"""

import json
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """
    Save dictionary as formatted JSON.

    Args:
        data: Dictionary to serialize.
        path: Destination file path.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Saved JSON to %s", path)
        return path

    except Exception as e:
        logger.error("Failed to save JSON to %s: %s", path, e)
        raise


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file and return as dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary.
    """
    path = Path(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Loaded JSON from %s", path)
        return data

    except Exception as e:
        logger.error("Failed to load JSON from %s: %s", path, e)
        raise


def save_image(image: np.ndarray, path: Union[str, Path]) -> Path:
    """
    Save an OpenCV BGR image to disk.

    Args:
        image: BGR numpy array.
        path: Destination file path.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        logger.info("Saved image to %s (%dx%d)", path, image.shape[1], image.shape[0])
        return path

    except Exception as e:
        logger.error("Failed to save image to %s: %s", path, e)
        raise
