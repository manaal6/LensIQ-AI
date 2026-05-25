"""
Caching Layer — Model and result caching for Streamlit.

Uses:
  • st.cache_resource for model singleton (survives reruns)
  • st.session_state for per-session result caching
  • Image hashing to avoid duplicate inference
"""

import logging
import streamlit as st
from typing import Dict, Any, Optional

from src.core.detector import get_model, is_model_loaded
from src.core.ocr import is_ocr_available
from src.utils.image_utils import compute_image_hash

logger = logging.getLogger(__name__)

_RESULT_CACHE_VERSION = "v3"


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Load and cache the detection model using Streamlit's resource cache.
    This persists across reruns and sessions — model loads only ONCE.

    Returns:
        The loaded cv2.dnn.Net model.
    """
    logger.info("Loading model into Streamlit cache_resource...")
    model = get_model()
    logger.info("Model cached successfully")
    return model


def get_system_status() -> Dict[str, bool]:
    """
    Return the current status of system components.

    Returns:
        Dict with component availability flags.
    """
    return {
        "model_loaded": is_model_loaded(),
        "ocr_available": is_ocr_available(),
    }


def get_cached_result(image_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached analysis result from session state.

    Args:
        image_hash: MD5 hash of the image.

    Returns:
        Cached result dict, or None if not cached.
    """
    cache_key = f"result_{_RESULT_CACHE_VERSION}_{image_hash}"

    if cache_key in st.session_state:
        logger.info("Cache HIT for image hash %s", image_hash[:8])
        return st.session_state[cache_key]

    logger.debug("Cache MISS for image hash %s", image_hash[:8])
    return None


def set_cached_result(image_hash: str, result: Dict[str, Any]) -> None:
    """
    Store an analysis result in session state.

    Args:
        image_hash: MD5 hash of the image.
        result: The pipeline result dict to cache.
    """
    cache_key = f"result_{_RESULT_CACHE_VERSION}_{image_hash}"
    st.session_state[cache_key] = result
    logger.info("Cached result for image hash %s", image_hash[:8])


def get_current_result() -> Optional[Dict[str, Any]]:
    """Get the most recent analysis result from session state."""
    return st.session_state.get("current_result", None)


def set_current_result(result: Dict[str, Any]) -> None:
    """Store the most recent analysis result in session state."""
    st.session_state["current_result"] = result


def get_chat_history() -> list:
    """Get the chat history from session state."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return st.session_state["chat_history"]


def add_chat_message(role: str, content: str) -> None:
    """Add a message to the chat history."""
    history = get_chat_history()
    history.append({"role": role, "content": content})
