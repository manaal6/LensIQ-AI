"""
AI Vision Recognition System — Production Streamlit Dashboard

Thin UI shell. All logic delegated to:
  • src.services.pipeline  — analysis orchestration
  • src.services.cache     — model + result caching
  • src.core.agent         — conversational Q&A
  • src.utils              — image processing + I/O
"""

import json
import cv2
import streamlit as st
from pathlib import Path

from src.services.pipeline import process_image, configure_logging
from src.services.cache import (
    get_cached_model,
    get_cached_result,
    set_cached_result,
    get_current_result,
    set_current_result,
    get_system_status,
    get_chat_history,
    add_chat_message,
)
from src.core.agent import ask_about_image
from src.utils.image_utils import decode_upload, draw_boxes, compute_image_hash
from src.utils.io_utils import save_json, save_image

# ── Bootstrap ─────────────────────────────────────────────────────────
configure_logging()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Vision Recognition System",
    page_icon="🔍",
    layout="centered",
)

# ── Eagerly load model into cache ─────────────────────────────────────
try:
    get_cached_model()
except Exception:
    pass  # handled in UI via status badges

# ── Production CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] { display: none; }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #6c757d;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* ── Status badges ── */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin: 0.75rem 0 1.5rem;
        flex-wrap: wrap;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #166534;
        transition: transform 0.15s;
    }
    .status-badge:hover { transform: scale(1.03); }
    .status-badge.warning {
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
    }

    /* ── Cards ── */
    .result-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .result-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transform: translateY(-1px);
    }
    .result-card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.6rem;
        margin-top: 0;
    }

    /* ── Summary card (accent) ── */
    .summary-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8edff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .summary-card h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #3730a3;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    .summary-card p {
        color: #312e81;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        max-width: 520px;
        margin: 0 auto;
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed #ced4da;
        border-radius: 14px;
        padding: 1rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #4361ee;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4361ee, #3a0ca3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67, 97, 238, 0.4);
        color: white;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Metric badge ── */
    .metric-badge {
        display: inline-block;
        background: #4361ee;
        color: #fff;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* ── OCR text box ── */
    .ocr-text-box {
        background: #f1f3f5;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #212529;
        white-space: pre-wrap;
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.55;
    }

    /* ── Processing time pill ── */
    .time-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #475569;
    }

    /* ── Chat panel ── */
    .chat-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px 14px 0 0;
        padding: 1rem 1.5rem;
        color: white;
    }
    .chat-header h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: white;
    }
    .chat-header p {
        margin: 0.2rem 0 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 2rem 0;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Vision Recognition System</h1>
    <p>Object Detection &nbsp;·&nbsp; OCR &nbsp;·&nbsp; Scene Analysis</p>
</div>
""", unsafe_allow_html=True)

# System status badges
status = get_system_status()
model_cls = "" if status["model_loaded"] else " warning"
model_icon = "✅" if status["model_loaded"] else "⚠️"
model_text = "Model Ready" if status["model_loaded"] else "Model Loading…"

ocr_cls = "" if status["ocr_available"] else " warning"
ocr_icon = "✅" if status["ocr_available"] else "⚠️"
ocr_text = "OCR Ready" if status["ocr_available"] else "OCR Unavailable"

st.markdown(f"""
<div class="status-bar">
    <span class="status-badge{model_cls}">{model_icon} {model_text}</span>
    <span class="status-badge{ocr_cls}">{ocr_icon} {ocr_text}</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, WEBP · Max 200 MB",
)

if uploaded_file is not None:
    # ── Decode image safely ───────────────────────────────────────────
    raw_bytes = uploaded_file.read()
    image = decode_upload(raw_bytes)

    if image is None:
        st.error("❌ Could not decode the uploaded image. The file may be corrupted — please try another.")
        st.stop()

    # ── Show uploaded image ───────────────────────────────────────────
    st.markdown('<div class="result-card"><h3>📷 Uploaded Image</h3></div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Compute image hash for caching ────────────────────────────────
    img_hash = compute_image_hash(image)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3 — PROCESSING
    # ═══════════════════════════════════════════════════════════════════
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run_clicked = st.button("🚀 Run AI Analysis", use_container_width=True)

    # Check cache first
    cached = get_cached_result(img_hash)

    if run_clicked or cached is not None:
        if cached is not None and not run_clicked:
            result = cached
        else:
            # Run the pipeline with progress indicators
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                with st.spinner("⚙️ Running full vision analysis pipeline…"):
                    result = process_image(image, confidence_threshold=0.4)

                    # Cache the result
                    if result["status"] == "success":
                        set_cached_result(img_hash, result)

            progress_placeholder.empty()

        # Store as current result for chat
        set_current_result(result)

        # ═══════════════════════════════════════════════════════════════
        # SECTION 4 — RESULTS DASHBOARD
        # ═══════════════════════════════════════════════════════════════

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if result["status"] == "failed":
            st.error(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            st.stop()

        # ── Processing time ───────────────────────────────────────────
        ms = result["processing_time_ms"]
        st.markdown(
            f'<div style="text-align:center; margin-bottom:1.5rem;">'
            f'<span class="time-pill">⚡ Processed in {ms:,} ms ({ms/1000:.1f}s)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Scene summary ─────────────────────────────────────────────
        st.markdown(
            f'<div class="summary-card">'
            f'<h3>🧠 Scene Summary</h3>'
            f'<p>{result["scene_summary"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Annotated image ───────────────────────────────────────────
        objects = result.get("objects", [])
        if objects:
            annotated = draw_boxes(image, objects)
            st.markdown('<div class="result-card"><h3>🎯 Annotated Image</h3></div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("No objects detected above 40% confidence.")
            annotated = image

        # ── Two-column results ────────────────────────────────────────
        col_ocr, col_det = st.columns(2)

        with col_ocr:
            st.markdown('<div class="result-card"><h3>📝 Extracted Text (OCR)</h3></div>', unsafe_allow_html=True)
            ocr_text = result.get("text", "")
            if ocr_text:
                st.markdown(f'<div class="ocr-text-box">{ocr_text}</div>', unsafe_allow_html=True)
            else:
                st.info("No text detected in the image.")

        with col_det:
            st.markdown('<div class="result-card"><h3>🏷️ Detected Objects</h3></div>', unsafe_allow_html=True)
            if objects:
                for d in objects:
                    st.markdown(
                        f'<span class="metric-badge">{d["confidence"]:.0%}</span> '
                        f'**{d["label"]}** &nbsp; '
                        f'`[{d["box"][0]}, {d["box"][1]}, {d["box"][2]}, {d["box"][3]}]`',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No objects detected above 40% confidence.")

        # ── Save outputs ──────────────────────────────────────────────
        output_data = {
            "status": result["status"],
            "image": uploaded_file.name,
            "scene_summary": result["scene_summary"],
            "objects": objects,
            "text": ocr_text,
            "processing_time_ms": ms,
        }

        save_json(output_data, OUTPUT_DIR / "result.json")
        save_image(annotated, OUTPUT_DIR / "annotated.jpg")

        # ── Download button ───────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download JSON Results",
            data=json.dumps(output_data, indent=2),
            file_name="result.json",
            mime="application/json",
            use_container_width=True,
        )

        st.success("✅ Results saved to `output/result.json` and `output/annotated.jpg`")

        # ═══════════════════════════════════════════════════════════════
        # SECTION 5 — CHAT PANEL ("Ask About Image")
        # ═══════════════════════════════════════════════════════════════

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        st.markdown("""
        <div class="chat-header">
            <h3>💬 Ask About This Image</h3>
            <p>Ask questions about what was detected, the text, objects, or scene</p>
        </div>
        """, unsafe_allow_html=True)

        # Display chat history
        chat_history = get_chat_history()
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about the image…")

        if user_question:
            # Show user message
            add_chat_message("user", user_question)
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate and show response
            current_result = get_current_result()
            answer = ask_about_image(user_question, current_result)

            add_chat_message("assistant", answer)
            with st.chat_message("assistant"):
                st.markdown(answer)
