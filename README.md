# LensIQ AI

LensIQ AI is a Streamlit-based vision analysis dashboard. Upload an image and the app detects objects, extracts readable text with OCR, generates a plain-language scene summary, and lets you ask simple questions about the processed image.

## Features

- Object detection with YOLOv8n through Ultralytics
- MobileNet-SSD fallback model support
- Face-based person detection fallback for difficult images
- OCR with Tesseract and `pytesseract`
- Rule-based scene summaries and image Q&A
- Annotated image preview and JSON result export
- Streamlit UI with cached models and cached image results

## Project Structure

```text
.
|-- app.py                    # Streamlit application
|-- download_models.py        # Optional MobileNet-SSD fallback model downloader
|-- requirements.txt          # Python dependencies
|-- Dockerfile                # Containerized app runtime
|-- test_detector.py          # Detector smoke test
`-- src
    |-- core
    |   |-- agent.py          # Scene summary and Q&A logic
    |   |-- detector.py       # YOLO/MobileNet/face fallback detection
    |   `-- ocr.py            # Tesseract OCR helpers
    |-- services
    |   |-- cache.py          # Streamlit cache/session helpers
    |   `-- pipeline.py       # Image analysis orchestration
    `-- utils
        |-- image_utils.py    # Image decoding, hashing, drawing, resizing
        `-- io_utils.py       # JSON/image output helpers
```

## Requirements

- Python 3.10+
- Tesseract OCR installed on your system
- Python packages from `requirements.txt`

The app uses `yolov8n.pt` by default. Ultralytics downloads this model automatically on first use when internet access is available.

## Setup

Clone the repository:

```bash
git clone https://github.com/manaal6/LensIQ-AI.git
cd LensIQ-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR:

- Windows: install from the official Tesseract installer and make sure `tesseract.exe` is available.
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

If Tesseract is installed in a custom location, set `TESSERACT_CMD` before running the app:

```bash
# Windows PowerShell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

# macOS/Linux
export TESSERACT_CMD=/usr/bin/tesseract
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Optional Fallback Models

YOLOv8n is the primary detector. If you want the local MobileNet-SSD fallback files available, run:

```bash
python download_models.py
```

This creates a `models/` directory containing:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

Force a re-download with:

```bash
python download_models.py --force
```

## Docker

Build the image:

```bash
docker build -t lensiq-ai .
```

Run the container:

```bash
docker run -p 8501:8501 lensiq-ai
```

Open:

```text
http://localhost:8501
```

## Usage

1. Upload an image in the Streamlit dashboard.
2. Run the analysis.
3. Review detected objects, confidence scores, OCR text, scene summary, and annotated output.
4. Ask questions about the processed image, such as:
   - `What do you see?`
   - `How many people are there?`
   - `What text is visible?`
   - `What are the confidence scores?`

Generated logs are written to `logs/app.log`. Processed outputs are saved under `output/`.

## Testing

Run the detector smoke test:

```bash
python test_detector.py
```

## Notes

- OCR quality depends heavily on image clarity, text size, contrast, and Tesseract availability.
- The Q&A agent is rule-based and uses the detection/OCR results from the current image. It does not call external language model APIs.
- On first run, YOLO may take extra time while the model weights are downloaded.
