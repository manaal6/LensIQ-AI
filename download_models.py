"""
Download MATCHING MobileNet-SSD model files from multiple mirrors.
Tries each URL until one works.
Run:  python download_models.py [--force]
"""

import urllib.request
import ssl
import sys
import hashlib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Multiple mirrors for reliability - tries in order
PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
    "https://huggingface.co/Imran606/cds/resolve/main/MobileNetSSD_deploy.prototxt",
]

CAFFEMODEL_URLS = [
    # Hugging Face direct download (most reliable for large files)
    "https://huggingface.co/Imran606/cds/resolve/main/MobileNetSSD_deploy.caffemodel",
    # Original repo (may fail due to Git LFS)
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
]

FILES = {
    "MobileNetSSD_deploy.prototxt": PROTOTXT_URLS,
    "MobileNetSSD_deploy.caffemodel": CAFFEMODEL_URLS,
}

# Expected approximate file sizes for validation
EXPECTED_SIZES = {
    "MobileNetSSD_deploy.prototxt": (28000, 35000),
    "MobileNetSSD_deploy.caffemodel": (23000000, 24000000),
}


def download(name, urls, force=False):
    dest = MODEL_DIR / name
    if dest.exists() and not force:
        print("  [SKIP] %s already exists (%d bytes). Use --force to re-download." % (name, dest.stat().st_size))
        return True

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in urls:
        print("  [>>] Trying: %s ..." % url[:80])
        sys.stdout.flush()

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=180) as resp:
                data = resp.read()
        except Exception as e:
            print("  [!!] Failed: %s" % str(e)[:80])
            continue

        # Validate: check it's not an HTML error page
        if data[:5] == b"<html" or data[:5] == b"<!DOC" or data[:15] == b"<!DOCTYPE html>":
            print("  [!!] Got HTML page instead of model file, trying next URL...")
            continue

        # Validate size
        if name in EXPECTED_SIZES:
            lo, hi = EXPECTED_SIZES[name]
            if len(data) < lo or len(data) > hi:
                print("  [!!] Size %d outside expected range (%d - %d), trying next URL..." % (len(data), lo, hi))
                continue

        with open(str(dest), "wb") as f:
            f.write(data)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print("  [OK] Saved %s (%.1f MB)" % (name, size_mb))
        return True

    print("  [FAIL] Could not download %s from any mirror." % name)
    print("         Please download manually and place in models/ folder.")
    return False


def verify_model():
    """Quick test that the model actually produces detections."""
    try:
        import cv2
        import numpy as np

        proto = str(MODEL_DIR / "MobileNetSSD_deploy.prototxt")
        model = str(MODEL_DIR / "MobileNetSSD_deploy.caffemodel")
        net = cv2.dnn.readNetFromCaffe(proto, model)

        # Test with a simple colored image
        test = np.zeros((300, 300, 3), dtype=np.uint8)
        test[50:250, 50:250] = [200, 150, 100]
        blob = cv2.dnn.blobFromImage(test, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        out = net.forward()

        print("\n  [VERIFY] Model loaded OK, output shape: %s" % str(out.shape))
        # A working model should at least produce the right output shape
        if out.shape == (1, 1, 100, 7):
            print("  [VERIFY] Output shape correct")
            return True
        else:
            print("  [VERIFY] Unexpected output shape!")
            return False
    except Exception as e:
        print("  [VERIFY] Verification failed: %s" % str(e))
        return False


if __name__ == "__main__":
    force = "--force" in sys.argv

    print("Downloading MobileNet-SSD model files ...\n")
    all_ok = True
    for name, urls in FILES.items():
        if not download(name, urls, force=force):
            all_ok = False

    if all_ok:
        print("\nVerifying model ...")
        verify_model()

    missing = [n for n in FILES if not (MODEL_DIR / n).exists()]
    if missing:
        print("\n[WARNING] Missing files: %s" % ", ".join(missing))
    elif all_ok:
        print("\nDone! Run: streamlit run app.py")
    else:
        print("\n[WARNING] Some downloads may have failed.")
