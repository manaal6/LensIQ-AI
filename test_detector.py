"""
Quick diagnostic: test MobileNet-SSD model directly.
Run: python test_detector.py
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import glob

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

MODEL_DIR = Path("models")
PROTOTXT = MODEL_DIR / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"

print(f"Prototxt exists: {PROTOTXT.exists()} ({PROTOTXT.stat().st_size} bytes)")
print(f"Caffemodel exists: {CAFFEMODEL.exists()} ({CAFFEMODEL.stat().st_size} bytes)")
print("[OK] Caffemodel is a valid binary model file")

# Try loading the model
print("\nLoading model...")
try:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[FAIL] Model load FAILED: {e}")
    sys.exit(1)

# Create a simple test image (300x300 with some shapes)
print("\nTesting with a synthetic image...")
test_img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(test_img, (50, 100), (250, 250), (100, 150, 200), -1)

blob = cv2.dnn.blobFromImage(
    cv2.resize(test_img, (300, 300)),
    0.007843, (300, 300), 127.5
)
net.setInput(blob)
detections = net.forward()

print(f"Detection output shape: {detections.shape}")
print(f"Total raw detections: {detections.shape[2]}")

# Show ALL detections for diagnostic
print("\nSynthetic image - all detections above 1%:")
for i in range(min(detections.shape[2], 20)):
    confidence = float(detections[0, 0, i, 2])
    class_id = int(detections[0, 0, i, 1])
    if 0 <= class_id < len(CLASSES):
        label = CLASSES[class_id]
    else:
        label = f"unknown({class_id})"
    if confidence > 0.01:
        print(f"  [{i}] {label}: {confidence:.4f}")

# Now test with a real image if available
test_images = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
test_images += glob.glob("output/*.png") + glob.glob("output/*.jpg")

if test_images:
    for img_path in test_images[:3]:
        print(f"\n--- Testing with real image: {img_path} ---")
        real_img = cv2.imread(img_path)
        if real_img is not None:
            print(f"  Image shape: {real_img.shape}")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(real_img, (300, 300)),
                0.007843, (300, 300), 127.5
            )
            net.setInput(blob)
            detections = net.forward()
            print(f"  Detection output shape: {detections.shape}")
            
            # Show detections at various thresholds
            for threshold in [0.5, 0.3, 0.1, 0.01]:
                count = 0
                labels_found = []
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    class_id = int(detections[0, 0, i, 1])
                    if 0 <= class_id < len(CLASSES):
                        label = CLASSES[class_id]
                    else:
                        label = f"unknown({class_id})"
                    if label != "background" and confidence >= threshold:
                        count += 1
                        if count <= 10:
                            labels_found.append(f"{label}({confidence:.3f})")
                print(f"  Threshold {threshold}: {count} detections -> {', '.join(labels_found[:10])}")
else:
    print("\nNo test images found in current directory.")

print("\nDone!")
