"""
visualize_detections.py

Runs the model on an image and saves an annotated version with bounding
boxes, class labels, and confidence scores drawn on it. Also prints each
detection's confidence score to the terminal so you can spot low-confidence
(likely false positive) detections.

Usage:
    python visualize_detections.py <image_path>

Example:
    python visualize_detections.py test_images/104_23.png
"""

import sys
import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CLASS_NAMES = {0: "bullet", 1: "bullseye", 2: "outerring"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_detections.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    print(f"\n📂 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"🎯 Running detection on: {image_path}\n")
    results = model(image_path, verbose=False)
    result = results[0]

    # Print each detection with its confidence score, sorted by confidence
    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append((cls, conf, cx, cy))

    detections.sort(key=lambda d: d[1])  # sort ascending — lowest confidence first

    print(f"{'CLASS':<12} {'CONFIDENCE':<12} {'CENTER (px)':<20}")
    print("-" * 45)
    for cls, conf, cx, cy in detections:
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        flag = "  ⚠️ LOW CONFIDENCE" if conf < 0.5 else ""
        print(f"{name:<12} {conf:<12.3f} ({cx:.1f}, {cy:.1f}){flag}")

    # Save annotated image with boxes drawn
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "annotated_" + os.path.basename(image_path))

    annotated = result.plot()  # numpy array with boxes drawn
    # pyrefly: ignore [missing-import]
    import cv2
    cv2.imwrite(save_path, annotated)

    print(f"\n💾 Annotated image saved to: {save_path}")
    print(f"   Open it to visually see which detection is the false positive.\n")


if __name__ == "__main__":
    main()
