"""
visualize_detections.py

Draws the detection results on the image so you can visually verify:
  - the black circle (found via classical CV, drawn in green)
  - each bullet hole detection AFTER confidence filtering + deduplication
    (drawn in blue, with confidence score)

This matches the same logic used in inference.py, so what you see here
is exactly what gets converted to mm coordinates.

Usage:
    python visualize_detections.py <image_path>

Example:
    python visualize_detections.py test_images/130_7.png
"""

import sys
import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import cv2

MODEL_PATH = "best.pt"
CLASS_BULLET = 0

CONFIDENCE_THRESHOLD  = 0.5
DUPLICATE_DISTANCE_PX = 8
BLACK_THRESHOLD = 60


def get_box_center_and_size(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def find_black_circle(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, thresh = cv2.threshold(img, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return cx, cy, radius


def deduplicate_by_distance(bullet_candidates, min_distance_px=DUPLICATE_DISTANCE_PX):
    bullet_candidates = sorted(bullet_candidates, key=lambda b: b[1], reverse=True)
    kept = []
    for box, conf, cx, cy in bullet_candidates:
        is_duplicate = False
        for _, _, kx, ky in kept:
            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            if dist < min_distance_px:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((box, conf, cx, cy))
    return kept  # keep full tuples here, we need conf for drawing labels


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

    # ── find black circle via classical CV ──
    circle_result = find_black_circle(image_path)
    if circle_result is None:
        print("❌ Could not find black circle")
        sys.exit(1)
    center_x, center_y, radius = circle_result
    print(f"Black circle — center: ({center_x:.1f}, {center_y:.1f})  radius: {radius:.1f}px  "
          f"diameter: {radius*2:.1f}px")

    # ── run YOLO for bullets only ──
    results = model(image_path, verbose=False)
    result = results[0]

    bullet_candidates = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != CLASS_BULLET or conf < CONFIDENCE_THRESHOLD:
            continue
        cx, cy, _, _ = get_box_center_and_size(box)
        bullet_candidates.append((box, conf, cx, cy))

    deduped = deduplicate_by_distance(bullet_candidates)
    print(f"Bullets — {len(bullet_candidates)} candidates → {len(deduped)} after dedup\n")

    for _, conf, cx, cy in sorted(deduped, key=lambda d: d[1]):
        flag = "  ⚠️ LOW CONFIDENCE" if conf < 0.6 else ""
        print(f"  bullet  conf={conf:.3f}  center=({cx:.1f}, {cy:.1f}){flag}")

    # ── draw everything on the image ──
    img = cv2.imread(image_path)

    # black circle in green
    cv2.circle(img, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)
    cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 255, 0), -1)  # center dot

    # bullets in blue, with confidence label
    for box, conf, cx, cy in deduped:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "annotated_" + os.path.basename(image_path))
    cv2.imwrite(save_path, img)

    print(f"\n💾 Annotated image saved to: {save_path}")
    print(f"   Green circle = detected black circle (center + scale reference)")
    print(f"   Blue boxes = bullet holes after filtering + deduplication\n")


if __name__ == "__main__":
    main()