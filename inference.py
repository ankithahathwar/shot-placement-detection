"""
inference.py

Runs the trained YOLOv8 model on a new target image and outputs bullet hole
coordinates in mm, relative to the target center — same format as the
manually collected ScattDB txt files.

Pipeline:
    1. Detect bullet holes using YOLOv8 (this class trained well — 0.95 mAP50)
    2. Find the black circle (rings 7-10) directly via classical CV contour
       detection — more reliable than the YOLO 'outerring' class, which
       suffered from inconsistent annotation sizing during training
    3. Filter out low-confidence bullet detections
    4. Deduplicate near-identical bullet detections (same physical hole
       detected twice) using center-distance
    5. Convert pixel coordinates to mm using the black circle's known
       real diameter (59.5mm, ISSF 10m air pistol spec)
    6. Save results as a txt file: one "x_mm, y_mm" line per bullet hole

Usage:
    python inference.py <image_path> <output_txt_path>

Example:
    python inference.py test_images/target_01.jpg output/target_01.txt
"""

import sys
import os
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import cv2

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
BLACK_CIRCLE_DIAMETER_MM = 59.5   # ISSF 10m air pistol — rings 7-10 boundary

CLASS_BULLET   = 0   # matches your training: 0 = bullets
CLASS_BULLSEYE = 1   # 1 = bullseye (unused — unreliable, ignored)
CLASS_OUTERRING = 2  # 2 = outerring (unused — unreliable, replaced by CV below)

CONFIDENCE_THRESHOLD  = 0.5   # discard bullet detections below this confidence
DUPLICATE_DISTANCE_PX = 8     # bullet centers closer than this = same hole, keep higher-conf one
DUPLICATE_DISTANCE_MM = 1.5
BLACK_THRESHOLD = 60   # pixel intensity below this = considered "black" for circle detection
# ────────────────────────────────────────────────────────────────────────────


def get_box_center_and_size(box):
    """Return (center_x, center_y, width, height) in pixels for a detection box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def find_black_circle(image_path):
    """
    Find the black circle (rings 7-10) directly from pixel data using
    contour detection. Returns (center_x, center_y, diameter_px) or None
    if no suitable circle was found.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, thresh = cv2.threshold(img, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # the black circle should be the largest black blob in the image
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)

    return cx, cy, radius * 2


def deduplicate_by_distance(bullet_candidates, min_distance_px=DUPLICATE_DISTANCE_PX):
    """
    Remove near-duplicate bullet detections. If two detected centers are
    closer than min_distance_px, they're almost certainly the same physical
    hole detected twice — keep only the higher-confidence one.

    bullet_candidates: list of (box, conf, cx, cy) tuples
    Returns: list of box objects (deduplicated)
    """
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

    return [item[0] for item in kept]


def process_image(model, image_path):
    """
    Run bullet detection + classical center/scale detection on one image.
    Returns list of (x_mm, y_mm) tuples, or None if the black circle
    couldn't be found (can't establish scale/center).
    """
    # ── Step 1: find center + scale using classical CV, NOT the YOLO outerring class ──
    circle_result = find_black_circle(image_path)
    if circle_result is None:
        print(f"  ⚠️  Could not find black circle in {os.path.basename(image_path)} — skipping")
        return None

    center_x, center_y, diameter_px = circle_result
    pixels_per_mm = diameter_px / BLACK_CIRCLE_DIAMETER_MM

    duplicate_distance_px = DUPLICATE_DISTANCE_MM * pixels_per_mm

    print(f"  Center: ({center_x:.1f}, {center_y:.1f}) px | "
          f"Black circle diameter: {diameter_px:.1f} px | "
          f"Scale: {pixels_per_mm:.3f} px/mm")

    # ── Step 2: run YOLO just for bullet hole detection ──
    results = model(image_path, verbose=False)
    result = results[0]
    boxes = result.boxes

    bullet_candidates = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != CLASS_BULLET:
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue

        cx, cy, _, _ = get_box_center_and_size(box)
        bullet_candidates.append((box, conf, cx, cy))

    # ── Step 3: deduplicate near-identical bullet detections ──
    bullet_boxes = deduplicate_by_distance(bullet_candidates, min_distance_px=duplicate_distance_px)
    if len(bullet_candidates) != len(bullet_boxes):
        print(f"  🔁 Removed {len(bullet_candidates) - len(bullet_boxes)} duplicate "
              f"detection(s) (centers within {duplicate_distance_px:.1f}px)")

    # ── Step 4: convert to mm, relative to black circle center ──
    coords_mm = []
    for hole_box in bullet_boxes:
        hole_x, hole_y, _, _ = get_box_center_and_size(hole_box)

        rel_x_px = hole_x - center_x
        rel_y_px = hole_y - center_y

        x_mm = rel_x_px / pixels_per_mm
        y_mm = rel_y_px / pixels_per_mm

        coords_mm.append((x_mm, y_mm))

    print(f"  ✅ {len(coords_mm)} bullet holes converted to mm coordinates")

    EXPECTED_SHOTS = 10
    if len(coords_mm) != EXPECTED_SHOTS:
        # pyrefly: ignore [parse-error]
        print(f"  ⚠️  Expected {EXPECTED_SHOTS} shots, got {len(coords_mm)} — FLAG FOR MANUAL REVIEW")
    return coords_mm


def save_coords(coords_mm, output_path):
    """Save coordinates as txt file, one 'x_mm, y_mm' per line."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for x_mm, y_mm in coords_mm:
            f.write(f"{x_mm:.2f}, {y_mm:.2f}\n")
    print(f"  💾 Saved to {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <output_txt_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model weights not found: {MODEL_PATH}")
        print(f"   Make sure best.pt is in the same folder as this script.")
        sys.exit(1)

    print(f"\n📂 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"🎯 Processing: {image_path}\n")
    coords_mm = process_image(model, image_path)

    if coords_mm is None:
        print("\n❌ Could not process image.")
        sys.exit(1)

    save_coords(coords_mm, output_path)
    print()


if __name__ == "__main__":
    main()