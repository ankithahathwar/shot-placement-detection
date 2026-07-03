"""
inference.py

Runs the trained YOLOv8 model on a new target image and outputs bullet hole
coordinates in mm, relative to the bullseye center — same format as the
manually collected ScattDB txt files.

Pipeline:
    1. Detect bullets, bullseye, outerring in the image
    2. Filter out low-confidence detections
    3. Deduplicate near-identical bullet detections (same physical hole
       detected twice) using center-distance, not IoU — safer for tight
       shot groups where genuinely separate holes can be close together
    4. Derive center from outerring (more reliable than bullseye class)
    5. Convert pixel coordinates to mm using outerring's known real diameter
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

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
OUTER_RING_DIAMETER_MM = 59.5   # confirmed: outerring = black circle (7-10 ring boundary)

CLASS_BULLET    = 0   # matches your training: 0 = bullets
CLASS_BULLSEYE  = 1   # 1 = bullseye
CLASS_OUTERRING = 2   # 2 = outerring

CONFIDENCE_THRESHOLD  = 0.5   # discard detections below this confidence
DUPLICATE_DISTANCE_PX = 8     # bullet centers closer than this = same hole, keep higher-conf one
# ────────────────────────────────────────────────────────────────────────────


def get_box_center_and_size(box):
    """Return (center_x, center_y, width, height) in pixels for a detection box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


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
    Run detection on one image, derive center + scale, convert bullet
    holes to mm coordinates. Returns list of (x_mm, y_mm) tuples, or None
    if no outerring was detected (can't establish scale/center).
    """
    results = model(image_path, verbose=False)  # default IoU — dedup handled separately below
    result = results[0]
    boxes = result.boxes

    bullet_candidates = []
    outerring_boxes = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD:
            continue

        if cls == CLASS_BULLET:
            cx, cy, _, _ = get_box_center_and_size(box)
            bullet_candidates.append((box, conf, cx, cy))
        elif cls == CLASS_OUTERRING:
            outerring_boxes.append(box)

    if not outerring_boxes:
        print(f"  ⚠️  No outerring detected in {os.path.basename(image_path)} — skipping")
        return None

    if len(outerring_boxes) > 1:
        outerring_boxes.sort(
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
            reverse=True
        )
        print(f"  ℹ️  {len(outerring_boxes)} outerring detections found, using largest")

    best_ring = outerring_boxes[0]
    center_x, center_y, ring_w, ring_h = get_box_center_and_size(best_ring)

    ring_diameter_px = (ring_w + ring_h) / 2
    pixels_per_mm = ring_diameter_px / OUTER_RING_DIAMETER_MM

    print(f"  Center: ({center_x:.1f}, {center_y:.1f}) px | "
          f"Ring diameter: {ring_diameter_px:.1f} px | "
          f"Scale: {pixels_per_mm:.3f} px/mm")

    bullet_boxes = deduplicate_by_distance(bullet_candidates)
    if len(bullet_candidates) != len(bullet_boxes):
        print(f"  🔁 Removed {len(bullet_candidates) - len(bullet_boxes)} duplicate "
              f"detection(s) (centers within {DUPLICATE_DISTANCE_PX}px)")

    coords_mm = []
    for hole_box in bullet_boxes:
        hole_x, hole_y, _, _ = get_box_center_and_size(hole_box)

        rel_x_px = hole_x - center_x
        rel_y_px = hole_y - center_y

        x_mm = rel_x_px / pixels_per_mm
        y_mm = rel_y_px / pixels_per_mm

        coords_mm.append((x_mm, y_mm))

    print(f"  ✅ {len(coords_mm)} bullet holes converted to mm coordinates")
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
        print("\n❌ Could not process image — no outerring detected.")
        sys.exit(1)

    save_coords(coords_mm, output_path)
    print()


if __name__ == "__main__":
    main()