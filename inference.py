"""
inference.py

Runs the trained YOLOv8 model on a new target image and outputs bullet hole
coordinates in mm, relative to the target center — same format as the
manually collected ScattDB txt files.

Pipeline:
    1. Find an approximate CENTER + rough radius of the black circle using
       contour + circularity filtering
    2. Refine the true RADIUS via radial profile scanning: sample pixel
       intensity along many rays from the center, apply a median filter to
       each ray's profile (smooths away anything thinner than the filter
       window — thin internal white ring lines, thin external black ring
       lines — while preserving the true sustained edge), then find where
       the smoothed profile transitions from black to white for good.
       Filter window size scales with the detected target size, so this
       works correctly regardless of image resolution.
    3. Detect bullet holes using YOLOv8
    4. Filter out low-confidence bullet detections
    5. Deduplicate near-identical bullet detections using a real-world mm
       distance (not raw pixels)
    6. Convert pixel coordinates to mm using the black circle's known
       real diameter (59.5mm, ISSF 10m air pistol spec)
    7. Save results as a txt file: one "x_mm, y_mm" line per bullet hole

Usage:
    python inference.py <image_path> <output_txt_path>
"""

import sys
import os
import math
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
BLACK_CIRCLE_DIAMETER_MM = 59.5   # ISSF 10m air pistol — rings 7-10 boundary

CLASS_BULLET = 0

CONFIDENCE_THRESHOLD  = 0.5
DUPLICATE_DISTANCE_MM = 1.5
EXPECTED_SHOTS = 10

BLACK_THRESHOLD = 60
MIN_CIRCLE_AREA_FRACTION = 0.01
MAX_CIRCLE_AREA_FRACTION = 0.85
MIN_CIRCULARITY = 0.6

NUM_RAYS = 360
MAX_SEARCH_RADIUS = 400
FILTER_WINDOW_FRACTION = 0.10   # median filter window as a fraction of seed
                                   # radius — smooths away thin lines while
                                   # preserving the true sustained edge;
                                   # scales with image resolution
MIN_FILTER_WINDOW = 5             # minimum window size (must end up odd)
# ────────────────────────────────────────────────────────────────────────────


def get_box_center_and_size(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def find_seed_center(gray_img):
    """
    Find an approximate center AND rough radius of the black circle using
    contour filtering. Used as a starting point for the radial scan below.

    KEY INSIGHT: background clutter (carpet, shadows, table edges) almost
    always touches the edge of the photo frame, since it extends beyond
    where the photographer aimed the camera. The actual target, properly
    framed, sits with margin on all sides. This is a more reliable signal
    than circularity — circularity degrades badly when a large overlapping
    bullet-hole cluster bites a notch out of the target's outline, but
    "does this blob touch the image border" doesn't care about that at all.
    """
    h, w = gray_img.shape[:2]
    image_area = h * w
    BORDER_MARGIN = 3   # pixels — small tolerance for edge-touching check

    _, thresh = cv2.threshold(gray_img, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CIRCLE_AREA_FRACTION * image_area:
            continue
        if area > MAX_CIRCLE_AREA_FRACTION * image_area:
            continue

        # reject anything touching the image border — background clutter
        bx, by, bw, bh = cv2.boundingRect(c)
        touches_border = (
            bx <= BORDER_MARGIN or by <= BORDER_MARGIN or
            (bx + bw) >= (w - BORDER_MARGIN) or (by + bh) >= (h - BORDER_MARGIN)
        )
        if touches_border:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        candidates.append((area, cx, cy, radius))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)  # largest surviving blob
    _, cx, cy, radius = candidates[0]
    return cx, cy, radius


def sample_ray_profile(gray_img, cx, cy, angle_deg, max_radius):
    """Sample raw pixel intensities along a ray from (cx, cy) outward."""
    h, w = gray_img.shape[:2]
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    profile = []
    for r in range(max_radius):
        x = int(cx + dx * r)
        y = int(cy + dy * r)
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        profile.append(gray_img[y, x])

    return np.array(profile, dtype=np.uint8)


def find_edge_from_profile(profile, filter_window):
    """
    Apply a median filter to the ray's intensity profile, then find the
    outermost point that's still "black" after smoothing. The median
    filter naturally erases anything thinner than filter_window (thin
    ring lines, whether black-on-white or white-on-black), leaving only
    the true sustained black/white regions intact.
    """
    if len(profile) < filter_window:
        return None

    ksize = filter_window if filter_window % 2 == 1 else filter_window + 1
    smoothed = cv2.medianBlur(profile.reshape(1, -1), ksize)[0]

    black_indices = np.where(smoothed < BLACK_THRESHOLD)[0]
    if len(black_indices) == 0:
        return None

    return int(black_indices[-1])


def find_target_circle(image_path):
    """
    Full two-step target circle detection:
        1. Find a seed center + rough radius via contour + circularity
        2. Refine the true radius via median-filtered radial scanning
           (robust to internal AND external thin ring lines)

    Returns (center_x, center_y, diameter_px) or None if detection failed.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    seed = find_seed_center(gray)
    if seed is None:
        return None
    cx, cy, seed_radius = seed

    filter_window = max(MIN_FILTER_WINDOW, int(seed_radius * FILTER_WINDOW_FRACTION))

    edge_distances = []
    for i in range(NUM_RAYS):
        angle = (360.0 / NUM_RAYS) * i
        profile = sample_ray_profile(gray, cx, cy, angle, MAX_SEARCH_RADIUS)
        dist = find_edge_from_profile(profile, filter_window)
        if dist is not None:
            edge_distances.append(dist)

    if not edge_distances:
        return None

    median_radius = float(np.median(edge_distances))
    return cx, cy, median_radius * 2


def deduplicate_by_distance(bullet_candidates, min_distance_px):
    """
    Remove near-duplicate bullet detections. Returns (kept_boxes, removed_info)
    where removed_info is a list of (removed_conf, removed_cx, removed_cy,
    kept_cx, kept_cy, distance_px) for anything that got merged — useful
    for diagnosing whether dedup is behaving correctly.
    """
    bullet_candidates = sorted(bullet_candidates, key=lambda b: b[1], reverse=True)
    kept = []
    removed_info = []

    for box, conf, cx, cy in bullet_candidates:
        is_duplicate = False
        for _, kconf, kx, ky in kept:
            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            if dist < min_distance_px:
                is_duplicate = True
                removed_info.append((conf, cx, cy, kx, ky, dist))
                break
        if not is_duplicate:
            kept.append((box, conf, cx, cy))

    return [item[0] for item in kept], removed_info


def process_image(model, image_path):
    circle_result = find_target_circle(image_path)
    if circle_result is None:
        print(f"  ⚠️  Could not find target circle in {os.path.basename(image_path)} — skipping")
        return None

    center_x, center_y, diameter_px = circle_result
    pixels_per_mm = diameter_px / BLACK_CIRCLE_DIAMETER_MM

    print(f"  Center: ({center_x:.1f}, {center_y:.1f}) px | "
          f"Diameter: {diameter_px:.1f} px | "
          f"Scale: {pixels_per_mm:.3f} px/mm")

    results = model(image_path, verbose=False)
    result = results[0]
    boxes = result.boxes

    bullet_candidates = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != CLASS_BULLET or conf < CONFIDENCE_THRESHOLD:
            continue

        cx, cy, _, _ = get_box_center_and_size(box)
        bullet_candidates.append((box, conf, cx, cy))

    duplicate_distance_px = DUPLICATE_DISTANCE_MM * pixels_per_mm
    bullet_boxes, removed_info = deduplicate_by_distance(bullet_candidates, min_distance_px=duplicate_distance_px)

    if removed_info:
        print(f"  🔁 Removed {len(removed_info)} duplicate detection(s) "
              f"(threshold: {DUPLICATE_DISTANCE_MM}mm = {duplicate_distance_px:.1f}px):")
        for conf, cx, cy, kx, ky, dist_px in removed_info:
            dist_mm = dist_px / pixels_per_mm
            print(f"      removed conf={conf:.3f} at ({cx:.1f},{cy:.1f}) — "
                  f"{dist_px:.1f}px ({dist_mm:.2f}mm) from kept hole at ({kx:.1f},{ky:.1f})")

    coords_mm = []
    for hole_box in bullet_boxes:
        hole_x, hole_y, _, _ = get_box_center_and_size(hole_box)

        rel_x_px = hole_x - center_x
        rel_y_px = hole_y - center_y

        x_mm = rel_x_px / pixels_per_mm
        y_mm = rel_y_px / pixels_per_mm

        coords_mm.append((x_mm, y_mm))

    print(f"  ✅ {len(coords_mm)} bullet holes converted to mm coordinates")

    if len(coords_mm) != EXPECTED_SHOTS:
        print(f"  ⚠️  Expected {EXPECTED_SHOTS} shots, got {len(coords_mm)} — FLAG FOR MANUAL REVIEW")

    return coords_mm


def save_coords(coords_mm, output_path):
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