"""
visualize_detections.py

Draws the detection results on the image so you can visually verify:
  - the target circle center + true radius (found via seed-center contour
    detection + median-filtered radial scanning, drawn in green) — robust
    against real-photo issues (backgrounds, internal AND external thin
    ring lines, perspective distortion) and scales correctly across
    different image resolutions
  - each bullet hole detection AFTER confidence filtering + mm-based
    deduplication (drawn in blue, with confidence score)

This matches the same logic used in inference.py, so what you see here
is exactly what gets converted to mm coordinates.

Usage:
    python visualize_detections.py <image_path>
"""

import sys
import os
import math
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

MODEL_PATH = "best.pt"
CLASS_BULLET = 0

CONFIDENCE_THRESHOLD  = 0.5
DUPLICATE_DISTANCE_MM = 1.5
BLACK_CIRCLE_DIAMETER_MM = 59.5

BLACK_THRESHOLD = 60
MIN_CIRCLE_AREA_FRACTION = 0.01
MAX_CIRCLE_AREA_FRACTION = 0.85
MIN_CIRCULARITY = 0.6

NUM_RAYS = 360
MAX_SEARCH_RADIUS = 400
FILTER_WINDOW_FRACTION = 0.10
MIN_FILTER_WINDOW = 5


def get_box_center_and_size(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def find_seed_center(gray_img):
    """
    Find an approximate center AND rough radius of the black circle.

    Background clutter (carpet, shadows, table edges) almost always
    touches the edge of the photo frame, since it extends beyond where
    the photographer aimed the camera. The actual target, properly framed,
    sits with margin on all sides. This is far more reliable than
    circularity, which degrades badly when a large overlapping bullet-hole
    cluster bites a notch out of the target's outline.
    """
    h, w = gray_img.shape[:2]
    image_area = h * w
    BORDER_MARGIN = 3

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

        bx, by, bw, bh = cv2.boundingRect(c)
        touches_border = (
            bx <= BORDER_MARGIN or by <= BORDER_MARGIN or
            (bx + bw) >= (w - BORDER_MARGIN) or (by + bh) >= (h - BORDER_MARGIN)
        )
        if touches_border:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        candidates.append((area, cx, cy, radius))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)  # largest surviving blob
    _, cx, cy, radius = candidates[0]
    return cx, cy, radius


def sample_ray_profile(gray_img, cx, cy, angle_deg, max_radius):
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
    if len(profile) < filter_window:
        return None

    ksize = filter_window if filter_window % 2 == 1 else filter_window + 1
    smoothed = cv2.medianBlur(profile.reshape(1, -1), ksize)[0]

    black_indices = np.where(smoothed < BLACK_THRESHOLD)[0]
    if len(black_indices) == 0:
        return None

    return int(black_indices[-1])


def find_target_circle(image_path):
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
    std_radius = float(np.std(edge_distances))
    return cx, cy, median_radius, std_radius, filter_window


def deduplicate_by_distance(bullet_candidates, min_distance_px):
    """
    Returns (kept, removed_info) where removed_info holds details of any
    detection dropped for being too close to an already-kept one — lets
    us print AND draw exactly what got merged for visual inspection.
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
                removed_info.append((box, conf, cx, cy, kx, ky, dist))
                break
        if not is_duplicate:
            kept.append((box, conf, cx, cy))
    return kept, removed_info


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

    circle_result = find_target_circle(image_path)
    if circle_result is None:
        print("❌ Could not find target circle")
        sys.exit(1)
    center_x, center_y, radius, std_radius, filter_window = circle_result
    pixels_per_mm = (radius * 2) / BLACK_CIRCLE_DIAMETER_MM

    print(f"Target circle — center: ({center_x:.1f}, {center_y:.1f})  "
          f"radius: {radius:.1f}px  diameter: {radius*2:.1f}px")
    print(f"  (std dev across rays: {std_radius:.1f}px, "
          f"median filter window used: {filter_window}px)")

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

    duplicate_distance_px = DUPLICATE_DISTANCE_MM * pixels_per_mm
    deduped, removed_info = deduplicate_by_distance(bullet_candidates, min_distance_px=duplicate_distance_px)

    print(f"Bullets — {len(bullet_candidates)} candidates → {len(deduped)} after dedup "
          f"(threshold: {DUPLICATE_DISTANCE_MM}mm = {duplicate_distance_px:.1f}px)\n")

    for _, conf, cx, cy in sorted(deduped, key=lambda d: d[1]):
        flag = "  ⚠️ LOW CONFIDENCE" if conf < 0.6 else ""
        print(f"  bullet  conf={conf:.3f}  center=({cx:.1f}, {cy:.1f}){flag}")

    if removed_info:
        print(f"\n🔁 Removed as duplicates:")
        for box, conf, cx, cy, kx, ky, dist_px in removed_info:
            dist_mm = dist_px / pixels_per_mm
            print(f"  conf={conf:.3f} at ({cx:.1f},{cy:.1f}) — "
                  f"{dist_px:.1f}px ({dist_mm:.2f}mm) from kept hole at ({kx:.1f},{ky:.1f})")

    img = cv2.imread(image_path)

    cv2.circle(img, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 3)
    cv2.circle(img, (int(center_x), int(center_y)), 4, (0, 255, 0), -1)

    # kept bullets in blue
    for box, conf, cx, cy in deduped:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # removed duplicates in red, so you can see exactly what got dropped
    for box, conf, cx, cy, kx, ky, dist_px in removed_info:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"removed {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # draw a line connecting removed detection to the one it was merged into
        cv2.line(img, (int(cx), int(cy)), (int(kx), int(ky)), (0, 0, 255), 1)

    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "annotated_" + os.path.basename(image_path))
    cv2.imwrite(save_path, img)

    print(f"\n💾 Annotated image saved to: {save_path}")
    print(f"   Blue = kept bullets | Red = removed as duplicates (line shows merge pair)\n")


if __name__ == "__main__":
    main()