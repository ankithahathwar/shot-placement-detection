"""
debug_radial_edge.py

More robust way to find the TRUE radius of the black circle: instead of
relying on one single contour shape (which can be thrown off by bullet
holes, printed numbers, or lighting), this casts rays outward from a
known center point at many angles, and finds where each ray transitions
from black to white. Taking the MEDIAN across all angles is robust to a
few bad rays (e.g. one that happens to cross a printed number right at
the boundary).

Usage:
    python debug_radial_edge.py <image_path> <center_x> <center_y>

Example (using the center already found by the contour method):
    python debug_radial_edge.py test_images/real.jpeg 549 935
"""

import sys
import os
import math
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

BLACK_THRESHOLD = 60
NUM_RAYS = 360          # one ray per degree
MAX_SEARCH_RADIUS = 400 # don't search further than this from center (pixels)


def find_edge_along_ray(img, cx, cy, angle_deg, max_radius):
    """
    Walk outward from (cx, cy) at the given angle for the FULL max_radius,
    and return the FURTHEST distance at which the pixel was still "black".

    This deliberately does NOT stop at the first black-to-white transition,
    because the target has thin white ring lines printed INSIDE the black
    area (marking the 8/9/10 ring boundaries) — stopping early would lock
    onto one of those instead of the true outer edge. By scanning the whole
    ray and taking the outermost black pixel, we correctly skip past those
    internal gaps and land on the real boundary where black ends for good.
    """
    h, w = img.shape[:2]
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    last_black_r = None
    for r in range(0, max_radius):
        x = int(cx + dx * r)
        y = int(cy + dy * r)
        if x < 0 or x >= w or y < 0 or y >= h:
            break

        pixel = img[y, x]
        if pixel < BLACK_THRESHOLD:
            last_black_r = r

    return last_black_r  # furthest black pixel found, or None if none at all


def main():
    if len(sys.argv) < 4:
        print("Usage: python debug_radial_edge.py <image_path> <center_x> <center_y>")
        sys.exit(1)

    image_path = sys.argv[1]
    cx = float(sys.argv[2])
    cy = float(sys.argv[3])

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image")
        sys.exit(1)

    print(f"\n📐 Scanning {NUM_RAYS} rays from center ({cx}, {cy})...\n")

    edge_distances = []
    for i in range(NUM_RAYS):
        angle = (360.0 / NUM_RAYS) * i
        dist = find_edge_along_ray(img, cx, cy, angle, MAX_SEARCH_RADIUS)
        if dist is not None:
            edge_distances.append(dist)

    if not edge_distances:
        print("❌ No edges found at all — check center coordinates or BLACK_THRESHOLD")
        return

    edge_distances = np.array(edge_distances)
    median_r = np.median(edge_distances)
    mean_r = np.mean(edge_distances)
    std_r = np.std(edge_distances)
    min_r = np.min(edge_distances)
    max_r = np.max(edge_distances)

    print(f"Rays that found an edge: {len(edge_distances)} / {NUM_RAYS}")
    print(f"  Median radius: {median_r:.1f}px  (diameter: {median_r*2:.1f}px)")
    print(f"  Mean radius:   {mean_r:.1f}px")
    print(f"  Std dev:       {std_r:.1f}px  (high = inconsistent edge, some rays hit noise)")
    print(f"  Range:         {min_r:.1f}px – {max_r:.1f}px")

    # draw the result for visual check
    color_img = cv2.imread(image_path)
    cv2.circle(color_img, (int(cx), int(cy)), int(median_r), (0, 255, 0), 3)
    cv2.circle(color_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    os.makedirs("debug_output", exist_ok=True)
    save_path = os.path.join("debug_output", "radial_" + os.path.basename(image_path))
    cv2.imwrite(save_path, color_img)
    print(f"\n💾 Saved visualization to: {save_path}\n")


if __name__ == "__main__":
    main()