"""
debug_circle_detection.py

Shows EVERY dark blob found in the image, along with its area fraction
and circularity score — even ones that get rejected by the current
thresholds. This tells us exactly why detection is failing, instead of
guessing at threshold values blindly.

Usage:
    python debug_circle_detection.py <image_path>
"""

import sys
import os
import math
# pyrefly: ignore [missing-import]
import cv2

BLACK_THRESHOLD = 60


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_circle_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image")
        sys.exit(1)

    h, w = img.shape[:2]
    image_area = h * w
    print(f"\n📐 Image size: {w}x{h}  (area: {image_area})")

    _, thresh = cv2.threshold(img, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔍 Found {len(contours)} total dark contours\n")

    if not contours:
        print("No contours found at all — try lowering BLACK_THRESHOLD (currently 60)")
        return

    results = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 10:  # skip tiny noise, not worth printing
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        area_fraction = area / image_area

        (cx, cy), radius = cv2.minEnclosingCircle(c)

        results.append({
            "area": area,
            "area_fraction": area_fraction,
            "circularity": circularity,
            "cx": cx, "cy": cy, "radius": radius
        })

    # sort by area, largest first — target circle is usually one of the biggest
    results.sort(key=lambda r: r["area"], reverse=True)

    print(f"{'AREA':<12} {'AREA %':<10} {'CIRCULARITY':<13} {'CENTER':<20} {'RADIUS'}")
    print("-" * 75)
    for r in results[:20]:  # top 20 largest blobs
        print(f"{r['area']:<12.0f} {r['area_fraction']*100:<10.2f} "
              f"{r['circularity']:<13.3f} ({r['cx']:.0f}, {r['cy']:.0f}){'':<10} {r['radius']:.1f}")

    print(f"\n💡 The real target black circle should be a large blob (several % of image)")
    print(f"   with circularity reasonably close to 1.0 (perfect circle = 1.0).")
    print(f"   If the real target's circularity is well below 0.7, that's why it's")
    print(f"   being rejected — likely due to camera angle/perspective distortion,")
    print(f"   or the dark region being broken up by glare/lighting.\n")


if __name__ == "__main__":
    main()
