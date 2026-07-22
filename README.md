# Snyptr PostShot: Target Detection & Bullet Hole Coordinate Converter

Snyptr PostShot is a computer vision pipeline that processes target images shot with **10m Air Pistol (ISSF Spec)** from the **ScattDB** dataset. It detects the target center, refines the black target area, detects bullet holes using a trained YOLOv8 model, and outputs the exact shot coordinates in real-world millimeters relative to the target center.

---

## Key Features
1. **Target Center Detection**: Locates the approximate target center and rough radius using contour extraction and circularity filters.
2. **Radial Edge Scan Refinement**: Refines the target boundary via median-filtered radial profile scanning (casting 360 rays from the center). This filters out thin internal ring lines (8, 9, 10 ring boundaries) and external ring lines, robustly locating the true target edge.
3. **YOLOv8 Bullet Detection**: Detects bullet holes using a custom-trained YOLOv8 model (`best.pt`).
4. **Post-Processing**:
   * Confidence thresholding to remove false positives.
   * Real-world millimeter-based deduplication to filter out near-identical duplicate boxes.
5. **Coordinate Conversion**: Maps pixel coordinates to real-world millimeters relative to the target center, using the known ISSF standard black circle diameter of **59.5mm** (covering rings 7 through 10).
6. **Outputs**: Saves coordinates as `x_mm, y_mm` lists in text files and generates visual debugging plots showing the detected circles and shots.

---

## Project Structure

```
Snyptr_PostShot/
├── best.pt                       # Trained weights of the YOLOv8 model (ignored by Git)
├── debug_circle_detection.py     # Debug tool for contour and circularity filtering
├── debug_hough_circle.py         # Debug tool for testing OpenCV Hough Circle Transform
├── debug_radial_edge.py          # Debug tool for analyzing radial ray intensity profiles
├── inference.py                  # Core pipeline CLI script
├── visualize_detections.py      # Core script to visually verify detection results
├── notebooks/
│   └── Snyptr_v8.ipynb           # Jupyter notebook used for YOLOv8 model training & validation
├── outputs/                      # Saved output text files containing mm coordinates (ignored)
├── debug_output/                 # Visual verification image files (ignored)
├── test_images/                  # Sample test target images (ignored)
├── requirements.txt              # Project dependencies
└── .gitignore                    # Git exclude configuration
```

---

## Setup & Installation

### 1. Clone the repository and navigate into it:
```bash
git clone https://github.com/ankithahathwar/Snyptr_PostShot.git
cd Snyptr_PostShot
```

### 2. Create and activate a Virtual Environment (named `bull`):
```bash
# On Windows
python -m venv bull
bull\Scripts\activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Inference Pipeline
To run the full pipeline on a target image and generate the millimeter coordinate text files:
```bash
python inference.py <path_to_target_image> <path_to_output_txt>
```
*Example:*
```bash
python inference.py test_images/real.jpeg outputs/real.txt
```

### Visual Verification
To visualize the target boundary detection (drawn in green) and bullet hole detections (drawn in blue with confidence scores):
```bash
python visualize_detections.py <path_to_target_image>
```
The visual output is saved in the `debug_output/` folder as `annotated_<image_name>`.

---

## Debugging Scripts

The repository includes three debugging/diagnostic scripts that were used to design the core pipeline algorithms:

1. **`debug_circle_detection.py`**:
   * Displays all detected dark contours, area fractions, and circularity scores.
   * Useful for diagnosing why a target circle might be rejected due to perspective distortion or lighting/glare.
   * *Usage*: `python debug_circle_detection.py <image_path>`

2. **`debug_radial_edge.py`**:
   * Simulates the 360-ray profile scan from a given center coordinate.
   * Saves a visualization showing the calculated median radius from the scan.
   * *Usage*: `python debug_radial_edge.py <image_path> <center_x> <center_y>`

3. **`debug_hough_circle.py`**:
   * Tests the OpenCV Hough Circle Transform on target images. This was explored as an alternative to contour filtering but is not used in the active pipeline.
   * *Usage*: `python debug_hough_circle.py <image_path>`
