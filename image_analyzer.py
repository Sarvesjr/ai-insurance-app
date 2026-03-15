import cv2
import numpy as np

def analyze_damage(image_path):

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return {"severity": "Unknown", "damage_ratio": 0}

    # Resize for consistency
    image = cv2.resize(image, (600, 400))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to remove noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect edges (possible damage lines)
    edges = cv2.Canny(blur, 50, 150)

    # Dilate edges to highlight damage areas
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Calculate damage ratio
    damage_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    damage_ratio = damage_pixels / total_pixels

    # Classify severity
    if damage_ratio < 0.02:
        severity = "Minor Dent"
    elif damage_ratio < 0.07:
        severity = "Moderate Damage"
    else:
        severity = "Severe Accident"

    return {
        "severity": severity,
        "damage_ratio": round(damage_ratio, 4)
    }