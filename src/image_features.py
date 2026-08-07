from __future__ import annotations

from pathlib import Path
from typing import Any


POTATO_IMAGE_THRESHOLDS = {
    "min_width": 160,
    "min_height": 120,
    "min_area": 30_000,
    "min_potato_pixel_ratio": 0.08,
    "max_blank_pixel_ratio": 0.82,
}


def analyze_potato_image_relevance(image: Any, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Return whether an image likely contains potato/chip pieces."""
    import numpy as np
    from PIL import Image

    thresholds = {**POTATO_IMAGE_THRESHOLDS, **(thresholds or {})}

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    area = width * height

    if width < thresholds["min_width"] or height < thresholds["min_height"] or area < thresholds["min_area"]:
        return {
            "is_relevant": False,
            "reason": "Image is too small to be a useful potato inspection photo.",
            "width": width,
            "height": height,
            "potato_pixel_ratio": 0.0,
            "blank_pixel_ratio": 0.0,
        }

    pixels = np.asarray(rgb_image, dtype=np.uint8)
    red = pixels[:, :, 0].astype(int)
    green = pixels[:, :, 1].astype(int)
    blue = pixels[:, :, 2].astype(int)

    max_channel = pixels.max(axis=2)
    min_channel = pixels.min(axis=2)
    saturation = max_channel - min_channel
    brightness = (red + green + blue) / 3

    blank_mask = (brightness > 235) & (saturation < 18)
    dark_blank_mask = (brightness < 20) & (saturation < 18)

    tan_mask = (
        (red >= 120)
        & (red <= 245)
        & (green >= 85)
        & (green <= 225)
        & (blue >= 35)
        & (blue <= 180)
        & (red >= green - 12)
        & (green >= blue + 8)
        & (saturation >= 18)
    )
    brown_mask = (
        (red >= 70)
        & (red <= 190)
        & (green >= 45)
        & (green <= 150)
        & (blue >= 20)
        & (blue <= 110)
        & (red >= green)
        & (green >= blue)
        & (saturation >= 15)
    )

    potato_ratio = float(np.mean(tan_mask | brown_mask))
    blank_ratio = float(np.mean(blank_mask | dark_blank_mask))

    if blank_ratio > thresholds["max_blank_pixel_ratio"]:
        is_relevant = False
        reason = "Image is mostly blank or background."
    elif potato_ratio < thresholds["min_potato_pixel_ratio"]:
        is_relevant = False
        reason = "Image does not contain enough potato-colored pixels."
    else:
        is_relevant = True
        reason = "Image appears to contain potato/chip-colored regions."

    return {
        "is_relevant": is_relevant,
        "reason": reason,
        "width": width,
        "height": height,
        "potato_pixel_ratio": round(potato_ratio, 4),
        "blank_pixel_ratio": round(blank_ratio, 4),
    }


def compute_image_features(image_path: str | Path) -> dict[str, Any]:
    """Compute simple explainable image features for early MVP insight."""
    import cv2
    import numpy as np

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    avg_color = np.mean(rgb.reshape(-1, 3), axis=0)
    dark_ratio = float(np.mean(gray < 65))
    pale_ratio = float(np.mean(gray > 225))

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    brown_mask = (hue >= 5) & (hue <= 30) & (saturation > 50) & (value < 180)
    potato_mask = (saturation > 20) & (value > 60)

    brown_ratio = float(np.mean(brown_mask))
    coverage_ratio = float(np.mean(potato_mask))

    return {
        "image_file": str(image_path),
        "avg_brightness": round(brightness, 2),
        "avg_red": round(float(avg_color[0]), 2),
        "avg_green": round(float(avg_color[1]), 2),
        "avg_blue": round(float(avg_color[2]), 2),
        "dark_pixel_ratio": round(dark_ratio, 4),
        "very_pale_pixel_ratio": round(pale_ratio, 4),
        "rough_chip_area_coverage": round(coverage_ratio, 4),
        "brown_pixel_ratio": round(brown_ratio, 4),
        "possible_defect_flag": bool(dark_ratio > 0.03 or brown_ratio > 0.02),
    }
