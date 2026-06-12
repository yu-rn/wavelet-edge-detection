from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .wavelet_edge import load_grayscale_image, normalize_to_uint8


def threshold_response(
    response: np.ndarray,
    threshold_method: str = "otsu",
    percentile: float = 95.0,
) -> np.ndarray:
    normalized = normalize_to_uint8(response)
    if threshold_method == "percentile":
        threshold = float(np.percentile(normalized, np.clip(percentile, 0.0, 100.0)))
        return (normalized >= threshold).astype(np.uint8) * 255

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def sobel_edges(
    image_source: str | Path | np.ndarray,
    threshold_method: str = "otsu",
    percentile: float = 95.0,
) -> dict[str, np.ndarray]:
    gray = load_grayscale_image(image_source)
    gray_float = gray.astype(np.float32)
    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    binary = threshold_response(magnitude, threshold_method=threshold_method, percentile=percentile)
    return {
        "gray_image": gray,
        "edge_response": magnitude,
        "normalized_edge_response": normalize_to_uint8(magnitude),
        "binary_edge_map": binary,
    }


def canny_edges(
    image_source: str | Path | np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> dict[str, np.ndarray]:
    gray = load_grayscale_image(image_source)
    binary = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)
    return {
        "gray_image": gray,
        "binary_edge_map": binary,
    }
