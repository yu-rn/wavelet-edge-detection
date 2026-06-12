from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_intermediate_images(
    image_stem: str,
    intermediate: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, value in intermediate.items():
        array = value
        if array.ndim != 2:
            continue

        if array.dtype != np.uint8:
            display = normalize_for_display(array)
        else:
            display = array

        cv2.imwrite(str(output_dir / f"{image_stem}_{key}.png"), display)


def save_comparison_figure(
    image_stem: str,
    original: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Predicted Edge")
    axes[2].imshow(ground_truth, cmap="gray")
    axes[2].set_title("Ground Truth")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_error_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_mask = prediction > 0
    gt_mask = ground_truth > 0

    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    true_positive = pred_mask & gt_mask
    false_positive = pred_mask & ~gt_mask
    false_negative = ~pred_mask & gt_mask

    error_map[true_positive] = (255, 255, 255)
    error_map[false_positive] = (255, 0, 0)
    error_map[false_negative] = (0, 0, 255)

    cv2.imwrite(str(output_path), cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)
