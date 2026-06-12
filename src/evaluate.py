from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def gt_path_from_image_name(image_name: str, gt_dir: Path) -> Path:
    stem = Path(image_name).stem
    return gt_dir / f"{stem}_gt_binary.png"


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def compute_precision_recall_f1(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_radius: int = 0,
) -> dict[str, float]:
    prediction = (prediction > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    if tolerance_radius > 0:
        kernel_size = 2 * tolerance_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_ground_truth = cv2.dilate(ground_truth, kernel)
        dilated_prediction = cv2.dilate(prediction, kernel)

        true_positive = int(np.sum((prediction == 1) & (dilated_ground_truth == 1)))
        false_positive = int(np.sum((prediction == 1) & (dilated_ground_truth == 0)))
        false_negative = int(np.sum((ground_truth == 1) & (dilated_prediction == 0)))
    else:
        true_positive = int(np.sum((prediction == 1) & (ground_truth == 1)))
        false_positive = int(np.sum((prediction == 1) & (ground_truth == 0)))
        false_negative = int(np.sum((prediction == 0) & (ground_truth == 1)))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tolerance_radius": float(tolerance_radius),
    }


def save_metrics_table(records: list[dict[str, float | str]], output_csv: Path) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if not frame.empty:
        mean_row = {
            "image_name": "mean",
            "precision": frame["precision"].mean(),
            "recall": frame["recall"].mean(),
            "f1_score": frame["f1_score"].mean(),
            "tolerance_radius": frame["tolerance_radius"].iloc[0] if "tolerance_radius" in frame else 0.0,
        }
        frame = pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
