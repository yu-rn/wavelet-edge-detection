from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import compute_precision_recall_f1, gt_path_from_image_name, load_binary_mask
from .wavelet_edge import detect_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a compact set of wavelet edge detection parameters.")
    parser.add_argument("--image-dir", type=Path, default=Path("samples/images"))
    parser.add_argument("--gt-dir", type=Path, default=Path("samples/groundTruth"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/metrics/parameter_search.csv"))
    parser.add_argument("--tolerance-radius", type=int, default=1)
    parser.add_argument("--max-configs", type=int, default=0)
    return parser.parse_args()


def candidate_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    percentiles = [92.0, 94.0, 95.0, 96.0, 97.0]
    response_modes = ["weighted", "coarse_enhanced", "multiscale_consistency"]
    texture_options = [
        {"use_texture_suppression": False, "texture_suppression_mode": "linear", "texture_weight": 0.0},
        {"use_texture_suppression": True, "texture_suppression_mode": "linear", "texture_weight": 0.15},
        {"use_texture_suppression": True, "texture_suppression_mode": "linear", "texture_weight": 0.25},
        {"use_texture_suppression": True, "texture_suppression_mode": "exp", "texture_strength": 0.8},
        {"use_texture_suppression": True, "texture_suppression_mode": "exp", "texture_strength": 1.0},
        {"use_texture_suppression": True, "texture_suppression_mode": "exp", "texture_strength": 1.5},
    ]
    min_skeleton_lengths = [0, 3, 5, 8, 12]
    min_component_mean_responses = [0.0, 5.0, 10.0, 15.0]
    level_weight_options = {
        1: [None],
        2: [None, [0.7, 0.3], [0.5, 0.5], [0.3, 0.7]],
    }
    tv_options = [
        {"use_tv_denoise": False, "tv_weight": 0.08},
        {"use_tv_denoise": True, "tv_weight": 0.05},
    ]

    def add_config(config: dict[str, object]) -> None:
        defaults: dict[str, object] = {
            "use_gradient_assist": True,
            "gradient_weight": 0.15,
            "gradient_mode": "multiply",
            "use_nms": True,
            "use_morphological_closing": False,
            "use_thinning": True,
            "min_object_size": 0,
            "use_endpoint_linking": False,
            "adaptive_use_hysteresis": True,
            "adaptive_low_ratio": 0.7,
            "use_texture_suppression": False,
            "texture_window_size": 15,
            "texture_weight": 0.0,
            "texture_suppression_mode": "linear",
            "texture_strength": 3.0,
            "texture_gamma": 2.0,
            "use_tv_denoise": False,
            "tv_weight": 0.08,
            "min_skeleton_length": 0,
            "min_component_mean_response": 0.0,
        }
        defaults.update(config)
        configs.append(defaults)

    for transform_mode in ["dwt", "swt"]:
        for wavelet in ["haar", "db2"]:
            for level in [1, 2]:
                for level_weights in level_weight_options[level]:
                    for response_mode in response_modes:
                        for percentile in percentiles:
                            for texture_option in texture_options:
                                for min_skeleton_length in min_skeleton_lengths:
                                    for min_component_mean_response in min_component_mean_responses:
                                        for tv_option in tv_options:
                                            add_config(
                                                {
                                                    "transform_mode": transform_mode,
                                                    "wavelet": wavelet,
                                                    "level": level,
                                                    "threshold_method": "percentile",
                                                    "percentile": percentile,
                                                    "low_threshold_ratio": 0.5,
                                                    "response_mode": response_mode,
                                                    "level_weights": level_weights,
                                                    "min_skeleton_length": min_skeleton_length,
                                                    "min_component_mean_response": min_component_mean_response,
                                                    **texture_option,
                                                    **tv_option,
                                                }
                                            )
    return configs


def evaluate_config(
    config: dict[str, object],
    image_paths: list[Path],
    gt_dir: Path,
    tolerance_radius: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metrics = []
    per_image_records = []
    for image_path in image_paths:
        result = detect_edges(image_path, **config)
        gt = load_binary_mask(gt_path_from_image_name(image_path.name, gt_dir))
        prediction = result["binary_edge_map"] > 0
        gt_mask = gt > 0
        edge_density = float(np.mean(prediction))
        gt_density = float(np.mean(gt_mask))
        density_ratio = edge_density / (gt_density + 1e-8)
        metric = compute_precision_recall_f1(
            result["binary_edge_map"],
            gt,
            tolerance_radius=tolerance_radius,
        )
        metrics.append(metric)
        per_image_records.append(
            {
                **config,
                "image_name": image_path.name,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1_score": metric["f1_score"],
                "edge_density": edge_density,
                "gt_density": gt_density,
                "density_ratio": density_ratio,
                "balanced_score": metric["f1_score"] - 0.15 * max(0.0, density_ratio - 2.5),
                "tolerance_radius": tolerance_radius,
            }
        )

    edge_density = sum(item["edge_density"] for item in per_image_records) / len(per_image_records)
    gt_density = sum(item["gt_density"] for item in per_image_records) / len(per_image_records)
    density_ratio = edge_density / (gt_density + 1e-8)
    f1_score = sum(item["f1_score"] for item in metrics) / len(metrics)
    aggregate = {
        **config,
        "precision": sum(item["precision"] for item in metrics) / len(metrics),
        "recall": sum(item["recall"] for item in metrics) / len(metrics),
        "f1_score": f1_score,
        "edge_density": edge_density,
        "gt_density": gt_density,
        "density_ratio": density_ratio,
        "balanced_score": f1_score - 0.15 * max(0.0, density_ratio - 2.5),
        "tolerance_radius": tolerance_radius,
    }
    return aggregate, per_image_records


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.image_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    configs = candidate_configs()
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    records = []
    per_image_records = []
    for config in configs:
        aggregate, per_image = evaluate_config(config, image_paths, args.gt_dir, args.tolerance_radius)
        records.append(aggregate)
        per_image_records.extend(per_image)

    frame = pd.DataFrame(records).sort_values("balanced_score", ascending=False)
    per_image_frame = pd.DataFrame(per_image_records)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    per_image_output = args.output_csv.with_name(f"{args.output_csv.stem}_per_image.csv")
    per_image_frame.to_csv(per_image_output, index=False)
    columns = [
        "transform_mode",
        "wavelet",
        "level",
        "threshold_method",
        "percentile",
        "response_mode",
        "texture_suppression_mode",
        "texture_strength",
        "min_skeleton_length",
        "min_component_mean_response",
        "precision",
        "recall",
        "f1_score",
        "edge_density",
        "density_ratio",
        "balanced_score",
    ]
    print(frame.head(20)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
