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
    wavelets = ["haar", "db2", "sym4"]
    levels = [1, 2, 3]
    use_diagonal_options = [True, False]
    diagonal_weights = [0.0, 0.5, 1.0]
    fusion_modes = ["weighted", "max"]
    threshold_methods = ["otsu", "percentile"]
    percentiles = [92.0, 94.0, 96.0, 98.0]
    thinning_options = [True, False]
    min_skeleton_lengths = [0, 5, 10]

    def add_config(config: dict[str, object]) -> None:
        defaults: dict[str, object] = {
            "transform_mode": "dwt",
            "response_mode": "weighted",
            "use_gradient_assist": False,
            "gradient_weight": 0.2,
            "gradient_mode": "multiply",
            "use_nms": True,
            "use_morphological_closing": False,
            "use_thinning": False,
            "min_object_size": 0,
            "use_endpoint_linking": False,
            "use_hysteresis": False,
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

    for wavelet in wavelets:
        for level in levels:
            for use_diagonal_detail in use_diagonal_options:
                for diagonal_weight in diagonal_weights:
                    for fusion_mode in fusion_modes:
                        for threshold_method in threshold_methods:
                            for percentile in percentiles:
                                for use_thinning in thinning_options:
                                    for min_skeleton_length in min_skeleton_lengths:
                                        add_config(
                                            {
                                                "wavelet": wavelet,
                                                "level": level,
                                                "use_diagonal_detail": use_diagonal_detail,
                                                "diagonal_weight": diagonal_weight,
                                                "fusion_mode": fusion_mode,
                                                "threshold_method": threshold_method,
                                                "percentile": percentile,
                                                "low_threshold_ratio": 0.45,
                                                "use_thinning": use_thinning,
                                                "min_skeleton_length": min_skeleton_length,
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
        "wavelet",
        "level",
        "use_diagonal_detail",
        "diagonal_weight",
        "fusion_mode",
        "threshold_method",
        "percentile",
        "use_thinning",
        "min_skeleton_length",
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
