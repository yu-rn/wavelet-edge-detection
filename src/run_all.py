from __future__ import annotations

import argparse
from pathlib import Path

from .baselines import canny_edges, sobel_edges
from .evaluate import (
    compute_precision_recall_f1,
    compute_strict_and_tolerant_metrics,
    gt_path_from_image_name,
    load_binary_mask,
    save_dual_metrics_tables,
    save_metrics_table,
)
from .visualize import (
    save_comparison_figure,
    save_error_map,
    save_image,
    save_intermediate_images,
    save_wavelet_process_figure,
)
from .wavelet_edge import detect_edges, parse_level_weights, read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wavelet-based edge detection on all sample images.")
    parser.add_argument("--image-dir", type=Path, default=Path("samples/images"))
    parser.add_argument("--gt-dir", type=Path, default=Path("samples/groundTruth"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--wavelet", type=str, default="haar", choices=["haar", "db2", "sym2", "sym4"])
    parser.add_argument("--transform-mode", type=str, default="dwt", choices=["dwt", "swt"])
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--threshold-method", type=str, default="percentile")
    parser.add_argument("--threshold-ratio", type=float, default=0.2)
    parser.add_argument("--fixed-threshold", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--low-threshold-ratio", type=float, default=0.45)
    parser.add_argument("--adaptive-block-size", type=int, default=31)
    parser.add_argument("--adaptive-percentile", type=float, default=95.0)
    parser.add_argument("--adaptive-use-hysteresis", action="store_true", default=True)
    parser.add_argument("--no-adaptive-hysteresis", dest="adaptive_use_hysteresis", action="store_false")
    parser.add_argument("--adaptive-low-ratio", type=float, default=0.7)
    parser.add_argument("--adaptive-global-floor-percentile", type=float, default=90.0)
    parser.add_argument("--adaptive-min-high-threshold", type=float, default=20.0)
    parser.add_argument("--use-hysteresis", action="store_true", default=False)
    parser.add_argument("--no-hysteresis", dest="use_hysteresis", action="store_false")
    parser.add_argument("--use-clahe", action="store_true")
    parser.add_argument("--use-tv-denoise", action="store_true")
    parser.add_argument("--tv-weight", type=float, default=0.08)
    parser.add_argument("--gaussian-kernel-size", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--use-median-filter", action="store_true")
    parser.add_argument("--median-kernel-size", type=int, default=3)
    parser.add_argument("--use-bilateral-filter", action="store_true")
    parser.add_argument("--bilateral-diameter", type=int, default=7)
    parser.add_argument("--bilateral-sigma-color", type=float, default=50.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=50.0)
    parser.add_argument("--min-object-size", type=int, default=0)
    parser.add_argument("--use-thinning", action="store_true", default=False)
    parser.add_argument("--no-thinning", dest="use_thinning", action="store_false")
    parser.add_argument("--fusion-mode", type=str, default="weighted")
    parser.add_argument("--use-diagonal-detail", action="store_true", default=True)
    parser.add_argument("--no-diagonal-detail", dest="use_diagonal_detail", action="store_false")
    parser.add_argument("--diagonal-weight", type=float, default=1.0)
    parser.add_argument("--response-mode", type=str, default="weighted")
    parser.add_argument("--normalize-each-level", action="store_true")
    parser.add_argument("--level-weights", type=str, default=None)
    parser.add_argument("--use-gradient-assist", action="store_true")
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--gradient-kernel-size", type=int, default=3)
    parser.add_argument("--gradient-mode", type=str, default="multiply")
    parser.add_argument("--use-nms", action="store_true", default=True)
    parser.add_argument("--no-nms", dest="use_nms", action="store_false")
    parser.add_argument("--use-morphological-closing", action="store_true", default=False)
    parser.add_argument("--no-morphological-closing", dest="use_morphological_closing", action="store_false")
    parser.add_argument("--closing-kernel-size", type=int, default=3)
    parser.add_argument("--use-morphological-opening", action="store_true")
    parser.add_argument("--opening-kernel-size", type=int, default=3)
    parser.add_argument("--use-texture-suppression", action="store_true", default=False)
    parser.add_argument("--texture-window-size", type=int, default=15)
    parser.add_argument("--texture-weight", type=float, default=0.2)
    parser.add_argument("--texture-suppression-mode", type=str, default="linear")
    parser.add_argument("--texture-strength", type=float, default=3.0)
    parser.add_argument("--texture-gamma", type=float, default=2.0)
    parser.add_argument("--use-endpoint-linking", action="store_true")
    parser.add_argument("--link-radius", type=int, default=6)
    parser.add_argument("--link-angle-threshold", type=float, default=35.0)
    parser.add_argument("--min-skeleton-length", type=int, default=0)
    parser.add_argument("--min-component-mean-response", type=float, default=0.0)
    parser.add_argument("--tolerance-radius", type=int, default=0)
    parser.add_argument("--run-baselines", action="store_true")
    return parser.parse_args()


def run_pipeline(
    image_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    wavelet: str,
    transform_mode: str,
    level: int,
    threshold_method: str,
    threshold_ratio: float,
    fixed_threshold: int | None,
    percentile: float,
    low_threshold_ratio: float,
    adaptive_block_size: int,
    adaptive_percentile: float,
    adaptive_use_hysteresis: bool,
    adaptive_low_ratio: float,
    adaptive_global_floor_percentile: float,
    adaptive_min_high_threshold: float,
    use_hysteresis: bool,
    use_clahe: bool,
    use_tv_denoise: bool,
    tv_weight: float,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
    use_median_filter: bool,
    median_kernel_size: int,
    use_bilateral_filter: bool,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    min_object_size: int,
    use_thinning: bool,
    fusion_mode: str,
    use_diagonal_detail: bool,
    diagonal_weight: float,
    response_mode: str,
    normalize_each_level: bool,
    level_weights: list[float] | None,
    use_gradient_assist: bool,
    gradient_weight: float,
    gradient_kernel_size: int,
    gradient_mode: str,
    use_nms: bool,
    use_morphological_closing: bool,
    closing_kernel_size: int,
    use_morphological_opening: bool,
    opening_kernel_size: int,
    use_texture_suppression: bool,
    texture_window_size: int,
    texture_weight: float,
    texture_suppression_mode: str,
    texture_strength: float,
    texture_gamma: float,
    use_endpoint_linking: bool,
    link_radius: int,
    link_angle_threshold: float,
    min_skeleton_length: int,
    min_component_mean_response: float,
    tolerance_radius: int,
    run_baselines: bool,
) -> None:
    edges_dir = output_dir / "edges"
    intermediate_dir = output_dir / "intermediate"
    comparisons_dir = output_dir / "comparisons"
    metrics_dir = output_dir / "metrics"

    image_paths = sorted(image_dir.glob("*"))
    records: list[dict[str, float | str]] = []
    strict_records: list[dict[str, float | str]] = []
    tolerant_records: list[dict[str, float | str]] = []
    baseline_strict_records: list[dict[str, float | str]] = []
    baseline_tolerant_records: list[dict[str, float | str]] = []

    for image_path in image_paths:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue

        image_stem = image_path.stem
        original_image = read_image(image_path)
        detection = detect_edges(
            image_path,
            wavelet=wavelet,
            transform_mode=transform_mode,
            level=level,
            threshold_method=threshold_method,
            threshold_ratio=threshold_ratio,
            fixed_threshold=fixed_threshold,
            percentile=percentile,
            low_threshold_ratio=low_threshold_ratio,
            adaptive_block_size=adaptive_block_size,
            adaptive_percentile=adaptive_percentile,
            adaptive_use_hysteresis=adaptive_use_hysteresis,
            adaptive_low_ratio=adaptive_low_ratio,
            adaptive_global_floor_percentile=adaptive_global_floor_percentile,
            adaptive_min_high_threshold=adaptive_min_high_threshold,
            use_hysteresis=use_hysteresis,
            use_clahe=use_clahe,
            use_tv_denoise=use_tv_denoise,
            tv_weight=tv_weight,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            use_median_filter=use_median_filter,
            median_kernel_size=median_kernel_size,
            use_bilateral_filter=use_bilateral_filter,
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            min_object_size=min_object_size,
            use_thinning=use_thinning,
            fusion_mode=fusion_mode,
            use_diagonal_detail=use_diagonal_detail,
            diagonal_weight=diagonal_weight,
            response_mode=response_mode,
            normalize_each_level=normalize_each_level,
            level_weights=level_weights,
            use_gradient_assist=use_gradient_assist,
            gradient_weight=gradient_weight,
            gradient_kernel_size=gradient_kernel_size,
            gradient_mode=gradient_mode,
            use_nms=use_nms,
            use_morphological_closing=use_morphological_closing,
            closing_kernel_size=closing_kernel_size,
            use_morphological_opening=use_morphological_opening,
            opening_kernel_size=opening_kernel_size,
            use_texture_suppression=use_texture_suppression,
            texture_window_size=texture_window_size,
            texture_weight=texture_weight,
            texture_suppression_mode=texture_suppression_mode,
            texture_strength=texture_strength,
            texture_gamma=texture_gamma,
            use_endpoint_linking=use_endpoint_linking,
            link_radius=link_radius,
            link_angle_threshold=link_angle_threshold,
            min_skeleton_length=min_skeleton_length,
            min_component_mean_response=min_component_mean_response,
        )
        gray_image = detection["gray_image"]
        intermediate = detection["intermediate"]
        edge_map = detection["binary_edge_map"]

        gt_path = gt_path_from_image_name(image_path.name, gt_dir)
        ground_truth = load_binary_mask(gt_path)

        strict_metrics, tolerant_metrics = compute_strict_and_tolerant_metrics(edge_map, ground_truth)
        metrics = compute_precision_recall_f1(edge_map, ground_truth, tolerance_radius=tolerance_radius)
        records.append({"image_name": image_path.name, **metrics})
        strict_records.append({"image_name": image_path.name, **strict_metrics})
        tolerant_records.append({"image_name": image_path.name, **tolerant_metrics})

        save_image(edge_map, edges_dir / f"{image_stem}_edge.png")
        save_intermediate_images(image_stem, intermediate, intermediate_dir)
        save_comparison_figure(
            image_stem=image_stem,
            original=gray_image,
            prediction=edge_map,
            ground_truth=ground_truth * 255,
            output_path=comparisons_dir / f"{image_stem}_comparison.png",
        )
        save_wavelet_process_figure(
            image_stem=image_stem,
            original=original_image,
            gray=gray_image,
            intermediate=intermediate,
            prediction=edge_map,
            ground_truth=ground_truth * 255,
            output_path=comparisons_dir / f"{image_stem}_wavelet_process.png",
        )
        save_error_map(
            prediction=edge_map,
            ground_truth=ground_truth * 255,
            output_path=comparisons_dir / f"{image_stem}_error_map.png",
        )

        if run_baselines:
            baseline_results = {
                "sobel": sobel_edges(image_path, threshold_method="otsu", percentile=percentile),
                "canny": canny_edges(image_path),
            }
            for method_name, baseline in baseline_results.items():
                baseline_map = baseline["binary_edge_map"]
                save_image(baseline_map, edges_dir / f"{image_stem}_{method_name}_edge.png")
                save_comparison_figure(
                    image_stem=f"{image_stem}_{method_name}",
                    original=gray_image,
                    prediction=baseline_map,
                    ground_truth=ground_truth * 255,
                    output_path=comparisons_dir / f"{image_stem}_{method_name}_comparison.png",
                )
                baseline_strict, baseline_tolerant = compute_strict_and_tolerant_metrics(baseline_map, ground_truth)
                baseline_strict_records.append(
                    {"image_name": image_path.name, "method": method_name, **baseline_strict}
                )
                baseline_tolerant_records.append(
                    {"image_name": image_path.name, "method": method_name, **baseline_tolerant}
                )

    save_metrics_table(records, metrics_dir / "metrics.csv")
    save_dual_metrics_tables(strict_records, tolerant_records, metrics_dir, prefix="metrics")
    if run_baselines:
        save_dual_metrics_tables(
            baseline_strict_records,
            baseline_tolerant_records,
            metrics_dir,
            prefix="baseline_metrics",
        )


def main() -> None:
    args = parse_args()
    run_pipeline(
        image_dir=args.image_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        wavelet=args.wavelet,
        transform_mode=args.transform_mode,
        level=args.level,
        threshold_method=args.threshold_method,
        threshold_ratio=args.threshold_ratio,
        fixed_threshold=args.fixed_threshold,
        percentile=args.percentile,
        low_threshold_ratio=args.low_threshold_ratio,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_percentile=args.adaptive_percentile,
        adaptive_use_hysteresis=args.adaptive_use_hysteresis,
        adaptive_low_ratio=args.adaptive_low_ratio,
        adaptive_global_floor_percentile=args.adaptive_global_floor_percentile,
        adaptive_min_high_threshold=args.adaptive_min_high_threshold,
        use_hysteresis=args.use_hysteresis,
        use_clahe=args.use_clahe,
        use_tv_denoise=args.use_tv_denoise,
        tv_weight=args.tv_weight,
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        use_median_filter=args.use_median_filter,
        median_kernel_size=args.median_kernel_size,
        use_bilateral_filter=args.use_bilateral_filter,
        bilateral_diameter=args.bilateral_diameter,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        min_object_size=args.min_object_size,
        use_thinning=args.use_thinning,
        fusion_mode=args.fusion_mode,
        use_diagonal_detail=args.use_diagonal_detail,
        diagonal_weight=args.diagonal_weight,
        response_mode=args.response_mode,
        normalize_each_level=args.normalize_each_level,
        level_weights=parse_level_weights(args.level_weights),
        use_gradient_assist=args.use_gradient_assist,
        gradient_weight=args.gradient_weight,
        gradient_kernel_size=args.gradient_kernel_size,
        gradient_mode=args.gradient_mode,
        use_nms=args.use_nms,
        use_morphological_closing=args.use_morphological_closing,
        closing_kernel_size=args.closing_kernel_size,
        use_morphological_opening=args.use_morphological_opening,
        opening_kernel_size=args.opening_kernel_size,
        use_texture_suppression=args.use_texture_suppression,
        texture_window_size=args.texture_window_size,
        texture_weight=args.texture_weight,
        texture_suppression_mode=args.texture_suppression_mode,
        texture_strength=args.texture_strength,
        texture_gamma=args.texture_gamma,
        use_endpoint_linking=args.use_endpoint_linking,
        link_radius=args.link_radius,
        link_angle_threshold=args.link_angle_threshold,
        min_skeleton_length=args.min_skeleton_length,
        min_component_mean_response=args.min_component_mean_response,
        tolerance_radius=args.tolerance_radius,
        run_baselines=args.run_baselines,
    )


if __name__ == "__main__":
    main()
