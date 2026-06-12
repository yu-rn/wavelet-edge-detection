from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pywt
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin
from skimage.restoration import denoise_tv_chambolle

SUPPORTED_WAVELETS = {"haar", "db2", "sym2", "sym4"}
SUPPORTED_TRANSFORM_MODES = {"dwt", "swt"}
SUPPORTED_FUSION_MODES = {"weighted", "max"}
SUPPORTED_RESPONSE_MODES = {"weighted", "multiscale_consistency", "coarse_enhanced", "coarse_ratio"}
SUPPORTED_TEXTURE_SUPPRESSION_MODES = {"linear", "exp", "power"}
SUPPORTED_GRADIENT_MODES = {"multiply", "blend"}
SUPPORTED_THRESHOLD_METHODS = {"otsu", "fixed", "percentile", "adaptive_percentile"}


def read_image(image_source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image_source, np.ndarray):
        return image_source.copy()

    image_path = Path(image_source)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def load_grayscale_image(image_source: str | Path | np.ndarray) -> np.ndarray:
    return to_grayscale(read_image(image_source))


def validate_choice(value: str, supported_values: set[str], label: str) -> str:
    value = value.lower()
    if value not in supported_values:
        supported = ", ".join(sorted(supported_values))
        raise ValueError(f"Unsupported {label} '{value}'. Supported values: {supported}")
    return value


def validate_wavelet_name(wavelet: str) -> str:
    return validate_choice(wavelet, SUPPORTED_WAVELETS, "wavelet")


def ensure_odd_kernel_size(kernel_size: int, minimum: int = 1) -> int:
    kernel_size = max(minimum, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def normalize_to_unit_float(image: np.ndarray) -> np.ndarray:
    return normalize_to_uint8(image).astype(np.float32) / 255.0


def normalize_signed_float(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    max_abs_value = float(np.max(np.abs(image)))
    if max_abs_value < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return image / max_abs_value


def resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def pad_for_swt(image: np.ndarray, level: int) -> tuple[np.ndarray, tuple[int, int]]:
    divisor = 2**level
    rows, cols = image.shape
    pad_rows = (divisor - rows % divisor) % divisor
    pad_cols = (divisor - cols % divisor) % divisor
    if pad_rows == 0 and pad_cols == 0:
        return image, (rows, cols)

    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_rows,
        0,
        pad_cols,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, (rows, cols)


def preprocess_gray_image(
    gray_image: np.ndarray,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    use_clahe: bool = False,
    use_tv_denoise: bool = False,
    tv_weight: float = 0.08,
    use_median_filter: bool = False,
    median_kernel_size: int = 3,
    use_bilateral_filter: bool = False,
    bilateral_diameter: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    processed = gray_image.astype(np.uint8)
    intermediates: dict[str, np.ndarray] = {"gray_image": gray_image.astype(np.uint8)}

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        intermediates["clahe_gray_image"] = processed.copy()

    if use_tv_denoise:
        denoised = denoise_tv_chambolle(processed.astype(np.float32) / 255.0, weight=max(0.0, tv_weight))
        processed = (denoised * 255).clip(0, 255).astype(np.uint8)
        intermediates["tv_denoised_gray_image"] = processed.copy()

    if use_median_filter:
        median_kernel_size = ensure_odd_kernel_size(median_kernel_size, minimum=3)
        processed = cv2.medianBlur(processed, median_kernel_size)
        intermediates["median_filtered_gray_image"] = processed.copy()

    if use_bilateral_filter:
        processed = cv2.bilateralFilter(
            processed,
            d=max(1, int(bilateral_diameter)),
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
        intermediates["bilateral_filtered_gray_image"] = processed.copy()

    if gaussian_kernel_size > 1:
        gaussian_kernel_size = ensure_odd_kernel_size(gaussian_kernel_size, minimum=3)
        processed = cv2.GaussianBlur(processed, (gaussian_kernel_size, gaussian_kernel_size), gaussian_sigma)
        intermediates["gaussian_filtered_gray_image"] = processed.copy()

    intermediates["preprocessed_gray_image"] = processed.copy()
    return processed, intermediates


def resolve_decomposition_level(image_shape: tuple[int, int], wavelet: str, level: int) -> int:
    if level < 1:
        raise ValueError("Decomposition level must be at least 1.")

    wavelet_obj = pywt.Wavelet(validate_wavelet_name(wavelet))
    max_level = pywt.dwtn_max_level(image_shape, wavelet_obj)
    if max_level < 1:
        raise ValueError("Image is too small for wavelet decomposition.")

    return min(level, max_level)


def decompose_image(
    gray_image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> list[Any]:
    resolved_level = resolve_decomposition_level(gray_image.shape, wavelet, level)
    normalized_gray = gray_image.astype(np.float32) / 255.0
    return pywt.wavedec2(normalized_gray, wavelet=wavelet, level=resolved_level)


def decompose_image_swt(
    gray_image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    resolved_level = resolve_decomposition_level(gray_image.shape, wavelet, level)
    padded_image, _ = pad_for_swt(gray_image, resolved_level)
    normalized_gray = padded_image.astype(np.float32) / 255.0
    return pywt.swt2(
        normalized_gray,
        wavelet=validate_wavelet_name(wavelet),
        level=resolved_level,
        trim_approx=False,
    )


def build_level_weights(num_levels: int, level_weights: list[float] | None = None) -> np.ndarray:
    if level_weights is None:
        weights = np.array([1.0 / (2**index) for index in range(num_levels)], dtype=np.float32)
    else:
        weights = np.array(level_weights, dtype=np.float32)
        if weights.size != num_levels:
            raise ValueError("level_weights must match the resolved decomposition level.")

    if float(weights.sum()) <= 0:
        raise ValueError("level_weights must sum to a positive value.")
    return weights / weights.sum()


def build_detail_response(
    detail_coeffs: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    use_diagonal_detail: bool = True,
    diagonal_weight: float = 1.0,
) -> dict[str, np.ndarray]:
    c_h, c_v, c_d = detail_coeffs
    diagonal_term = max(0.0, float(diagonal_weight)) * c_d**2 if use_diagonal_detail else 0.0
    response = np.sqrt(c_h**2 + c_v**2 + diagonal_term)

    return {
        "cH_horizontal_detail": resize_to_shape(c_h, target_shape).astype(np.float32),
        "cV_vertical_detail": resize_to_shape(c_v, target_shape).astype(np.float32),
        "cD_diagonal_detail": resize_to_shape(c_d, target_shape).astype(np.float32),
        "edge_response": resize_to_shape(response, target_shape).astype(np.float32),
    }


def aggregate_multilevel_responses(
    coeffs: list[Any],
    target_shape: tuple[int, int],
    level_weights: list[float] | None = None,
    fusion_mode: str = "weighted",
    normalize_each_level: bool = False,
    use_diagonal_detail: bool = True,
    diagonal_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fusion_mode = validate_choice(fusion_mode, SUPPORTED_FUSION_MODES, "fusion mode")
    weights = build_level_weights(len(coeffs) - 1, level_weights)

    fused_c_h = np.zeros(target_shape, dtype=np.float32)
    fused_c_v = np.zeros(target_shape, dtype=np.float32)
    fused_c_d = np.zeros(target_shape, dtype=np.float32)
    fused_response = np.zeros(target_shape, dtype=np.float32)
    intermediates: dict[str, np.ndarray] = {}

    # PyWavelets returns details from coarse to fine; reversed order makes level_1 the finest scale.
    for level_index, detail_coeffs in enumerate(reversed(coeffs[1:]), start=1):
        detail_maps = build_detail_response(
            detail_coeffs,
            target_shape,
            use_diagonal_detail=use_diagonal_detail,
            diagonal_weight=diagonal_weight,
        )
        c_h = detail_maps["cH_horizontal_detail"]
        c_v = detail_maps["cV_vertical_detail"]
        c_d = detail_maps["cD_diagonal_detail"]
        response = detail_maps["edge_response"]

        if normalize_each_level:
            c_h = normalize_signed_float(c_h)
            c_v = normalize_signed_float(c_v)
            c_d = normalize_signed_float(c_d)
            response = normalize_to_uint8(response).astype(np.float32)

        intermediates[f"cH_horizontal_detail_level_{level_index}"] = c_h
        intermediates[f"cV_vertical_detail_level_{level_index}"] = c_v
        intermediates[f"cD_diagonal_detail_level_{level_index}"] = c_d
        intermediates[f"LH_level_{level_index}"] = c_h
        intermediates[f"HL_level_{level_index}"] = c_v
        intermediates[f"HH_level_{level_index}"] = c_d
        intermediates[f"edge_response_level_{level_index}"] = response

        if level_index == 1:
            intermediates["cH_horizontal_detail"] = c_h
            intermediates["cV_vertical_detail"] = c_v
            intermediates["cD_diagonal_detail"] = c_d

        if fusion_mode == "weighted":
            weight = float(weights[level_index - 1])
            fused_c_h += weight * c_h
            fused_c_v += weight * c_v
            fused_c_d += weight * c_d
            fused_response += weight * response
        else:
            fused_c_h = np.where(np.abs(c_h) >= np.abs(fused_c_h), c_h, fused_c_h)
            fused_c_v = np.where(np.abs(c_v) >= np.abs(fused_c_v), c_v, fused_c_v)
            fused_c_d = np.where(np.abs(c_d) >= np.abs(fused_c_d), c_d, fused_c_d)
            fused_response = np.maximum(fused_response, response)

    # Keep the final response explicitly tied to the fused high-frequency subbands.
    diagonal_term = max(0.0, float(diagonal_weight)) * fused_c_d**2 if use_diagonal_detail else 0.0
    fused_response = np.sqrt(np.maximum(fused_c_h**2 + fused_c_v**2 + diagonal_term, 0.0))
    intermediates["fused_cH_horizontal_detail"] = fused_c_h
    intermediates["fused_cV_vertical_detail"] = fused_c_v
    intermediates["fused_cD_diagonal_detail"] = fused_c_d
    intermediates["fused_edge_response_raw"] = fused_response
    intermediates["fused_edge_response"] = fused_response
    return fused_response, fused_c_h, fused_c_v, intermediates


def aggregate_swt_responses(
    coeffs: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    target_shape: tuple[int, int],
    level_weights: list[float] | None = None,
    fusion_mode: str = "weighted",
    normalize_each_level: bool = False,
    use_diagonal_detail: bool = True,
    diagonal_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fusion_mode = validate_choice(fusion_mode, SUPPORTED_FUSION_MODES, "fusion mode")
    weights = build_level_weights(len(coeffs), level_weights)

    fused_c_h = np.zeros(target_shape, dtype=np.float32)
    fused_c_v = np.zeros(target_shape, dtype=np.float32)
    fused_c_d = np.zeros(target_shape, dtype=np.float32)
    fused_response = np.zeros(target_shape, dtype=np.float32)
    intermediates: dict[str, np.ndarray] = {}

    for level_index, (approximation, detail_coeffs) in enumerate(coeffs, start=1):
        c_h, c_v, c_d = detail_coeffs
        c_h = resize_to_shape(c_h, target_shape).astype(np.float32)
        c_v = resize_to_shape(c_v, target_shape).astype(np.float32)
        c_d = resize_to_shape(c_d, target_shape).astype(np.float32)
        diagonal_term = max(0.0, float(diagonal_weight)) * c_d**2 if use_diagonal_detail else 0.0
        response = np.sqrt(np.maximum(c_h**2 + c_v**2 + diagonal_term, 0.0))

        if normalize_each_level:
            c_h = normalize_signed_float(c_h)
            c_v = normalize_signed_float(c_v)
            c_d = normalize_signed_float(c_d)
            response = normalize_to_uint8(response).astype(np.float32)

        intermediates[f"cA_approximation_level_{level_index}"] = resize_to_shape(approximation, target_shape)
        intermediates[f"cH_horizontal_detail_level_{level_index}"] = c_h
        intermediates[f"cV_vertical_detail_level_{level_index}"] = c_v
        intermediates[f"cD_diagonal_detail_level_{level_index}"] = c_d
        intermediates[f"LL_level_{level_index}"] = resize_to_shape(approximation, target_shape)
        intermediates[f"LH_level_{level_index}"] = c_h
        intermediates[f"HL_level_{level_index}"] = c_v
        intermediates[f"HH_level_{level_index}"] = c_d
        intermediates[f"edge_response_level_{level_index}"] = response

        if level_index == 1:
            intermediates["cA_approximation"] = resize_to_shape(approximation, target_shape)
            intermediates["cH_horizontal_detail"] = c_h
            intermediates["cV_vertical_detail"] = c_v
            intermediates["cD_diagonal_detail"] = c_d

        if fusion_mode == "weighted":
            weight = float(weights[level_index - 1])
            fused_c_h += weight * c_h
            fused_c_v += weight * c_v
            fused_c_d += weight * c_d
            fused_response += weight * response
        else:
            fused_c_h = np.where(np.abs(c_h) >= np.abs(fused_c_h), c_h, fused_c_h)
            fused_c_v = np.where(np.abs(c_v) >= np.abs(fused_c_v), c_v, fused_c_v)
            fused_c_d = np.where(np.abs(c_d) >= np.abs(fused_c_d), c_d, fused_c_d)
            fused_response = np.maximum(fused_response, response)

    diagonal_term = max(0.0, float(diagonal_weight)) * fused_c_d**2 if use_diagonal_detail else 0.0
    fused_response = np.sqrt(np.maximum(fused_c_h**2 + fused_c_v**2 + diagonal_term, 0.0))
    intermediates["fused_cH_horizontal_detail"] = fused_c_h
    intermediates["fused_cV_vertical_detail"] = fused_c_v
    intermediates["fused_cD_diagonal_detail"] = fused_c_d
    intermediates["fused_edge_response_raw"] = fused_response
    intermediates["fused_edge_response"] = fused_response
    return fused_response, fused_c_h, fused_c_v, intermediates


def collect_level_responses(wavelet_maps: dict[str, np.ndarray]) -> list[np.ndarray]:
    level_keys = sorted(
        key for key in wavelet_maps if key.startswith("edge_response_level_")
    )
    return [wavelet_maps[key].astype(np.float32) for key in level_keys]


def compute_response_variants(
    fused_response: np.ndarray,
    level_responses: list[np.ndarray],
) -> dict[str, np.ndarray]:
    if not level_responses:
        return {
            "multiscale_consistency_response": fused_response,
            "coarse_enhanced_response": fused_response,
            "coarse_ratio_response": fused_response,
        }

    normalized_levels = [normalize_to_unit_float(response) + 1e-6 for response in level_responses]
    product = np.ones_like(normalized_levels[0], dtype=np.float32)
    for response in normalized_levels:
        product *= response
    multiscale_consistency = np.power(product, 1.0 / len(normalized_levels)) * 255.0

    weights = np.array([2.0**index for index in range(len(level_responses))], dtype=np.float32)
    weights /= weights.sum()
    coarse_enhanced = np.zeros_like(level_responses[0], dtype=np.float32)
    for weight, response in zip(weights, level_responses):
        coarse_enhanced += float(weight) * normalize_to_uint8(response).astype(np.float32)

    fine = level_responses[0].astype(np.float32)
    coarse = level_responses[-1].astype(np.float32)
    ratio = np.clip(coarse / (fine + 1e-6), 0.0, 1.0)
    coarse_ratio = coarse * ratio

    return {
        "multiscale_consistency_response": multiscale_consistency.astype(np.float32),
        "coarse_enhanced_response": coarse_enhanced.astype(np.float32),
        "coarse_ratio_response": coarse_ratio.astype(np.float32),
    }


def apply_response_mode(
    fused_response: np.ndarray,
    wavelet_maps: dict[str, np.ndarray],
    response_mode: str = "weighted",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    response_mode = validate_choice(response_mode, SUPPORTED_RESPONSE_MODES, "response mode")
    variants = compute_response_variants(fused_response, collect_level_responses(wavelet_maps))

    if response_mode == "multiscale_consistency":
        response = variants["multiscale_consistency_response"]
    elif response_mode == "coarse_enhanced":
        response = variants["coarse_enhanced_response"]
    elif response_mode == "coarse_ratio":
        response = variants["coarse_ratio_response"]
    else:
        response = fused_response

    return response.astype(np.float32), variants


def apply_texture_suppression(
    edge_response: np.ndarray,
    gray_image: np.ndarray,
    use_texture_suppression: bool = False,
    texture_window_size: int = 15,
    texture_weight: float = 0.35,
    texture_suppression_mode: str = "linear",
    texture_strength: float = 3.0,
    texture_gamma: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not use_texture_suppression:
        return edge_response.astype(np.float32), {}

    texture_suppression_mode = validate_choice(
        texture_suppression_mode,
        SUPPORTED_TEXTURE_SUPPRESSION_MODES,
        "texture suppression mode",
    )

    window_size = ensure_odd_kernel_size(texture_window_size, minimum=3)
    gray_float = gray_image.astype(np.float32)
    mean = cv2.blur(gray_float, (window_size, window_size))
    mean_square = cv2.blur(gray_float * gray_float, (window_size, window_size))
    local_variance = np.maximum(mean_square - mean * mean, 0.0)

    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_density = cv2.blur(normalize_to_unit_float(cv2.magnitude(gradient_x, gradient_y)), (window_size, window_size))

    texture_map = 0.5 * normalize_to_unit_float(local_variance) + 0.5 * gradient_density
    texture_map = np.clip(texture_map, 0.0, 1.0).astype(np.float32)
    if texture_suppression_mode == "exp":
        suppression = np.exp(-max(0.0, texture_strength) * texture_map)
    elif texture_suppression_mode == "power":
        suppression = np.power(np.clip(1.0 - texture_map, 0.0, 1.0), max(0.0, texture_gamma))
    else:
        suppression = 1.0 - float(np.clip(texture_weight, 0.0, 1.0)) * texture_map

    suppressed_response = edge_response.astype(np.float32) * suppression

    return suppressed_response, {
        "texture_map": texture_map * 255.0,
        "texture_suppression_factor": suppression * 255.0,
        "texture_suppressed_response": suppressed_response,
    }


def compute_gradient_orientation(c_h: np.ndarray, c_v: np.ndarray) -> np.ndarray:
    angles = np.rad2deg(np.arctan2(c_v, c_h))
    angles[angles < 0] += 180
    return angles


def compute_sobel_orientation(gray_image: np.ndarray, kernel_size: int = 3) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if kernel_size not in {1, 3, 5, 7}:
        kernel_size = 3

    gray_float = gray_image.astype(np.float32)
    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
    orientation = compute_gradient_orientation(gradient_x, gradient_y)
    return orientation, {
        "sobel_gradient_x": gradient_x,
        "sobel_gradient_y": gradient_y,
        "sobel_gradient_magnitude": cv2.magnitude(gradient_x, gradient_y),
    }


def non_maximum_suppression(response: np.ndarray, angle: np.ndarray) -> np.ndarray:
    rows, cols = response.shape
    suppressed = np.zeros((rows, cols), dtype=np.float32)

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            current_angle = angle[row, col]
            current_value = response[row, col]

            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                neighbor_1 = response[row, col - 1]
                neighbor_2 = response[row, col + 1]
            elif 22.5 <= current_angle < 67.5:
                neighbor_1 = response[row - 1, col + 1]
                neighbor_2 = response[row + 1, col - 1]
            elif 67.5 <= current_angle < 112.5:
                neighbor_1 = response[row - 1, col]
                neighbor_2 = response[row + 1, col]
            else:
                neighbor_1 = response[row - 1, col - 1]
                neighbor_2 = response[row + 1, col + 1]

            if current_value >= neighbor_1 and current_value >= neighbor_2:
                suppressed[row, col] = current_value

    return suppressed


def compute_gradient_assisted_response(
    wavelet_response: np.ndarray,
    gray_image: np.ndarray,
    use_gradient_assist: bool = False,
    gradient_weight: float = 0.2,
    gradient_kernel_size: int = 3,
    gradient_mode: str = "multiply",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not use_gradient_assist:
        return wavelet_response.astype(np.float32), {}

    gradient_mode = validate_choice(gradient_mode, SUPPORTED_GRADIENT_MODES, "gradient mode")
    if gradient_kernel_size not in {1, 3, 5, 7}:
        gradient_kernel_size = 3

    gray_float = gray_image.astype(np.float32)
    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=gradient_kernel_size)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=gradient_kernel_size)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    wavelet_norm = normalize_to_unit_float(wavelet_response)
    gradient_norm = normalize_to_unit_float(gradient_magnitude)
    weight = float(np.clip(gradient_weight, 0.0, 1.0))

    if gradient_mode == "multiply":
        assisted = wavelet_norm * (1.0 + weight * gradient_norm)
    else:
        assisted = (1.0 - weight) * wavelet_norm + weight * gradient_norm

    assisted_response = assisted * 255.0
    return assisted_response.astype(np.float32), {
        "gradient_magnitude": gradient_magnitude,
        "gradient_assisted_response": assisted_response,
    }


def threshold_edge_map(
    response: np.ndarray,
    method: str = "otsu",
    threshold_ratio: float = 0.2,
    fixed_threshold: int | None = None,
    percentile: float = 90.0,
    low_threshold_ratio: float = 0.2,
    use_hysteresis: bool = False,
    adaptive_block_size: int = 31,
    adaptive_percentile: float = 95.0,
    adaptive_use_hysteresis: bool = True,
    adaptive_low_ratio: float = 0.7,
    adaptive_global_floor_percentile: float = 90.0,
    adaptive_min_high_threshold: float = 20.0,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    method = validate_choice(method, SUPPORTED_THRESHOLD_METHODS, "threshold method")
    normalized = normalize_to_uint8(response)

    if method == "otsu":
        high_threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "fixed":
        if fixed_threshold is None:
            fixed_threshold = int(np.clip(threshold_ratio, 0.0, 1.0) * 255)
        high_threshold = float(np.clip(fixed_threshold, 0, 255))
    elif method == "percentile":
        high_threshold = float(np.percentile(normalized, np.clip(percentile, 0.0, 100.0)))
    else:
        binary, threshold_map, low_threshold_map = adaptive_percentile_threshold(
            normalized,
            adaptive_block_size=adaptive_block_size,
            adaptive_percentile=adaptive_percentile,
            adaptive_use_hysteresis=adaptive_use_hysteresis,
            adaptive_low_ratio=adaptive_low_ratio,
            adaptive_global_floor_percentile=adaptive_global_floor_percentile,
            adaptive_min_high_threshold=adaptive_min_high_threshold,
        )
        return binary, {
            "threshold_input_response": normalized,
            "adaptive_high_threshold_map": threshold_map,
            "adaptive_low_threshold_map": low_threshold_map,
            "threshold_binary_raw": binary,
            "high_threshold_value": float(np.mean(threshold_map)),
            "low_threshold_value": float(np.mean(low_threshold_map)),
        }

    low_threshold = float(np.clip(low_threshold_ratio, 0.0, 1.0) * high_threshold)

    if use_hysteresis:
        edge_mask = apply_hysteresis_threshold(normalized.astype(np.float32), low_threshold, high_threshold)
        binary = edge_mask.astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(normalized, high_threshold, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

    return binary, {
        "threshold_input_response": normalized,
        "threshold_binary_raw": binary,
        "high_threshold_value": float(high_threshold),
        "low_threshold_value": float(low_threshold),
    }


def adaptive_percentile_threshold(
    normalized_response: np.ndarray,
    adaptive_block_size: int = 31,
    adaptive_percentile: float = 95.0,
    adaptive_use_hysteresis: bool = True,
    adaptive_low_ratio: float = 0.7,
    adaptive_global_floor_percentile: float = 90.0,
    adaptive_min_high_threshold: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_size = ensure_odd_kernel_size(adaptive_block_size, minimum=3)
    percentile_value = float(np.clip(adaptive_percentile, 0.0, 100.0))
    pad = block_size // 2
    padded = cv2.copyMakeBorder(
        normalized_response,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_REFLECT_101,
    )

    threshold_map = np.zeros_like(normalized_response, dtype=np.float32)
    for row in range(normalized_response.shape[0]):
        for col in range(normalized_response.shape[1]):
            window = padded[row : row + block_size, col : col + block_size]
            threshold_map[row, col] = np.percentile(window, percentile_value)

    response_float = normalized_response.astype(np.float32)
    global_floor = float(np.percentile(normalized_response, np.clip(adaptive_global_floor_percentile, 0.0, 100.0)))
    high_threshold_map = np.maximum(threshold_map, global_floor)
    high_threshold_map = np.maximum(high_threshold_map, float(np.clip(adaptive_min_high_threshold, 0.0, 255.0)))
    low_threshold_map = high_threshold_map * float(np.clip(adaptive_low_ratio, 0.0, 1.0))
    strong = response_float >= high_threshold_map

    if not adaptive_use_hysteresis:
        return strong.astype(np.uint8) * 255, high_threshold_map, low_threshold_map

    weak = response_float >= low_threshold_map
    num_labels, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(weak, dtype=bool)
    strong_labels = np.unique(labels[strong])
    for label in strong_labels:
        if label != 0:
            keep[labels == label] = True

    return keep.astype(np.uint8) * 255, high_threshold_map, low_threshold_map


def remove_small_connected_components(edge_mask: np.ndarray, min_object_size: int) -> np.ndarray:
    binary = edge_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)

    for label_index in range(1, num_labels):
        if stats[label_index, cv2.CC_STAT_AREA] >= min_object_size:
            filtered[labels == label_index] = 1

    return filtered.astype(bool)


def find_skeleton_endpoints(edge_mask: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    skeleton = edge_mask.astype(np.uint8)
    padded = np.pad(skeleton, 1, mode="constant")
    endpoints: list[tuple[int, int, np.ndarray]] = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    rows, cols = skeleton.shape
    for row in range(rows):
        for col in range(cols):
            if skeleton[row, col] == 0:
                continue

            neighbors = []
            for d_row, d_col in neighbor_offsets:
                if padded[row + 1 + d_row, col + 1 + d_col] > 0:
                    neighbors.append((d_row, d_col))

            if len(neighbors) == 1:
                d_row, d_col = neighbors[0]
                # Direction points outward from the existing skeleton.
                direction = np.array([-d_col, -d_row], dtype=np.float32)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction /= norm
                endpoints.append((row, col, direction))

    return endpoints


def angle_between_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def link_skeleton_endpoints(
    edge_mask: np.ndarray,
    link_radius: int = 6,
    link_angle_threshold: float = 35.0,
) -> np.ndarray:
    linked = edge_mask.astype(np.uint8).copy()
    endpoints = find_skeleton_endpoints(linked > 0)
    used: set[int] = set()

    for index, (row_a, col_a, direction_a) in enumerate(endpoints):
        if index in used:
            continue

        best_index: int | None = None
        best_distance = float(link_radius) + 1.0
        for candidate_index, (row_b, col_b, direction_b) in enumerate(endpoints):
            if candidate_index == index or candidate_index in used:
                continue

            delta = np.array([col_b - col_a, row_b - row_a], dtype=np.float32)
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6 or distance > link_radius or distance >= best_distance:
                continue

            direction_ab = delta / distance
            direction_ba = -direction_ab
            angle_a = angle_between_degrees(direction_a, direction_ab)
            angle_b = angle_between_degrees(direction_b, direction_ba)
            if angle_a <= link_angle_threshold and angle_b <= link_angle_threshold:
                best_index = candidate_index
                best_distance = distance

        if best_index is not None:
            row_b, col_b, _ = endpoints[best_index]
            cv2.line(linked, (col_a, row_a), (col_b, row_b), color=1, thickness=1)
            used.add(index)
            used.add(best_index)

    return linked.astype(bool)


def filter_skeleton_components(
    edge_mask: np.ndarray,
    response: np.ndarray | None,
    min_skeleton_length: int = 0,
    min_component_mean_response: float = 0.0,
) -> np.ndarray:
    if min_skeleton_length <= 0 and min_component_mean_response <= 0:
        return edge_mask

    binary = edge_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    response_uint8 = normalize_to_uint8(response) if response is not None else None
    filtered = np.zeros_like(binary)

    for label_index in range(1, num_labels):
        component_mask = labels == label_index
        length = int(stats[label_index, cv2.CC_STAT_AREA])
        mean_response = 255.0
        if response_uint8 is not None:
            mean_response = float(np.mean(response_uint8[component_mask]))

        if length >= min_skeleton_length and mean_response >= min_component_mean_response:
            filtered[component_mask] = 1

    return filtered.astype(bool)


def postprocess_binary_edge_map(
    binary_edge_map: np.ndarray,
    response: np.ndarray | None = None,
    min_object_size: int = 0,
    use_thinning: bool = False,
    use_morphological_closing: bool = False,
    closing_kernel_size: int = 3,
    use_morphological_opening: bool = False,
    opening_kernel_size: int = 3,
    use_endpoint_linking: bool = False,
    link_radius: int = 6,
    link_angle_threshold: float = 35.0,
    min_skeleton_length: int = 0,
    min_component_mean_response: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    edge_mask = binary_edge_map > 0
    intermediates: dict[str, np.ndarray] = {}

    if use_morphological_closing:
        kernel_size = ensure_odd_kernel_size(closing_kernel_size, minimum=1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        intermediates["morphological_closing_edge_map"] = edge_mask.astype(np.uint8) * 255

    if use_morphological_opening:
        kernel_size = ensure_odd_kernel_size(opening_kernel_size, minimum=1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        intermediates["morphological_opening_edge_map"] = edge_mask.astype(np.uint8) * 255

    if min_object_size > 1:
        edge_mask = remove_small_connected_components(edge_mask, min_object_size)
        intermediates["small_components_removed_edge_map"] = edge_mask.astype(np.uint8) * 255

    if use_thinning:
        edge_mask = thin(edge_mask)
        intermediates["thinned_edge_map"] = edge_mask.astype(np.uint8) * 255

    edge_mask = filter_skeleton_components(
        edge_mask,
        response=response,
        min_skeleton_length=min_skeleton_length,
        min_component_mean_response=min_component_mean_response,
    )
    if min_skeleton_length > 0 or min_component_mean_response > 0:
        intermediates["filtered_skeleton_edge_map"] = edge_mask.astype(np.uint8) * 255

    if use_endpoint_linking:
        edge_mask = link_skeleton_endpoints(
            edge_mask,
            link_radius=max(1, int(link_radius)),
            link_angle_threshold=float(np.clip(link_angle_threshold, 0.0, 180.0)),
        )
        intermediates["endpoint_linked_edge_map"] = edge_mask.astype(np.uint8) * 255

    binary = edge_mask.astype(np.uint8) * 255
    intermediates["postprocessed_binary_edge_map"] = binary
    return binary, intermediates


def parse_level_weights(level_weights: str | None) -> list[float] | None:
    if not level_weights:
        return None
    weights = [float(item.strip()) for item in level_weights.split(",") if item.strip()]
    return weights or None


def prepare_intermediate_images(intermediates: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
    image_maps: dict[str, np.ndarray] = {}
    for key, value in intermediates.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim != 2:
            continue
        image_maps[key] = value if value.dtype == np.uint8 else normalize_to_uint8(value)
    return image_maps


def compute_wavelet_response(
    gray_image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
    use_diagonal_detail: bool = True,
    diagonal_weight: float = 1.0,
    use_clahe: bool = False,
    use_tv_denoise: bool = False,
    tv_weight: float = 0.08,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    use_median_filter: bool = False,
    median_kernel_size: int = 3,
    use_bilateral_filter: bool = False,
    bilateral_diameter: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
    level_weights: list[float] | None = None,
    fusion_mode: str = "weighted",
    normalize_each_level: bool = False,
    use_gradient_assist: bool = False,
    gradient_weight: float = 0.2,
    gradient_kernel_size: int = 3,
    gradient_mode: str = "multiply",
    use_nms: bool = True,
    transform_mode: str = "dwt",
    response_mode: str = "weighted",
    use_texture_suppression: bool = False,
    texture_window_size: int = 15,
    texture_weight: float = 0.35,
    texture_suppression_mode: str = "linear",
    texture_strength: float = 3.0,
    texture_gamma: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    preprocessed, preprocessing_maps = preprocess_gray_image(
        gray_image,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
        use_clahe=use_clahe,
        use_tv_denoise=use_tv_denoise,
        tv_weight=tv_weight,
        use_median_filter=use_median_filter,
        median_kernel_size=median_kernel_size,
        use_bilateral_filter=use_bilateral_filter,
        bilateral_diameter=bilateral_diameter,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    transform_mode = validate_choice(transform_mode, SUPPORTED_TRANSFORM_MODES, "transform mode")
    if transform_mode == "swt":
        swt_coeffs = decompose_image_swt(preprocessed, wavelet=wavelet, level=level)
        fused_response, fused_c_h, fused_c_v, wavelet_maps = aggregate_swt_responses(
            swt_coeffs,
            gray_image.shape,
            level_weights=level_weights,
            fusion_mode=fusion_mode,
            normalize_each_level=normalize_each_level,
            use_diagonal_detail=use_diagonal_detail,
            diagonal_weight=diagonal_weight,
        )
    else:
        coeffs = decompose_image(preprocessed, wavelet=wavelet, level=level)
        fused_response, fused_c_h, fused_c_v, wavelet_maps = aggregate_multilevel_responses(
            coeffs,
            gray_image.shape,
            level_weights=level_weights,
            fusion_mode=fusion_mode,
            normalize_each_level=normalize_each_level,
            use_diagonal_detail=use_diagonal_detail,
            diagonal_weight=diagonal_weight,
        )
        resolved_level = len(coeffs) - 1
        approximation = resize_to_shape(coeffs[0], gray_image.shape)
        wavelet_maps["cA_approximation"] = approximation
        wavelet_maps[f"LL_level_{resolved_level}"] = approximation

    wavelet_maps["transform_mode_dwt_or_swt"] = np.full(gray_image.shape, 255 if transform_mode == "swt" else 0, dtype=np.uint8)
    mode_response, response_mode_maps = apply_response_mode(
        fused_response,
        wavelet_maps,
        response_mode=response_mode,
    )
    texture_response, texture_maps = apply_texture_suppression(
        mode_response,
        preprocessed,
        use_texture_suppression=use_texture_suppression,
        texture_window_size=texture_window_size,
        texture_weight=texture_weight,
        texture_suppression_mode=texture_suppression_mode,
        texture_strength=texture_strength,
        texture_gamma=texture_gamma,
    )
    assisted_response, gradient_maps = compute_gradient_assisted_response(
        texture_response,
        preprocessed,
        use_gradient_assist=use_gradient_assist,
        gradient_weight=gradient_weight,
        gradient_kernel_size=gradient_kernel_size,
        gradient_mode=gradient_mode,
    )

    if use_nms:
        orientation, sobel_orientation_maps = compute_sobel_orientation(preprocessed, kernel_size=gradient_kernel_size)
        edge_response = non_maximum_suppression(assisted_response, orientation)
    else:
        sobel_orientation_maps = {}
        edge_response = assisted_response.astype(np.float32)

    raw_intermediates: dict[str, np.ndarray | float] = {
        **preprocessing_maps,
        **wavelet_maps,
        **response_mode_maps,
        **texture_maps,
        **gradient_maps,
        **sobel_orientation_maps,
        "gradient_oriented_edge_response": assisted_response,
        "nms_edge_response": edge_response,
        "normalized_edge_response": normalize_to_uint8(edge_response),
    }
    return edge_response, prepare_intermediate_images(raw_intermediates)


def detect_edges(
    image_source: str | Path | np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
    threshold_method: str = "percentile",
    threshold_ratio: float = 0.2,
    fixed_threshold: int | None = None,
    percentile: float = 95.0,
    low_threshold_ratio: float = 0.2,
    use_hysteresis: bool = False,
    adaptive_block_size: int = 31,
    adaptive_percentile: float = 95.0,
    adaptive_use_hysteresis: bool = True,
    adaptive_low_ratio: float = 0.7,
    adaptive_global_floor_percentile: float = 90.0,
    adaptive_min_high_threshold: float = 20.0,
    use_clahe: bool = False,
    use_tv_denoise: bool = False,
    tv_weight: float = 0.08,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    use_median_filter: bool = False,
    median_kernel_size: int = 3,
    use_bilateral_filter: bool = False,
    bilateral_diameter: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
    min_object_size: int = 0,
    use_thinning: bool = False,
    level_weights: list[float] | None = None,
    fusion_mode: str = "weighted",
    use_diagonal_detail: bool = True,
    diagonal_weight: float = 1.0,
    normalize_each_level: bool = False,
    use_gradient_assist: bool = False,
    gradient_weight: float = 0.2,
    gradient_kernel_size: int = 3,
    gradient_mode: str = "multiply",
    use_nms: bool = True,
    use_morphological_closing: bool = False,
    closing_kernel_size: int = 3,
    use_morphological_opening: bool = False,
    opening_kernel_size: int = 3,
    transform_mode: str = "dwt",
    response_mode: str = "weighted",
    use_texture_suppression: bool = False,
    texture_window_size: int = 15,
    texture_weight: float = 0.35,
    texture_suppression_mode: str = "linear",
    texture_strength: float = 3.0,
    texture_gamma: float = 2.0,
    use_endpoint_linking: bool = False,
    link_radius: int = 6,
    link_angle_threshold: float = 35.0,
    min_skeleton_length: int = 0,
    min_component_mean_response: float = 0.0,
) -> dict[str, np.ndarray]:
    gray_image = load_grayscale_image(image_source)
    edge_response, intermediate = compute_wavelet_response(
        gray_image,
        wavelet=wavelet,
        level=level,
        use_diagonal_detail=use_diagonal_detail,
        diagonal_weight=diagonal_weight,
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
        level_weights=level_weights,
        fusion_mode=fusion_mode,
        normalize_each_level=normalize_each_level,
        use_gradient_assist=use_gradient_assist,
        gradient_weight=gradient_weight,
        gradient_kernel_size=gradient_kernel_size,
        gradient_mode=gradient_mode,
        use_nms=use_nms,
        transform_mode=transform_mode,
        response_mode=response_mode,
        use_texture_suppression=use_texture_suppression,
        texture_window_size=texture_window_size,
        texture_weight=texture_weight,
        texture_suppression_mode=texture_suppression_mode,
        texture_strength=texture_strength,
        texture_gamma=texture_gamma,
    )
    thresholded, threshold_maps = threshold_edge_map(
        edge_response,
        method=threshold_method,
        threshold_ratio=threshold_ratio,
        fixed_threshold=fixed_threshold,
        percentile=percentile,
        low_threshold_ratio=low_threshold_ratio,
        use_hysteresis=use_hysteresis,
        adaptive_block_size=adaptive_block_size,
        adaptive_percentile=adaptive_percentile,
        adaptive_use_hysteresis=adaptive_use_hysteresis,
        adaptive_low_ratio=adaptive_low_ratio,
        adaptive_global_floor_percentile=adaptive_global_floor_percentile,
        adaptive_min_high_threshold=adaptive_min_high_threshold,
    )
    binary_edge_map, postprocess_maps = postprocess_binary_edge_map(
        thresholded,
        response=edge_response,
        min_object_size=min_object_size,
        use_thinning=use_thinning,
        use_morphological_closing=use_morphological_closing,
        closing_kernel_size=closing_kernel_size,
        use_morphological_opening=use_morphological_opening,
        opening_kernel_size=opening_kernel_size,
        use_endpoint_linking=use_endpoint_linking,
        link_radius=link_radius,
        link_angle_threshold=link_angle_threshold,
        min_skeleton_length=min_skeleton_length,
        min_component_mean_response=min_component_mean_response,
    )

    intermediate.update(prepare_intermediate_images(threshold_maps))
    intermediate.update(postprocess_maps)
    intermediate["binary_edge_map"] = binary_edge_map

    return {
        "gray_image": gray_image,
        "preprocessed_gray_image": intermediate["preprocessed_gray_image"],
        "edge_response": edge_response,
        "normalized_edge_response": intermediate["normalized_edge_response"],
        "binary_edge_map": binary_edge_map,
        "intermediate": intermediate,
    }
