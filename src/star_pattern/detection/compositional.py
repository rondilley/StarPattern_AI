"""Compositional detection: evolved processing pipelines from primitive operations.

10 primitive operations that can be composed into variable-length pipelines.
Each operation takes (image, context, params) -> (image, context).
Pipelines are evolved via PipelineGenome (discovery/pipeline_genome.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.compositional")


@dataclass
class OperationSpec:
    """Specification for a single pipeline operation."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSpec:
    """Specification for a composed detection pipeline."""

    operations: list[OperationSpec] = field(default_factory=list)
    score_method: str = "component_count"


def _mask_sources(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Zero out detected source positions."""
    radius = int(params.get("radius_px", 5))
    positions = context.get("positions")
    result = image.copy()

    if positions is not None and len(positions) > 0:
        pos = np.asarray(positions)
        h, w = image.shape[:2]
        for p in pos:
            y, x = int(p[0]), int(p[1]) if len(p) > 1 else (int(p[0]), 0)
            y_lo = max(0, y - radius)
            y_hi = min(h, y + radius + 1)
            x_lo = max(0, x - radius)
            x_hi = min(w, x + radius + 1)
            result[y_lo:y_hi, x_lo:x_hi] = 0

    context["mask_applied"] = True
    return result, context


def _subtract_model(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract smooth model (Gaussian-filtered version) from image."""
    sigma = float(params.get("smooth_sigma", 5.0))
    smooth = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)
    residual = image.astype(np.float64) - smooth
    context["model_subtracted"] = True
    return residual, context


def _wavelet_residual(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract wavelet residual at a specific scale via a-trous decomposition."""
    scale = int(params.get("scale", 3))
    data = image.astype(np.float64)

    # Simple a-trous: progressively larger Gaussian filters
    prev = data.copy()
    for s in range(scale):
        sigma = 2 ** s
        smoothed = ndimage.gaussian_filter(prev, sigma=sigma)
        residual = prev - smoothed
        prev = smoothed

    context["wavelet_scale"] = scale
    return residual, context


def _threshold_image(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Binary threshold at a given percentile."""
    percentile = float(params.get("percentile", 90))
    data = image.astype(np.float64)
    threshold = np.percentile(data, percentile)
    binary = (data > threshold).astype(np.float64)
    context["threshold"] = float(threshold)
    context["n_above"] = int(np.sum(binary > 0))
    return binary, context


def _convolve_kernel(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convolve with a parameterized kernel."""
    kernel_size = int(params.get("kernel_size", 5))
    kernel_type = params.get("type", "gaussian")

    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_type == "laplacian":
        # Laplacian of Gaussian approximation
        result = ndimage.laplace(image.astype(np.float64))
    elif kernel_type == "tophat":
        # Circular tophat kernel
        y, x = np.ogrid[
            -kernel_size // 2 : kernel_size // 2 + 1,
            -kernel_size // 2 : kernel_size // 2 + 1,
        ]
        r = np.sqrt(x ** 2 + y ** 2)
        kernel = (r <= kernel_size / 2).astype(np.float64)
        kernel /= max(kernel.sum(), 1)
        result = ndimage.convolve(image.astype(np.float64), kernel)
    else:
        # Gaussian (default)
        sigma = kernel_size / 4.0
        result = ndimage.gaussian_filter(image.astype(np.float64), sigma=sigma)

    context["convolved_with"] = kernel_type
    return result, context


def _cross_correlate(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-correlate image with source density map."""
    sigma = float(params.get("smooth_sigma", 5.0))
    positions = context.get("positions")

    # Build density map from source positions
    density = np.zeros_like(image, dtype=np.float64)
    if positions is not None and len(positions) > 0:
        pos = np.asarray(positions)
        h, w = image.shape[:2]
        for p in pos:
            y = int(np.clip(p[0], 0, h - 1))
            x = int(np.clip(p[1], 0, w - 1)) if len(p) > 1 else 0
            density[y, x] += 1.0

    density = ndimage.gaussian_filter(density, sigma=sigma)
    if density.max() > 0:
        density /= density.max()

    # Correlate
    result = image.astype(np.float64) * density
    context["density_correlated"] = True
    return result, context


def _combine_masks(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine current image (as mask) with context mask via AND/OR/XOR."""
    mode = params.get("mode", "and")
    prev_mask = context.get("prev_mask")

    binary = (image > 0).astype(np.float64)

    if prev_mask is not None and prev_mask.shape == binary.shape:
        prev_binary = (prev_mask > 0).astype(np.float64)
        if mode == "or":
            result = np.maximum(binary, prev_binary)
        elif mode == "xor":
            result = np.abs(binary - prev_binary)
        else:  # "and"
            result = binary * prev_binary
    else:
        result = binary

    context["prev_mask"] = binary
    return result, context


def _region_statistics(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute statistics on masked/thresholded regions."""
    stat_type = params.get("stat_type", "count")
    data = image.astype(np.float64)

    mask = data > 0
    if mask.any():
        if stat_type == "mean":
            context["region_stat"] = float(np.mean(data[mask]))
        elif stat_type == "std":
            context["region_stat"] = float(np.std(data[mask]))
        else:  # count
            labels, n_labels = ndimage.label(mask)
            context["region_stat"] = float(n_labels)
            context["n_components"] = n_labels
    else:
        context["region_stat"] = 0.0
        context["n_components"] = 0

    return image, context


def _edge_detect(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Edge detection via Sobel or Canny-like thresholding."""
    method = params.get("method", "sobel")
    threshold = float(params.get("threshold", 3.0))

    data = image.astype(np.float64)

    if method == "canny":
        # Simplified Canny: Sobel + double threshold
        gy = ndimage.sobel(data, axis=0)
        gx = ndimage.sobel(data, axis=1)
        edges = np.hypot(gx, gy)
        high = np.percentile(edges, 100 - threshold)
        low = high * 0.4
        result = np.where(edges > high, 1.0, np.where(edges > low, 0.5, 0.0))
    else:
        # Sobel (default)
        gy = ndimage.sobel(data, axis=0)
        gx = ndimage.sobel(data, axis=1)
        result = np.hypot(gx, gy)
        std = np.std(result)
        if std > 0:
            result = result / std

    context["edge_method"] = method
    return result, context


def _radial_profile_residual(
    image: np.ndarray, context: dict[str, Any], params: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract azimuthally-averaged radial profile."""
    center_method = params.get("center_method", "peak")
    data = image.astype(np.float64)

    h, w = data.shape[:2]
    if center_method == "centroid":
        total = np.sum(np.abs(data))
        if total > 0:
            yy, xx = np.mgrid[0:h, 0:w]
            cy = float(np.sum(yy * np.abs(data)) / total)
            cx = float(np.sum(xx * np.abs(data)) / total)
        else:
            cy, cx = h / 2, w / 2
    else:
        # Peak
        idx = np.unravel_index(np.argmax(data), data.shape)
        cy, cx = float(idx[0]), float(idx[1])

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Build radial profile
    max_r = int(np.max(r)) + 1
    radial_profile = np.zeros(max_r)
    radial_counts = np.zeros(max_r)
    r_int = r.astype(int)
    for ri in range(max_r):
        mask = r_int == ri
        if mask.any():
            radial_profile[ri] = np.mean(data[mask])
            radial_counts[ri] = np.sum(mask)

    # Subtract radial profile
    model = radial_profile[np.clip(r_int, 0, max_r - 1)]
    residual = data - model

    context["radial_center"] = (cy, cx)
    return residual, context


# Registry of all operations
_OPERATION_MAP: dict[str, Callable] = {
    "mask_sources": _mask_sources,
    "subtract_model": _subtract_model,
    "wavelet_residual": _wavelet_residual,
    "threshold_image": _threshold_image,
    "convolve_kernel": _convolve_kernel,
    "cross_correlate": _cross_correlate,
    "combine_masks": _combine_masks,
    "region_statistics": _region_statistics,
    "edge_detect": _edge_detect,
    "radial_profile_residual": _radial_profile_residual,
}

# Operation parameter constraints: name -> {param: (min, max, dtype)}
OPERATION_PARAMS: dict[str, dict[str, tuple[float, float, str]]] = {
    "mask_sources": {"radius_px": (2, 20, "int")},
    "subtract_model": {"smooth_sigma": (1, 10, "float")},
    "wavelet_residual": {"scale": (1, 7, "int")},
    "threshold_image": {"percentile": (50, 99, "float")},
    "convolve_kernel": {
        "kernel_size": (3, 15, "int"),
        "type": (0, 2, "choice"),  # 0=gaussian, 1=laplacian, 2=tophat
    },
    "cross_correlate": {"smooth_sigma": (1, 10, "float")},
    "combine_masks": {"mode": (0, 2, "choice")},  # 0=and, 1=or, 2=xor
    "region_statistics": {"stat_type": (0, 2, "choice")},  # 0=mean, 1=std, 2=count
    "edge_detect": {
        "method": (0, 1, "choice"),  # 0=sobel, 1=canny
        "threshold": (1, 10, "float"),
    },
    "radial_profile_residual": {"center_method": (0, 1, "choice")},  # 0=peak, 1=centroid
}

_CHOICE_MAPS: dict[str, list[str]] = {
    "convolve_kernel.type": ["gaussian", "laplacian", "tophat"],
    "combine_masks.mode": ["and", "or", "xor"],
    "region_statistics.stat_type": ["mean", "std", "count"],
    "edge_detect.method": ["sobel", "canny"],
    "radial_profile_residual.center_method": ["peak", "centroid"],
}

ALL_OPERATIONS = list(_OPERATION_MAP.keys())


class OperationRegistry:
    """Registry of all primitive operations."""

    def __init__(self) -> None:
        self._ops = dict(_OPERATION_MAP)

    @property
    def names(self) -> list[str]:
        return list(self._ops.keys())

    def execute(
        self,
        op: OperationSpec,
        image: np.ndarray,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute a single operation.

        Args:
            op: Operation specification.
            image: 2D image array.
            context: Mutable context dict (carries state between ops).

        Returns:
            (processed_image, updated_context)
        """
        func = self._ops.get(op.name)
        if func is None:
            logger.warning(f"Unknown operation: {op.name}")
            return image, context

        try:
            return func(image, context, op.params)
        except Exception as e:
            logger.debug(f"Operation {op.name} failed: {e}")
            return image, context


class ComposedPipelineScorer:
    """Scoring methods for composed pipeline outputs."""

    @staticmethod
    def component_count(image: np.ndarray, context: dict[str, Any]) -> float:
        """Score based on number of connected components in thresholded output."""
        n = context.get("n_components")
        if n is not None:
            return min(float(n) / 10.0, 1.0)

        binary = image > np.percentile(image, 95)
        labels, n_labels = ndimage.label(binary)
        return min(float(n_labels) / 10.0, 1.0)

    @staticmethod
    def max_residual(image: np.ndarray, context: dict[str, Any]) -> float:
        """Score based on maximum residual value (normalized)."""
        data = image.astype(np.float64)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        max_val = np.max(np.abs(data))
        return float(min(max_val / (5.0 * std), 1.0))

    @staticmethod
    def area_fraction(image: np.ndarray, context: dict[str, Any]) -> float:
        """Score based on fraction of image above threshold."""
        binary = image > np.percentile(image, 90)
        fraction = np.mean(binary)
        # Optimal: 1-10% of image has signal
        if fraction < 0.005:
            return 0.0
        if fraction > 0.2:
            return max(0, 1.0 - (fraction - 0.2) * 5)
        return min(fraction * 10, 1.0)


_SCORERS: dict[str, Callable] = {
    "component_count": ComposedPipelineScorer.component_count,
    "max_residual": ComposedPipelineScorer.max_residual,
    "area_fraction": ComposedPipelineScorer.area_fraction,
}


class ComposedPipeline:
    """A composed detection pipeline: sequence of primitive operations + scorer."""

    def __init__(
        self,
        spec: PipelineSpec,
        registry: OperationRegistry | None = None,
    ):
        self.spec = spec
        self._registry = registry or OperationRegistry()

    def run(
        self,
        image: Any,
        detection_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the composed pipeline on an image.

        Args:
            image: FITSImage or ndarray.
            detection_results: Optional detection dict for source positions.

        Returns:
            Dict with composed_score, pipeline_description, intermediate_results.
        """
        data = image.data if hasattr(image, "data") else image
        data = np.asarray(data, dtype=np.float64)

        # Initialize context with source positions if available
        context: dict[str, Any] = {}
        if detection_results:
            sources = detection_results.get("sources", {})
            positions = sources.get("positions")
            if positions is not None:
                context["positions"] = positions

        intermediate: list[dict[str, Any]] = []

        # Execute operations sequentially
        current = data.copy()
        for op in self.spec.operations:
            current, context = self._registry.execute(op, current, context)
            intermediate.append({
                "op": op.name,
                "params": op.params,
                "shape": list(current.shape),
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
            })

        # Score the final output
        scorer = _SCORERS.get(
            self.spec.score_method, ComposedPipelineScorer.component_count
        )
        try:
            score = scorer(current, context)
        except Exception as e:
            logger.debug(f"Composed scoring failed: {e}")
            score = 0.0

        description = " -> ".join(
            f"{op.name}({', '.join(f'{k}={v}' for k, v in op.params.items())})"
            if op.params
            else op.name
            for op in self.spec.operations
        )
        description += f" -> score:{self.spec.score_method}"

        return {
            "composed_score": float(np.clip(score, 0, 1)),
            "pipeline_description": description,
            "intermediate_results": intermediate,
            "context": {
                k: v
                for k, v in context.items()
                if not isinstance(v, np.ndarray)
            },
        }
