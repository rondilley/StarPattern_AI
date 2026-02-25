"""Wavelet multi-scale analysis for astronomical image processing.

The a-trous (stationary) wavelet transform is the standard multi-scale
decomposition in astronomy, used by SExtractor, MR/1 (Starck & Murtagh),
and many survey pipelines. It decomposes an image into detail coefficient
planes at dyadic scales (1, 2, 4, 8, ... pixels) plus a smooth residual.

Key advantages over Gabor filters:
- Perfect reconstruction (no information loss)
- Scale-dependent significance testing with well-defined noise properties
- Point sources appear at fine scales, extended emission at coarse scales
- Standard B3 spline kernel preserves positivity

This module provides:
- A-trous wavelet decomposition
- Scale-dependent significance maps (wavelet-based source detection)
- Multi-scale anomaly detection (features that deviate from expected behavior)
- Scale spectrum analysis (power distribution across scales)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from star_pattern.utils.gpu import get_array_module, to_device, to_numpy
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.wavelet")

# B3 spline scaling function kernel (1D)
# This is the standard kernel used in astronomical wavelet transforms
B3_SPLINE_1D = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def _atrous_convolve_2d(data: np.ndarray, scale: int) -> np.ndarray:
    """Apply a-trous convolution at a given scale.

    The a-trous algorithm inserts (2^scale - 1) zeros between kernel elements,
    equivalent to convolving with a dilated kernel. Implemented via separable
    1D convolution with a sparse kernel.

    Uses CuPy GPU acceleration when available.

    Args:
        data: 2D image array.
        scale: Scale level (0, 1, 2, ...). Dilation = 2^scale.

    Returns:
        Smoothed image at this scale.
    """
    step = 2 ** scale
    # Build dilated 1D kernel
    kernel_1d = np.zeros(1 + (len(B3_SPLINE_1D) - 1) * step, dtype=np.float64)
    kernel_1d[::step] = B3_SPLINE_1D

    xp, use_gpu = get_array_module()
    if use_gpu:
        try:
            import cupyx.scipy.ndimage as cu_ndimage

            gpu_data = to_device(data, xp) if isinstance(data, np.ndarray) else data
            gpu_kernel = to_device(kernel_1d, xp)
            smoothed = cu_ndimage.convolve1d(gpu_data, gpu_kernel, axis=1, mode="reflect")
            smoothed = cu_ndimage.convolve1d(smoothed, gpu_kernel, axis=0, mode="reflect")
            return to_numpy(smoothed)
        except Exception:
            pass

    # Separable 2D convolution (row then column)
    smoothed = ndimage.convolve1d(data, kernel_1d, axis=1, mode="reflect")
    smoothed = ndimage.convolve1d(smoothed, kernel_1d, axis=0, mode="reflect")
    return smoothed


def atrous_decompose(
    data: np.ndarray,
    n_scales: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """A-trous (stationary) wavelet decomposition.

    Decomposes image into n_scales wavelet detail planes W_j plus a smooth
    residual c_J. The original image = sum(W_j) + c_J (perfect reconstruction).

    Args:
        data: 2D image array.
        n_scales: Number of wavelet scales.

    Returns:
        Tuple of (detail_planes, smooth_residual) where detail_planes[j]
        contains features at scale 2^j pixels.
    """
    current = data.astype(np.float64)
    details = []

    for j in range(n_scales):
        smoothed = _atrous_convolve_2d(current, j)
        detail = current - smoothed
        details.append(detail)
        current = smoothed

    return details, current


class WaveletAnalyzer:
    """Multi-scale wavelet analysis for astronomical images.

    Detects features at each spatial scale and computes scale-dependent
    significance maps. Identifies:
    - Point sources (stars) at scale 0-1
    - Compact extended sources (small galaxies) at scale 2-3
    - Diffuse emission (nebulae, tidal features) at scale 3-5
    - Large-scale structure gradients at highest scales

    Parameters:
        n_scales: Number of wavelet decomposition scales.
        significance_threshold: Sigma threshold for significant coefficients.
        noise_estimation: Method for noise estimation ('mad' or 'std').
    """

    def __init__(
        self,
        n_scales: int = 5,
        significance_threshold: float = 3.0,
        noise_estimation: str = "mad",
    ):
        self.n_scales = n_scales
        self.significance_threshold = significance_threshold
        self.noise_estimation = noise_estimation

    def analyze(
        self,
        data: np.ndarray,
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Run full wavelet analysis on an image.

        Args:
            data: 2D image array.
            pixel_scale_arcsec: Pixel scale for physical scale reporting.

        Returns:
            Dict with wavelet decomposition results and wavelet_score.
        """
        # Limit n_scales based on image size (scale 2^j must fit in image)
        max_scales = max(1, int(np.log2(min(data.shape))) - 2)
        n_scales = min(self.n_scales, max_scales)

        details, smooth = atrous_decompose(data, n_scales)

        results: dict[str, Any] = {
            "n_scales": n_scales,
            "scale_analysis": [],
        }

        # Analyze each scale
        total_significant = 0
        scale_energies = []
        multiscale_detections = []

        for j, detail in enumerate(details):
            scale_px = 2 ** j
            scale_arcsec = scale_px * pixel_scale_arcsec if pixel_scale_arcsec else None

            # Noise estimation at this scale
            noise = self._estimate_noise(detail)

            # Significance map
            sig_map = np.abs(detail) / max(noise, 1e-10)

            # Count significant coefficients
            significant = sig_map > self.significance_threshold
            n_significant = int(np.sum(significant))
            total_significant += n_significant

            # Energy at this scale
            energy = float(np.sum(detail ** 2))
            scale_energies.append(energy)

            # Find significant connected regions at this scale
            labeled, n_features = ndimage.label(significant)
            scale_features = []

            if n_features > 0:
                # Compute areas for ALL labels in one pass (avoids
                # raster-scan bias from capping label indices)
                label_indices = np.arange(1, n_features + 1)
                areas = ndimage.sum(
                    significant, labeled, label_indices,
                ).astype(int)

                # Filter by minimum area
                valid_mask = areas >= 3
                valid_labels = label_indices[valid_mask]
                valid_areas = areas[valid_mask]

                # Use find_objects for efficient bounding-box access
                slices = ndimage.find_objects(labeled)

                # If still very many, pre-filter by peak significance
                # using bounding-box slices (cheap)
                candidates = []
                for idx, label_idx in enumerate(valid_labels):
                    sl = slices[label_idx - 1]
                    if sl is None:
                        continue
                    local_region = labeled[sl] == label_idx
                    local_sig = sig_map[sl]
                    peak_sig = float(np.max(local_sig[local_region]))
                    candidates.append((label_idx, sl, valid_areas[idx], peak_sig))

                # Sort by peak significance and take top 50
                candidates.sort(key=lambda c: c[3], reverse=True)
                for label_idx, sl, area, peak_sig in candidates[:50]:
                    local_region = labeled[sl] == label_idx
                    local_ys, local_xs = np.where(local_region)
                    mean_x = float(np.mean(local_xs)) + sl[1].start
                    mean_y = float(np.mean(local_ys)) + sl[0].start

                    feature = {
                        "scale": j,
                        "scale_px": scale_px,
                        "x": mean_x,
                        "y": mean_y,
                        "area_px": int(area),
                        "peak_significance": peak_sig,
                        "peak_snr": peak_sig,
                        "is_positive": bool(
                            np.mean(detail[sl][local_region]) > 0
                        ),
                    }
                    if scale_arcsec is not None:
                        feature["scale_arcsec"] = scale_arcsec
                    scale_features.append(feature)

            scale_features.sort(key=lambda f: f["peak_significance"], reverse=True)

            scale_info = {
                "scale": j,
                "scale_px": scale_px,
                "noise_level": float(noise),
                "energy": energy,
                "n_significant": n_significant,
                "significant_fraction": n_significant / max(detail.size, 1),
                "n_features": len(scale_features),
            }
            if scale_arcsec is not None:
                scale_info["scale_arcsec"] = scale_arcsec

            results["scale_analysis"].append(scale_info)
            multiscale_detections.extend(scale_features[:10])

        results["detections"] = multiscale_detections

        # Scale spectrum: energy distribution across scales
        total_energy = sum(scale_energies) + 1e-10
        scale_spectrum = [e / total_energy for e in scale_energies]
        results["scale_spectrum"] = scale_spectrum

        # Detect multi-scale anomalies (features present at multiple scales)
        multiscale_objects = self._find_multiscale_objects(
            multiscale_detections, n_scales,
        )
        results["multiscale_objects"] = multiscale_objects

        # Scale concentration: where is most energy?
        if scale_energies:
            weighted_scale = sum(
                j * e for j, e in enumerate(scale_energies)
            ) / total_energy
            results["mean_scale"] = float(weighted_scale)
        else:
            results["mean_scale"] = 0.0

        # Compute composite score
        results["wavelet_score"] = self._compute_score(results, n_scales)

        logger.info(
            f"Wavelet analysis: {n_scales} scales, "
            f"{len(multiscale_detections)} features, "
            f"{len(multiscale_objects)} multi-scale objects, "
            f"score={results['wavelet_score']:.3f}"
        )
        return results

    def _estimate_noise(self, detail: np.ndarray) -> float:
        """Estimate noise level in a wavelet detail plane.

        Uses Median Absolute Deviation (MAD) which is robust to signal
        contamination, following Starck & Murtagh (2002).
        """
        if self.noise_estimation == "mad":
            # MAD estimator: sigma = 1.4826 * median(|x - median(x)|)
            median_val = np.median(detail)
            mad = np.median(np.abs(detail - median_val))
            return float(1.4826 * mad)
        else:
            return float(np.std(detail))

    def _find_multiscale_objects(
        self,
        detections: list[dict[str, Any]],
        n_scales: int,
    ) -> list[dict[str, Any]]:
        """Find objects that appear significant across multiple wavelet scales.

        Objects detected at multiple scales are more likely to be real
        astronomical sources rather than noise or artifacts.
        """
        if not detections:
            return []

        from scipy.spatial import cKDTree

        # Group nearby detections across scales using spatial index
        objects: list[dict[str, Any]] = []
        used = set()

        # Sort by peak significance
        sorted_dets = sorted(
            detections, key=lambda d: d["peak_significance"], reverse=True
        )

        # Build KD-tree for O(n log n) spatial queries
        positions = np.array([[d["x"], d["y"]] for d in sorted_dets])
        tree = cKDTree(positions)

        for i, det in enumerate(sorted_dets):
            if i in used:
                continue

            # Find detections at other scales near this position
            group = [det]
            scales_present = {det["scale"]}
            used.add(i)

            # Query tree for nearby detections (use max possible match radius)
            max_match_radius = max(2 ** n_scales * 2, 5)
            neighbors = tree.query_ball_point([det["x"], det["y"]], max_match_radius)

            for j in neighbors:
                if j in used:
                    continue
                other = sorted_dets[j]
                if other["scale"] in scales_present:
                    continue
                # Check scale-dependent match radius
                match_radius = max(2 ** other["scale"] * 2, 5)
                dist = np.sqrt((det["x"] - other["x"]) ** 2 + (det["y"] - other["y"]) ** 2)
                if dist < match_radius:
                    group.append(other)
                    scales_present.add(other["scale"])
                    used.add(j)

            if len(scales_present) >= 2:
                # Multi-scale detection
                objects.append({
                    "x": float(np.mean([d["x"] for d in group])),
                    "y": float(np.mean([d["y"] for d in group])),
                    "n_scales": len(scales_present),
                    "scales": sorted(scales_present),
                    "max_significance": float(
                        max(d["peak_significance"] for d in group)
                    ),
                    "total_area": sum(d["area_px"] for d in group),
                    "is_extended": max(scales_present) >= 3,
                })

        objects.sort(key=lambda o: o["max_significance"], reverse=True)
        return objects[:30]

    def _compute_score(self, results: dict[str, Any], n_scales: int) -> float:
        """Compute wavelet_score [0, 1] reflecting multi-scale interest."""
        score = 0.0

        # Multi-scale objects are interesting
        n_multiscale = len(results.get("multiscale_objects", []))
        score += min(n_multiscale / 10, 0.3)

        # Extended objects (detected at coarse scales) are interesting
        n_extended = sum(
            1 for obj in results.get("multiscale_objects", [])
            if obj.get("is_extended", False)
        )
        score += min(n_extended / 5, 0.2)

        # Unusual scale spectrum (not dominated by noise at scale 0)
        spectrum = results.get("scale_spectrum", [])
        if len(spectrum) >= 3:
            # Energy ratio: coarse scales vs fine scales
            fine_energy = sum(spectrum[:2])
            coarse_energy = sum(spectrum[2:])
            if fine_energy > 0:
                ratio = coarse_energy / max(fine_energy, 1e-10)
                # High ratio = unusual extended emission
                score += min(ratio / 2, 0.2)

        # Total significant features
        total_features = len(results.get("detections", []))
        score += min(total_features / 30, 0.15)

        # Highly significant detections
        max_sig = max(
            (d["peak_significance"] for d in results.get("detections", [])),
            default=0,
        )
        if max_sig > 10:
            score += 0.15

        return float(np.clip(score, 0, 1))
