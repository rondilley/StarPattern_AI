"""Galaxy-specific feature detection: tidal features, mergers, color anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.galaxy")


class GalaxyDetector:
    """Detect galaxy interaction signatures beyond basic CAS morphology.

    Analyzes FITS image data and optional catalog data for:
    - Tidal features (arcs, tails, shells from interactions)
    - Merger signatures (double nuclei, high asymmetry)
    - Color anomalies (unusual BP-RP or g-r vs magnitude)
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.tidal_threshold = self.config.galaxy_tidal_threshold
        self.color_sigma = self.config.galaxy_color_sigma
        self.asymmetry_threshold = self.config.galaxy_asymmetry_threshold

    def detect(
        self,
        data: np.ndarray,
        catalog: StarCatalog | None = None,
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Run all galaxy-specific detection on image data and optional catalog.

        Args:
            data: 2D image array.
            catalog: Optional StarCatalog with photometric properties.
            pixel_scale_arcsec: Pixel scale in arcsec/pixel. If provided,
                filter kernels scale to physical sizes.

        Returns:
            Dict with galaxy_score and sub-results.
        """
        self._pixel_scale = pixel_scale_arcsec
        results: dict[str, Any] = {}

        try:
            tidal = self._detect_tidal_features(data)
            results["tidal_features"] = tidal
        except Exception as e:
            logger.warning(f"Tidal feature detection failed: {e}")
            results["tidal_features"] = []

        try:
            mergers = self._detect_mergers(data)
            results["merger_candidates"] = mergers
        except Exception as e:
            logger.warning(f"Merger detection failed: {e}")
            results["merger_candidates"] = []

        color_outliers: list[dict[str, Any]] = []
        if catalog is not None:
            try:
                color_outliers = self._detect_color_anomalies(catalog)
                results["color_outliers"] = color_outliers
            except Exception as e:
                logger.warning(f"Color anomaly detection failed: {e}")
                results["color_outliers"] = []
        else:
            results["color_outliers"] = []

        # Composite score
        n_tidal = len(results["tidal_features"])
        n_mergers = len(results["merger_candidates"])
        n_color = len(results["color_outliers"])
        total_detections = n_tidal + n_mergers + n_color

        # Score: saturates around 1.0 with multiple detections
        galaxy_score = float(np.clip(
            0.3 * min(n_tidal, 3) / 3.0
            + 0.4 * min(n_mergers, 2) / 2.0
            + 0.3 * min(n_color, 5) / 5.0,
            0.0, 1.0,
        ))

        results["galaxy_score"] = galaxy_score
        results["n_detections"] = total_detections

        logger.info(
            f"Galaxy detection: score={galaxy_score:.3f}, "
            f"tidal={n_tidal}, mergers={n_mergers}, color={n_color}"
        )
        return results

    def _detect_tidal_features(self, data: np.ndarray) -> list[dict[str, Any]]:
        """Detect tidal tails, shells, and streams via residual analysis.

        Subtracts a heavily smoothed model (proxy for smooth galaxy light)
        and searches for extended arc-like residuals using Gabor filters
        at large spatial scales.
        """
        # Smooth model subtraction
        smooth_sigma = max(data.shape) / 20.0
        smooth_model = ndimage.gaussian_filter(data.astype(np.float64), sigma=smooth_sigma)
        residual = data.astype(np.float64) - smooth_model

        # Normalize residual by local RMS using fast uniform_filter
        # (replaces generic_filter with np.std which is O(n*window^2) in Python)
        filter_size = max(3, int(smooth_sigma))
        abs_residual = np.abs(residual)
        local_mean = ndimage.uniform_filter(abs_residual, size=filter_size)
        local_sq_mean = ndimage.uniform_filter(abs_residual ** 2, size=filter_size)
        local_rms = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        local_rms = np.where(local_rms > 0, local_rms, 1.0)
        snr_map = residual / local_rms

        # Gabor filter at large scale to pick up extended features
        features = []
        from scipy.signal import fftconvolve

        freq = 1.0 / (smooth_sigma * 2)
        for angle_idx in range(4):
            theta = angle_idx * np.pi / 4
            kernel_size = int(smooth_sigma * 3) | 1  # ensure odd
            half = kernel_size // 2
            y, x = np.mgrid[-half:half + 1, -half:half + 1]
            sigma_gabor = smooth_sigma
            gabor = np.exp(-(x**2 + y**2) / (2 * sigma_gabor**2)) * np.cos(
                2 * np.pi * freq * (x * np.cos(theta) + y * np.sin(theta))
            )
            response = fftconvolve(snr_map, gabor, mode="same")

            # Threshold for significant detections
            # Use both fraction-of-max AND absolute noise floor (RMS)
            # to prevent noise in blank fields from triggering detections
            resp_std = np.std(response)
            if resp_std < 1e-10:
                continue
            frac_threshold = self.tidal_threshold * np.max(np.abs(response))
            abs_threshold = 3.0 * resp_std
            threshold = max(frac_threshold, abs_threshold)
            peaks = np.abs(response) > threshold
            labeled, n_features = ndimage.label(peaks)

            for label_idx in range(1, n_features + 1):
                region = labeled == label_idx
                area = np.sum(region)
                # Tidal features are extended -- require minimum area
                if area < smooth_sigma * 2:
                    continue
                ys, xs = np.where(region)
                peak_strength = float(np.max(np.abs(response[region])))
                features.append({
                    "type": "tidal",
                    "x": float(np.mean(xs)),
                    "y": float(np.mean(ys)),
                    "area": int(area),
                    "orientation": float(theta),
                    "strength": peak_strength,
                    "tidal_snr": float(peak_strength / max(resp_std, 1e-10)),
                })

        # Deduplicate nearby detections
        features = _deduplicate_detections(features, min_dist=smooth_sigma)
        return features

    def _detect_mergers(self, data: np.ndarray) -> list[dict[str, Any]]:
        """Detect merger candidates via double-nucleus and asymmetry analysis.

        Finds bright compact cores within a galaxy footprint. Two or more
        cores within a characteristic galaxy radius suggest an ongoing merger.
        """
        from scipy.spatial import cKDTree

        mergers = []

        # Find bright peaks (potential nuclei)
        # Scale filter kernel by pixel scale: ~4 arcsec filter radius
        filter_size = 11
        ps = getattr(self, "_pixel_scale", None)
        if ps and ps > 0:
            filter_size = max(5, int(4.0 / ps)) | 1  # ensure odd
        smooth = ndimage.gaussian_filter(data.astype(np.float64), sigma=2.0)
        local_max = ndimage.maximum_filter(smooth, size=filter_size)
        threshold = np.percentile(smooth, 95)
        peaks = (smooth == local_max) & (smooth > threshold)

        peak_coords = np.column_stack(np.where(peaks))
        if len(peak_coords) < 2:
            return mergers

        # Cap peaks to the brightest 200 to avoid O(n^2) blowup
        max_peaks = 200
        if len(peak_coords) > max_peaks:
            brightnesses = smooth[peak_coords[:, 0], peak_coords[:, 1]]
            top_idx = np.argsort(brightnesses)[-max_peaks:]
            peak_coords = peak_coords[top_idx]

        # Typical galaxy spans ~20-60 pixels at survey resolution
        max_sep = max(data.shape) / 8.0

        # Use KDTree for efficient neighbor search instead of O(n^2)
        tree = cKDTree(peak_coords)
        pairs = tree.query_pairs(r=max_sep)

        for i, j in pairs:
            dist = np.linalg.norm(peak_coords[i] - peak_coords[j])

            # Check asymmetry in the region between the two peaks
            cy = int((peak_coords[i][0] + peak_coords[j][0]) / 2)
            cx = int((peak_coords[i][1] + peak_coords[j][1]) / 2)
            r = int(dist)
            y0 = max(0, cy - r)
            y1 = min(data.shape[0], cy + r)
            x0 = max(0, cx - r)
            x1 = min(data.shape[1], cx + r)
            cutout = data[y0:y1, x0:x1].astype(np.float64)

            if cutout.size == 0:
                continue

            # Subtract local background using border pixels (5px ring)
            # This prevents sky background from inflating the denominator
            border_width = min(5, min(cutout.shape) // 4)
            if border_width >= 1:
                border_mask = np.ones(cutout.shape, dtype=bool)
                if (cutout.shape[0] > 2 * border_width
                        and cutout.shape[1] > 2 * border_width):
                    border_mask[border_width:-border_width,
                                border_width:-border_width] = False
                bg_level = np.median(cutout[border_mask])
                bg_rms = np.std(cutout[border_mask])
                cutout = cutout - bg_level

                # Skip cutouts where peak signal is below 5-sigma above noise
                # (noise peaks in large cutouts routinely reach 3-4 sigma)
                if bg_rms > 0 and np.max(cutout) < 5.0 * bg_rms:
                    continue

            # Asymmetry: compare cutout with 180-degree rotation
            rotated = np.rot90(cutout, 2)
            min_h = min(cutout.shape[0], rotated.shape[0])
            min_w = min(cutout.shape[1], rotated.shape[1])
            cutout_trim = cutout[:min_h, :min_w]
            rotated_trim = rotated[:min_h, :min_w]
            total_flux = np.sum(np.abs(cutout_trim))
            if total_flux < 1e-10:
                continue
            asymmetry = float(
                np.sum(np.abs(cutout_trim - rotated_trim)) / (2.0 * total_flux)
            )

            if asymmetry > self.asymmetry_threshold:
                p1_brightness = float(smooth[peak_coords[i][0], peak_coords[i][1]])
                p2_brightness = float(smooth[peak_coords[j][0], peak_coords[j][1]])
                mergers.append({
                    "type": "merger",
                    "nucleus_1": {
                        "y": int(peak_coords[i][0]),
                        "x": int(peak_coords[i][1]),
                    },
                    "nucleus_2": {
                        "y": int(peak_coords[j][0]),
                        "x": int(peak_coords[j][1]),
                    },
                    "separation_px": float(dist),
                    "asymmetry": asymmetry,
                    "asymmetry_sigma": float(
                        asymmetry / max(self.asymmetry_threshold, 1e-10)
                    ),
                    "flux_ratio": float(
                        min(p1_brightness, p2_brightness)
                        / max(p1_brightness, p2_brightness)
                    ) if max(p1_brightness, p2_brightness) > 0 else 0.0,
                })

            # Cap output to avoid memory blowup
            if len(mergers) >= 50:
                break

        # Sort by asymmetry descending, return top candidates
        mergers.sort(key=lambda m: m["asymmetry"], reverse=True)
        return mergers[:50]

    def _detect_color_anomalies(self, catalog: StarCatalog) -> list[dict[str, Any]]:
        """Flag objects with unusual colors relative to the population.

        Uses BP-RP color from Gaia or g-r from SDSS stored in
        entry.properties. Objects beyond color_sigma from the median
        color-magnitude relation are flagged.
        """
        outliers = []

        # Try BP-RP first (Gaia), then g-r (SDSS)
        colors = []
        mags = []
        indices = []

        for i, entry in enumerate(catalog.entries):
            props = entry.properties
            bp_rp = props.get("bp_rp")
            g_r = props.get("g_r")
            color = bp_rp if bp_rp is not None else g_r
            mag = entry.mag

            if color is not None and mag is not None:
                try:
                    c = float(color)
                    m = float(mag)
                    if np.isfinite(c) and np.isfinite(m):
                        colors.append(c)
                        mags.append(m)
                        indices.append(i)
                except (ValueError, TypeError):
                    continue

        if len(colors) < 10:
            return outliers

        colors_arr = np.array(colors)
        mags_arr = np.array(mags)

        # Bin by magnitude and compute median color in each bin
        n_bins = max(3, len(colors) // 20)
        mag_bins = np.linspace(mags_arr.min(), mags_arr.max(), n_bins + 1)

        for b in range(n_bins):
            mask = (mags_arr >= mag_bins[b]) & (mags_arr < mag_bins[b + 1])
            if np.sum(mask) < 3:
                continue

            bin_colors = colors_arr[mask]
            median_color = np.median(bin_colors)
            std_color = np.std(bin_colors)
            if std_color < 1e-6:
                continue

            deviations = np.abs(bin_colors - median_color) / std_color
            bin_indices = np.array(indices)[mask]

            for k, (dev, idx) in enumerate(zip(deviations, bin_indices)):
                if dev > self.color_sigma:
                    entry = catalog.entries[idx]
                    outliers.append({
                        "type": "color_outlier",
                        "source_id": entry.source_id,
                        "ra": entry.ra,
                        "dec": entry.dec,
                        "mag": entry.mag,
                        "color": float(bin_colors[k]),
                        "median_color": float(median_color),
                        "deviation_sigma": float(dev),
                    })

        return outliers


def _deduplicate_detections(
    detections: list[dict[str, Any]],
    min_dist: float,
) -> list[dict[str, Any]]:
    """Remove duplicate detections within min_dist pixels, keeping strongest."""
    if not detections:
        return detections

    # Sort by strength descending
    sorted_dets = sorted(detections, key=lambda d: d.get("strength", 0), reverse=True)
    kept = []

    for det in sorted_dets:
        too_close = False
        for existing in kept:
            dx = det["x"] - existing["x"]
            dy = det["y"] - existing["y"]
            if np.sqrt(dx**2 + dy**2) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(det)

    return kept
