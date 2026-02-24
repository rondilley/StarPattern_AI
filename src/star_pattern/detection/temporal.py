"""Temporal change detection via multi-epoch image differencing.

Builds a reference image from multiple epochs, subtracts each epoch,
and detects residuals (new sources, disappearances, brightenings,
fadings, moving objects) from the difference images.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label

from star_pattern.core.config import DetectionConfig
from star_pattern.core.sky_region import EpochImage
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.temporal")


class TemporalDetector:
    """Detect changes across multiple epochs of imaging.

    Uses WCS reprojection to align epochs, median-stacking to build a
    reference (which naturally rejects transients), and connected-component
    labeling on SNR-thresholded difference images to find residuals.
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self._snr_threshold = getattr(config, "temporal_snr_threshold", 5.0) if config else 5.0
        self._min_epochs = getattr(config, "temporal_min_epochs", 2) if config else 2
        self._dipole_max_sep = getattr(config, "temporal_dipole_max_sep", 5.0) if config else 5.0
        self.diagnostics: dict[str, Any] | None = None

    def analyze(
        self,
        epoch_images: list[EpochImage],
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Run temporal change detection on a list of epoch images.

        Args:
            epoch_images: List of EpochImage, each with .image (FITSImage) and .mjd.
            pixel_scale_arcsec: Pixel scale for dipole separation calculation.

        Returns:
            Dict with temporal_score and classified findings.
        """
        # Clear diagnostics so stale data never persists from a prior call
        self.diagnostics = None

        if len(epoch_images) < self._min_epochs:
            return {"temporal_score": 0}

        # Sort by MJD
        sorted_epochs = sorted(epoch_images, key=lambda e: e.mjd)

        # Build reference image
        reference, ref_wcs = self._build_reference(sorted_epochs)
        if reference is None:
            return {"temporal_score": 0}

        # Get pixel scale from reference WCS or parameter
        if pixel_scale_arcsec is None:
            try:
                ps = sorted_epochs[0].image.pixel_scale()
                pixel_scale_arcsec = ps if ps and ps > 0 else 1.0
            except Exception:
                pixel_scale_arcsec = 1.0

        # Subtract each epoch and collect residuals + diagnostics
        all_residuals: list[dict[str, Any]] = []
        diag_diffs: list[dict[str, Any]] = []
        diag_snr_maps: list[dict[str, Any]] = []
        n_residuals_per_epoch: list[int] = []
        for epoch in sorted_epochs:
            diff, noise = self._subtract_epoch(epoch, reference, ref_wcs)
            if diff is None:
                continue
            # SNR map: NaN in non-overlap regions for clean visualization
            snr_map = np.abs(diff) / noise
            residuals = self._detect_residuals(
                diff, noise, epoch.image.wcs, epoch.mjd
            )
            # Cap residuals per epoch to prevent O(n^2) cross-matching blowup
            if len(residuals) > 200:
                residuals.sort(key=lambda r: r["peak_snr"], reverse=True)
                residuals = residuals[:200]
            all_residuals.append({
                "mjd": epoch.mjd,
                "residuals": residuals,
            })
            diag_diffs.append({
                "mjd": epoch.mjd, "data": diff, "noise": noise,
            })
            diag_snr_maps.append({
                "mjd": epoch.mjd, "data": snr_map,
            })
            n_residuals_per_epoch.append(len(residuals))

        if not all_residuals:
            return {"temporal_score": 0}

        # Store diagnostics for visualization
        self.diagnostics = {
            "reference_image": reference,
            "reference_wcs": ref_wcs,
            "diff_images": diag_diffs,
            "snr_maps": diag_snr_maps,
            "n_residuals_per_epoch": n_residuals_per_epoch,
        }

        # Classify residuals across epochs
        classified = self._classify_residuals(
            all_residuals, pixel_scale_arcsec
        )

        # Compute temporal score
        temporal_score = self._compute_temporal_score(classified)

        # Compute baseline
        baseline_days = sorted_epochs[-1].mjd - sorted_epochs[0].mjd

        return {
            "temporal_score": temporal_score,
            "n_epochs_analyzed": len(all_residuals),
            "baseline_days": baseline_days,
            "new_sources": classified.get("new_sources", []),
            "disappeared": classified.get("disappeared", []),
            "brightenings": classified.get("brightenings", []),
            "fadings": classified.get("fadings", []),
            "moving_objects": classified.get("moving_objects", []),
            "n_new_sources": len(classified.get("new_sources", [])),
            "n_disappeared": len(classified.get("disappeared", [])),
            "n_brightenings": len(classified.get("brightenings", [])),
            "n_fadings": len(classified.get("fadings", [])),
            "n_moving": len(classified.get("moving_objects", [])),
        }

    def _build_reference(
        self, sorted_epochs: list[EpochImage]
    ) -> tuple[np.ndarray | None, Any]:
        """Build a median-stack reference image from all epochs.

        Median naturally rejects single-epoch transients.
        All epochs are reprojected onto the WCS of the first epoch.

        Returns:
            (reference_array, reference_wcs) or (None, None) on failure.
        """
        try:
            from reproject import reproject_interp
        except ImportError:
            logger.warning("reproject not available for temporal analysis")
            return None, None

        ref_epoch = sorted_epochs[0]
        ref_wcs = ref_epoch.image.wcs
        ref_shape = ref_epoch.image.data.shape

        if ref_wcs is None:
            # No WCS -- fall back to direct pixel stacking
            return self._build_reference_pixel(sorted_epochs)

        reprojected: list[np.ndarray] = []
        for epoch in sorted_epochs:
            if epoch.image.wcs is None:
                # Stack directly if no WCS (assume aligned)
                data = epoch.image.data
                if data.shape == ref_shape:
                    reprojected.append(data.astype(np.float64))
                continue

            try:
                from astropy.io.fits import Header
                from astropy.wcs import WCS
                reproj, footprint = reproject_interp(
                    (epoch.image.data.astype(np.float64), epoch.image.wcs),
                    ref_wcs,
                    shape_out=ref_shape,
                )
                # Mask out-of-footprint pixels
                reproj[footprint < 0.5] = np.nan
                reprojected.append(reproj)
            except Exception as e:
                logger.debug(f"Reprojection failed for epoch MJD={epoch.mjd:.2f}: {e}")
                # Fall back to direct stacking if shapes match
                if epoch.image.data.shape == ref_shape:
                    reprojected.append(epoch.image.data.astype(np.float64))

        if len(reprojected) < 2:
            return None, None

        stack = np.stack(reprojected)
        # Count how many epochs contributed valid data at each pixel
        coverage = np.sum(np.isfinite(stack), axis=0)
        reference = np.nanmedian(stack, axis=0)
        # Keep NaN where fewer than 2 epochs overlap -- these pixels
        # have no meaningful reference and must not produce false signals
        reference[coverage < 2] = np.nan

        return reference, ref_wcs

    def _build_reference_pixel(
        self, sorted_epochs: list[EpochImage]
    ) -> tuple[np.ndarray | None, Any]:
        """Build reference by direct pixel stacking (no WCS alignment)."""
        ref_shape = sorted_epochs[0].image.data.shape
        stack = []
        for epoch in sorted_epochs:
            if epoch.image.data.shape == ref_shape:
                stack.append(epoch.image.data.astype(np.float64))
        if len(stack) < 2:
            return None, None
        reference = np.median(np.stack(stack), axis=0)
        return reference, None

    def _subtract_epoch(
        self,
        epoch: EpochImage,
        reference: np.ndarray,
        ref_wcs: Any,
    ) -> tuple[np.ndarray | None, float]:
        """Subtract reference from an epoch, return (diff, noise_estimate).

        Reprojects the epoch onto the reference WCS, then subtracts.
        Noise is estimated via MAD (robust to real signals).
        """
        data = epoch.image.data.astype(np.float64)

        if ref_wcs is not None and epoch.image.wcs is not None:
            try:
                from reproject import reproject_interp
                reproj, footprint = reproject_interp(
                    (data, epoch.image.wcs),
                    ref_wcs,
                    shape_out=reference.shape,
                )
                reproj[footprint < 0.5] = np.nan
                data = reproj
            except Exception as e:
                logger.debug(f"Epoch reprojection failed: {e}")
                if data.shape != reference.shape:
                    return None, 0.0
        elif data.shape != reference.shape:
            return None, 0.0

        # Difference -- NaN propagates where either epoch or reference
        # has no valid data, preventing false signals at partial-overlap edges.
        diff = data - reference

        # Estimate noise only from the valid (overlapping) region
        valid_mask = np.isfinite(diff)
        n_valid = int(np.sum(valid_mask))
        if n_valid < 100:
            return None, 0.0

        valid_diff = diff[valid_mask]

        # Robust noise estimate via MAD of valid pixels only
        noise = 1.4826 * float(np.median(np.abs(valid_diff - np.median(valid_diff))))

        # Floor at the RMS of the quietest 80% of valid pixels
        flat = np.sort(np.abs(valid_diff))
        n80 = int(0.8 * len(flat))
        if n80 > 10:
            rms_80 = float(np.sqrt(np.mean(flat[:n80] ** 2)))
            noise = max(noise, rms_80)

        # Absolute floor based on valid reference pixels
        ref_valid = reference[np.isfinite(reference)]
        img_rms = float(np.std(ref_valid)) if len(ref_valid) > 0 else 1.0
        noise = max(noise, img_rms * 0.01, 1e-6)

        return diff, noise

    def _detect_residuals(
        self,
        diff: np.ndarray,
        noise: float,
        wcs: Any,
        mjd: float,
    ) -> list[dict[str, Any]]:
        """Detect significant residuals in a difference image.

        Uses connected-component labeling on |diff/noise| > threshold.
        """
        # Replace NaN with 0 for labeling -- NaN pixels are non-overlap
        # regions and must not trigger detections
        diff_safe = np.nan_to_num(diff, nan=0.0)
        snr_map = np.abs(diff_safe) / noise
        binary = snr_map > self._snr_threshold
        labeled, n_components = label(binary)

        residuals: list[dict[str, Any]] = []
        for comp_id in range(1, n_components + 1):
            mask = labeled == comp_id
            pixels = diff_safe[mask]
            abs_snr = snr_map[mask]

            # Properties
            total_flux = float(np.sum(pixels))
            peak_snr = float(np.max(abs_snr))
            area = int(np.sum(mask))
            sign = "positive" if total_flux > 0 else "negative"

            # Flux-weighted centroid in pixel coords
            ys, xs = np.where(mask)
            weights = np.abs(pixels)
            w_sum = np.sum(weights)
            if w_sum > 0:
                cx = float(np.sum(xs * weights) / w_sum)
                cy = float(np.sum(ys * weights) / w_sum)
            else:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))

            # Convert to sky coords
            sky_ra, sky_dec = None, None
            if wcs is not None:
                try:
                    coord = wcs.pixel_to_world(cx, cy)
                    sky_ra = float(coord.ra.deg)
                    sky_dec = float(coord.dec.deg)
                except Exception:
                    pass

            residuals.append({
                "cx": cx,
                "cy": cy,
                "sky_ra": sky_ra,
                "sky_dec": sky_dec,
                "peak_snr": peak_snr,
                "total_flux": total_flux,
                "area": area,
                "sign": sign,
                "mjd": mjd,
            })

        return residuals

    def _classify_residuals(
        self,
        all_residuals: list[dict[str, Any]],
        pixel_scale_arcsec: float,
    ) -> dict[str, list[dict[str, Any]]]:
        """Cross-match residuals across epochs and classify temporal changes.

        Classification types:
        - temporal_new_source: positive residual in later epoch(s), absent earlier
        - temporal_disappeared: negative residual in later epoch(s)
        - temporal_brightening: positive residual at same position across epochs
        - temporal_fading: negative residual at same position across epochs
        - temporal_moving: positive+negative dipole that shifts between epochs
        """
        if not all_residuals:
            return {}

        dipole_max_pix = self._dipole_max_sep / pixel_scale_arcsec if pixel_scale_arcsec and pixel_scale_arcsec > 0 else 5.0

        # Flatten all residuals with epoch index
        flat: list[dict[str, Any]] = []
        for epoch_data in all_residuals:
            mjd = epoch_data["mjd"]
            for r in epoch_data["residuals"]:
                flat.append({**r, "mjd": mjd})

        if not flat:
            return {}

        # Group by spatial position (cross-match within dipole_max_pix)
        matched_groups: list[list[dict[str, Any]]] = []
        used = set()

        for i, r in enumerate(flat):
            if i in used:
                continue
            group = [r]
            used.add(i)
            for j in range(i + 1, len(flat)):
                if j in used:
                    continue
                dx = r["cx"] - flat[j]["cx"]
                dy = r["cy"] - flat[j]["cy"]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < dipole_max_pix:
                    group.append(flat[j])
                    used.add(j)
            matched_groups.append(group)

        new_sources: list[dict[str, Any]] = []
        disappeared: list[dict[str, Any]] = []
        brightenings: list[dict[str, Any]] = []
        fadings: list[dict[str, Any]] = []
        moving_objects: list[dict[str, Any]] = []

        n_epochs = len(all_residuals)

        for group in matched_groups:
            positive = [r for r in group if r["sign"] == "positive"]
            negative = [r for r in group if r["sign"] == "negative"]
            unique_mjds = set(r["mjd"] for r in group)

            # Use the best residual for location
            best = max(group, key=lambda r: abs(r["peak_snr"]))
            finding = {
                "sky_ra": best.get("sky_ra"),
                "sky_dec": best.get("sky_dec"),
                "cx": best["cx"],
                "cy": best["cy"],
                "peak_snr": best["peak_snr"],
                "n_epochs_detected": len(unique_mjds),
            }

            if positive and negative:
                # Both positive and negative residuals at same position.
                # This is expected from median subtraction: a source present
                # in only some epochs will leave positive residuals in those
                # epochs and negative residuals in the others.
                pos_mjds = sorted(set(r["mjd"] for r in positive))
                neg_mjds = sorted(set(r["mjd"] for r in negative))

                # Check if positive residuals are in later epochs (new/brightening)
                # vs earlier epochs (fading/disappeared)
                if pos_mjds and neg_mjds:
                    mean_pos_t = np.mean(pos_mjds)
                    mean_neg_t = np.mean(neg_mjds)
                    if mean_pos_t > mean_neg_t:
                        # Source appeared/brightened in later epochs
                        if len(pos_mjds) >= 2:
                            brightenings.append(finding)
                        else:
                            new_sources.append(finding)
                    elif mean_neg_t > mean_pos_t:
                        # Source faded/disappeared in later epochs
                        if len(neg_mjds) >= 2:
                            fadings.append(finding)
                        else:
                            disappeared.append(finding)
                    else:
                        # Simultaneous positive+negative at different positions = moving
                        # Check if centroids differ significantly
                        pos_cx = np.mean([r["cx"] for r in positive])
                        neg_cx = np.mean([r["cx"] for r in negative])
                        pos_cy = np.mean([r["cy"] for r in positive])
                        neg_cy = np.mean([r["cy"] for r in negative])
                        sep = np.sqrt((pos_cx - neg_cx)**2 + (pos_cy - neg_cy)**2)
                        if sep > 1.0:
                            moving_objects.append(finding)
                        else:
                            brightenings.append(finding)
                continue

            if len(positive) > 0 and len(negative) == 0:
                if len(unique_mjds) >= 2:
                    brightenings.append(finding)
                else:
                    new_sources.append(finding)
            elif len(negative) > 0 and len(positive) == 0:
                if len(unique_mjds) >= 2:
                    fadings.append(finding)
                else:
                    disappeared.append(finding)

        return {
            "new_sources": new_sources,
            "disappeared": disappeared,
            "brightenings": brightenings,
            "fadings": fadings,
            "moving_objects": moving_objects,
        }

    def _compute_temporal_score(
        self, classified: dict[str, list[dict[str, Any]]]
    ) -> float:
        """Compute an overall temporal anomaly score from classified findings.

        Weights: new sources and moving objects score highest, then
        brightenings/fadings, then disappeared.
        """
        n_new = len(classified.get("new_sources", []))
        n_disappeared = len(classified.get("disappeared", []))
        n_bright = len(classified.get("brightenings", []))
        n_fading = len(classified.get("fadings", []))
        n_moving = len(classified.get("moving_objects", []))

        total_count = n_new + n_disappeared + n_bright + n_fading + n_moving
        if total_count == 0:
            return 0.0

        # Peak SNR across all findings
        all_findings = []
        for findings_list in classified.values():
            all_findings.extend(findings_list)
        max_snr = max((f.get("peak_snr", 0) for f in all_findings), default=0)

        # Count-based score (saturates at ~10 findings)
        count_score = min(total_count / 10.0, 1.0)

        # SNR-based score (saturates at SNR ~20)
        snr_score = min(max_snr / 20.0, 1.0) if max_snr > 0 else 0.0

        # Weight by type importance
        type_score = min(
            (n_new * 0.3 + n_moving * 0.25 + n_bright * 0.2
             + n_fading * 0.15 + n_disappeared * 0.1) / 3.0,
            1.0,
        )

        return float(np.clip(
            0.4 * snr_score + 0.35 * type_score + 0.25 * count_score,
            0.0, 1.0,
        ))
