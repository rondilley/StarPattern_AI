"""Transient and variability detection from catalog data."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.transient")


class TransientDetector:
    """Detect variability and transient indicators from catalog data.

    Uses Gaia astrometric excess noise, photometric scatter, and
    parallax anomalies to flag variable stars, unresolved binaries,
    and other transient phenomena.
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.noise_threshold = self.config.transient_noise_threshold
        self.parallax_snr = self.config.transient_parallax_snr

    def analyze(self, catalog: StarCatalog) -> dict[str, Any]:
        """Run transient/variability analysis on catalog entries.

        Args:
            catalog: StarCatalog with Gaia-derived properties.

        Returns:
            Dict with transient_score and flagged candidates.
        """
        results: dict[str, Any] = {}

        try:
            astro_outliers = self._flag_astrometric_outliers(catalog)
            results["astrometric_outliers"] = astro_outliers
        except Exception as e:
            logger.warning(f"Astrometric outlier detection failed: {e}")
            results["astrometric_outliers"] = []

        try:
            photo_outliers = self._flag_photometric_outliers(catalog)
            results["photometric_outliers"] = photo_outliers
        except Exception as e:
            logger.warning(f"Photometric outlier detection failed: {e}")
            results["photometric_outliers"] = []

        try:
            parallax_anom = self._flag_parallax_anomalies(catalog)
            results["parallax_anomalies"] = parallax_anom
        except Exception as e:
            logger.warning(f"Parallax anomaly detection failed: {e}")
            results["parallax_anomalies"] = []

        n_astro = len(results["astrometric_outliers"])
        n_photo = len(results["photometric_outliers"])
        n_parallax = len(results["parallax_anomalies"])

        # Build combined variable_candidates list
        results["variable_candidates"] = (
            results["astrometric_outliers"]
            + results["photometric_outliers"]
        )

        transient_score = float(np.clip(
            0.4 * min(n_astro, 5) / 5.0
            + 0.3 * min(n_photo, 5) / 5.0
            + 0.3 * min(n_parallax, 5) / 5.0,
            0.0, 1.0,
        ))

        results["transient_score"] = transient_score
        results["n_detections"] = n_astro + n_photo + n_parallax

        logger.info(
            f"Transient detection: score={transient_score:.3f}, "
            f"astro={n_astro}, photo={n_photo}, parallax={n_parallax}"
        )
        return results

    def _flag_astrometric_outliers(self, catalog: StarCatalog) -> list[dict[str, Any]]:
        """Flag objects with high astrometric excess noise.

        Gaia's astrometric_excess_noise indicates a poor single-star
        astrometric fit. High values suggest unresolved binaries,
        variable stars, or extended objects.
        """
        outliers = []
        noise_values = []
        entries_with_noise = []

        for entry in catalog.entries:
            noise = entry.properties.get("astrometric_excess_noise")
            if noise is None:
                noise = entry.properties.get("astro_noise")
            if noise is not None:
                try:
                    noise_f = float(noise)
                    if np.isfinite(noise_f):
                        noise_values.append(noise_f)
                        entries_with_noise.append(entry)
                except (ValueError, TypeError):
                    continue

        if not noise_values:
            logger.debug(
                f"No sources with astrometric noise data "
                f"(checked {len(catalog.entries)} entries)"
            )
            return outliers

        noise_arr = np.array(noise_values)
        median_noise = np.median(noise_arr)
        mad = np.median(np.abs(noise_arr - median_noise))
        sigma = mad * 1.4826 if mad > 0 else np.std(noise_arr)

        if sigma < 1e-10:
            return outliers

        for i, entry in enumerate(entries_with_noise):
            deviation = (noise_arr[i] - median_noise) / sigma
            if deviation > self.noise_threshold:
                outliers.append({
                    "type": "astrometric_outlier",
                    "source_id": entry.source_id,
                    "ra": entry.ra,
                    "dec": entry.dec,
                    "astrometric_noise": float(noise_arr[i]),
                    "deviation_sigma": float(deviation),
                })

        return outliers

    def _flag_photometric_outliers(self, catalog: StarCatalog) -> list[dict[str, Any]]:
        """Flag objects with unusual color relative to magnitude.

        Large scatter between BP and RP magnitudes vs the expected
        color-magnitude relation can indicate variability.
        """
        outliers = []
        colors = []
        mags = []
        valid_entries = []

        for entry in catalog.entries:
            props = entry.properties
            # Gaia BP/RP: check both raw column names and short keys
            bp_mag = props.get("phot_bp_mean_mag") or props.get("BP")
            rp_mag = props.get("phot_rp_mean_mag") or props.get("RP")
            g_mag = entry.mag

            color = None
            mag_val = None

            if bp_mag is not None and rp_mag is not None and g_mag is not None:
                try:
                    bp = float(bp_mag)
                    rp = float(rp_mag)
                    g = float(g_mag)
                    if np.isfinite(bp) and np.isfinite(rp) and np.isfinite(g):
                        color = bp - rp
                        mag_val = g
                except (ValueError, TypeError):
                    pass

            # Fallback: SDSS g-r color
            if color is None and g_mag is not None:
                g_sdss = props.get("g")
                r_sdss = props.get("r")
                if g_sdss is not None and r_sdss is not None:
                    try:
                        gv = float(g_sdss)
                        rv = float(r_sdss)
                        mg = float(g_mag)
                        if np.isfinite(gv) and np.isfinite(rv) and np.isfinite(mg):
                            color = gv - rv
                            mag_val = mg
                    except (ValueError, TypeError):
                        pass

            if color is not None and mag_val is not None:
                colors.append(color)
                mags.append(mag_val)
                valid_entries.append(entry)

        if len(colors) < 10:
            return outliers

        colors_arr = np.array(colors)
        mags_arr = np.array(mags)

        # Fit simple linear color-magnitude relation
        try:
            slope, intercept, _, _, _ = np.polynomial.polynomial.polyfit(
                mags_arr, colors_arr, deg=1, full=True
            )[:2] if False else (0, 0, 0, 0, 0)
        except Exception:
            pass

        # Use robust approach: median color in magnitude bins
        n_bins = max(3, len(colors) // 15)
        mag_bins = np.linspace(mags_arr.min(), mags_arr.max(), n_bins + 1)

        for b in range(n_bins):
            mask = (mags_arr >= mag_bins[b]) & (mags_arr < mag_bins[b + 1])
            if np.sum(mask) < 3:
                continue

            bin_colors = colors_arr[mask]
            median_color = np.median(bin_colors)
            mad = np.median(np.abs(bin_colors - median_color))
            sigma = mad * 1.4826 if mad > 0 else np.std(bin_colors)
            if sigma < 1e-6:
                continue

            bin_indices = np.where(mask)[0]
            for k, idx in enumerate(bin_indices):
                deviation = abs(bin_colors[k] - median_color) / sigma
                if deviation > self.noise_threshold:
                    entry = valid_entries[idx]
                    outliers.append({
                        "type": "photometric_outlier",
                        "source_id": entry.source_id,
                        "ra": entry.ra,
                        "dec": entry.dec,
                        "bp_rp_color": float(colors_arr[idx]),
                        "expected_color": float(median_color),
                        "deviation_sigma": float(deviation),
                    })

        return outliers

    def _flag_parallax_anomalies(self, catalog: StarCatalog) -> list[dict[str, Any]]:
        """Flag objects with anomalous parallax measurements.

        Negative parallax or parallax_error >> parallax may indicate
        distant AGN, QSOs, or problematic astrometric solutions (e.g.,
        unresolved binaries causing excess astrometric scatter).
        """
        anomalies = []

        for entry in catalog.entries:
            props = entry.properties
            parallax = props.get("parallax")
            parallax_error = props.get("parallax_error")

            if parallax is None:
                continue

            try:
                plx = float(parallax)
                plx_err = float(parallax_error) if parallax_error is not None else None
            except (ValueError, TypeError):
                continue

            if not np.isfinite(plx):
                continue

            is_anomalous = False
            reason = ""

            # Negative parallax
            if plx < 0:
                is_anomalous = True
                reason = "negative_parallax"
            # Very low SNR parallax
            elif plx_err is not None and np.isfinite(plx_err) and plx_err > 0:
                snr = abs(plx) / plx_err
                if snr < (1.0 / self.parallax_snr):
                    is_anomalous = True
                    reason = "low_parallax_snr"

            if is_anomalous:
                anomalies.append({
                    "type": "parallax_anomaly",
                    "source_id": entry.source_id,
                    "ra": entry.ra,
                    "dec": entry.dec,
                    "parallax": plx,
                    "parallax_error": plx_err,
                    "reason": reason,
                })

        return anomalies
