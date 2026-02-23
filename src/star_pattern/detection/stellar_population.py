"""Stellar population analysis via color-magnitude diagrams (CMDs).

The color-magnitude diagram (CMD, also known as the Hertzsprung-Russell diagram
for observational data) is the fundamental tool for stellar population analysis.
By plotting color (e.g., Gaia BP-RP or SDSS g-r) against magnitude (G or g),
distinct evolutionary sequences become visible:

- Main Sequence (MS): hydrogen-burning stars, diagonal band
- Red Giant Branch (RGB): evolved stars ascending from MS turnoff
- Horizontal Branch (HB): core-helium-burning, horizontal feature
- Blue Stragglers (BS): anomalously blue MS stars (binary mergers/mass transfer)
- White Dwarf Sequence (WD): faint, blue objects below MS

This module identifies these populations, detects unusual CMD features,
and estimates basic stellar population properties from catalog photometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from star_pattern.core.catalog import StarCatalog
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.stellar_population")

# Empirical main sequence locus in Gaia (BP-RP, M_G) space
# Derived from Gaia DR3 solar neighborhood CMD
# (BP-RP color, approximate absolute magnitude M_G)
MS_LOCUS_GAIA = np.array([
    (-0.2, -1.0),
    (0.3, 1.0),
    (0.6, 3.0),
    (0.9, 4.5),
    (1.2, 6.0),
    (1.5, 7.5),
    (2.0, 9.0),
    (2.5, 11.0),
    (3.0, 13.0),
    (3.5, 15.0),
])


class StellarPopulationAnalyzer:
    """Analyze stellar populations through color-magnitude diagrams.

    Uses catalog photometry (Gaia BP-RP/G or SDSS g-r/g) to identify
    stellar evolutionary sequences and population anomalies.

    Parameters:
        ms_width: Half-width of main sequence band in color (magnitudes).
        blue_straggler_offset: Color offset blueward of MS turnoff for BS detection.
        density_grid_size: Grid size for CMD density estimation.
        min_sources: Minimum catalog entries needed for analysis.
    """

    def __init__(
        self,
        ms_width: float = 0.3,
        blue_straggler_offset: float = 0.3,
        density_grid_size: int = 50,
        min_sources: int = 20,
    ):
        self.ms_width = ms_width
        self.blue_straggler_offset = blue_straggler_offset
        self.density_grid_size = density_grid_size
        self.min_sources = min_sources

    def analyze(self, catalog: StarCatalog) -> dict[str, Any]:
        """Analyze stellar populations in a catalog.

        Args:
            catalog: StarCatalog with photometric properties.

        Returns:
            Dict with population analysis, CMD features, and population_score.
        """
        results: dict[str, Any] = {}

        # Extract color-magnitude data
        colors, mags, source_ids = self._extract_cmd_data(catalog)

        if len(colors) < self.min_sources:
            results["population_score"] = 0.0
            results["reason"] = "insufficient_photometry"
            results["n_photometric"] = len(colors)
            return results

        results["n_photometric"] = len(colors)
        results["color_range"] = [float(colors.min()), float(colors.max())]
        results["mag_range"] = [float(mags.min()), float(mags.max())]

        # CMD density map
        cmd_density = self._cmd_density(colors, mags)
        results["cmd_density"] = cmd_density

        # Identify main sequence
        ms_result = self._identify_main_sequence(colors, mags)
        results["main_sequence"] = ms_result

        # Detect red giant branch
        rgb_result = self._detect_red_giants(colors, mags, ms_result)
        results["red_giants"] = rgb_result

        # Detect blue stragglers
        bs_result = self._detect_blue_stragglers(colors, mags, ms_result)
        results["blue_stragglers"] = bs_result

        # Detect multiple populations (bimodal color distribution)
        pop_result = self._detect_multiple_populations(colors, mags)
        results["multiple_populations"] = pop_result

        # Color distribution analysis
        color_result = self._analyze_color_distribution(colors)
        results["color_distribution"] = color_result

        # Compute composite score
        results["population_score"] = self._compute_score(results)

        logger.info(
            f"Population analysis: {len(colors)} stars, "
            f"MS={ms_result['n_ms_stars']}, "
            f"RGB={rgb_result['n_red_giants']}, "
            f"BS={bs_result['n_blue_stragglers']}, "
            f"score={results['population_score']:.3f}"
        )
        return results

    def _extract_cmd_data(
        self, catalog: StarCatalog,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract color and magnitude arrays from catalog.

        Tries Gaia BP-RP first, falls back to SDSS g-r.
        """
        colors = []
        mags = []
        source_ids = []

        for entry in catalog.entries:
            props = entry.properties
            mag = entry.mag

            # Try Gaia BP-RP
            bp_rp = props.get("bp_rp")
            # Try SDSS g-r (pre-computed or compute from individual bands)
            g_r = props.get("g_r")
            if g_r is None:
                g_mag = props.get("g")
                r_mag = props.get("r")
                if g_mag is not None and r_mag is not None:
                    try:
                        g_val = float(g_mag)
                        r_val = float(r_mag)
                        if np.isfinite(g_val) and np.isfinite(r_val):
                            g_r = g_val - r_val
                    except (ValueError, TypeError):
                        pass

            color = bp_rp if bp_rp is not None else g_r

            if color is not None and mag is not None:
                try:
                    c = float(color)
                    m = float(mag)
                    if np.isfinite(c) and np.isfinite(m):
                        colors.append(c)
                        mags.append(m)
                        source_ids.append(entry.source_id or "")
                except (ValueError, TypeError):
                    continue

        return np.array(colors), np.array(mags), source_ids

    def _cmd_density(
        self, colors: np.ndarray, mags: np.ndarray,
    ) -> dict[str, Any]:
        """Compute 2D density map of the CMD using kernel density estimation."""
        n = self.density_grid_size
        c_range = (colors.min() - 0.2, colors.max() + 0.2)
        m_range = (mags.min() - 0.5, mags.max() + 0.5)

        # 2D histogram
        hist, c_edges, m_edges = np.histogram2d(
            colors, mags, bins=n, range=[c_range, m_range],
        )

        # Smooth for density estimation
        density = ndimage.gaussian_filter(hist.astype(np.float64), sigma=1.5)

        # Find density peaks (CMD sequences)
        local_max = ndimage.maximum_filter(density, size=5)
        peaks = (density == local_max) & (density > np.percentile(density, 90))
        peak_coords = np.argwhere(peaks)

        cmd_peaks = []
        for py, px in peak_coords:
            cmd_peaks.append({
                "color": float((c_edges[py] + c_edges[py + 1]) / 2),
                "mag": float((m_edges[px] + m_edges[px + 1]) / 2),
                "density": float(density[py, px]),
            })

        return {
            "n_peaks": len(cmd_peaks),
            "peaks": cmd_peaks[:10],
            "max_density": float(density.max()),
        }

    def _identify_main_sequence(
        self, colors: np.ndarray, mags: np.ndarray,
    ) -> dict[str, Any]:
        """Identify the main sequence in the CMD.

        Uses a running median fit to trace the main locus, then identifies
        stars within ms_width of the median relation. Turnoff is estimated
        from the bright end of the high-density MS region (not the single
        brightest star, which may be a blue straggler).
        """
        # Sort by magnitude and compute running median color
        sort_idx = np.argsort(mags)
        sorted_mags = mags[sort_idx]
        sorted_colors = colors[sort_idx]

        # Running median in magnitude bins
        n_bins = max(5, len(mags) // 20)
        bin_edges = np.linspace(mags.min(), mags.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        median_colors = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (sorted_mags >= bin_edges[i]) & (sorted_mags < bin_edges[i + 1])
            bin_counts[i] = in_bin.sum()
            if in_bin.sum() >= 3:
                median_colors[i] = np.median(sorted_colors[in_bin])
            elif i > 0:
                median_colors[i] = median_colors[i - 1]

        # Interpolate MS locus for all stars
        ms_color_at_mag = np.interp(mags, bin_centers, median_colors)

        # Main sequence membership: within ms_width of median relation
        color_offset = colors - ms_color_at_mag
        ms_mask = np.abs(color_offset) < self.ms_width

        # Estimate turnoff from the density peak: find the brightest magnitude
        # bin that has at least 30% of the peak bin count. This avoids
        # sparse blue straggler bins from defining the turnoff.
        peak_count = bin_counts.max()
        density_threshold = peak_count * 0.3
        turnoff_bin = 0
        for i in range(n_bins):
            if bin_counts[i] >= density_threshold:
                turnoff_bin = i
                break

        if ms_mask.any():
            # Turnoff magnitude = bright edge of the high-density region
            ms_turnoff_mag = float(bin_centers[turnoff_bin])
            ms_turnoff_color = float(median_colors[turnoff_bin])
        else:
            ms_turnoff_mag = float(mags.min())
            ms_turnoff_color = float(np.median(colors))

        return {
            "n_ms_stars": int(ms_mask.sum()),
            "ms_fraction": float(ms_mask.sum() / len(mags)),
            "turnoff_mag": ms_turnoff_mag,
            "turnoff_color": ms_turnoff_color,
            "ms_width_measured": float(np.std(color_offset[ms_mask]))
            if ms_mask.sum() > 3
            else self.ms_width,
            "ms_mask": ms_mask,
            "ms_color_at_mag": ms_color_at_mag,
        }

    def _detect_red_giants(
        self,
        colors: np.ndarray,
        mags: np.ndarray,
        ms_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect red giant branch stars.

        RGBs are redder and brighter than the MS turnoff.
        """
        turnoff_color = ms_result["turnoff_color"]
        turnoff_mag = ms_result["turnoff_mag"]

        # RGB: brighter than turnoff AND redder than turnoff + offset
        rgb_mask = (
            (mags < turnoff_mag + 0.5)
            & (colors > turnoff_color + 0.3)
            & ~ms_result["ms_mask"]
        )

        rgb_colors = colors[rgb_mask]
        rgb_mags = mags[rgb_mask]

        # Detect RGB tip (brightest red giant)
        rgb_tip_mag = float(rgb_mags.min()) if len(rgb_mags) > 0 else None

        return {
            "n_red_giants": int(rgb_mask.sum()),
            "rgb_fraction": float(rgb_mask.sum() / len(mags)),
            "rgb_tip_mag": rgb_tip_mag,
            "mean_rgb_color": float(rgb_colors.mean()) if len(rgb_colors) > 0 else None,
        }

    def _detect_blue_stragglers(
        self,
        colors: np.ndarray,
        mags: np.ndarray,
        ms_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect blue straggler candidates.

        Blue stragglers are stars bluer and brighter than the MS turnoff,
        thought to result from binary mergers or mass transfer.
        """
        turnoff_color = ms_result["turnoff_color"]
        turnoff_mag = ms_result["turnoff_mag"]

        # Blue stragglers: brighter than turnoff AND bluer than turnoff - offset
        bs_mask = (
            (mags < turnoff_mag)
            & (colors < turnoff_color - self.blue_straggler_offset)
        )

        return {
            "n_blue_stragglers": int(bs_mask.sum()),
            "bs_fraction": float(bs_mask.sum() / len(mags)),
            "candidates": [
                {
                    "color": float(colors[i]),
                    "mag": float(mags[i]),
                    "color_offset": float(turnoff_color - colors[i]),
                }
                for i in np.where(bs_mask)[0][:20]  # Top 20
            ],
        }

    def _detect_multiple_populations(
        self,
        colors: np.ndarray,
        mags: np.ndarray,
    ) -> dict[str, Any]:
        """Detect multiple stellar populations via color bimodality.

        Multiple populations (different ages or metallicities) appear as
        parallel or offset sequences in the CMD.
        """
        # Test for bimodality in color at fixed magnitude
        n_bins = max(3, len(mags) // 30)
        mag_bins = np.linspace(mags.min(), mags.max(), n_bins + 1)

        bimodal_bins = 0
        total_tested = 0

        for i in range(n_bins):
            in_bin = (mags >= mag_bins[i]) & (mags < mag_bins[i + 1])
            if in_bin.sum() < 10:
                continue

            total_tested += 1
            bin_colors = colors[in_bin]

            # Test bimodality via Hartigan's dip test approximation
            # Use the gap between sorted values
            sorted_c = np.sort(bin_colors)
            n = len(sorted_c)
            gaps = sorted_c[1:] - sorted_c[:-1]

            # A bimodal distribution has a large gap near the middle
            mid_range = slice(n // 4, 3 * n // 4)
            mid_gaps = gaps[mid_range]

            if len(mid_gaps) > 0:
                max_gap = mid_gaps.max()
                typical_gap = np.median(gaps)
                # Bimodal if the largest mid-range gap is >3x the typical gap
                if typical_gap > 0 and max_gap / typical_gap > 3:
                    bimodal_bins += 1

        bimodality_fraction = bimodal_bins / max(total_tested, 1)

        return {
            "bimodal_bins": bimodal_bins,
            "total_tested": total_tested,
            "bimodality_fraction": bimodality_fraction,
            "is_multiple_population": bimodality_fraction > 0.3,
        }

    def _analyze_color_distribution(
        self, colors: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze the overall color distribution for anomalies."""
        median_color = float(np.median(colors))
        std_color = float(np.std(colors))

        # Skewness: asymmetry in color distribution
        if std_color > 1e-6:
            skewness = float(
                np.mean(((colors - median_color) / std_color) ** 3)
            )
        else:
            skewness = 0.0

        # Kurtosis: peakedness
        if std_color > 1e-6:
            kurtosis = float(
                np.mean(((colors - median_color) / std_color) ** 4) - 3
            )
        else:
            kurtosis = 0.0

        # Color outliers (beyond 3 sigma)
        outlier_mask = np.abs(colors - median_color) > 3 * std_color
        n_outliers = int(outlier_mask.sum())

        return {
            "median_color": median_color,
            "color_spread": std_color,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "n_outliers": n_outliers,
            "outlier_fraction": n_outliers / len(colors),
        }

    def _compute_score(self, results: dict[str, Any]) -> float:
        """Compute population_score [0, 1] reflecting CMD interest."""
        score = 0.0

        # Blue stragglers are always interesting
        n_bs = results.get("blue_stragglers", {}).get("n_blue_stragglers", 0)
        score += min(n_bs / 5, 0.25)

        # Multiple populations are interesting
        if results.get("multiple_populations", {}).get("is_multiple_population"):
            score += 0.2

        # Red giant branch presence indicates evolved population
        rgb_frac = results.get("red_giants", {}).get("rgb_fraction", 0)
        if rgb_frac > 0.1:
            score += 0.15

        # Narrow main sequence (cluster-like) is interesting
        ms_width = results.get("main_sequence", {}).get("ms_width_measured", 1.0)
        if ms_width < 0.15:
            score += 0.15

        # Wide color spread (diverse population)
        color_spread = results.get("color_distribution", {}).get("color_spread", 0)
        if color_spread > 1.0:
            score += 0.1

        # Color outliers
        n_outliers = results.get("color_distribution", {}).get("n_outliers", 0)
        score += min(n_outliers / 10, 0.15)

        return float(np.clip(score, 0, 1))
