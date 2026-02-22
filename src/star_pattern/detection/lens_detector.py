"""Gravitational lens detection: arc and ring finding."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.lens")


class LensDetector:
    """Detect gravitational lens candidates (arcs, rings, multiple images)."""

    def __init__(
        self,
        arc_min_length: int = 15,
        arc_max_width: int = 8,
        ring_min_radius: int = 10,
        ring_max_radius: int = 80,
        snr_threshold: float = 3.0,
    ):
        self.arc_min_length = arc_min_length
        self.arc_max_width = arc_max_width
        self.ring_min_radius = ring_min_radius
        self.ring_max_radius = ring_max_radius
        self.snr_threshold = snr_threshold

    def detect(
        self,
        image: np.ndarray,
        pixel_scale_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Run lens detection on an image.

        Args:
            image: 2D image array.
            pixel_scale_arcsec: Pixel scale in arcsec/pixel. If provided,
                arc/ring radii are converted from physical to pixel units.

        Returns:
            Dict with arc candidates, ring candidates, and overall lens score.
        """
        # Adapt radii from physical arcsec to pixels if scale is known
        if pixel_scale_arcsec and pixel_scale_arcsec > 0:
            self.ring_min_radius = max(3, int(3.0 / pixel_scale_arcsec))
            self.ring_max_radius = max(10, int(25.0 / pixel_scale_arcsec))
            self.arc_min_length = max(5, int(5.0 / pixel_scale_arcsec))

        data = image.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0)

        # Background estimation
        bkg = np.median(data)
        rms = np.std(data)

        # Subtract background
        data_sub = data - bkg

        # Find central bright source (potential lens galaxy)
        central = self._find_central_source(data_sub)

        # Look for arcs around center
        arcs = self._detect_arcs(data_sub, central, rms)

        # Look for ring-like residuals
        rings = self._detect_rings(data_sub, central, rms)

        # Compute lens score
        lens_score = self._compute_lens_score(arcs, rings, central)

        return {
            "central_source": central,
            "arcs": arcs,
            "rings": rings,
            "lens_score": lens_score,
            "is_candidate": lens_score > 0.3,
        }

    def _get_cutout(
        self, data: np.ndarray, cx: int, cy: int, radius: int
    ) -> tuple[np.ndarray, int, int]:
        """Extract a square cutout around (cx, cy) with padding = radius.

        Returns (cutout, cx_local, cy_local) where local coords are
        the center position within the cutout.
        """
        pad = radius + 5
        y0 = max(0, cy - pad)
        y1 = min(data.shape[0], cy + pad)
        x0 = max(0, cx - pad)
        x1 = min(data.shape[1], cx + pad)
        cutout = data[y0:y1, x0:x1]
        return cutout, cx - x0, cy - y0

    def _find_central_source(self, data: np.ndarray) -> dict[str, Any]:
        """Find the brightest central source."""
        # Smooth to find peak
        smoothed = ndimage.gaussian_filter(data, sigma=3.0)
        cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        peak_flux = float(smoothed[cy, cx])

        # Work in a cutout around the peak for efficiency
        cutout, lx, ly = self._get_cutout(data, cx, cy, self.ring_max_radius)
        y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]
        r = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
        mask = r < self.ring_max_radius
        if mask.any():
            sorted_r = np.sort(r[mask & (cutout > 0)])
            total_flux = np.sum(cutout[mask & (cutout > 0)])
            cumflux = np.cumsum(np.sort(cutout[mask & (cutout > 0)])[::-1])
            if len(cumflux) > 0 and total_flux > 0:
                half_idx = np.searchsorted(cumflux, 0.5 * total_flux)
                half_light_r = float(sorted_r[min(half_idx, len(sorted_r) - 1)])
            else:
                half_light_r = 5.0
        else:
            half_light_r = 5.0

        return {
            "x": int(cx),
            "y": int(cy),
            "peak_flux": peak_flux,
            "half_light_radius": half_light_r,
        }

    def _detect_arcs(
        self, data: np.ndarray, central: dict[str, Any], rms: float
    ) -> list[dict[str, Any]]:
        """Detect arc-like features around the central source."""
        cx, cy = central["x"], central["y"]
        hlr = central["half_light_radius"]

        # Work in a cutout around the central source for efficiency
        cutout, lx, ly = self._get_cutout(data, cx, cy, self.ring_max_radius)
        y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]
        r = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
        model = central["peak_flux"] * np.exp(-0.5 * (r / max(hlr, 1)) ** 2)
        residual = cutout - model
        theta = np.arctan2(y - ly, x - lx)

        # Adaptive step: at least 10 radii, at most ~30
        radius_range = self.ring_max_radius - self.ring_min_radius
        arc_step = max(5, radius_range // 20)

        # Pre-compute sector starts for vectorized angular tests
        sector_starts = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        sector_width = np.pi / 3  # 60-degree sectors

        arcs = []
        for r_inner in range(self.ring_min_radius, self.ring_max_radius, arc_step):
            r_outer = r_inner + self.arc_max_width
            annulus = (r >= r_inner) & (r < r_outer)

            if not annulus.any():
                continue

            # Extract annulus pixels once for all sectors
            annulus_theta = theta[annulus]
            annulus_residual = residual[annulus]

            # Vectorized sector checks
            for sector_start in sector_starts:
                sector_end = sector_start + sector_width
                sector_mask = (annulus_theta >= sector_start) & (annulus_theta < sector_end)
                n_sector = sector_mask.sum()

                if n_sector < self.arc_min_length:
                    continue

                sector_flux = annulus_residual[sector_mask]
                snr = float(np.mean(sector_flux) / max(rms, 1e-10))

                if snr > self.snr_threshold:
                    arcs.append(
                        {
                            "radius": (r_inner + r_outer) / 2,
                            "angle_start": float(np.degrees(sector_start)),
                            "angle_span": 60.0,
                            "snr": snr,
                            "mean_flux": float(np.mean(sector_flux)),
                            "n_pixels": int(n_sector),
                        }
                    )

        # Sort by SNR
        arcs.sort(key=lambda a: a["snr"], reverse=True)
        return arcs[:10]

    def _detect_rings(
        self, data: np.ndarray, central: dict[str, Any], rms: float
    ) -> list[dict[str, Any]]:
        """Detect ring-like residuals (Einstein rings)."""
        cx, cy = central["x"], central["y"]
        hlr = central["half_light_radius"]

        # Work in a cutout around the central source for efficiency
        cutout, lx, ly = self._get_cutout(data, cx, cy, self.ring_max_radius)
        y, x = np.mgrid[: cutout.shape[0], : cutout.shape[1]]
        r = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
        model = central["peak_flux"] * np.exp(-0.5 * (r / max(hlr, 1)) ** 2)
        residual = cutout - model

        # Adaptive step: at least 10 radii, at most ~40
        radius_range = self.ring_max_radius - self.ring_min_radius
        ring_step = max(2, radius_range // 30)

        rings = []
        for radius in range(self.ring_min_radius, self.ring_max_radius, ring_step):
            ring_mask = (r >= radius - 2) & (r < radius + 2)
            if ring_mask.sum() < 20:
                continue

            ring_flux = residual[ring_mask]
            mean_flux = float(np.mean(ring_flux))
            snr = mean_flux / max(rms, 1e-10)

            if snr > self.snr_threshold:
                # Check azimuthal completeness (vectorized)
                theta_ring = np.arctan2(y[ring_mask] - ly, x[ring_mask] - lx)
                n_sectors = 8
                sector_edges = np.linspace(-np.pi, np.pi, n_sectors + 1)
                # Digitize theta values into sector bins
                sector_idx = np.digitize(theta_ring, sector_edges) - 1
                sector_idx = np.clip(sector_idx, 0, n_sectors - 1)
                filled_sectors = 0
                for i in range(n_sectors):
                    s_mask = sector_idx == i
                    if s_mask.any() and np.mean(ring_flux[s_mask]) > rms:
                        filled_sectors += 1

                completeness = filled_sectors / n_sectors

                rings.append(
                    {
                        "radius": radius,
                        "snr": float(snr),
                        "mean_flux": mean_flux,
                        "completeness": completeness,
                        "is_complete_ring": completeness > 0.6,
                    }
                )

        rings.sort(key=lambda r: r["snr"], reverse=True)
        return rings[:5]

    def _compute_lens_score(
        self,
        arcs: list[dict[str, Any]],
        rings: list[dict[str, Any]],
        central: dict[str, Any],
    ) -> float:
        """Compute overall gravitational lens likelihood score [0, 1]."""
        score = 0.0

        # Arc contribution
        if arcs:
            best_arc_snr = arcs[0]["snr"]
            score += min(best_arc_snr / 10, 0.4)

        # Ring contribution
        if rings:
            best_ring = rings[0]
            ring_score = min(best_ring["snr"] / 10, 0.3)
            if best_ring.get("is_complete_ring"):
                ring_score += 0.2
            score += ring_score

        # Central source (bright = good lens candidate)
        if central["peak_flux"] > 0:
            score += 0.1

        return float(np.clip(score, 0, 1))
