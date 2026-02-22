"""Inject synthetic anomalies for calibration and testing."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.utils.logging import get_logger

logger = get_logger("evaluation.synthetic")


class SyntheticInjector:
    """Inject synthetic anomalies into real images for calibration."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def inject_arc(
        self,
        image: FITSImage,
        center: tuple[int, int] | None = None,
        radius: float = 30.0,
        arc_angle: float = 90.0,
        arc_width: float = 3.0,
        brightness: float = 200.0,
    ) -> tuple[FITSImage, dict[str, Any]]:
        """Inject a synthetic arc (gravitational lens) into an image.

        Returns:
            Tuple of (modified image, injection metadata).
        """
        data = image.data.copy()
        h, w = data.shape

        if center is None:
            center = (h // 2, w // 2)

        cy, cx = center
        y, x = np.mgrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        # Arc mask
        angle_start = self.rng.uniform(-np.pi, np.pi)
        angle_end = angle_start + np.radians(arc_angle)

        # Handle angle wrapping
        if angle_end > np.pi:
            arc_mask = (
                (np.abs(r - radius) < arc_width)
                & ((theta >= angle_start) | (theta < angle_end - 2 * np.pi))
            )
        else:
            arc_mask = (
                (np.abs(r - radius) < arc_width)
                & (theta >= angle_start)
                & (theta < angle_end)
            )

        # Gaussian profile across arc width
        profile = brightness * np.exp(-0.5 * ((r - radius) / (arc_width / 2)) ** 2)
        data[arc_mask] += profile[arc_mask]

        metadata = {
            "type": "arc",
            "center": center,
            "radius": radius,
            "angle_start_deg": float(np.degrees(angle_start)),
            "arc_angle_deg": arc_angle,
            "brightness": brightness,
        }

        return FITSImage(data=data, header=image.header, wcs=image.wcs), metadata

    def inject_ring(
        self,
        image: FITSImage,
        center: tuple[int, int] | None = None,
        radius: float = 25.0,
        width: float = 3.0,
        brightness: float = 150.0,
        completeness: float = 0.8,
    ) -> tuple[FITSImage, dict[str, Any]]:
        """Inject a synthetic Einstein ring."""
        data = image.data.copy()
        h, w = data.shape

        if center is None:
            center = (h // 2, w // 2)

        cy, cx = center
        y, x = np.mgrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        theta = np.arctan2(y - cy, x - cx)

        # Ring with gaps
        ring = brightness * np.exp(-0.5 * ((r - radius) / (width / 2)) ** 2)

        # Add gaps for incomplete rings
        n_gaps = int((1 - completeness) * 4)
        for _ in range(n_gaps):
            gap_center = self.rng.uniform(-np.pi, np.pi)
            gap_width = self.rng.uniform(0.3, 0.8)
            gap_mask = np.abs(theta - gap_center) < gap_width
            ring[gap_mask] *= 0.1

        data += ring

        metadata = {
            "type": "ring",
            "center": center,
            "radius": radius,
            "brightness": brightness,
            "completeness": completeness,
        }

        return FITSImage(data=data, header=image.header, wcs=image.wcs), metadata

    def inject_overdensity(
        self,
        image: FITSImage,
        center: tuple[int, int] | None = None,
        n_sources: int = 20,
        spread: float = 15.0,
        source_brightness: float = 300.0,
    ) -> tuple[FITSImage, dict[str, Any]]:
        """Inject a cluster of point sources (stellar overdensity)."""
        data = image.data.copy()
        h, w = data.shape

        if center is None:
            center = (
                self.rng.integers(50, h - 50),
                self.rng.integers(50, w - 50),
            )

        cy, cx = center
        positions = []

        for _ in range(n_sources):
            sy = int(cy + self.rng.normal(0, spread))
            sx = int(cx + self.rng.normal(0, spread))
            if 2 <= sy < h - 2 and 2 <= sx < w - 2:
                sigma = self.rng.uniform(1.0, 2.5)
                flux = source_brightness * self.rng.uniform(0.3, 1.0)
                yy, xx = np.mgrid[-3:4, -3:4]
                psf = flux * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                data[sy - 3 : sy + 4, sx - 3 : sx + 4] += psf
                positions.append((sy, sx))

        metadata = {
            "type": "overdensity",
            "center": center,
            "n_sources": len(positions),
            "spread": spread,
        }

        return FITSImage(data=data, header=image.header, wcs=image.wcs), metadata

    def inject_random(self, image: FITSImage) -> tuple[FITSImage, dict[str, Any]]:
        """Inject a random synthetic anomaly."""
        anomaly_type = self.rng.choice(["arc", "ring", "overdensity"])

        if anomaly_type == "arc":
            return self.inject_arc(image)
        elif anomaly_type == "ring":
            return self.inject_ring(image)
        else:
            return self.inject_overdensity(image)
