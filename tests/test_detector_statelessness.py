"""Detectors must not carry per-image state between calls.

The ensemble builds each detector once and reuses it for every image in
a run. A detector that stores a per-image quantity on self therefore
leaks that value into the next image, which made detection results
depend on the order images happened to arrive in. These tests fail
against the previous implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from star_pattern.core.config import DetectionConfig
from star_pattern.detection.classical import ClassicalDetector
from star_pattern.detection.galaxy_detector import GalaxyDetector
from star_pattern.detection.lens_detector import LensDetector


def _ring_image(seed: int = 7) -> np.ndarray:
    """Synthetic frame with a bright core and a faint ring around it."""
    rng = np.random.default_rng(seed)
    size = 128
    image = rng.standard_normal((size, size)) * 0.5
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    image += 50.0 * np.exp(-0.5 * (r / 4.0) ** 2)
    image += 8.0 * np.exp(-0.5 * ((r - 24.0) / 2.0) ** 2)
    return image


# SDSS pixel scale: small enough that the scaled radii differ sharply
# from the defaults, which is what exposed the leak.
SDSS_PIXEL_SCALE = 0.396


class TestLensDetectorStatelessness:
    def test_result_does_not_depend_on_call_order(self):
        image = _ring_image()

        baseline = LensDetector().detect(image, pixel_scale_arcsec=None)

        shared = LensDetector()
        before = shared.detect(image, pixel_scale_arcsec=None)
        # An image with a WCS in between must not change later results.
        shared.detect(image, pixel_scale_arcsec=SDSS_PIXEL_SCALE)
        after = shared.detect(image, pixel_scale_arcsec=None)

        assert before["lens_score"] == pytest.approx(baseline["lens_score"])
        assert after["lens_score"] == pytest.approx(baseline["lens_score"])
        assert len(after["rings"]) == len(before["rings"])
        assert len(after["arcs"]) == len(before["arcs"])

    def test_configured_radii_survive_a_scaled_call(self):
        detector = LensDetector()
        original = (
            detector.ring_min_radius,
            detector.ring_max_radius,
            detector.arc_min_length,
        )

        detector.detect(_ring_image(), pixel_scale_arcsec=SDSS_PIXEL_SCALE)

        assert (
            detector.ring_min_radius,
            detector.ring_max_radius,
            detector.arc_min_length,
        ) == original

    def test_pixel_scale_still_changes_the_radii_used(self):
        """The fix must not silently disable pixel-scale adaptation."""
        detector = LensDetector()
        unscaled = detector._radii_for(None)
        scaled = detector._radii_for(SDSS_PIXEL_SCALE)
        assert scaled.ring_max != unscaled.ring_max
        assert scaled.ring_max == max(10, int(25.0 / SDSS_PIXEL_SCALE))


class TestClassicalDetectorStatelessness:
    def test_hough_detector_is_not_replaced(self):
        detector = ClassicalDetector()
        hough = detector.hough
        radii = (hough.min_radius, hough.max_radius)

        detector.detect(_ring_image(), pixel_scale_arcsec=SDSS_PIXEL_SCALE)

        assert detector.hough is hough
        assert (detector.hough.min_radius, detector.hough.max_radius) == radii

    def test_result_does_not_depend_on_call_order(self):
        image = _ring_image()

        baseline = ClassicalDetector().detect(image, pixel_scale_arcsec=None)

        shared = ClassicalDetector()
        shared.detect(image, pixel_scale_arcsec=SDSS_PIXEL_SCALE)
        after = shared.detect(image, pixel_scale_arcsec=None)

        assert after["classical_score"] == pytest.approx(baseline["classical_score"])


class TestGalaxyDetectorStatelessness:
    def test_stores_no_per_image_pixel_scale(self):
        detector = GalaxyDetector(DetectionConfig())
        detector.detect(_ring_image(), pixel_scale_arcsec=SDSS_PIXEL_SCALE)
        assert not hasattr(detector, "_pixel_scale")

    def test_result_does_not_depend_on_call_order(self):
        image = _ring_image()
        config = DetectionConfig()

        baseline = GalaxyDetector(config).detect(image, pixel_scale_arcsec=None)

        shared = GalaxyDetector(config)
        shared.detect(image, pixel_scale_arcsec=SDSS_PIXEL_SCALE)
        after = shared.detect(image, pixel_scale_arcsec=None)

        assert after["galaxy_score"] == pytest.approx(baseline["galaxy_score"])
        assert len(after["merger_candidates"]) == len(baseline["merger_candidates"])
