"""Source extraction must give the same catalog for the same image, every time.

sep's deblender is not reproducible: given a byte-identical array it returns
identical positions and counts, then assigns blended pixels to neighbours
differently on every call. No sep version fixes it and no deblend parameter
fixes it without switching deblending off entirely.

SourceExtractor therefore uses sep for detection only and measures flux and
shape itself, with a fixed circular aperture and fixed-window flux-weighted
moments. Both are plain numpy over fixed pixel sets, so both are exactly
reproducible. These tests hold that boundary in place.
"""

from __future__ import annotations

import numpy as np
import pytest

from star_pattern.detection.source_extraction import (
    SourceExtractor,
    _aperture_moments,
)


def _field(seed: int = 20260801, size: int = 192, n_sources: int = 40) -> np.ndarray:
    """Star field with an extended core, an arc, and point sources."""
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 8.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    data += 400.0 * np.exp(-0.5 * (r / 6.0) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    data[(np.abs(r - 28.0) < 2.5) & (np.abs(theta - 0.6) < 1.1)] += 180.0
    for _ in range(n_sources):
        sy, sx = rng.integers(12, size - 12, size=2)
        sigma = rng.uniform(1.5, 3.0)
        py, px = np.mgrid[-8:9, -8:9]
        data[sy - 8 : sy + 9, sx - 8 : sx + 9] += 600.0 * np.exp(-(px**2 + py**2) / (2 * sigma**2))
    return data.astype(np.float32)


FIELDS = {
    "dense": _field(),
    "sparse": np.random.default_rng(3).normal(100.0, 1.0, (96, 96)).astype(np.float32),
}


class TestExtractorIsDeterministic:
    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_every_output_field_is_reproducible(self, field: str):
        """This is the whole point of the aperture-photometry design."""
        extractor = SourceExtractor(threshold=3.0)
        runs = [extractor.extract(FIELDS[field].copy()) for _ in range(12)]

        for key in (
            "n_sources",
            "positions",
            "fluxes",
            "ellipticity",
            "fwhm",
            "star_mask",
            "background_rms",
        ):
            distinct = {np.asarray(r[key]).tobytes() for r in runs}
            assert len(distinct) == 1, f"{key} varies across identical extractions"

    def test_separate_instances_agree(self):
        """Determinism must not depend on reusing one extractor object."""
        image = FIELDS["dense"]
        first = SourceExtractor(threshold=3.0).extract(image.copy())
        second = SourceExtractor(threshold=3.0).extract(image.copy())
        np.testing.assert_array_equal(first["fluxes"], second["fluxes"])
        np.testing.assert_array_equal(first["star_mask"], second["star_mask"])

    def test_input_image_is_not_modified(self):
        image = FIELDS["dense"].copy()
        before = image.copy()
        SourceExtractor(threshold=3.0).extract(image)
        np.testing.assert_array_equal(image, before)


class TestApertureMoments:
    """The shape estimator that replaces sep's a, b and theta."""

    @staticmethod
    def _blob(size=61, sx=3.0, sy=3.0, cx=30.0, cy=30.0):
        yy, xx = np.mgrid[0:size, 0:size]
        return 1000.0 * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))

    def test_round_source_has_low_ellipticity(self):
        a, b, _ = _aperture_moments(self._blob(), np.array([30.0]), np.array([30.0]), 8.0)
        assert 1 - b[0] / a[0] < 0.05

    def test_elongated_source_has_high_ellipticity(self):
        image = self._blob(sx=6.0, sy=2.0)
        a, b, _ = _aperture_moments(image, np.array([30.0]), np.array([30.0]), 8.0)
        assert 1 - b[0] / a[0] > 0.3

    def test_position_angle_follows_the_elongation(self):
        horizontal = self._blob(sx=6.0, sy=2.0)
        _, _, theta = _aperture_moments(horizontal, np.array([30.0]), np.array([30.0]), 8.0)
        # Elongated along x, so the major axis is near 0 (or pi) radians.
        assert min(abs(theta[0]), abs(abs(theta[0]) - np.pi)) < 0.2

        vertical = self._blob(sx=2.0, sy=6.0)
        _, _, theta_v = _aperture_moments(vertical, np.array([30.0]), np.array([30.0]), 8.0)
        assert abs(abs(theta_v[0]) - np.pi / 2) < 0.2

    def test_wider_source_gives_larger_axes(self):
        narrow = _aperture_moments(
            self._blob(sx=2.0, sy=2.0), np.array([30.0]), np.array([30.0]), 10.0
        )
        wide = _aperture_moments(
            self._blob(sx=5.0, sy=5.0), np.array([30.0]), np.array([30.0]), 10.0
        )
        assert wide[0][0] > narrow[0][0]

    def test_source_at_the_edge_does_not_raise(self):
        """Zero-padding means an edge source needs no special case."""
        image = self._blob(cx=1.0, cy=1.0)
        a, b, _ = _aperture_moments(image, np.array([1.0]), np.array([1.0]), 8.0)
        assert np.isfinite(a[0]) and np.isfinite(b[0])

    def test_empty_aperture_does_not_divide_by_zero(self):
        image = np.zeros((41, 41))
        a, b, theta = _aperture_moments(image, np.array([20.0]), np.array([20.0]), 6.0)
        assert np.isfinite(a[0]) and np.isfinite(b[0]) and np.isfinite(theta[0])
        assert a[0] > 0

    def test_no_sources_returns_empty(self):
        a, b, theta = _aperture_moments(np.zeros((10, 10)), np.array([]), np.array([]), 5.0)
        assert len(a) == len(b) == len(theta) == 0

    def test_is_deterministic(self):
        image = self._blob(sx=5.0, sy=2.0)
        x = np.array([30.0, 12.0])
        y = np.array([30.0, 44.0])
        results = [_aperture_moments(image, x, y, 8.0) for _ in range(8)]
        for index in range(3):
            distinct = {results[i][index].tobytes() for i in range(len(results))}
            assert len(distinct) == 1


class TestClassificationIsPreserved:
    """The new shape estimator must still separate stars from galaxies.

    Measuring shape in the same aperture used for flux caps the visible
    extent and collapses the classifier to "everything is a star". The
    moment window is deliberately wider for this reason.
    """

    def test_star_galaxy_split_is_not_degenerate(self):
        result = SourceExtractor(threshold=3.0).extract(FIELDS["dense"].copy())
        n_stars = int(result["star_mask"].sum())
        n_total = result["n_sources"]
        assert 0 < n_stars < n_total, (
            f"classifier returned {n_stars}/{n_total} stars, which means it has "
            f"stopped separating. Check MOMENT_RADIUS_FACTOR."
        )

    def test_moment_window_is_wider_than_the_photometric_aperture(self):
        extractor = SourceExtractor(threshold=3.0)
        assert extractor.moment_radius_px > extractor.aperture_radius_px

    def test_equal_radii_would_degenerate(self):
        """Documents why the two radii differ, so nobody 'simplifies' it."""
        same = SourceExtractor(threshold=3.0, moment_radius_px=5.0)
        result = same.extract(FIELDS["dense"].copy())
        assert int(result["star_mask"].sum()) == result["n_sources"]
