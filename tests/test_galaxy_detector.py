"""Tests for galaxy-specific feature detection."""

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.detection.galaxy_detector import GalaxyDetector


@pytest.fixture
def galaxy_detector():
    return GalaxyDetector()


@pytest.fixture
def image_with_tidal_feature():
    """Synthetic image with a tidal tail-like residual arc."""
    rng = np.random.default_rng(42)
    data = rng.normal(100, 5, (256, 256)).astype(np.float64)
    # Central galaxy
    yy, xx = np.mgrid[-128:128, -128:128]
    data += 500 * np.exp(-(xx**2 + yy**2) / (2 * 15**2))
    # Tidal tail: extended arc feature at large radius
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    tidal_mask = (np.abs(r - 60) < 5) & (theta > 0.5) & (theta < 2.0)
    data[tidal_mask] += 150
    return data.astype(np.float32)


@pytest.fixture
def image_with_double_nucleus():
    """Synthetic image with two bright nuclei (merger candidate).

    Nuclei are placed asymmetrically so the cutout around them has
    high rotational asymmetry (needed for merger detection).
    """
    rng = np.random.default_rng(42)
    data = rng.normal(100, 5, (256, 256)).astype(np.float64)
    # Extended galaxy envelope offset from center
    yy, xx = np.mgrid[-128:128, -128:128]
    data += 200 * np.exp(-((xx - 5)**2 + (yy - 5)**2) / (2 * 30**2))
    # Nucleus 1 -- strong, compact, off-center
    data += 1500 * np.exp(-((xx - 5)**2 + (yy - 5)**2) / (2 * 2**2))
    # Nucleus 2 -- weaker, further from envelope center, creating asymmetry
    data += 800 * np.exp(-((xx + 15)**2 + (yy + 10)**2) / (2 * 2**2))
    return data.astype(np.float32)


@pytest.fixture
def catalog_with_color_outliers():
    """Synthetic catalog with known color outliers."""
    rng = np.random.default_rng(42)
    entries = []
    for i in range(100):
        mag = rng.uniform(14, 20)
        # Normal color-magnitude relation: color ~ 0.05 * mag + 0.5
        color = 0.05 * mag + 0.5 + rng.normal(0, 0.1)
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=mag,
                mag_band="g",
                obj_type="galaxy",
                source="test",
                source_id=f"gal_{i}",
                properties={"bp_rp": color},
            )
        )
    # Add 3 color outliers with extreme colors
    for i in range(3):
        mag = rng.uniform(16, 18)
        color = 5.0 + rng.normal(0, 0.1)  # Far from normal
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=mag,
                mag_band="g",
                obj_type="galaxy",
                source="test",
                source_id=f"outlier_{i}",
                properties={"bp_rp": color},
            )
        )
    return StarCatalog(entries=entries, source="test")


class TestGalaxyDetector:
    def test_detect_returns_expected_keys(self, galaxy_detector):
        data = np.random.default_rng(0).normal(100, 10, (128, 128)).astype(np.float32)
        result = galaxy_detector.detect(data)
        assert "galaxy_score" in result
        assert "tidal_features" in result
        assert "merger_candidates" in result
        assert "color_outliers" in result
        assert 0 <= result["galaxy_score"] <= 1

    def test_detect_tidal_features(self, galaxy_detector, image_with_tidal_feature):
        result = galaxy_detector.detect(image_with_tidal_feature)
        tidal = result["tidal_features"]
        assert isinstance(tidal, list)
        # The strong tidal feature should be detected
        assert len(tidal) > 0

    def test_detect_mergers(self, galaxy_detector, image_with_double_nucleus):
        config = DetectionConfig(galaxy_asymmetry_threshold=0.1)
        detector = GalaxyDetector(config)
        result = detector.detect(image_with_double_nucleus)
        mergers = result["merger_candidates"]
        assert isinstance(mergers, list)
        # Double nucleus should be detected as merger candidate
        assert len(mergers) > 0
        if mergers:
            assert "asymmetry" in mergers[0]
            assert "separation_px" in mergers[0]

    def test_detect_color_anomalies(self, galaxy_detector, catalog_with_color_outliers):
        result = galaxy_detector.detect(
            np.random.default_rng(0).normal(100, 10, (64, 64)).astype(np.float32),
            catalog=catalog_with_color_outliers,
        )
        outliers = result["color_outliers"]
        assert isinstance(outliers, list)
        assert len(outliers) > 0
        # At least some of the injected outliers should be flagged
        outlier_ids = {o["source_id"] for o in outliers}
        injected_ids = {f"outlier_{i}" for i in range(3)}
        assert len(outlier_ids & injected_ids) > 0

    def test_no_catalog_returns_empty_colors(self, galaxy_detector):
        data = np.random.default_rng(0).normal(100, 10, (64, 64)).astype(np.float32)
        result = galaxy_detector.detect(data, catalog=None)
        assert result["color_outliers"] == []

    def test_noise_image_low_score(self, galaxy_detector):
        data = np.random.default_rng(0).normal(100, 10, (128, 128)).astype(np.float32)
        result = galaxy_detector.detect(data)
        # Pure noise should have low galaxy score
        assert result["galaxy_score"] < 0.5
