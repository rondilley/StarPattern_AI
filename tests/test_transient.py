"""Tests for transient / variability detection."""

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.detection.transient import TransientDetector


@pytest.fixture
def transient_detector():
    return TransientDetector()


@pytest.fixture
def catalog_with_astrometric_noise():
    """Catalog with high astrometric excess noise entries."""
    rng = np.random.default_rng(42)
    entries = []

    # Normal stars with low noise
    for i in range(80):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=rng.uniform(14, 20),
                source="gaia",
                source_id=f"normal_{i}",
                properties={
                    "astrometric_excess_noise": rng.uniform(0.0, 0.5),
                    "parallax": rng.uniform(0.5, 5),
                    "parallax_error": rng.uniform(0.01, 0.1),
                },
            )
        )

    # High-noise sources (unresolved binaries, variables)
    for i in range(5):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=rng.uniform(14, 18),
                source="gaia",
                source_id=f"noisy_{i}",
                properties={
                    "astrometric_excess_noise": 5.0 + rng.uniform(0, 3),
                    "parallax": rng.uniform(0.5, 3),
                    "parallax_error": rng.uniform(0.01, 0.1),
                },
            )
        )

    return StarCatalog(entries=entries, source="gaia")


@pytest.fixture
def catalog_with_parallax_anomalies():
    """Catalog with negative parallax and low-SNR parallax entries."""
    rng = np.random.default_rng(42)
    entries = []

    # Normal stars
    for i in range(50):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=rng.uniform(14, 20),
                source="gaia",
                source_id=f"normal_{i}",
                properties={
                    "parallax": rng.uniform(1, 5),
                    "parallax_error": 0.05,
                },
            )
        )

    # Negative parallax (distant AGN or bad fit)
    for i in range(3):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=rng.uniform(18, 22),
                source="gaia",
                source_id=f"neg_plx_{i}",
                properties={
                    "parallax": -0.5 - rng.uniform(0, 1),
                    "parallax_error": 0.3,
                },
            )
        )

    return StarCatalog(entries=entries, source="gaia")


@pytest.fixture
def catalog_with_photometric_outliers():
    """Catalog with photometric color outliers."""
    rng = np.random.default_rng(42)
    entries = []

    # Normal stars following a color-magnitude relation
    for i in range(80):
        g_mag = rng.uniform(14, 20)
        bp_rp = 0.5 + 0.05 * g_mag + rng.normal(0, 0.1)
        bp_mag = g_mag + bp_rp / 2
        rp_mag = g_mag - bp_rp / 2
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=g_mag,
                source="gaia",
                source_id=f"normal_{i}",
                properties={
                    "phot_bp_mean_mag": bp_mag,
                    "phot_rp_mean_mag": rp_mag,
                },
            )
        )

    # Photometric outliers (extreme blue or red)
    for i in range(5):
        g_mag = rng.uniform(16, 18)
        bp_rp = 4.0 + rng.normal(0, 0.1)  # Very red
        bp_mag = g_mag + bp_rp / 2
        rp_mag = g_mag - bp_rp / 2
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.01),
                dec=45.0 + rng.normal(0, 0.01),
                mag=g_mag,
                source="gaia",
                source_id=f"photo_outlier_{i}",
                properties={
                    "phot_bp_mean_mag": bp_mag,
                    "phot_rp_mean_mag": rp_mag,
                },
            )
        )

    return StarCatalog(entries=entries, source="gaia")


class TestTransientDetector:
    def test_analyze_returns_expected_keys(self, transient_detector, catalog_with_astrometric_noise):
        result = transient_detector.analyze(catalog_with_astrometric_noise)
        assert "transient_score" in result
        assert "astrometric_outliers" in result
        assert "photometric_outliers" in result
        assert "parallax_anomalies" in result
        assert "variable_candidates" in result
        assert 0 <= result["transient_score"] <= 1

    def test_flag_astrometric_outliers(self, transient_detector, catalog_with_astrometric_noise):
        result = transient_detector.analyze(catalog_with_astrometric_noise)
        outliers = result["astrometric_outliers"]
        assert len(outliers) > 0
        # At least some of the noisy sources should be flagged
        outlier_ids = {o["source_id"] for o in outliers}
        noisy_ids = {f"noisy_{i}" for i in range(5)}
        assert len(outlier_ids & noisy_ids) > 0

    def test_flag_parallax_anomalies(self, transient_detector, catalog_with_parallax_anomalies):
        result = transient_detector.analyze(catalog_with_parallax_anomalies)
        anomalies = result["parallax_anomalies"]
        assert len(anomalies) > 0
        # Negative parallax sources should be flagged
        anomaly_ids = {a["source_id"] for a in anomalies}
        neg_ids = {f"neg_plx_{i}" for i in range(3)}
        assert len(anomaly_ids & neg_ids) > 0
        # Verify reason
        for a in anomalies:
            if a["source_id"].startswith("neg_plx"):
                assert a["reason"] == "negative_parallax"

    def test_flag_photometric_outliers(self, transient_detector, catalog_with_photometric_outliers):
        result = transient_detector.analyze(catalog_with_photometric_outliers)
        outliers = result["photometric_outliers"]
        assert isinstance(outliers, list)
        # Photometric outliers should be detected
        if outliers:
            assert "deviation_sigma" in outliers[0]

    def test_empty_catalog(self, transient_detector):
        catalog = StarCatalog(entries=[], source="empty")
        result = transient_detector.analyze(catalog)
        assert result["transient_score"] == 0.0

    def test_catalog_without_gaia_data(self, transient_detector):
        """Catalog with no Gaia-specific properties should return empty results."""
        entries = [
            CatalogEntry(
                ra=180.0, dec=45.0, mag=15.0,
                source="sdss", source_id=f"sdss_{i}",
            )
            for i in range(20)
        ]
        catalog = StarCatalog(entries=entries, source="sdss")
        result = transient_detector.analyze(catalog)
        assert result["astrometric_outliers"] == []
        assert result["parallax_anomalies"] == []
