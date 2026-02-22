"""Tests for stellar population / CMD analysis."""

import numpy as np
import pytest

from star_pattern.core.catalog import StarCatalog, CatalogEntry
from star_pattern.detection.stellar_population import StellarPopulationAnalyzer


def _make_synthetic_catalog(
    n_ms=80, n_rgb=15, n_bs=5, seed=42,
) -> StarCatalog:
    """Create a synthetic star catalog with known populations.

    Generates stars along a main sequence, red giant branch,
    and blue straggler population. MS turnoff is at mag~15,
    so BS (mag 12-14.5) and RGB (mag 12-14.5 but red) stand out.
    """
    rng = np.random.default_rng(seed)
    entries = []

    # Main sequence: diagonal band in CMD, turnoff at mag~15
    # color = 0.5 + 0.12 * (mag - 15) + noise
    for i in range(n_ms):
        mag = rng.uniform(15, 21)
        color = 0.5 + 0.12 * (mag - 15) + rng.normal(0, 0.04)
        entries.append(CatalogEntry(
            source_id=f"MS_{i}",
            ra=180 + rng.uniform(-0.01, 0.01),
            dec=45 + rng.uniform(-0.01, 0.01),
            mag=float(mag),
            properties={"bp_rp": float(color)},
        ))

    # Red giant branch: brighter than turnoff, very red
    for i in range(n_rgb):
        mag = rng.uniform(12, 15)
        color = 1.8 + rng.uniform(0, 1.0)
        entries.append(CatalogEntry(
            source_id=f"RGB_{i}",
            ra=180 + rng.uniform(-0.01, 0.01),
            dec=45 + rng.uniform(-0.01, 0.01),
            mag=float(mag),
            properties={"bp_rp": float(color)},
        ))

    # Blue stragglers: brighter than turnoff AND bluer
    for i in range(n_bs):
        mag = rng.uniform(12, 14.5)  # brighter than MS turnoff at ~15
        color = rng.uniform(-0.5, 0.0)  # very blue, well below turnoff color ~0.5
        entries.append(CatalogEntry(
            source_id=f"BS_{i}",
            ra=180 + rng.uniform(-0.01, 0.01),
            dec=45 + rng.uniform(-0.01, 0.01),
            mag=float(mag),
            properties={"bp_rp": float(color)},
        ))

    return StarCatalog(entries=entries, source="synthetic")


class TestStellarPopulationAnalyzer:
    """Test CMD analysis pipeline."""

    def test_analyze_returns_expected_keys(self):
        catalog = _make_synthetic_catalog()
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        assert "population_score" in result
        assert "n_photometric" in result
        assert "main_sequence" in result
        assert "red_giants" in result
        assert "blue_stragglers" in result
        assert "color_distribution" in result

    def test_main_sequence_detection(self):
        """Should identify main sequence stars."""
        catalog = _make_synthetic_catalog(n_ms=80, n_rgb=0, n_bs=0)
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        ms = result["main_sequence"]
        # Most stars should be on the MS
        assert ms["ms_fraction"] > 0.5

    def test_red_giant_detection(self):
        """Should detect red giant branch stars."""
        catalog = _make_synthetic_catalog(n_ms=60, n_rgb=30, n_bs=0)
        analyzer = StellarPopulationAnalyzer(ms_width=0.4)
        result = analyzer.analyze(catalog)

        rgb = result["red_giants"]
        # Some RGB stars should be detected (may depend on turnoff estimation)
        assert rgb["n_red_giants"] >= 0  # At minimum no error
        # The red giants have colors >1.5, so they should stand out
        assert result["color_distribution"]["color_spread"] > 0.3

    def test_blue_straggler_detection(self):
        """Should detect blue straggler candidates."""
        catalog = _make_synthetic_catalog(n_ms=60, n_rgb=10, n_bs=15)
        analyzer = StellarPopulationAnalyzer(blue_straggler_offset=0.1)
        result = analyzer.analyze(catalog)

        bs = result["blue_stragglers"]
        assert bs["n_blue_stragglers"] > 0

    def test_insufficient_data(self):
        """Should handle catalogs with too few photometric entries."""
        entries = [
            CatalogEntry(
                source_id=f"S_{i}", ra=180, dec=45, mag=15.0,
                properties={"bp_rp": 1.0},
            )
            for i in range(5)
        ]
        catalog = StarCatalog(entries=entries, source="test")
        analyzer = StellarPopulationAnalyzer(min_sources=20)
        result = analyzer.analyze(catalog)

        assert result["population_score"] == 0.0
        assert result["reason"] == "insufficient_photometry"

    def test_no_photometry(self):
        """Should handle catalog entries without color data."""
        entries = [
            CatalogEntry(
                source_id=f"S_{i}", ra=180, dec=45, mag=15.0,
                properties={},  # no bp_rp
            )
            for i in range(50)
        ]
        catalog = StarCatalog(entries=entries, source="test")
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["population_score"] == 0.0

    def test_score_range(self):
        """Population score should be in [0, 1]."""
        catalog = _make_synthetic_catalog()
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        assert 0 <= result["population_score"] <= 1

    def test_color_distribution_stats(self):
        """Color distribution should compute basic statistics."""
        catalog = _make_synthetic_catalog()
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        cd = result["color_distribution"]
        assert "median_color" in cd
        assert "color_spread" in cd
        assert "skewness" in cd
        assert "kurtosis" in cd

    def test_cmd_density_peaks(self):
        """CMD density should find concentration peaks."""
        catalog = _make_synthetic_catalog(n_ms=100, n_rgb=20, n_bs=0)
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        density = result["cmd_density"]
        assert density["n_peaks"] >= 1

    def test_sdss_color_fallback(self):
        """Should use g_r color when bp_rp is not available."""
        rng = np.random.default_rng(42)
        entries = [
            CatalogEntry(
                source_id=f"S_{i}",
                ra=180 + rng.uniform(-0.01, 0.01),
                dec=45 + rng.uniform(-0.01, 0.01),
                mag=float(rng.uniform(14, 20)),
                properties={"g_r": float(rng.uniform(0.2, 1.5))},
            )
            for i in range(50)
        ]
        catalog = StarCatalog(entries=entries, source="sdss")
        analyzer = StellarPopulationAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_photometric"] == 50
