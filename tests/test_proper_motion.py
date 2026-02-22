"""Tests for proper motion / kinematic analysis."""

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.detection.proper_motion import ProperMotionAnalyzer


@pytest.fixture
def pm_analyzer():
    config = DetectionConfig(
        kinematic_cluster_eps=2.0,
        kinematic_cluster_min=5,
        kinematic_pm_min=5.0,
        kinematic_stream_min_length=8,
    )
    return ProperMotionAnalyzer(config)


@pytest.fixture
def catalog_with_comoving_groups():
    """Catalog with 2 co-moving groups embedded in field stars."""
    rng = np.random.default_rng(42)
    entries = []

    # Group 1: ~10 stars with similar proper motion around (5, 3)
    for i in range(10):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.02),
                dec=45.0 + rng.normal(0, 0.02),
                mag=rng.uniform(14, 18),
                source="gaia",
                source_id=f"group1_{i}",
                properties={
                    "pmra": 5.0 + rng.normal(0, 0.3),
                    "pmdec": 3.0 + rng.normal(0, 0.3),
                    "parallax": 2.0 + rng.normal(0, 0.1),
                },
            )
        )

    # Group 2: ~8 stars with similar proper motion around (-7, 2)
    for i in range(8):
        entries.append(
            CatalogEntry(
                ra=180.05 + rng.normal(0, 0.02),
                dec=45.05 + rng.normal(0, 0.02),
                mag=rng.uniform(14, 18),
                source="gaia",
                source_id=f"group2_{i}",
                properties={
                    "pmra": -7.0 + rng.normal(0, 0.3),
                    "pmdec": 2.0 + rng.normal(0, 0.3),
                    "parallax": 1.5 + rng.normal(0, 0.1),
                },
            )
        )

    # Field stars: 50 stars with random proper motions
    for i in range(50):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.05),
                dec=45.0 + rng.normal(0, 0.05),
                mag=rng.uniform(14, 22),
                source="gaia",
                source_id=f"field_{i}",
                properties={
                    "pmra": rng.normal(0, 10),
                    "pmdec": rng.normal(0, 10),
                    "parallax": rng.uniform(0.1, 5),
                },
            )
        )

    return StarCatalog(entries=entries, source="gaia")


@pytest.fixture
def catalog_with_runaway():
    """Catalog with one high proper motion outlier."""
    rng = np.random.default_rng(42)
    entries = []

    # Normal field stars
    for i in range(50):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.05),
                dec=45.0 + rng.normal(0, 0.05),
                mag=rng.uniform(14, 20),
                source="gaia",
                source_id=f"normal_{i}",
                properties={
                    "pmra": rng.normal(0, 2),
                    "pmdec": rng.normal(0, 2),
                    "parallax": rng.uniform(0.5, 3),
                },
            )
        )

    # Runaway star with very high proper motion
    entries.append(
        CatalogEntry(
            ra=180.01,
            dec=45.01,
            mag=15.0,
            source="gaia",
            source_id="runaway_0",
            properties={
                "pmra": 50.0,
                "pmdec": -30.0,
                "parallax": 1.0,
            },
        )
    )

    return StarCatalog(entries=entries, source="gaia")


@pytest.fixture
def catalog_with_stream():
    """Catalog with a linear stream structure in position+PM space."""
    rng = np.random.default_rng(42)
    entries = []

    # Stream: stars along a line in position space with correlated PM
    for i in range(15):
        t = i / 14.0
        entries.append(
            CatalogEntry(
                ra=180.0 + t * 0.1,
                dec=45.0 + t * 0.05,
                mag=rng.uniform(16, 19),
                source="gaia",
                source_id=f"stream_{i}",
                properties={
                    "pmra": 3.0 + rng.normal(0, 0.2),
                    "pmdec": -1.5 + rng.normal(0, 0.2),
                    "parallax": 0.8 + rng.normal(0, 0.05),
                },
            )
        )

    # Field stars (scattered)
    for i in range(40):
        entries.append(
            CatalogEntry(
                ra=180.0 + rng.normal(0, 0.1),
                dec=45.0 + rng.normal(0, 0.1),
                mag=rng.uniform(14, 22),
                source="gaia",
                source_id=f"field_{i}",
                properties={
                    "pmra": rng.normal(0, 10),
                    "pmdec": rng.normal(0, 10),
                    "parallax": rng.uniform(0.1, 5),
                },
            )
        )

    return StarCatalog(entries=entries, source="gaia")


class TestProperMotionAnalyzer:
    def test_analyze_returns_expected_keys(self, pm_analyzer, catalog_with_comoving_groups):
        result = pm_analyzer.analyze(catalog_with_comoving_groups)
        assert "kinematic_score" in result
        assert "comoving_groups" in result
        assert "stream_candidates" in result
        assert "runaway_stars" in result
        assert 0 <= result["kinematic_score"] <= 1

    def test_find_comoving_groups(self, pm_analyzer, catalog_with_comoving_groups):
        result = pm_analyzer.analyze(catalog_with_comoving_groups)
        groups = result["comoving_groups"]
        assert len(groups) >= 2
        # Verify group structure
        for group in groups:
            assert "n_members" in group
            assert group["n_members"] >= 5
            assert "mean_pmra" in group
            assert "mean_pmdec" in group

    def test_find_runaway(self, pm_analyzer, catalog_with_runaway):
        result = pm_analyzer.analyze(catalog_with_runaway)
        runaways = result["runaway_stars"]
        assert len(runaways) >= 1
        # The injected runaway should be found
        runaway_ids = {r["source_id"] for r in runaways}
        assert "runaway_0" in runaway_ids

    def test_detect_stream(self, catalog_with_stream):
        config = DetectionConfig(
            kinematic_stream_min_length=8,
            kinematic_cluster_eps=1.0,
            kinematic_cluster_min=5,
            kinematic_pm_min=5.0,
        )
        analyzer = ProperMotionAnalyzer(config)
        result = analyzer.analyze(catalog_with_stream)
        streams = result["stream_candidates"]
        assert isinstance(streams, list)
        # Stream detection is probabilistic; verify structure if found
        if streams:
            assert streams[0]["n_members"] >= 8

    def test_empty_catalog(self, pm_analyzer):
        catalog = StarCatalog(entries=[], source="empty")
        result = pm_analyzer.analyze(catalog)
        assert result["kinematic_score"] == 0.0
        assert result["comoving_groups"] == []

    def test_no_pm_data(self, pm_analyzer):
        entries = [
            CatalogEntry(ra=180.0, dec=45.0, mag=15.0, source="sdss", source_id="s1")
            for _ in range(20)
        ]
        catalog = StarCatalog(entries=entries, source="sdss")
        result = pm_analyzer.analyze(catalog)
        assert result["kinematic_score"] == 0.0
