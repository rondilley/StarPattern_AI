"""Tests for SDSS data source footprint gating."""

import pytest

from star_pattern.core.sky_region import SkyRegion
from star_pattern.data.sdss import SDSSDataSource


class TestSDSSFootprintGate:
    """Tests for SDSS survey footprint gating."""

    def test_in_main_footprint_accepts_north(self):
        """Coordinates well within SDSS coverage should pass."""
        assert SDSSDataSource._in_main_footprint(180.0, 45.0) is True

    def test_in_main_footprint_accepts_equator(self):
        """Equatorial coordinates should pass."""
        assert SDSSDataSource._in_main_footprint(0.0, 0.0) is True

    def test_in_main_footprint_accepts_boundary(self):
        """Dec = -15 (boundary) should pass."""
        assert SDSSDataSource._in_main_footprint(120.0, -15.0) is True

    def test_in_main_footprint_rejects_deep_south(self):
        """Dec well below -15 should be rejected."""
        assert SDSSDataSource._in_main_footprint(184.9, -25.5) is False

    def test_in_main_footprint_rejects_south_pole(self):
        """South pole should be rejected."""
        assert SDSSDataSource._in_main_footprint(0.0, -90.0) is False

    def test_in_main_footprint_rejects_just_below(self):
        """Dec just below -15 should be rejected."""
        assert SDSSDataSource._in_main_footprint(60.0, -15.1) is False

    def test_images_skipped_out_of_footprint(self):
        """Image fetch should return empty dict for out-of-footprint regions
        without making any network calls."""
        src = SDSSDataSource()
        region = SkyRegion(ra=184.9, dec=-25.5, radius=3.0)
        try:
            result = src.fetch_images(region)
            assert result == {}
        except Exception:
            pytest.fail("Out-of-footprint image fetch should not raise")

    def test_declination_range(self):
        """SDSS declination range should match its footprint gate."""
        src = SDSSDataSource()
        assert src.declination_range == (-15.0, 90.0)

    def test_catalog_skipped_out_of_footprint(self):
        """Catalog fetch should return empty StarCatalog for out-of-footprint
        regions without making any network calls."""
        src = SDSSDataSource()
        region = SkyRegion(ra=184.9, dec=-25.5, radius=3.0)
        try:
            catalog = src.fetch_catalog(region)
            assert catalog.source == "sdss"
            assert len(catalog.entries) == 0
        except Exception:
            pytest.fail("Out-of-footprint catalog fetch should not raise")
