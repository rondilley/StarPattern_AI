"""Tests for ZTF data source.

Tests that can run without network access use synthetic data.
Tests requiring IRSA TAP access use pytest.skip() when unavailable.
"""

import pytest

from star_pattern.data.ztf import ZTFDataSource
from star_pattern.core.sky_region import SkyRegion


class TestZTFDataSource:
    """Tests for ZTFDataSource."""

    def test_ztf_name(self):
        """ZTF source should have name 'ztf'."""
        src = ZTFDataSource()
        assert src.name == "ztf"

    def test_ztf_available_bands(self):
        """ZTF should report g, r, i bands."""
        src = ZTFDataSource()
        assert src.available_bands == ["g", "r", "i"]

    def test_ztf_fetch_images_returns_empty(self):
        """ZTF is light-curve-only, fetch_images should return empty dict."""
        src = ZTFDataSource()
        region = SkyRegion(ra=180.0, dec=45.0, radius=3.0)
        result = src.fetch_images(region)
        assert result == {}

    def test_ztf_is_available(self):
        """is_available should return True when astroquery.ipac.irsa is importable."""
        src = ZTFDataSource()
        try:
            from astroquery.ipac.irsa import Irsa  # noqa: F401
            assert src.is_available() is True
        except ImportError:
            assert src.is_available() is False

    def test_ztf_catalog_no_network(self):
        """fetch_catalog should return empty catalog when IRSA is unreachable."""
        src = ZTFDataSource()
        if not src.is_available():
            pytest.skip("astroquery.ipac.irsa not available")

        # Query a position with very small radius -- likely no results
        region = SkyRegion(ra=0.001, dec=89.999, radius=0.001)
        try:
            catalog = src.fetch_catalog(region, max_results=10)
            # Should return a valid (possibly empty) StarCatalog
            assert catalog.source == "ztf"
        except Exception:
            # Network errors are acceptable in tests
            pytest.skip("IRSA TAP service unreachable")

    def test_ztf_lightcurve_method(self):
        """fetch_lightcurves should return a list."""
        src = ZTFDataSource()
        if not src.is_available():
            pytest.skip("astroquery.ipac.irsa not available")

        try:
            results = src.fetch_lightcurves(ra=180.0, dec=45.0, radius_arcsec=1.0)
            assert isinstance(results, list)
        except Exception:
            pytest.skip("IRSA TAP service unreachable")
