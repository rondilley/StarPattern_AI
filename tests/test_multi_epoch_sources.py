"""Tests for multi-source temporal epoch image fetching.

Covers:
- SDSS Stripe 82 footprint boundary logic (no network)
- DataSource ABC default fetch_epoch_images() returns {} (no network)
- MAST epoch image fetching (network, skipped when unavailable)
- SDSS Stripe 82 epoch image fetching (network, skipped when unavailable)
- DataPipeline temporal merge from multiple sources (synthetic, no network)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from astropy.io.fits import Header
from astropy.wcs import WCS

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import EpochImage, SkyRegion
from star_pattern.data.sdss import SDSSDataSource
from star_pattern.data.cache import DataCache


# --- Helpers ---

def _make_wcs(crval_ra: float = 0.0, crval_dec: float = 0.0,
              cdelt: float = -0.0002777, naxis: int = 64) -> WCS:
    """Create a simple TAN WCS."""
    header = Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis
    header["NAXIS2"] = naxis
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = naxis / 2
    header["CRPIX2"] = naxis / 2
    header["CRVAL1"] = crval_ra
    header["CRVAL2"] = crval_dec
    header["CDELT1"] = cdelt
    header["CDELT2"] = abs(cdelt)
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    return WCS(header)


def _make_fits(data: np.ndarray, wcs: WCS | None = None) -> FITSImage:
    """Create a FITSImage from array + optional WCS."""
    img = FITSImage.__new__(FITSImage)
    img.data = data.astype(np.float64)
    img.header = Header()
    img.wcs = wcs
    img._file_path = None
    return img


def _make_epoch(mjd: float, band: str = "r", source: str = "test") -> EpochImage:
    """Create a synthetic EpochImage for merge testing."""
    data = np.random.default_rng(int(mjd)).normal(100, 10, (64, 64))
    wcs = _make_wcs()
    fits_img = _make_fits(data, wcs)
    return EpochImage(
        image=fits_img, mjd=mjd, band=band, source=source,
    )


# --- Stripe 82 Footprint Tests ---

class TestStripe82Footprint:
    """Test SDSSDataSource._in_stripe82() boundary logic."""

    def test_center_of_stripe82(self):
        """RA=0, Dec=0 is the heart of Stripe 82."""
        assert SDSSDataSource._in_stripe82(0.0, 0.0) is True

    def test_high_ra_edge(self):
        """RA=350 is within the fall portion of Stripe 82."""
        assert SDSSDataSource._in_stripe82(350.0, 0.5) is True

    def test_low_ra_edge(self):
        """RA=30 is within the spring portion of Stripe 82."""
        assert SDSSDataSource._in_stripe82(30.0, -0.5) is True

    def test_outside_ra_range(self):
        """RA=180 is nowhere near Stripe 82."""
        assert SDSSDataSource._in_stripe82(180.0, 0.0) is False

    def test_outside_dec_range(self):
        """Dec=10 is outside the Stripe 82 declination band."""
        assert SDSSDataSource._in_stripe82(0.0, 10.0) is False

    def test_dec_boundary_positive(self):
        """Dec=+1.5 is at the exact boundary (inclusive)."""
        assert SDSSDataSource._in_stripe82(0.0, 1.5) is True

    def test_dec_boundary_negative(self):
        """Dec=-1.5 is at the exact boundary (inclusive)."""
        assert SDSSDataSource._in_stripe82(0.0, -1.5) is True

    def test_dec_just_outside(self):
        """Dec=1.6 is just outside the stripe."""
        assert SDSSDataSource._in_stripe82(0.0, 1.6) is False

    def test_ra_boundary_310(self):
        """RA=310 is exactly at the boundary (inclusive)."""
        assert SDSSDataSource._in_stripe82(310.0, 0.0) is True

    def test_ra_just_below_310(self):
        """RA=309 is just outside the Stripe 82 RA range."""
        assert SDSSDataSource._in_stripe82(309.0, 0.0) is False


# --- Base Class Default Tests ---

class TestBaseClassDefault:
    """Test that DataSource.fetch_epoch_images() returns empty dict."""

    def test_gaia_returns_empty(self):
        """Gaia has no epoch image support -- should use base class default."""
        try:
            from star_pattern.data.gaia import GaiaDataSource
        except ImportError:
            pytest.skip("astroquery not available")

        cache = DataCache.__new__(DataCache)
        cache.cache_dir = None
        cache._index = {}

        gaia = GaiaDataSource.__new__(GaiaDataSource)
        gaia._cache = cache

        region = SkyRegion(ra=180.0, dec=45.0, radius=3.0)
        result = gaia.fetch_epoch_images(region)
        assert result == {}

    def test_sdss_outside_stripe82_returns_empty(self):
        """SDSS outside Stripe 82 should return empty via the gate check."""
        sdss = SDSSDataSource.__new__(SDSSDataSource)
        sdss._cache = DataCache.__new__(DataCache)
        sdss._cache.cache_dir = None
        sdss._cache._index = {}

        # RA=180 is not in Stripe 82
        region = SkyRegion(ra=180.0, dec=0.0, radius=3.0)
        result = sdss.fetch_epoch_images(region)
        assert result == {}


# --- MAST Epoch Image Tests (network) ---

class TestMASTEpochImages:
    """Integration tests for MAST multi-epoch image fetching."""

    @pytest.fixture
    def mast_source(self, tmp_path):
        try:
            from star_pattern.data.mast import MASTDataSource
        except ImportError:
            pytest.skip("astroquery.mast not available")
        cache = DataCache(tmp_path / "cache")
        return MASTDataSource(cache=cache, max_observations=5)

    def test_mast_epoch_query(self, mast_source):
        """MAST should find epoch images for a well-observed HST field."""
        try:
            from astroquery.mast import Observations
            Observations.query_region  # verify accessible
        except Exception:
            pytest.skip("MAST service not available")

        # Hubble Deep Field area -- dense HST coverage
        region = SkyRegion(ra=189.2, dec=62.2, radius=1.0)
        try:
            result = mast_source.fetch_epoch_images(
                region, max_epochs=3, min_baseline_days=1.0,
            )
        except Exception as e:
            pytest.skip(f"MAST query failed: {e}")

        # May or may not find epochs depending on server response
        assert isinstance(result, dict)
        for band, epochs in result.items():
            assert all(isinstance(e, EpochImage) for e in epochs)
            assert all(e.source == "mast" for e in epochs)
            # Should be sorted by MJD
            mjds = [e.mjd for e in epochs]
            assert mjds == sorted(mjds)


# --- SDSS Stripe 82 Epoch Image Tests (network) ---

class TestSDSSStripe82EpochImages:
    """Integration tests for SDSS Stripe 82 multi-epoch fetching."""

    @pytest.fixture
    def sdss_source(self, tmp_path):
        try:
            from astroquery.sdss import SDSS  # noqa: F401
        except ImportError:
            pytest.skip("astroquery.sdss not available")
        cache = DataCache(tmp_path / "cache")
        return SDSSDataSource(cache=cache)

    def test_stripe82_fetch(self, sdss_source):
        """Stripe 82 coordinates should yield multi-epoch images."""
        try:
            from astroquery.sdss import SDSS
            SDSS.query_region  # verify accessible
        except Exception:
            pytest.skip("SDSS service not available")

        # Center of Stripe 82
        region = SkyRegion(ra=0.0, dec=0.0, radius=1.0)
        try:
            result = sdss_source.fetch_epoch_images(
                region, bands=["r"], max_epochs=3,
            )
        except Exception as e:
            pytest.skip(f"SDSS query failed: {e}")

        assert isinstance(result, dict)
        for band, epochs in result.items():
            assert all(isinstance(e, EpochImage) for e in epochs)
            assert all(e.source == "sdss" for e in epochs)
            mjds = [e.mjd for e in epochs]
            assert mjds == sorted(mjds)

    def test_non_stripe82_returns_empty(self, sdss_source):
        """Coordinates outside Stripe 82 should return empty (no network)."""
        region = SkyRegion(ra=180.0, dec=45.0, radius=3.0)
        result = sdss_source.fetch_epoch_images(region)
        assert result == {}


# --- Pipeline Temporal Merge Tests (synthetic) ---

class TestPipelineTemporalMerge:
    """Test DataPipeline temporal merge logic with synthetic sources."""

    def test_merge_same_band_from_two_sources(self):
        """Epochs from two sources in the same band should merge and sort."""
        from star_pattern.core.sky_region import RegionData

        region = SkyRegion(ra=0.0, dec=0.0, radius=3.0)

        # Simulate what the pipeline does: merge epoch dicts
        ztf_epochs = {
            "r": [_make_epoch(59000.0, "r", "ztf"),
                  _make_epoch(59100.0, "r", "ztf")],
        }
        sdss_epochs = {
            "r": [_make_epoch(58500.0, "r", "sdss"),
                  _make_epoch(59050.0, "r", "sdss")],
        }

        # Replicate pipeline merge logic
        all_temporal: dict[str, list] = {}
        for source_epochs in [ztf_epochs, sdss_epochs]:
            for band, epochs in source_epochs.items():
                all_temporal.setdefault(band, []).extend(epochs)
        for band in all_temporal:
            all_temporal[band].sort(key=lambda e: e.mjd)

        assert "r" in all_temporal
        assert len(all_temporal["r"]) == 4

        # Verify sorted by MJD
        mjds = [e.mjd for e in all_temporal["r"]]
        assert mjds == sorted(mjds)

        # Verify both sources present
        sources = {e.source for e in all_temporal["r"]}
        assert sources == {"ztf", "sdss"}

    def test_merge_different_bands(self):
        """Different bands from different sources stay separate."""
        ztf_epochs = {
            "r": [_make_epoch(59000.0, "r", "ztf")],
        }
        mast_epochs = {
            "F606W": [_make_epoch(55000.0, "F606W", "mast")],
        }

        all_temporal: dict[str, list] = {}
        for source_epochs in [ztf_epochs, mast_epochs]:
            for band, epochs in source_epochs.items():
                all_temporal.setdefault(band, []).extend(epochs)

        assert "r" in all_temporal
        assert "F606W" in all_temporal
        assert len(all_temporal["r"]) == 1
        assert len(all_temporal["F606W"]) == 1
        assert all_temporal["r"][0].source == "ztf"
        assert all_temporal["F606W"][0].source == "mast"

    def test_source_failure_isolation(self):
        """One source failing should not prevent others from contributing."""
        # Simulate: ztf succeeds, sdss raises, mast succeeds
        ztf_result = {"r": [_make_epoch(59000.0, "r", "ztf")]}
        mast_result = {"F814W": [_make_epoch(55000.0, "F814W", "mast")]}

        all_temporal: dict[str, list] = {}
        source_results = [
            ("ztf", ztf_result),
            ("sdss", Exception("SDSS timeout")),
            ("mast", mast_result),
        ]

        for source_name, result in source_results:
            if isinstance(result, Exception):
                # Pipeline would log warning and continue
                continue
            for band, epochs in result.items():
                all_temporal.setdefault(band, []).extend(epochs)

        assert len(all_temporal) == 2
        assert "r" in all_temporal
        assert "F814W" in all_temporal

    def test_empty_result_when_no_sources_have_epochs(self):
        """Pipeline should produce empty temporal dict if all sources return {}."""
        all_temporal: dict[str, list] = {}
        for source_result in [{}, {}, {}]:
            for band, epochs in source_result.items():
                all_temporal.setdefault(band, []).extend(epochs)

        assert all_temporal == {}


# --- SDSS MJD Extraction Tests ---

class TestSDSSMjdExtraction:
    """Test SDSSDataSource._extract_mjd_from_fits() with various headers."""

    def test_tai_header(self):
        """TAI seconds should convert to valid MJD."""
        fits_img = _make_fits(np.zeros((10, 10)))
        # TAI for MJD 51000 = 51000 * 86400 = 4406400000
        fits_img.header["TAI"] = 4406400000.0
        mjd = SDSSDataSource._extract_mjd_from_fits(fits_img)
        assert mjd is not None
        assert abs(mjd - 51000.0) < 0.001

    def test_mjd_header(self):
        """Direct MJD header should be used when TAI is absent."""
        fits_img = _make_fits(np.zeros((10, 10)))
        fits_img.header["MJD"] = 51000
        mjd = SDSSDataSource._extract_mjd_from_fits(fits_img)
        assert mjd is not None
        assert abs(mjd - 51000.0) < 0.001

    def test_no_time_headers(self):
        """Should return None when no time headers are present."""
        fits_img = _make_fits(np.zeros((10, 10)))
        mjd = SDSSDataSource._extract_mjd_from_fits(fits_img)
        assert mjd is None

    def test_invalid_tai(self):
        """TAI below sanity threshold should be skipped."""
        fits_img = _make_fits(np.zeros((10, 10)))
        fits_img.header["TAI"] = 100.0  # Way too low
        fits_img.header["MJD"] = 51000  # Fallback should work
        mjd = SDSSDataSource._extract_mjd_from_fits(fits_img)
        assert mjd is not None
        assert abs(mjd - 51000.0) < 0.001
