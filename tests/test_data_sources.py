"""Tests for data sources and caching."""

import numpy as np
import pytest

from star_pattern.core.sky_region import SkyRegion
from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.data.sdss import SDSSDataSource
from star_pattern.data.cache import DataCache


class TestDataCache:
    def test_cache_miss(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        assert cache.get_path("sdss", 180.0, 45.0, 3.0) is None

    def test_cache_put_and_get(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        test_file = tmp_path / "test.fits"
        test_file.write_text("data")
        cache.put("sdss", 180.0, 45.0, 3.0, test_file, band="r")
        result = cache.get_path("sdss", 180.0, 45.0, 3.0, band="r")
        assert result is not None
        assert result.exists()

    def test_cache_size(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        assert cache.size == 0
        test_file = tmp_path / "test.fits"
        test_file.write_text("data")
        cache.put("sdss", 1.0, 2.0, 3.0, test_file)
        assert cache.size == 1

    def test_cache_clear(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        test_file = tmp_path / "cache" / "test.fits"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("data")
        cache.put("sdss", 1.0, 2.0, 3.0, test_file)
        removed = cache.clear()
        assert removed == 1
        assert cache.size == 0

    def test_cache_key_uniqueness(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        # Different bands should produce different keys
        path1 = cache.cache_path_for("sdss", 180.0, 45.0, 3.0, band="r")
        path2 = cache.cache_path_for("sdss", 180.0, 45.0, 3.0, band="g")
        assert path1 != path2

    def test_cache_stale_entry_removed(self, tmp_path):
        cache = DataCache(tmp_path / "cache")
        # Put a path that doesn't actually exist on disk
        nonexistent = tmp_path / "gone.fits"
        cache._index["fakekey"] = {"path": str(nonexistent), "source": "test",
                                    "ra": 0, "dec": 0, "radius": 1, "band": ""}
        cache._save_index()
        # get_path should clean up the stale entry
        result = cache.get_path("test", 0, 0, 1)
        assert result is None


class TestSkyRegion:
    def test_random_region(self):
        rng = np.random.default_rng(42)
        region = SkyRegion.random(min_galactic_lat=20.0, rng=rng)
        assert 0 <= region.ra <= 360
        assert -90 <= region.dec <= 90
        assert region.is_high_latitude(20.0)

    def test_separation(self):
        r1 = SkyRegion(ra=180.0, dec=45.0, radius=3.0)
        r2 = SkyRegion(ra=180.1, dec=45.0, radius=3.0)
        sep = r1.separation_to(r2)
        assert sep > 0

    def test_galactic_coordinates(self):
        region = SkyRegion(ra=0.0, dec=0.0, radius=3.0)
        assert -90 <= region.galactic_lat <= 90
        assert 0 <= region.galactic_lon <= 360

    def test_repr(self):
        region = SkyRegion(ra=180.0, dec=45.0, radius=3.0)
        assert "180" in repr(region)
        assert "45" in repr(region)


class TestSDSSDataSource:
    def test_properties(self):
        ds = SDSSDataSource()
        assert ds.name == "sdss"
        assert "r" in ds.available_bands
        assert len(ds.available_bands) == 5

    def test_is_available(self):
        ds = SDSSDataSource()
        # astroquery should be installed
        assert ds.is_available()

    @pytest.mark.network
    def test_fetch_catalog_real(self):
        """Integration test - fetches real catalog data from SDSS."""
        ds = SDSSDataSource()
        region = SkyRegion(ra=180.0, dec=45.0, radius=1.0)
        cat = ds.fetch_catalog(region, max_results=10)
        assert isinstance(cat, StarCatalog)
        if len(cat) > 0:
            entry = cat[0]
            assert entry.source == "sdss"
            assert entry.ra > 0
            assert entry.mag_band == "r"

    @pytest.mark.network
    def test_fetch_images_real(self):
        """Integration test - fetches real image data from SDSS."""
        ds = SDSSDataSource()
        region = SkyRegion(ra=180.0, dec=45.0, radius=1.0)
        images = ds.fetch_images(region, bands=["r"])
        # SDSS may or may not have coverage here, but shouldn't error
        assert isinstance(images, dict)
