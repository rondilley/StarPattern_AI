"""Tests for catalog caching across runs."""

import json
from pathlib import Path

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.sky_region import SkyRegion
from star_pattern.data.cache import DataCache


class TestCatalogEntrySerialization:
    """Test CatalogEntry to_dict/from_dict roundtrip."""

    def test_basic_roundtrip(self):
        entry = CatalogEntry(
            ra=180.0,
            dec=45.0,
            mag=18.5,
            mag_band="r",
            obj_type="star",
            source="sdss",
            source_id="obj123",
            properties={"u": 19.0, "g": 18.8},
        )
        d = entry.to_dict()
        restored = CatalogEntry.from_dict(d)

        assert restored.ra == 180.0
        assert restored.dec == 45.0
        assert restored.mag == 18.5
        assert restored.mag_band == "r"
        assert restored.obj_type == "star"
        assert restored.source == "sdss"
        assert restored.source_id == "obj123"
        assert restored.properties["u"] == 19.0

    def test_none_mag(self):
        entry = CatalogEntry(ra=10.0, dec=20.0, mag=None)
        d = entry.to_dict()
        restored = CatalogEntry.from_dict(d)
        assert restored.mag is None

    def test_complex_properties(self):
        """Properties with nested structures (like ZTF light curves) survive."""
        lc_data = {
            "g": [[59000.0, 18.5, 0.02], [59001.0, 18.6, 0.03]],
            "r": [[59000.0, 17.9, 0.01]],
        }
        entry = CatalogEntry(
            ra=10.0, dec=20.0,
            source="ztf",
            properties={"ztf_lightcurve": lc_data, "ztf_n_epochs": 3},
        )
        d = entry.to_dict()
        # Must survive JSON roundtrip
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = CatalogEntry.from_dict(d2)

        assert restored.properties["ztf_n_epochs"] == 3
        assert len(restored.properties["ztf_lightcurve"]["g"]) == 2

    def test_defaults_on_missing_fields(self):
        d = {"ra": 1.0, "dec": 2.0}
        entry = CatalogEntry.from_dict(d)
        assert entry.mag is None
        assert entry.obj_type == "unknown"
        assert entry.source == ""
        assert entry.properties == {}


class TestDataCacheCatalog:
    """Test DataCache catalog get/put methods."""

    def test_put_and_get_catalog(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path / "cache")
        entries = [
            {"ra": 180.0, "dec": 45.0, "mag": 18.0, "source": "sdss"},
            {"ra": 180.1, "dec": 45.1, "mag": 19.0, "source": "sdss"},
        ]
        cache.put_catalog("sdss", 180.0, 45.0, 3.0, entries)
        result = cache.get_catalog("sdss", 180.0, 45.0, 3.0)

        assert result is not None
        assert len(result) == 2
        assert result[0]["ra"] == 180.0
        assert result[1]["mag"] == 19.0

    def test_cache_miss(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path / "cache")
        result = cache.get_catalog("sdss", 100.0, 50.0, 3.0)
        assert result is None

    def test_different_sources_independent(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path / "cache")
        sdss_entries = [{"ra": 180.0, "dec": 45.0, "source": "sdss"}]
        gaia_entries = [{"ra": 180.0, "dec": 45.0, "source": "gaia"}]

        cache.put_catalog("sdss", 180.0, 45.0, 3.0, sdss_entries)
        cache.put_catalog("gaia", 180.0, 45.0, 3.0, gaia_entries)

        assert cache.get_catalog("sdss", 180.0, 45.0, 3.0)[0]["source"] == "sdss"
        assert cache.get_catalog("gaia", 180.0, 45.0, 3.0)[0]["source"] == "gaia"

    def test_different_regions_independent(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path / "cache")
        cache.put_catalog("sdss", 180.0, 45.0, 3.0, [{"ra": 180.0}])
        cache.put_catalog("sdss", 200.0, 50.0, 3.0, [{"ra": 200.0}])

        r1 = cache.get_catalog("sdss", 180.0, 45.0, 3.0)
        r2 = cache.get_catalog("sdss", 200.0, 50.0, 3.0)
        assert r1[0]["ra"] == 180.0
        assert r2[0]["ra"] == 200.0

    def test_catalog_persists_across_instances(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache1 = DataCache(cache_dir=cache_dir)
        cache1.put_catalog("gaia", 180.0, 45.0, 3.0, [{"ra": 180.0}])

        # New cache instance reads same directory
        cache2 = DataCache(cache_dir=cache_dir)
        result = cache2.get_catalog("gaia", 180.0, 45.0, 3.0)
        assert result is not None
        assert len(result) == 1

    def test_catalog_not_confused_with_images(self, tmp_path):
        """Catalog cache entries don't collide with image cache entries."""
        cache = DataCache(cache_dir=tmp_path / "cache")

        # Cache an image entry
        img_path = tmp_path / "test.fits"
        img_path.write_bytes(b"fake fits")
        cache.put("sdss", 180.0, 45.0, 3.0, img_path, band="r")

        # Cache a catalog entry for same region
        cache.put_catalog("sdss", 180.0, 45.0, 3.0, [{"ra": 180.0}])

        # Both should be retrievable independently
        assert cache.get_path("sdss", 180.0, 45.0, 3.0, "r") is not None
        assert cache.get_catalog("sdss", 180.0, 45.0, 3.0) is not None

    def test_clear_removes_catalogs(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path / "cache")
        cache.put_catalog("sdss", 180.0, 45.0, 3.0, [{"ra": 180.0}])
        assert cache.get_catalog("sdss", 180.0, 45.0, 3.0) is not None

        cache.clear()
        assert cache.get_catalog("sdss", 180.0, 45.0, 3.0) is None

    def test_large_catalog(self, tmp_path):
        """Handles catalogs with many entries."""
        cache = DataCache(cache_dir=tmp_path / "cache")
        entries = [{"ra": float(i), "dec": float(i), "mag": 18.0 + i * 0.01} for i in range(5000)]
        cache.put_catalog("gaia", 180.0, 45.0, 3.0, entries)
        result = cache.get_catalog("gaia", 180.0, 45.0, 3.0)
        assert len(result) == 5000

    def test_ztf_lightcurve_roundtrip(self, tmp_path):
        """ZTF light curve data (tuples as lists) survives caching."""
        cache = DataCache(cache_dir=tmp_path / "cache")
        entries = [
            {
                "ra": 180.0,
                "dec": 45.0,
                "source": "ztf",
                "properties": {
                    "ztf_lightcurve": {
                        "g": [[59000.0, 18.5, 0.02], [59001.0, 18.6, 0.03]],
                        "r": [[59000.0, 17.9, 0.01]],
                    },
                    "ztf_n_epochs": 3,
                    "ztf_baseline_days": 1.0,
                },
            },
        ]
        cache.put_catalog("ztf", 180.0, 45.0, 3.0, entries)
        result = cache.get_catalog("ztf", 180.0, 45.0, 3.0)
        assert result is not None
        lc = result[0]["properties"]["ztf_lightcurve"]
        assert len(lc["g"]) == 2
        assert lc["g"][0][0] == 59000.0
