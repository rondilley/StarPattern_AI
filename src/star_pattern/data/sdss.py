"""SDSS data source using astroquery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion
from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.data.base import DataSource
from star_pattern.data.cache import DataCache
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("data.sdss")


class SDSSDataSource(DataSource):
    """Data source for the Sloan Digital Sky Survey."""

    BANDS = ["u", "g", "r", "i", "z"]

    def __init__(self, cache: DataCache | None = None):
        self._cache = cache or DataCache()

    @property
    def name(self) -> str:
        return "sdss"

    @property
    def available_bands(self) -> list[str]:
        return self.BANDS

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
    ) -> dict[str, FITSImage]:
        """Fetch SDSS images for a sky region."""
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        bands = bands or ["r"]  # Default to r-band
        images: dict[str, FITSImage] = {}

        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")

        for band in bands:
            # Check cache
            cached = self._cache.get_path("sdss", region.ra, region.dec, region.radius, band)
            if cached:
                logger.info(f"Loading cached SDSS {band}-band for ({region.ra:.3f}, {region.dec:.3f})")
                images[band] = FITSImage.from_file(cached)
                continue

            logger.info(f"Fetching SDSS {band}-band for ({region.ra:.3f}, {region.dec:.3f})")
            try:
                hdu_list = SDSS.get_images(
                    coordinates=coord,
                    radius=region.radius * u.arcmin,
                    band=band,
                    timeout=120,
                )
                if hdu_list and len(hdu_list) > 0:
                    # Save to cache
                    cache_path = self._cache.cache_path_for(
                        "sdss", region.ra, region.dec, region.radius, band
                    )
                    hdu_list[0].writeto(str(cache_path), overwrite=True)
                    self._cache.put("sdss", region.ra, region.dec, region.radius, cache_path, band)

                    # Load as FITSImage
                    img = FITSImage.from_file(cache_path)
                    images[band] = img
                    logger.info(f"  Got {band}-band image: {img.shape}")
                else:
                    logger.warning(f"  No SDSS {band}-band image found")
            except Exception as e:
                logger.error(f"  Failed to fetch SDSS {band}-band: {e}")

        return images

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def fetch_catalog(
        self,
        region: SkyRegion,
        max_results: int = 10000,
    ) -> StarCatalog:
        """Fetch SDSS photometric catalog for a region."""
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Check catalog cache
        cached_entries = self._cache.get_catalog("sdss", region.ra, region.dec, region.radius)
        if cached_entries is not None:
            entries = [CatalogEntry.from_dict(d) for d in cached_entries]
            logger.info(f"Loaded {len(entries)} cached SDSS catalog entries")
            return StarCatalog(entries=entries, source="sdss")

        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")

        logger.info(f"Fetching SDSS catalog for ({region.ra:.3f}, {region.dec:.3f})")
        try:
            table = SDSS.query_region(
                coord,
                radius=region.radius * u.arcmin,
                photoobj_fields=[
                    "objid", "ra", "dec", "type",
                    "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
                ],
            )
        except Exception as e:
            logger.error(f"  SDSS catalog query failed: {e}")
            return StarCatalog(source="sdss")

        if table is None or len(table) == 0:
            logger.warning("  No SDSS catalog entries found")
            return StarCatalog(source="sdss")

        entries = []
        for row in table[:max_results]:
            # SDSS type: 3=galaxy, 6=star
            obj_type = "galaxy" if row["type"] == 3 else "star" if row["type"] == 6 else "unknown"
            r_mag = float(row["psfMag_r"]) if row["psfMag_r"] is not None else None
            entries.append(
                CatalogEntry(
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    mag=r_mag,
                    mag_band="r",
                    obj_type=obj_type,
                    source="sdss",
                    source_id=str(row["objid"]),
                    properties={
                        "u": float(row["psfMag_u"]) if row["psfMag_u"] is not None else None,
                        "g": float(row["psfMag_g"]) if row["psfMag_g"] is not None else None,
                        "r": float(row["psfMag_r"]) if row["psfMag_r"] is not None else None,
                        "i": float(row["psfMag_i"]) if row["psfMag_i"] is not None else None,
                        "z": float(row["psfMag_z"]) if row["psfMag_z"] is not None else None,
                    },
                )
            )

        logger.info(f"  Got {len(entries)} catalog entries")
        catalog = StarCatalog(entries=entries, source="sdss")

        # Cache the catalog
        self._cache.put_catalog(
            "sdss", region.ra, region.dec, region.radius,
            [e.to_dict() for e in entries],
        )

        return catalog

    def is_available(self) -> bool:
        try:
            from astroquery.sdss import SDSS

            return True
        except ImportError:
            return False
