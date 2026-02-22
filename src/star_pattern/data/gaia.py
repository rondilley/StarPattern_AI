"""Gaia DR3 data source via TAP+."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion
from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.data.base import DataSource
from star_pattern.data.cache import DataCache
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("data.gaia")


class GaiaDataSource(DataSource):
    """Data source for Gaia DR3 (catalog-only, no images)."""

    def __init__(self, cache: DataCache | None = None):
        self._cache = cache or DataCache()

    @property
    def name(self) -> str:
        return "gaia"

    @property
    def available_bands(self) -> list[str]:
        return ["G", "BP", "RP"]

    def fetch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
    ) -> dict[str, FITSImage]:
        """Gaia has no images - returns empty dict."""
        logger.debug("Gaia is catalog-only, no images available")
        return {}

    @retry_with_backoff(max_retries=3, base_delay=5.0)
    def fetch_catalog(
        self,
        region: SkyRegion,
        max_results: int = 10000,
    ) -> StarCatalog:
        """Fetch Gaia DR3 catalog data using TAP+."""
        # Check catalog cache
        try:
            cached_entries = self._cache.get_catalog("gaia", region.ra, region.dec, region.radius)
            if cached_entries is not None:
                entries = [CatalogEntry.from_dict(d) for d in cached_entries]
                logger.info(f"Loaded {len(entries)} cached Gaia catalog entries")
                return StarCatalog(entries=entries, source="gaia")
        except Exception as e:
            logger.debug(f"Gaia catalog cache check failed: {e}")

        from astroquery.gaia import Gaia

        logger.info(f"Fetching Gaia DR3 catalog for ({region.ra:.3f}, {region.dec:.3f})")

        query = f"""
        SELECT TOP {max_results}
            source_id, ra, dec, phot_g_mean_mag,
            phot_bp_mean_mag, phot_rp_mean_mag,
            parallax, parallax_error, pmra, pmdec,
            bp_rp, astrometric_excess_noise
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {region.ra}, {region.dec}, {region.radius / 60})
        )
        ORDER BY phot_g_mean_mag ASC
        """

        try:
            job = Gaia.launch_job(query)
            table = job.get_results()
        except Exception as e:
            logger.error(f"Gaia query failed: {e}")
            return StarCatalog(source="gaia")

        if table is None or len(table) == 0:
            logger.warning("No Gaia sources found")
            return StarCatalog(source="gaia")

        entries = []
        for row in table:
            g_mag = float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] is not None else None
            entries.append(
                CatalogEntry(
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    mag=g_mag,
                    mag_band="G",
                    obj_type="star",
                    source="gaia",
                    source_id=str(row["source_id"]),
                    properties={
                        "G": g_mag,
                        "BP": float(row["phot_bp_mean_mag"]) if row["phot_bp_mean_mag"] else None,
                        "RP": float(row["phot_rp_mean_mag"]) if row["phot_rp_mean_mag"] else None,
                        "parallax": float(row["parallax"]) if row["parallax"] else None,
                        "parallax_error": float(row["parallax_error"]) if row["parallax_error"] else None,
                        "pmra": float(row["pmra"]) if row["pmra"] else None,
                        "pmdec": float(row["pmdec"]) if row["pmdec"] else None,
                        "bp_rp": float(row["bp_rp"]) if row["bp_rp"] else None,
                        "astro_noise": float(row["astrometric_excess_noise"]) if row["astrometric_excess_noise"] else None,
                    },
                )
            )

        logger.info(f"Got {len(entries)} Gaia sources")
        catalog = StarCatalog(entries=entries, source="gaia")

        # Cache the catalog
        self._cache.put_catalog(
            "gaia", region.ra, region.dec, region.radius,
            [e.to_dict() for e in entries],
        )

        return catalog

    def is_available(self) -> bool:
        try:
            from astroquery.gaia import Gaia

            return True
        except ImportError:
            return False
