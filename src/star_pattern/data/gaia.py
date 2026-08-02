"""Gaia DR3 data source via TAP+."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion
from star_pattern.data.base import DataSource
from star_pattern.data.cache import DataCache
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("data.gaia")


def _float_or_none(value: Any) -> float | None:
    """Convert a catalog cell to float, or None when it holds no value.

    Truthiness is the wrong test here. A proper motion, parallax, or
    colour index of exactly 0.0 is a real measurement, and `if value`
    discards it, which silently drops the source from every kinematic
    and photometric analysis downstream. Astropy also returns masked
    cells and NaN rather than None for absent values, so both need an
    explicit check.
    """
    if value is None or value is np.ma.masked:
        return None
    if np.ma.is_masked(value):
        return None
    result = float(value)
    return None if np.isnan(result) else result


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

        # No try/except around the query: catching here would stop the
        # @retry_with_backoff decorator from ever seeing the failure, and
        # would turn a Gaia outage into an empty catalog that reads as
        # "no sources in this field".
        job = Gaia.launch_job(query)
        table = job.get_results()

        if table is None or len(table) == 0:
            logger.warning("No Gaia sources found")
            return StarCatalog(source="gaia")

        entries = []
        for row in table:
            g_mag = _float_or_none(row["phot_g_mean_mag"])
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
                        "BP": _float_or_none(row["phot_bp_mean_mag"]),
                        "RP": _float_or_none(row["phot_rp_mean_mag"]),
                        "parallax": _float_or_none(row["parallax"]),
                        "parallax_error": _float_or_none(row["parallax_error"]),
                        "pmra": _float_or_none(row["pmra"]),
                        "pmdec": _float_or_none(row["pmdec"]),
                        "bp_rp": _float_or_none(row["bp_rp"]),
                        "astro_noise": _float_or_none(row["astrometric_excess_noise"]),
                    },
                )
            )

        logger.info(f"Got {len(entries)} Gaia sources")
        catalog = StarCatalog(entries=entries, source="gaia")

        # Cache the catalog
        self._cache.put_catalog(
            "gaia",
            region.ra,
            region.dec,
            region.radius,
            [e.to_dict() for e in entries],
        )

        return catalog

    def is_available(self) -> bool:
        try:
            from astroquery.gaia import Gaia  # noqa: F401 - availability probe

            return True
        except ImportError:
            return False
