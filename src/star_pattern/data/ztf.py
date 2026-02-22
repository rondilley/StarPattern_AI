"""Zwicky Transient Facility light curve data via IRSA TAP."""

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

logger = get_logger("data.ztf")

# ZTF data release number for TAP queries
ZTF_DR = 22


class ZTFDataSource(DataSource):
    """Zwicky Transient Facility light curve data via IRSA TAP.

    ZTF is a light-curve-only source for our purposes. It provides
    multi-epoch photometry in g, r, and i bands from the Palomar 48-inch
    telescope. Light curves are stored in CatalogEntry.properties for
    downstream variability analysis.
    """

    def __init__(self, cache: DataCache | None = None):
        self._cache = cache or DataCache()

    @property
    def name(self) -> str:
        return "ztf"

    @property
    def available_bands(self) -> list[str]:
        return ["g", "r", "i"]

    def fetch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
    ) -> dict[str, FITSImage]:
        """ZTF is light-curve-only -- returns empty dict."""
        logger.debug("ZTF is light-curve-only, no images available")
        return {}

    @retry_with_backoff(max_retries=2, base_delay=5.0)
    def fetch_catalog(
        self,
        region: SkyRegion,
        max_results: int = 10000,
    ) -> StarCatalog:
        """Fetch ZTF sources with light curves as catalog entries.

        For each source in the region, fetches the full light curve
        and stores it in CatalogEntry.properties["ztf_lightcurve"].
        """
        # Check catalog cache
        try:
            cached_entries = self._cache.get_catalog("ztf", region.ra, region.dec, region.radius)
            if cached_entries is not None:
                entries = [CatalogEntry.from_dict(d) for d in cached_entries]
                logger.info(f"Loaded {len(entries)} cached ZTF catalog entries")
                return StarCatalog(entries=entries, source="ztf")
        except Exception as e:
            logger.debug(f"ZTF catalog cache check failed: {e}")

        try:
            from astroquery.ipac.irsa import Irsa
        except ImportError:
            logger.warning("astroquery.ipac.irsa not available for ZTF queries")
            return StarCatalog(source="ztf")

        logger.info(
            f"Fetching ZTF light curves for ({region.ra:.3f}, {region.dec:.3f})"
        )

        # Query the ZTF objects table for sources in the region
        radius_deg = region.radius / 60.0
        obj_query = (
            f"SELECT TOP {max_results} "
            f"oid, ra, dec, meanmag, nobs "
            f"FROM ztf_objects_dr{ZTF_DR} "
            f"WHERE CONTAINS("
            f"POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {region.ra}, {region.dec}, {radius_deg})"
            f") = 1 "
            f"ORDER BY meanmag ASC"
        )

        try:
            obj_table = Irsa.query_tap(obj_query).to_table()
        except Exception as e:
            logger.warning(f"ZTF object query failed: {e}")
            return StarCatalog(source="ztf")

        if obj_table is None or len(obj_table) == 0:
            logger.info("No ZTF sources found in region")
            return StarCatalog(source="ztf")

        # Fetch light curves for all sources in one bulk query
        lightcurves = self._fetch_bulk_lightcurves(
            region.ra, region.dec, radius_deg
        )

        entries = []
        for row in obj_table:
            oid = str(row["oid"])
            ra_val = float(row["ra"])
            dec_val = float(row["dec"])
            mean_mag = float(row["meanmag"]) if row["meanmag"] is not None else None
            n_obs = int(row["nobs"]) if row["nobs"] is not None else 0

            # Get light curve data for this source
            source_lc = lightcurves.get(oid, {})

            # Compute baseline if we have light curve data
            baseline_days = 0.0
            total_epochs = 0
            for band_lc in source_lc.values():
                if len(band_lc) > 0:
                    times = [pt[0] for pt in band_lc]
                    baseline_days = max(baseline_days, max(times) - min(times))
                    total_epochs += len(band_lc)

            entries.append(
                CatalogEntry(
                    ra=ra_val,
                    dec=dec_val,
                    mag=mean_mag,
                    mag_band="r",
                    obj_type="unknown",
                    source="ztf",
                    source_id=f"ztf_{oid}",
                    properties={
                        "ztf_lightcurve": source_lc,
                        "ztf_n_epochs": total_epochs,
                        "ztf_baseline_days": baseline_days,
                        "ztf_n_obs": n_obs,
                    },
                )
            )

        logger.info(
            f"Got {len(entries)} ZTF sources, "
            f"{sum(1 for e in entries if e.properties.get('ztf_n_epochs', 0) > 0)} "
            f"with light curves"
        )
        catalog = StarCatalog(entries=entries, source="ztf")

        # Cache the catalog (including light curve data)
        self._cache.put_catalog(
            "ztf", region.ra, region.dec, region.radius,
            [e.to_dict() for e in entries],
        )

        return catalog

    def _fetch_bulk_lightcurves(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
    ) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
        """Fetch all light curves in a region with a single TAP query.

        Returns dict mapping oid -> {band: [(mjd, mag, magerr), ...]}.
        """
        try:
            from astroquery.ipac.irsa import Irsa
        except ImportError:
            return {}

        # filtercode mapping: 1=g, 2=r, 3=i
        filter_map = {"1": "g", "2": "r", "3": "i"}

        lc_query = (
            f"SELECT oid, filtercode, mjd, mag, magerr "
            f"FROM ztf_dr{ZTF_DR} "
            f"WHERE CONTAINS("
            f"POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})"
            f") = 1 "
            f"AND catflags = 0 "
            f"ORDER BY oid, filtercode, mjd"
        )

        try:
            lc_table = Irsa.query_tap(lc_query).to_table()
        except Exception as e:
            logger.warning(f"ZTF light curve query failed: {e}")
            return {}

        if lc_table is None or len(lc_table) == 0:
            return {}

        # Group by oid and filtercode
        result: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
        for row in lc_table:
            oid = str(row["oid"])
            fc = str(row["filtercode"])
            band = filter_map.get(fc, fc)
            mjd = float(row["mjd"])
            mag = float(row["mag"])
            magerr = float(row["magerr"]) if row["magerr"] is not None else 0.1

            if oid not in result:
                result[oid] = {}
            if band not in result[oid]:
                result[oid][band] = []
            result[oid][band].append((mjd, mag, magerr))

        logger.info(
            f"Fetched light curves for {len(result)} ZTF sources"
        )
        return result

    def fetch_lightcurves(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Fetch raw ZTF light curves for a position.

        Convenience method for targeted light curve retrieval.
        """
        radius_deg = radius_arcsec / 3600.0
        lc_data = self._fetch_bulk_lightcurves(ra, dec, radius_deg)

        results = []
        for oid, bands in lc_data.items():
            total_epochs = sum(len(pts) for pts in bands.values())
            results.append({
                "oid": oid,
                "bands": bands,
                "n_epochs": total_epochs,
            })
        return results

    def is_available(self) -> bool:
        """Check if IRSA TAP is importable."""
        try:
            from astroquery.ipac.irsa import Irsa  # noqa: F401

            return True
        except ImportError:
            return False
