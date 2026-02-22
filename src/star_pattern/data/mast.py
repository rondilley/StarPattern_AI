"""MAST data source for HST and JWST observations."""

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

logger = get_logger("data.mast")


class MASTDataSource(DataSource):
    """Data source for HST/JWST via MAST (Mikulski Archive)."""

    def __init__(
        self,
        cache: DataCache | None = None,
        missions: list[str] | None = None,
        max_observations: int = 10,
    ):
        self._cache = cache or DataCache()
        self.missions = missions or ["HST", "JWST"]
        self.max_observations = max_observations

    @property
    def name(self) -> str:
        return "mast"

    @property
    def available_bands(self) -> list[str]:
        return ["optical", "infrared"]

    @retry_with_backoff(max_retries=2, base_delay=5.0)
    def fetch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
    ) -> dict[str, FITSImage]:
        """Fetch HST/JWST images from MAST."""
        from astroquery.mast import Observations
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")
        images: dict[str, FITSImage] = {}

        logger.info(f"Querying MAST for ({region.ra:.3f}, {region.dec:.3f})")

        try:
            obs_table = Observations.query_region(
                coord,
                radius=region.radius * u.arcmin,
            )
        except Exception as e:
            logger.error(f"MAST query failed: {e}")
            return images

        if obs_table is None or len(obs_table) == 0:
            logger.warning("No MAST observations found")
            return images

        # Filter for desired missions and image products
        mission_mask = np.isin(obs_table["obs_collection"], self.missions)
        type_mask = obs_table["dataproduct_type"] == "image"
        filtered = obs_table[mission_mask & type_mask]

        if len(filtered) == 0:
            logger.warning("No image products found after filtering")
            return images

        # Take first few observations
        for row in filtered[:self.max_observations]:
            obs_id = str(row["obs_id"])
            mission = str(row["obs_collection"])

            # Check cache
            cached = self._cache.get_path("mast", region.ra, region.dec, region.radius, obs_id)
            if cached:
                images[f"{mission}_{obs_id}"] = FITSImage.from_file(cached)
                continue

            try:
                products = Observations.get_product_list(row)
                # Filter for science images
                sci = Observations.filter_products(
                    products,
                    productType="SCIENCE",
                    extension="fits",
                )

                if len(sci) == 0:
                    continue

                # Download first product
                download = Observations.download_products(
                    sci[:1], download_dir=str(self._cache.cache_dir)
                )

                if download and len(download) > 0:
                    dl_path = Path(str(download["Local Path"][0]))
                    if dl_path.exists():
                        img = FITSImage.from_file(dl_path)
                        images[f"{mission}_{obs_id}"] = img
                        self._cache.put(
                            "mast", region.ra, region.dec, region.radius, dl_path, obs_id
                        )
                        logger.info(f"  Got {mission} {obs_id}: {img.shape}")
            except Exception as e:
                logger.warning(f"  Failed to download {obs_id}: {e}")

        return images

    @retry_with_backoff(max_retries=2, base_delay=5.0)
    def fetch_catalog(
        self,
        region: SkyRegion,
        max_results: int = 10000,
    ) -> StarCatalog:
        """Fetch MAST catalog data."""
        # Check catalog cache
        try:
            cached_entries = self._cache.get_catalog("mast", region.ra, region.dec, region.radius)
            if cached_entries is not None:
                entries = [CatalogEntry.from_dict(d) for d in cached_entries]
                logger.info(f"Loaded {len(entries)} cached MAST catalog entries")
                return StarCatalog(entries=entries, source="mast")
        except Exception as e:
            logger.debug(f"MAST catalog cache check failed: {e}")

        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")

        logger.info(f"Fetching MAST catalog for ({region.ra:.3f}, {region.dec:.3f})")

        try:
            table = Catalogs.query_region(
                coord,
                radius=region.radius * u.arcmin,
                catalog="HSC",  # Hubble Source Catalog
            )
        except Exception as e:
            logger.warning(f"MAST catalog query failed: {e}")
            return StarCatalog(source="mast")

        if table is None or len(table) == 0:
            return StarCatalog(source="mast")

        entries = []
        for row in table[:max_results]:
            try:
                entries.append(
                    CatalogEntry(
                        ra=float(row["MatchRA"]),
                        dec=float(row["MatchDec"]),
                        mag=float(row.get("W3_F606W", np.nan)),
                        mag_band="F606W",
                        obj_type="unknown",
                        source="mast",
                        source_id=str(row.get("MatchID", "")),
                    )
                )
            except (KeyError, TypeError):
                continue

        logger.info(f"Got {len(entries)} MAST catalog entries")
        catalog = StarCatalog(entries=entries, source="mast")

        # Cache the catalog
        self._cache.put_catalog(
            "mast", region.ra, region.dec, region.radius,
            [e.to_dict() for e in entries],
        )

        return catalog

    def is_available(self) -> bool:
        try:
            from astroquery.mast import Observations

            return True
        except ImportError:
            return False
