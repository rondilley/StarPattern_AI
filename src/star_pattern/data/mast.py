"""MAST data source for HST and JWST observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, EpochImage
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

    @retry_with_backoff(max_retries=2, base_delay=5.0)
    def fetch_epoch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
        max_epochs: int = 10,
        min_baseline_days: float = 1.0,
        max_baseline_days: float = 2000.0,
    ) -> dict[str, list[EpochImage]]:
        """Fetch multi-epoch HST/JWST images from MAST.

        Queries MAST for calibrated image observations across multiple epochs,
        groups by filter, and downloads FITS cutouts for temporal analysis.

        Args:
            region: Sky region to query.
            bands: Filter names to include (e.g. ["F606W", "F814W"]).
                None = all available filters.
            max_epochs: Maximum number of epochs per filter.
            min_baseline_days: Minimum time span between first and last epoch.
            max_baseline_days: Maximum time span between first and last epoch.

        Returns:
            Dict mapping filter name -> list of EpochImage sorted by MJD.
        """
        try:
            from astroquery.mast import Observations
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except ImportError:
            logger.warning("astroquery.mast not available for MAST epoch queries")
            return {}

        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")

        logger.info(
            f"Querying MAST epoch images for ({region.ra:.3f}, {region.dec:.3f})"
        )

        try:
            obs_table = Observations.query_region(
                coord,
                radius=region.radius * u.arcmin,
            )
        except Exception as e:
            logger.warning(f"MAST epoch query failed: {e}")
            return {}

        if obs_table is None or len(obs_table) == 0:
            logger.debug("No MAST observations found for epoch images")
            return {}

        # Filter: target missions, image products, calibrated (level >= 2)
        mission_mask = np.isin(obs_table["obs_collection"], self.missions)
        type_mask = obs_table["dataproduct_type"] == "image"
        calib_mask = obs_table["calib_level"] >= 2
        filtered = obs_table[mission_mask & type_mask & calib_mask]

        if len(filtered) == 0:
            logger.debug("No calibrated image products found for epoch images")
            return {}

        # Group by filter
        filter_groups: dict[str, list] = {}
        for row in filtered:
            filter_name = str(row["filters"]).strip()
            if not filter_name or filter_name == "--":
                continue
            # Apply band filter if specified
            if bands is not None and filter_name not in bands:
                continue
            try:
                t_min = float(row["t_min"])
                if not np.isfinite(t_min) or t_min <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            filter_groups.setdefault(filter_name, []).append(row)

        if not filter_groups:
            logger.debug("No valid filter groups found in MAST observations")
            return {}

        result: dict[str, list[EpochImage]] = {}
        cutout_arcmin = min(2 * region.radius, 5.0)

        for filter_name, rows in filter_groups.items():
            # Sort by observation time (t_min is MJD)
            rows.sort(key=lambda r: float(r["t_min"]))

            # Apply baseline constraints
            if len(rows) < 2:
                continue
            t_first = float(rows[0]["t_min"])
            t_last = float(rows[-1]["t_min"])
            baseline = t_last - t_first
            if baseline < min_baseline_days:
                logger.debug(
                    f"MAST baseline too short for {filter_name}: "
                    f"{baseline:.1f} < {min_baseline_days} days"
                )
                continue

            # Trim to max baseline
            if baseline > max_baseline_days:
                cutoff = t_first + max_baseline_days
                rows = [r for r in rows if float(r["t_min"]) <= cutoff]

            # Subsample to max_epochs
            if len(rows) > max_epochs:
                indices = np.linspace(0, len(rows) - 1, max_epochs, dtype=int)
                rows = [rows[i] for i in indices]

            # Download each observation
            epoch_images: list[EpochImage] = []
            for row in rows:
                obs_id = str(row["obs_id"])
                t_min = float(row["t_min"])

                # Check cache
                cached = self._cache.get_path(
                    "mast_epoch", region.ra, region.dec, region.radius,
                    band=filter_name, epoch=obs_id,
                )
                if cached is not None:
                    try:
                        fits_img = FITSImage.from_file(str(cached))
                        fits_img = self._extract_cutout(
                            fits_img, region.ra, region.dec,
                            cutout_arcmin,
                        )
                        epoch_images.append(EpochImage(
                            image=fits_img,
                            mjd=t_min,
                            band=filter_name,
                            source="mast",
                            metadata={"obs_id": obs_id},
                        ))
                        continue
                    except Exception:
                        pass

                try:
                    products = Observations.get_product_list(row)
                    sci = Observations.filter_products(
                        products,
                        productType="SCIENCE",
                        extension="fits",
                    )

                    if len(sci) == 0:
                        continue

                    # Sort by file size (ascending) and take smallest to
                    # avoid downloading large raw frames
                    if "size" in sci.colnames:
                        sci.sort("size")

                    download = Observations.download_products(
                        sci[:1], download_dir=str(self._cache.cache_dir)
                    )

                    if download is None or len(download) == 0:
                        continue

                    dl_path = Path(str(download["Local Path"][0]))
                    if not dl_path.exists():
                        continue

                    fits_img = FITSImage.from_file(dl_path)

                    # Cache the downloaded file
                    self._cache.put(
                        "mast_epoch", region.ra, region.dec, region.radius,
                        dl_path, band=filter_name, epoch=obs_id,
                        metadata={"t_min": t_min, "obs_id": obs_id},
                    )

                    fits_img = self._extract_cutout(
                        fits_img, region.ra, region.dec,
                        cutout_arcmin,
                    )
                    epoch_images.append(EpochImage(
                        image=fits_img,
                        mjd=t_min,
                        band=filter_name,
                        source="mast",
                        metadata={"obs_id": obs_id},
                    ))
                except Exception as e:
                    logger.debug(f"MAST epoch download failed for {obs_id}: {e}")
                    continue

            if epoch_images:
                epoch_images.sort(key=lambda e: e.mjd)
                result[filter_name] = epoch_images
                logger.info(
                    f"Fetched {len(epoch_images)} MAST epoch images in {filter_name} "
                    f"(baseline: {epoch_images[-1].mjd - epoch_images[0].mjd:.1f} days)"
                )

        return result

    @staticmethod
    def _extract_cutout(
        fits_img: FITSImage, ra: float, dec: float, size_arcmin: float,
    ) -> FITSImage:
        """Extract a WCS-aware cutout from a full MAST frame.

        Returns the original image unchanged if WCS is missing or cutout fails.
        """
        if fits_img.wcs is None:
            return fits_img

        try:
            from astropy.coordinates import SkyCoord
            from astropy.nddata import Cutout2D
            import astropy.units as u

            center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            size = size_arcmin * u.arcmin
            cutout = Cutout2D(
                fits_img.data, center, size, wcs=fits_img.wcs, mode="partial",
            )
            result = FITSImage.__new__(FITSImage)
            result.data = cutout.data.astype(np.float64)
            result.header = fits_img.header.copy()
            result.wcs = cutout.wcs
            result._file_path = None
            return result
        except Exception as e:
            logger.debug(f"MAST cutout extraction failed: {e}")
            return fits_img

    def is_available(self) -> bool:
        try:
            from astroquery.mast import Observations

            return True
        except ImportError:
            return False
