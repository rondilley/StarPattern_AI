"""SDSS data source using astroquery."""

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

    @staticmethod
    def _in_stripe82(ra: float, dec: float) -> bool:
        """Check if coordinates fall within the SDSS Stripe 82 footprint.

        Stripe 82 covers Dec in [-1.5, +1.5] and RA >= 310 or RA <= 60.
        Slightly wider than the true footprint to account for search radius.
        """
        if not (-1.5 <= dec <= 1.5):
            return False
        return ra >= 310.0 or ra <= 60.0

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def fetch_epoch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
        max_epochs: int = 10,
        min_baseline_days: float = 1.0,
        max_baseline_days: float = 2000.0,
    ) -> dict[str, list[EpochImage]]:
        """Fetch multi-epoch SDSS images from Stripe 82.

        Stripe 82 was observed repeatedly (~36+ runs), providing a temporal
        baseline for transient and variability detection in the equatorial
        stripe (Dec ~ 0, RA 310-60 deg).

        Args:
            region: Sky region to query.
            bands: SDSS bands to fetch (default: ["r"]).
            max_epochs: Maximum number of epochs per band.
            min_baseline_days: Minimum time span between first and last epoch.
            max_baseline_days: Maximum time span between first and last epoch.

        Returns:
            Dict mapping band -> list of EpochImage sorted by MJD.
        """
        # Gate: only Stripe 82 has multi-epoch coverage
        if not self._in_stripe82(region.ra, region.dec):
            return {}

        try:
            from astroquery.sdss import SDSS
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except ImportError:
            logger.warning("astroquery.sdss not available for Stripe 82 queries")
            return {}

        bands = bands or ["r"]
        coord = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg, frame="icrs")

        logger.info(
            f"Querying SDSS Stripe 82 epoch images for "
            f"({region.ra:.3f}, {region.dec:.3f})"
        )

        # Query for all photoobj entries to get unique (run, rerun, camcol, field)
        try:
            table = SDSS.query_region(
                coord,
                radius=region.radius * u.arcmin,
                photoobj_fields=["run", "rerun", "camcol", "field"],
            )
        except Exception as e:
            logger.warning(f"SDSS Stripe 82 query failed: {e}")
            return {}

        if table is None or len(table) == 0:
            logger.debug("No SDSS sources found in Stripe 82 region")
            return {}

        # Extract unique (run, rerun, camcol, field) tuples
        seen: set[tuple[int, int, int, int]] = set()
        run_tuples: list[tuple[int, int, int, int]] = []
        for row in table:
            try:
                key = (
                    int(row["run"]),
                    int(row["rerun"]),
                    int(row["camcol"]),
                    int(row["field"]),
                )
            except (ValueError, TypeError):
                continue
            if key not in seen:
                seen.add(key)
                run_tuples.append(key)

        if not run_tuples:
            logger.debug("No valid SDSS runs found")
            return {}

        logger.info(f"Found {len(run_tuples)} unique SDSS runs in Stripe 82")

        result: dict[str, list[EpochImage]] = {}
        # Cutout size: 2x search radius, capped at 5 arcmin
        cutout_arcmin = min(2 * region.radius, 5.0)

        for band in bands:
            epoch_images: list[EpochImage] = []

            for run, rerun, camcol, field_id in run_tuples:
                epoch_key = str(run)

                # Check cache
                cached = self._cache.get_path(
                    "sdss_epoch", region.ra, region.dec, region.radius,
                    band=band, epoch=epoch_key,
                )
                if cached is not None:
                    try:
                        fits_img = FITSImage.from_file(str(cached))
                        fits_img = self._extract_cutout(
                            fits_img, region.ra, region.dec,
                            cutout_arcmin,
                        )
                        # Read MJD from cached metadata or FITS header
                        mjd = self._extract_mjd_from_fits(fits_img)
                        if mjd is not None:
                            epoch_images.append(EpochImage(
                                image=fits_img,
                                mjd=mjd,
                                band=band,
                                source="sdss",
                                metadata={"run": run},
                            ))
                        continue
                    except Exception:
                        pass

                try:
                    hdu_list = SDSS.get_images(
                        run=run, rerun=rerun, camcol=camcol,
                        field=field_id, band=band, timeout=120,
                    )
                except Exception as e:
                    logger.debug(f"SDSS epoch image download failed for run {run}: {e}")
                    continue

                if not hdu_list or len(hdu_list) == 0:
                    continue

                # Save to cache
                cache_path = self._cache.cache_path_for(
                    "sdss_epoch", region.ra, region.dec, region.radius,
                    band=band, epoch=epoch_key,
                )
                try:
                    hdu_list[0].writeto(str(cache_path), overwrite=True)
                    self._cache.put(
                        "sdss_epoch", region.ra, region.dec, region.radius,
                        cache_path, band=band, epoch=epoch_key,
                        metadata={"run": run},
                    )

                    fits_img = FITSImage.from_file(str(cache_path))
                    fits_img = self._extract_cutout(
                        fits_img, region.ra, region.dec,
                        cutout_arcmin,
                    )
                    mjd = self._extract_mjd_from_fits(fits_img)
                    if mjd is not None:
                        epoch_images.append(EpochImage(
                            image=fits_img,
                            mjd=mjd,
                            band=band,
                            source="sdss",
                            metadata={"run": run},
                        ))
                except Exception as e:
                    logger.debug(f"SDSS epoch image save/load failed: {e}")
                    continue

            if not epoch_images:
                continue

            # Sort by MJD
            epoch_images.sort(key=lambda e: e.mjd)

            # Apply baseline constraints
            if len(epoch_images) >= 2:
                baseline = epoch_images[-1].mjd - epoch_images[0].mjd
                if baseline < min_baseline_days:
                    logger.debug(
                        f"SDSS baseline too short for {band}: "
                        f"{baseline:.1f} < {min_baseline_days} days"
                    )
                    continue

                # Trim to max baseline
                if baseline > max_baseline_days:
                    cutoff = epoch_images[0].mjd + max_baseline_days
                    epoch_images = [e for e in epoch_images if e.mjd <= cutoff]

            # Subsample to max_epochs
            if len(epoch_images) > max_epochs:
                indices = np.linspace(
                    0, len(epoch_images) - 1, max_epochs, dtype=int
                )
                epoch_images = [epoch_images[i] for i in indices]

            if epoch_images:
                result[band] = epoch_images
                logger.info(
                    f"Fetched {len(epoch_images)} SDSS Stripe 82 epoch images "
                    f"in band {band} "
                    f"(baseline: {epoch_images[-1].mjd - epoch_images[0].mjd:.1f} days)"
                )

        return result

    @staticmethod
    def _extract_mjd_from_fits(fits_img: FITSImage) -> float | None:
        """Extract MJD from an SDSS FITS header.

        Tries TAI (seconds since MJD epoch), then MJD, then DATE-OBS headers.
        """
        header = getattr(fits_img, 'header', None)
        if header is None:
            return None

        # TAI is seconds since MJD 0 -- convert to days
        tai = header.get("TAI")
        if tai is not None:
            try:
                mjd = float(tai) / 86400.0
                if mjd > 40000:  # Sanity check: must be a valid MJD
                    return mjd
            except (ValueError, TypeError):
                pass

        # Direct MJD header
        mjd_val = header.get("MJD")
        if mjd_val is not None:
            try:
                mjd = float(mjd_val)
                if mjd > 40000:
                    return mjd
            except (ValueError, TypeError):
                pass

        # DATE-OBS fallback (ISO format)
        date_obs = header.get("DATE-OBS")
        if date_obs is not None:
            try:
                from astropy.time import Time
                t = Time(date_obs, format="isot", scale="utc")
                return float(t.mjd)
            except Exception:
                pass

        return None

    @staticmethod
    def _extract_cutout(
        fits_img: FITSImage, ra: float, dec: float, size_arcmin: float,
    ) -> FITSImage:
        """Extract a WCS-aware cutout from a full SDSS frame.

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
            logger.debug(f"Cutout extraction failed: {e}")
            return fits_img

    def is_available(self) -> bool:
        try:
            from astroquery.sdss import SDSS

            return True
        except ImportError:
            return False
