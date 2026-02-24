"""Zwicky Transient Facility light curve data and epoch images via IRSA."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import requests

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, EpochImage
from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.data.base import DataSource
from star_pattern.data.cache import DataCache
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("data.ztf")

# ZTF data release number for TAP queries
ZTF_DR = 24


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

        # Query the ZTF objects table for sources in the region,
        # including pre-computed variability statistics.
        # The objects table has one row per (oid, filtercode), so we
        # fetch all rows and group by oid below.
        radius_deg = region.radius / 60.0
        obj_query = (
            f"SELECT TOP {max_results} "
            f"oid, ra, dec, meanmag, nobs, filtercode, "
            f"magrms, medmagerr, medianabsdev, vonneumannratio, "
            f"stetsonj, stetsonk, skewness, smallkurtosis, "
            f"maxslope, maxmag, minmag, ngoodobs "
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

        # Group rows by oid (objects table has one row per oid+filtercode)
        oid_rows: dict[str, list] = {}
        for row in obj_table:
            oid = str(row["oid"])
            if oid not in oid_rows:
                oid_rows[oid] = []
            oid_rows[oid].append(row)

        # Fetch light curves for all sources in one bulk query
        lightcurves = self._fetch_bulk_lightcurves(
            region.ra, region.dec, radius_deg
        )

        entries = []
        n_with_stats = 0
        for oid, rows in oid_rows.items():
            # Use first row for positional/mag info
            first_row = rows[0]
            ra_val = float(first_row["ra"])
            dec_val = float(first_row["dec"])
            mean_mag = float(first_row["meanmag"]) if first_row["meanmag"] is not None else None
            n_obs = int(first_row["nobs"]) if first_row["nobs"] is not None else 0

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

            # Extract pre-computed variability stats from the filter
            # with the most good observations
            ztf_stats = self._extract_best_stats(rows)
            if ztf_stats:
                n_with_stats += 1

            properties: dict[str, Any] = {
                "ztf_lightcurve": source_lc,
                "ztf_n_epochs": total_epochs,
                "ztf_baseline_days": baseline_days,
                "ztf_n_obs": n_obs,
            }
            if ztf_stats:
                properties["ztf_stats"] = ztf_stats

            entries.append(
                CatalogEntry(
                    ra=ra_val,
                    dec=dec_val,
                    mag=mean_mag,
                    mag_band="r",
                    obj_type="unknown",
                    source="ztf",
                    source_id=f"ztf_{oid}",
                    properties=properties,
                )
            )

        logger.info(
            f"Got {len(entries)} ZTF sources, "
            f"{sum(1 for e in entries if e.properties.get('ztf_n_epochs', 0) > 0)} "
            f"with light curves, {n_with_stats} with stats"
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
            logger.debug(f"ZTF light curve query failed (expected for recent DRs): {e}")
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

    @staticmethod
    def _extract_best_stats(rows: list) -> dict[str, Any] | None:
        """Extract variability stats from the filter with most good observations.

        The objects table has one row per (oid, filtercode). We pick the row
        with the highest ngoodobs to get the most reliable statistics.
        """
        best_row = None
        best_ngoodobs = -1
        for row in rows:
            ngoodobs = int(row["ngoodobs"]) if row["ngoodobs"] is not None else 0
            if ngoodobs > best_ngoodobs:
                best_ngoodobs = ngoodobs
                best_row = row

        if best_row is None or best_ngoodobs <= 0:
            return None

        def _safe_float(val: Any) -> float | None:
            if val is None:
                return None
            try:
                v = float(val)
                return v if np.isfinite(v) else None
            except (ValueError, TypeError):
                return None

        # filtercode mapping: zg=g, zr=r, zi=i (DR objects table uses string codes)
        filter_map = {"zg": "g", "zr": "r", "zi": "i",
                      "1": "g", "2": "r", "3": "i"}
        raw_fc = str(best_row["filtercode"]).strip()
        filtercode = filter_map.get(raw_fc, raw_fc)

        return {
            "magrms": _safe_float(best_row["magrms"]) or 0.0,
            "medmagerr": _safe_float(best_row["medmagerr"]) or 0.0,
            "medianabsdev": _safe_float(best_row["medianabsdev"]) or 0.0,
            "vonneumannratio": _safe_float(best_row["vonneumannratio"]),
            "stetsonj": _safe_float(best_row["stetsonj"]),
            "stetsonk": _safe_float(best_row["stetsonk"]),
            "skewness": _safe_float(best_row["skewness"]) or 0.0,
            "maxslope": _safe_float(best_row["maxslope"]) or 0.0,
            "maxmag": _safe_float(best_row["maxmag"]) or 0.0,
            "minmag": _safe_float(best_row["minmag"]) or 0.0,
            "filtercode": filtercode,
            "ngoodobs": best_ngoodobs,
        }

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

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def fetch_epoch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
        max_epochs: int = 10,
        min_baseline_days: float = 1.0,
        max_baseline_days: float = 2000.0,
    ) -> dict[str, list[EpochImage]]:
        """Fetch ZTF science image cutouts at multiple epochs.

        Queries the IRSA Image Browser Engine (IBE) for ZTF science images
        covering the region, then downloads cutouts for each epoch.

        Args:
            region: Sky region to query.
            bands: Filters to fetch (default: ["r"]). ZTF codes: zg=g, zr=r, zi=i.
            max_epochs: Maximum number of epochs per band.
            min_baseline_days: Minimum time between first and last epoch.
            max_baseline_days: Maximum time between first and last epoch.

        Returns:
            Dict mapping band -> list of EpochImage sorted by MJD.
        """
        bands = bands or ["r"]
        ibe_filter_map = {"g": "zg", "r": "zr", "i": "zi"}
        result: dict[str, list[EpochImage]] = {}

        radius_deg = region.radius / 60.0
        cutout_arcsec = min(2 * region.radius * 60, 300)

        for band in bands:
            ztf_filter = ibe_filter_map.get(band, band)

            # Search IRSA IBE for science images
            search_url = (
                f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
                f"?POS={region.ra},{region.dec}&SIZE={radius_deg}"
                f"&ct=csv"
            )
            try:
                resp = requests.get(search_url, timeout=90)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"ZTF IBE search failed for band {band}: {e}")
                continue

            # Parse CSV response
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            if not rows:
                logger.debug(f"No ZTF science images found for band {band}")
                continue

            # Filter by band
            band_rows = [
                r for r in rows
                if r.get("filtercode", "").strip() == ztf_filter
            ]
            if not band_rows:
                logger.debug(f"No ZTF images in filter {ztf_filter}")
                continue

            # Parse observation dates and sort by time
            for r in band_rows:
                try:
                    r["_obsjd"] = float(r.get("obsjd", 0))
                except (ValueError, TypeError):
                    r["_obsjd"] = 0
            band_rows.sort(key=lambda r: r["_obsjd"])

            # Filter by baseline constraints
            if len(band_rows) >= 2:
                t_first = band_rows[0]["_obsjd"]
                t_last = band_rows[-1]["_obsjd"]
                baseline = t_last - t_first
                if baseline < min_baseline_days:
                    logger.debug(
                        f"ZTF baseline too short for band {band}: "
                        f"{baseline:.1f} < {min_baseline_days} days"
                    )
                    continue

                # Trim to max baseline
                if baseline > max_baseline_days:
                    cutoff = t_first + max_baseline_days
                    band_rows = [r for r in band_rows if r["_obsjd"] <= cutoff]

            # Select up to max_epochs, spread across the baseline
            if len(band_rows) > max_epochs:
                indices = np.linspace(0, len(band_rows) - 1, max_epochs, dtype=int)
                band_rows = [band_rows[i] for i in indices]

            # Download cutouts
            epoch_images: list[EpochImage] = []
            for row in band_rows:
                filefracday = row.get("filefracday", "").strip()
                field_id = row.get("field", "").strip()
                ccdid = row.get("ccdid", "").strip()
                qid = row.get("qid", "").strip()
                obsjd = row["_obsjd"]

                if not all([filefracday, field_id, ccdid, qid]):
                    continue

                # Check cache
                epoch_key = filefracday
                cached = self._cache.get_path(
                    "ztf_epoch", region.ra, region.dec, region.radius,
                    band=band, epoch=epoch_key,
                )
                if cached is not None:
                    try:
                        fits_img = FITSImage.from_file(str(cached))
                        epoch_images.append(EpochImage(
                            image=fits_img,
                            mjd=obsjd - 2400000.5,  # JD to MJD
                            band=band,
                            source="ztf",
                            metadata={"filefracday": filefracday},
                        ))
                        continue
                    except Exception:
                        pass

                # Build download path
                # ZTF IBE path: /{year}/{mmdd}/{fracday}/ztf_{filefracday}_{field}_{filtercode}_c{ccdid}_o_q{qid}_sciimg.fits
                # Directory uses the fractional day only (chars 8+), not the full filefracday
                year = filefracday[:4]
                mmdd = filefracday[4:8]
                fracday = filefracday[8:]
                sci_path = (
                    f"{year}/{mmdd}/{fracday}/"
                    f"ztf_{filefracday}_{int(field_id):06d}_{ztf_filter}"
                    f"_c{int(ccdid):02d}_o_q{int(qid)}_sciimg.fits"
                )
                cutout_url = (
                    f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{sci_path}"
                    f"?center={region.ra},{region.dec}"
                    f"&size={cutout_arcsec}arcsec&gzip=false"
                )

                try:
                    img_resp = requests.get(cutout_url, timeout=60)
                    img_resp.raise_for_status()
                except requests.RequestException as e:
                    logger.debug(f"ZTF cutout download failed: {e}")
                    continue

                # Save to cache
                cache_path = self._cache.cache_path_for(
                    "ztf_epoch", region.ra, region.dec, region.radius,
                    band=band, epoch=epoch_key,
                )
                try:
                    cache_path.write_bytes(img_resp.content)
                    self._cache.put(
                        "ztf_epoch", region.ra, region.dec, region.radius,
                        cache_path, band=band, epoch=epoch_key,
                        metadata={"obsjd": obsjd, "filefracday": filefracday},
                    )
                    fits_img = FITSImage.from_file(str(cache_path))
                    epoch_images.append(EpochImage(
                        image=fits_img,
                        mjd=obsjd - 2400000.5,
                        band=band,
                        source="ztf",
                        metadata={"filefracday": filefracday},
                    ))
                except Exception as e:
                    logger.debug(f"ZTF epoch image load failed: {e}")
                    continue

            if epoch_images:
                epoch_images.sort(key=lambda e: e.mjd)
                result[band] = epoch_images
                logger.info(
                    f"Fetched {len(epoch_images)} ZTF epoch images in band {band} "
                    f"(baseline: {epoch_images[-1].mjd - epoch_images[0].mjd:.1f} days)"
                )

        return result

    def is_available(self) -> bool:
        """Check if IRSA TAP is importable."""
        try:
            from astroquery.ipac.irsa import Irsa  # noqa: F401

            return True
        except ImportError:
            return False
