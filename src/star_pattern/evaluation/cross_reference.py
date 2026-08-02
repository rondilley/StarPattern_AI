"""Cross-reference detections with SIMBAD, NED, and TNS catalogs.

A catalog miss and a catalog outage are different facts and must not
produce the same answer. If SIMBAD is unreachable, the position is
UNVERIFIED, not novel. The previous version returned an empty list for
both cases, so a 30-second outage was enough to report a catalogued
galaxy as a new discovery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("evaluation.cross_reference")


class CatalogUnavailable(RuntimeError):
    """A catalog could not be queried, so its coverage is unknown."""


class CatalogCrossReferencer:
    """Cross-match detected patterns against known astronomical catalogs."""

    def __init__(self, search_radius_arcsec: float = 30.0):
        self.search_radius = search_radius_arcsec

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def query_simbad(self, ra: float, dec: float) -> list[dict[str, Any]]:
        """Query SIMBAD for known objects near a position.

        Raises:
            Exception: propagated after the retries are exhausted, so the
                caller can tell an outage from an empty field.
        """
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.simbad import Simbad

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        simbad = Simbad()
        simbad.add_votable_fields("otype", "V", "morphtype")

        # No try/except here. Catching the exception inside the function
        # means the decorator above never sees it, so no retry can fire.
        table = simbad.query_region(coord, radius=self.search_radius * u.arcsec)

        if table is None or len(table) == 0:
            return []

        matches = []
        for row in table:
            matches.append(
                {
                    "name": str(row["MAIN_ID"]),
                    "object_type": str(row.get("OTYPE", "unknown")),
                    "catalog": "SIMBAD",
                }
            )

        logger.info(f"SIMBAD: {len(matches)} matches near ({ra:.4f}, {dec:.4f})")
        return matches

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def query_ned(self, ra: float, dec: float) -> list[dict[str, Any]]:
        """Query NED for known objects near a position."""
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.ned import Ned

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        table = Ned.query_region(coord, radius=self.search_radius * u.arcsec)

        if table is None or len(table) == 0:
            return []

        matches = []
        for row in table:
            matches.append(
                {
                    "name": str(row["Object Name"]),
                    "object_type": str(row.get("Type", "unknown")),
                    "catalog": "NED",
                }
            )
        logger.info(f"NED: {len(matches)} matches near ({ra:.4f}, {dec:.4f})")
        return matches

    @staticmethod
    def _tns_api_key() -> str | None:
        """Read the TNS bot key from tns.key.txt or the environment."""
        env_key = os.environ.get("TNS_API_KEY", "").strip()
        if env_key:
            return env_key
        for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
            key_file = root / "tns.key.txt"
            try:
                if key_file.is_file():
                    key = key_file.read_text(encoding="utf-8").strip()
                    if key:
                        return key
            except OSError as exc:
                logger.debug("Cannot read %s: %r", key_file, exc)
        return None

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def query_tns(self, ra: float, dec: float) -> list[dict[str, Any]]:
        """Query the Transient Name Server for known transients.

        Raises:
            CatalogUnavailable: when no TNS bot key is configured. TNS
                rejects unauthenticated searches, so without a key the
                honest answer is "not checked", not "no transient here".
        """
        import requests

        api_key = self._tns_api_key()
        if api_key is None:
            raise CatalogUnavailable(
                "No TNS API key. Put a TNS bot key in tns.key.txt or set "
                "TNS_API_KEY to enable transient cross-matching."
            )

        tns_url = "https://www.wis-tns.org/api/get/search"
        bot_id = os.environ.get("TNS_BOT_ID", "0")
        bot_name = os.environ.get("TNS_BOT_NAME", "star_pattern_ai")

        headers = {
            "User-Agent": (
                f'tns_marker{{"tns_id": {bot_id}, "type": "bot", ' f'"name": "{bot_name}"}}'
            ),
        }

        search_data = {
            "ra": str(ra),
            "dec": str(dec),
            "radius": str(self.search_radius),
            "units": "arcsec",
        }

        # The TNS API expects JSON in the data field. str(dict) emits a
        # Python repr with single quotes, which TNS rejects.
        resp = requests.post(
            tns_url,
            headers=headers,
            data={"api_key": api_key, "data": json.dumps(search_data)},
            timeout=15,
        )
        if resp.status_code != 200:
            raise CatalogUnavailable(f"TNS query returned status {resp.status_code}")

        result = resp.json()
        reply = result.get("data", {}).get("reply", [])

        matches = []
        for item in reply:
            matches.append(
                {
                    "name": item.get("objname", ""),
                    "object_type": item.get("type_name", "unknown"),
                    "catalog": "TNS",
                    "discovery_date": item.get("discoverydate", ""),
                    "redshift": item.get("redshift"),
                }
            )

        if matches:
            logger.info(f"TNS: {len(matches)} transients near ({ra:.4f}, {dec:.4f})")
        return matches

    def cross_reference(self, ra: float, dec: float) -> dict[str, Any]:
        """Cross-reference a position against all available catalogs.

        Returns:
            Dict with matches from each catalog, an 'is_known' flag, and
            the coverage that flag rests on. When 'coverage_complete' is
            False at least one catalog could not be reached, so
            'is_known: False' means "not confirmed", not "confirmed new".
        """
        all_matches: list[dict[str, Any]] = []
        queried: list[str] = []
        failed: dict[str, str] = {}

        for catalog, query in (
            ("SIMBAD", self.query_simbad),
            ("NED", self.query_ned),
            ("TNS", self.query_tns),
        ):
            try:
                all_matches.extend(query(ra, dec))
                queried.append(catalog)
            except CatalogUnavailable as exc:
                logger.info("%s not queried: %s", catalog, exc)
                failed[catalog] = str(exc)
            except Exception as exc:  # noqa: BLE001 - vendor-specific types
                # Retries are already exhausted by the time this fires.
                logger.warning("%s query failed after retries: %r", catalog, exc)
                failed[catalog] = repr(exc)

        is_known = len(all_matches) > 0
        # sorted(), not list(set(...)): set iteration order varies between
        # processes under hash randomization, which made the same sky
        # position produce different report bytes across runs.
        known_types = sorted({m.get("object_type", "") for m in all_matches})

        lens_types = {"GrL", "LeG", "LensingEv", "GravLens"}
        is_known_lens = any(m.get("object_type", "") in lens_types for m in all_matches)

        transient_checked = "TNS" in queried
        is_known_transient = any(m.get("catalog") == "TNS" for m in all_matches)

        return {
            "matches": all_matches,
            "n_matches": len(all_matches),
            "is_known": is_known,
            "is_known_lens": is_known_lens,
            "is_known_transient": is_known_transient if transient_checked else None,
            "known_types": known_types,
            "catalogs_queried": queried,
            "catalogs_failed": failed,
            "coverage_complete": not failed,
        }
