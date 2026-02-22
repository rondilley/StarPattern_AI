"""Cross-reference detections with SIMBAD, NED, and VizieR catalogs."""

from __future__ import annotations

from typing import Any

from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("evaluation.cross_reference")


class CatalogCrossReferencer:
    """Cross-match detected patterns against known astronomical catalogs."""

    def __init__(self, search_radius_arcsec: float = 30.0):
        self.search_radius = search_radius_arcsec

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def query_simbad(self, ra: float, dec: float) -> list[dict[str, Any]]:
        """Query SIMBAD for known objects near a position."""
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        simbad = Simbad()
        simbad.add_votable_fields("otype", "flux(V)", "morphtype")

        try:
            table = simbad.query_region(coord, radius=self.search_radius * u.arcsec)
        except Exception as e:
            logger.warning(f"SIMBAD query failed: {e}")
            return []

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
        try:
            from astroquery.ned import Ned
            from astropy.coordinates import SkyCoord
            import astropy.units as u

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
        except Exception as e:
            logger.warning(f"NED query failed: {e}")
            return []

    @retry_with_backoff(max_retries=2, base_delay=3.0)
    def query_tns(self, ra: float, dec: float) -> list[dict[str, Any]]:
        """Query the Transient Name Server for known transients.

        Uses the TNS public API cone search endpoint.
        Returns matches with: name, type, discovery_date, redshift.
        """
        import requests

        tns_url = "https://www.wis-tns.org/api/get/search"
        radius_arcsec = self.search_radius

        headers = {
            "User-Agent": 'tns_marker{"tns_id": 0, "type": "bot", "name": "star_pattern_ai"}',
        }

        search_data = {
            "ra": str(ra),
            "dec": str(dec),
            "radius": str(radius_arcsec),
            "units": "arcsec",
            "format": "json",
        }

        try:
            resp = requests.post(
                tns_url,
                headers=headers,
                data={"api_key": "", "data": str(search_data)},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.debug(f"TNS query returned status {resp.status_code}")
                return []

            result = resp.json()
            reply = result.get("data", {}).get("reply", [])

            matches = []
            for item in reply:
                matches.append({
                    "name": item.get("objname", ""),
                    "object_type": item.get("type_name", "unknown"),
                    "catalog": "TNS",
                    "discovery_date": item.get("discoverydate", ""),
                    "redshift": item.get("redshift"),
                })

            if matches:
                logger.info(f"TNS: {len(matches)} transients near ({ra:.4f}, {dec:.4f})")
            return matches

        except Exception as e:
            logger.debug(f"TNS query failed: {e}")
            return []

    def cross_reference(self, ra: float, dec: float) -> dict[str, Any]:
        """Cross-reference a position against all available catalogs.

        Returns:
            Dict with matches from each catalog and a 'known' flag.
        """
        all_matches = []

        simbad = self.query_simbad(ra, dec)
        all_matches.extend(simbad)

        ned = self.query_ned(ra, dec)
        all_matches.extend(ned)

        tns = self.query_tns(ra, dec)
        all_matches.extend(tns)

        # Determine if this is a known object
        is_known = len(all_matches) > 0
        known_types = list(set(m.get("object_type", "") for m in all_matches))

        # Check if it's a known lens
        lens_types = {"GrL", "LeG", "LensingEv", "GravLens"}
        is_known_lens = any(
            m.get("object_type", "") in lens_types for m in all_matches
        )

        # Check if it's a known transient
        is_known_transient = any(
            m.get("catalog") == "TNS" for m in all_matches
        )

        return {
            "matches": all_matches,
            "n_matches": len(all_matches),
            "is_known": is_known,
            "is_known_lens": is_known_lens,
            "is_known_transient": is_known_transient,
            "known_types": known_types,
        }
