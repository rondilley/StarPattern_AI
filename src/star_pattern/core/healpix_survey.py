"""HEALPix grid survey for systematic sky coverage.

Uses HEALPix equal-area pixelization to divide the sky into tiles,
providing resumable whole-sky survey coverage with galactic plane
filtering and configurable visit ordering.

Requires astropy-healpix (optional dependency): pip install star-pattern-ai[survey]
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from star_pattern.core.config import SurveyConfig
from star_pattern.core.sky_region import SkyRegion
from star_pattern.utils.logging import get_logger

logger = get_logger("core.healpix_survey")


def _nside2npix(nside: int) -> int:
    """Number of pixels for a given NSIDE."""
    return 12 * nside * nside


class HEALPixSurvey:
    """Systematic sky survey using HEALPix equal-area pixelization.

    Uses astropy_healpix (optional dependency) for pixel operations.
    Persists visited/pending state to JSON for cross-session resume.
    """

    def __init__(
        self,
        config: SurveyConfig,
        state_dir: Path | None = None,
    ):
        self.config = config
        self.state_dir = Path(state_dir) if state_dir else Path(".")

        # Lazy import astropy_healpix
        try:
            from astropy_healpix import HEALPix as AstropyHEALPix
            from astropy.coordinates import ICRS
            self._healpix = AstropyHEALPix(
                nside=config.nside, order="ring", frame=ICRS(),
            )
        except ImportError:
            raise ImportError(
                "astropy-healpix is required for HEALPix survey mode. "
                "Install it with: pip install star-pattern-ai[survey] "
                "or: pip install astropy-healpix"
            )

        self._visited: set[int] = set()
        self._findings_per_pixel: dict[int, int] = {}
        self._pending: list[int] = []

        # Build pixel list and try loading existing state
        all_pixels = self._build_pixel_list()
        state_path = self.state_dir / self.config.state_file
        if state_path.exists():
            self.load_state()
        else:
            self._pending = list(all_pixels)

        self._all_pixels_set = set(all_pixels)

    def _build_pixel_list(self) -> list[int]:
        """All HEALPix pixels passing galactic latitude filter, ordered by config.order."""
        from astropy.coordinates import SkyCoord, Galactic
        import astropy.units as u

        npix = _nside2npix(self.config.nside)
        all_indices = np.arange(npix)

        # Get coordinates for all pixels at once (vectorized)
        coords = self._healpix.healpix_to_skycoord(all_indices)
        galactic = coords.galactic
        gal_lats = galactic.b.deg

        # Filter by galactic latitude
        mask = np.abs(gal_lats) >= self.config.min_galactic_lat
        filtered = all_indices[mask].tolist()

        # Apply ordering
        filtered = self._apply_ordering(filtered)

        return filtered

    def _apply_ordering(self, pixels: list[int]) -> list[int]:
        """Sort pixel list according to config.order strategy."""
        from astropy.coordinates import Galactic
        import astropy.units as u

        if self.config.order == "galactic_latitude":
            # High |b| first (cleaner fields, less extinction)
            pixel_arr = np.array(pixels)
            coords = self._healpix.healpix_to_skycoord(pixel_arr)
            gal_lats = np.abs(coords.galactic.b.deg)
            # Sort descending by |galactic latitude|
            order = np.argsort(-gal_lats)
            pixels = pixel_arr[order].tolist()

        elif self.config.order == "dec_sweep":
            # Declination bands from equator outward
            pixel_arr = np.array(pixels)
            coords = self._healpix.healpix_to_skycoord(pixel_arr)
            decs = np.abs(coords.dec.deg)
            # Sort ascending by |dec|
            order = np.argsort(decs)
            pixels = pixel_arr[order].tolist()

        elif self.config.order == "random_shuffle":
            random.shuffle(pixels)

        return pixels

    def next_region(self) -> SkyRegion | None:
        """Return next unvisited region as SkyRegion, or None if survey complete."""
        while self._pending:
            pixel_idx = self._pending[0]
            if pixel_idx not in self._visited:
                return self.pixel_to_region(pixel_idx)
            # Already visited (e.g. from loaded state), skip
            self._pending.pop(0)

        return None

    def mark_visited(self, pixel_idx: int, findings_count: int = 0) -> None:
        """Record a pixel as visited with optional finding count."""
        self._visited.add(pixel_idx)
        if findings_count > 0:
            self._findings_per_pixel[pixel_idx] = findings_count

        # Remove from pending
        if self._pending and self._pending[0] == pixel_idx:
            self._pending.pop(0)
        elif pixel_idx in self._pending:
            self._pending.remove(pixel_idx)

    def save_state(self) -> None:
        """Persist visited set + pending queue to JSON."""
        state = {
            "config": asdict(self.config),
            "visited": sorted(self._visited),
            "pending": self._pending,
            "findings_per_pixel": {
                str(k): v for k, v in self._findings_per_pixel.items()
            },
        }
        state_path = self.state_dir / self.config.state_file
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(
            f"Survey state saved: {len(self._visited)} visited, "
            f"{len(self._pending)} pending"
        )

    def load_state(self) -> None:
        """Resume from saved state file."""
        state_path = self.state_dir / self.config.state_file
        with open(state_path) as f:
            state = json.load(f)

        self._visited = set(state.get("visited", []))
        self._pending = state.get("pending", [])
        self._findings_per_pixel = {
            int(k): v for k, v in state.get("findings_per_pixel", {}).items()
        }

        logger.info(
            f"Survey state loaded: {len(self._visited)} visited, "
            f"{len(self._pending)} pending"
        )

    def coverage_stats(self) -> dict:
        """Return n_total, n_visited, n_remaining, percent_complete, n_with_findings."""
        n_total = len(self._all_pixels_set)
        n_visited = len(self._visited & self._all_pixels_set)
        n_remaining = n_total - n_visited
        percent_complete = (n_visited / n_total * 100.0) if n_total > 0 else 0.0
        n_with_findings = sum(
            1 for pix in self._findings_per_pixel
            if pix in self._all_pixels_set and self._findings_per_pixel[pix] > 0
        )

        return {
            "n_total": n_total,
            "n_visited": n_visited,
            "n_remaining": n_remaining,
            "percent_complete": round(percent_complete, 2),
            "n_with_findings": n_with_findings,
        }

    def pixel_to_region(self, pixel_idx: int) -> SkyRegion:
        """Convert HEALPix pixel index to SkyRegion(ra, dec, radius)."""
        coord = self._healpix.healpix_to_skycoord(pixel_idx)
        ra_deg = float(coord.ra.deg)
        dec_deg = float(coord.dec.deg)

        region = SkyRegion(
            ra=ra_deg,
            dec=dec_deg,
            radius=self.config.radius_arcmin,
        )
        # Attach pixel index as metadata for mark_visited
        region._healpix_pixel = pixel_idx  # type: ignore[attr-defined]
        return region

    @property
    def current_pixel(self) -> int | None:
        """Return the current (next unvisited) pixel index, or None."""
        for pixel_idx in self._pending:
            if pixel_idx not in self._visited:
                return pixel_idx
        return None
