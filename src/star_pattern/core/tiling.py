"""Sky tiling: decompose large fields into overlapping circular tiles."""

from __future__ import annotations

import numpy as np

from star_pattern.core.sky_region import SkyRegion
from star_pattern.utils.logging import get_logger

logger = get_logger("core.tiling")


class TileGrid:
    """Decompose a large sky area into overlapping circular tiles.

    Uses hex-packed tiling for efficient coverage. Each tile row is offset
    by half a tile width to minimize gaps.
    """

    def __init__(
        self,
        center_ra: float,
        center_dec: float,
        field_radius_arcmin: float,
        tile_radius_arcmin: float = 3.0,
        overlap_fraction: float = 0.2,
        max_tiles: int = 500,
    ):
        self.center_ra = center_ra
        self.center_dec = center_dec
        self.field_radius_arcmin = field_radius_arcmin
        self.tile_radius_arcmin = tile_radius_arcmin
        self.overlap_fraction = overlap_fraction
        self.max_tiles = max_tiles
        self._tiles: list[SkyRegion] = []
        self._generate_hex_tiles()

    def _generate_hex_tiles(self) -> None:
        """Generate hex-packed tile centers using tangent-plane projection."""
        # If field fits in a single tile, just return the center
        if self.field_radius_arcmin <= self.tile_radius_arcmin:
            self._tiles = [
                SkyRegion(
                    ra=self.center_ra,
                    dec=self.center_dec,
                    radius=self.tile_radius_arcmin,
                )
            ]
            return

        step = self.tile_radius_arcmin * 2 * (1 - self.overlap_fraction)
        step_deg = step / 60.0
        radius_deg = self.field_radius_arcmin / 60.0

        # cos(dec) correction for RA spacing
        cos_dec = np.cos(np.radians(self.center_dec))
        ra_step = step_deg / max(cos_dec, 0.1)

        n_ra = int(np.ceil(2 * radius_deg / ra_step)) + 1
        n_dec = int(np.ceil(2 * radius_deg / step_deg)) + 1

        tiles: list[SkyRegion] = []
        for j in range(n_dec):
            dec = self.center_dec - radius_deg + j * step_deg
            if dec < -90 or dec > 90:
                continue
            # Hex offset: odd rows shifted by half step
            ra_offset = (ra_step / 2) if j % 2 else 0
            for i in range(n_ra):
                ra = (
                    self.center_ra
                    - radius_deg / max(cos_dec, 0.1)
                    + i * ra_step
                    + ra_offset
                )
                ra = ra % 360  # Wrap RA

                # Check if tile center is within field radius
                sep = self._angular_sep(
                    ra, dec, self.center_ra, self.center_dec
                )
                if sep <= self.field_radius_arcmin:
                    tiles.append(
                        SkyRegion(
                            ra=ra, dec=dec, radius=self.tile_radius_arcmin
                        )
                    )

                if len(tiles) >= self.max_tiles:
                    break
            if len(tiles) >= self.max_tiles:
                break

        self._tiles = tiles
        logger.info(
            f"TileGrid: {len(tiles)} tiles for "
            f"{self.field_radius_arcmin}' field "
            f"(tile_r={self.tile_radius_arcmin}', "
            f"overlap={self.overlap_fraction})"
        )

    @staticmethod
    def _angular_sep(
        ra1: float, dec1: float, ra2: float, dec2: float
    ) -> float:
        """Angular separation in arcminutes using Vincenty formula."""
        ra1_r = np.radians(ra1)
        dec1_r = np.radians(dec1)
        ra2_r = np.radians(ra2)
        dec2_r = np.radians(dec2)
        dra = ra2_r - ra1_r

        num = np.sqrt(
            (np.cos(dec2_r) * np.sin(dra)) ** 2
            + (
                np.cos(dec1_r) * np.sin(dec2_r)
                - np.sin(dec1_r) * np.cos(dec2_r) * np.cos(dra)
            )
            ** 2
        )
        den = np.sin(dec1_r) * np.sin(dec2_r) + np.cos(dec1_r) * np.cos(
            dec2_r
        ) * np.cos(dra)
        sep_rad = np.arctan2(num, den)
        return float(np.degrees(sep_rad) * 60.0)  # arcminutes

    @property
    def tiles(self) -> list[SkyRegion]:
        """Return the list of tile regions."""
        return self._tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def __repr__(self) -> str:
        return (
            f"TileGrid(center=({self.center_ra:.3f}, {self.center_dec:.3f}), "
            f"field_r={self.field_radius_arcmin}', "
            f"n_tiles={len(self._tiles)})"
        )
