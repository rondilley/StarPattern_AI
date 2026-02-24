"""Sky region definitions and data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from astropy.coordinates import SkyCoord, Galactic
    import astropy.units as u
except ImportError as e:
    raise ImportError("astropy is required") from e

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.catalog import StarCatalog


@dataclass
class EpochImage:
    """A single-epoch image with timestamp and filter metadata."""

    image: FITSImage
    mjd: float          # Modified Julian Date
    band: str           # Filter (g, r, i)
    source: str = ""    # ztf, mast
    metadata: dict = field(default_factory=dict)


@dataclass
class SkyRegion:
    """A circular region on the sky."""

    ra: float  # degrees
    dec: float  # degrees
    radius: float  # arcminutes

    @property
    def center(self) -> SkyCoord:
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame="icrs")

    @property
    def galactic_lat(self) -> float:
        """Galactic latitude in degrees."""
        coord = self.center.galactic
        return float(coord.b.deg)

    @property
    def galactic_lon(self) -> float:
        """Galactic longitude in degrees."""
        coord = self.center.galactic
        return float(coord.l.deg)

    def is_high_latitude(self, min_lat: float = 20.0) -> bool:
        """Check if region is above the galactic plane (less crowded)."""
        return abs(self.galactic_lat) >= min_lat

    def separation_to(self, other: SkyRegion) -> float:
        """Angular separation to another region in arcminutes."""
        sep = self.center.separation(other.center)
        return float(sep.arcmin)

    @classmethod
    def random(
        cls,
        min_galactic_lat: float = 20.0,
        radius: float = 3.0,
        rng: np.random.Generator | None = None,
    ) -> SkyRegion:
        """Generate a random sky region avoiding the galactic plane."""
        rng = rng or np.random.default_rng()
        while True:
            ra = rng.uniform(0, 360)
            # Uniform on sphere: dec = arcsin(uniform(-1,1))
            dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
            region = cls(ra=ra, dec=dec, radius=radius)
            if region.is_high_latitude(min_galactic_lat):
                return region

    def __repr__(self) -> str:
        return f"SkyRegion(ra={self.ra:.4f}, dec={self.dec:.4f}, r={self.radius}')"


@dataclass
class RegionData:
    """All data acquired for a sky region."""

    region: SkyRegion
    images: dict[str, FITSImage] = field(default_factory=dict)  # band -> image
    catalogs: dict[str, StarCatalog] = field(default_factory=dict)  # source -> catalog
    metadata: dict[str, Any] = field(default_factory=dict)
    temporal_images: dict[str, list[EpochImage]] = field(default_factory=dict)  # band -> epochs sorted by MJD

    @property
    def primary_image(self) -> FITSImage | None:
        """Return the first available image."""
        if not self.images:
            return None
        return next(iter(self.images.values()))

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_catalogs(self) -> bool:
        return len(self.catalogs) > 0

    def has_temporal_images(self) -> bool:
        """Check if any band has multi-epoch images."""
        return any(len(epochs) >= 2 for epochs in self.temporal_images.values())
