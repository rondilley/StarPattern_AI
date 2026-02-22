"""Star and galaxy catalog data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CatalogEntry:
    """A single catalog object (star or galaxy)."""

    ra: float  # degrees
    dec: float  # degrees
    mag: float | None = None  # magnitude
    mag_band: str | None = None
    obj_type: str = "unknown"  # star, galaxy, qso, unknown
    source: str = ""  # which survey
    source_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> tuple[float, float]:
        return (self.ra, self.dec)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "ra": self.ra,
            "dec": self.dec,
            "mag": self.mag,
            "mag_band": self.mag_band,
            "obj_type": self.obj_type,
            "source": self.source,
            "source_id": self.source_id,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CatalogEntry:
        """Deserialize from a dict."""
        return cls(
            ra=d["ra"],
            dec=d["dec"],
            mag=d.get("mag"),
            mag_band=d.get("mag_band"),
            obj_type=d.get("obj_type", "unknown"),
            source=d.get("source", ""),
            source_id=d.get("source_id", ""),
            properties=d.get("properties", {}),
        )


@dataclass
class StarCatalog:
    """A collection of catalog entries for a region."""

    entries: list[CatalogEntry] = field(default_factory=list)
    source: str = ""  # SDSS, Gaia, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> CatalogEntry:
        return self.entries[idx]

    def __iter__(self):
        return iter(self.entries)

    @property
    def positions(self) -> np.ndarray:
        """Return Nx2 array of (ra, dec) positions."""
        if not self.entries:
            return np.empty((0, 2))
        return np.array([[e.ra, e.dec] for e in self.entries])

    @property
    def magnitudes(self) -> np.ndarray:
        """Return array of magnitudes (NaN for missing)."""
        return np.array([e.mag if e.mag is not None else np.nan for e in self.entries])

    def filter_by_type(self, obj_type: str) -> StarCatalog:
        """Return a new catalog with only objects of a given type."""
        return StarCatalog(
            entries=[e for e in self.entries if e.obj_type == obj_type],
            source=self.source,
            metadata=self.metadata,
        )

    def merge(self, other: StarCatalog) -> StarCatalog:
        """Merge another catalog, deduplicating by source_id."""
        seen = {e.source_id for e in self.entries if e.source_id}
        merged = list(self.entries)
        for entry in other.entries:
            if entry.source_id and entry.source_id in seen:
                continue
            merged.append(entry)
            if entry.source_id:
                seen.add(entry.source_id)
        return StarCatalog(entries=merged, source=self.source)

    def filter_by_magnitude(self, min_mag: float = -30, max_mag: float = 30) -> StarCatalog:
        """Return entries within a magnitude range."""
        return StarCatalog(
            entries=[
                e
                for e in self.entries
                if e.mag is not None and min_mag <= e.mag <= max_mag
            ],
            source=self.source,
            metadata=self.metadata,
        )

    def to_table(self) -> Any:
        """Convert to an astropy Table."""
        from astropy.table import Table

        if not self.entries:
            return Table(names=["ra", "dec", "mag", "obj_type", "source_id"])
        rows = []
        for e in self.entries:
            rows.append(
                {
                    "ra": e.ra,
                    "dec": e.dec,
                    "mag": e.mag if e.mag is not None else np.nan,
                    "obj_type": e.obj_type,
                    "source_id": e.source_id,
                }
            )
        return Table(rows=rows)
