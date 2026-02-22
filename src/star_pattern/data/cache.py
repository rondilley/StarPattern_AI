"""Download caching and deduplication.

Caches both FITS image files and catalog data (serialized as JSON) so that
repeated runs against the same sky regions avoid redundant network calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("cache")


class DataCache:
    """Cache manager for downloaded data files and catalog queries.

    Caches:
    - FITS images (via get_path / put)
    - StarCatalog data (via get_catalog / put_catalog)

    Both use the same SHA256 key scheme based on (source, ra, dec, radius, band).
    Catalog entries use band="__catalog__" to distinguish from image entries.
    """

    def __init__(self, cache_dir: str | Path = "output/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_file.exists():
            self._index = json.loads(self._index_file.read_text())

    def _save_index(self) -> None:
        self._index_file.write_text(json.dumps(self._index, indent=2))

    @staticmethod
    def _make_key(source: str, ra: float, dec: float, radius: float, band: str = "") -> str:
        """Create a unique cache key."""
        raw = f"{source}:{ra:.6f}:{dec:.6f}:{radius:.4f}:{band}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_path(self, source: str, ra: float, dec: float, radius: float, band: str = "") -> Path | None:
        """Get cached file path if it exists."""
        key = self._make_key(source, ra, dec, radius, band)
        if key in self._index:
            path = Path(self._index[key]["path"])
            if path.exists():
                logger.debug(f"Cache hit: {key}")
                return path
            else:
                del self._index[key]
                self._save_index()
        return None

    def put(
        self,
        source: str,
        ra: float,
        dec: float,
        radius: float,
        path: Path,
        band: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a file in the cache."""
        key = self._make_key(source, ra, dec, radius, band)
        self._index[key] = {
            "path": str(path),
            "source": source,
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "band": band,
            **(metadata or {}),
        }
        self._save_index()
        logger.debug(f"Cached: {key} -> {path}")

    def cache_path_for(
        self, source: str, ra: float, dec: float, radius: float, band: str = "", ext: str = ".fits"
    ) -> Path:
        """Generate a cache file path (does not create the file)."""
        key = self._make_key(source, ra, dec, radius, band)
        return self.cache_dir / f"{source}_{key}{ext}"

    # --- Catalog caching ---

    def get_catalog(
        self, source: str, ra: float, dec: float, radius: float
    ) -> list[dict[str, Any]] | None:
        """Load a cached catalog for a region.

        Returns:
            List of serialized CatalogEntry dicts, or None on cache miss.
        """
        key = self._make_key(source, ra, dec, radius, band="__catalog__")
        if key not in self._index:
            return None

        path = Path(self._index[key]["path"])
        if not path.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            data = json.loads(path.read_text())
            n = len(data.get("entries", []))
            logger.info(f"Catalog cache hit: {source} at ({ra:.3f}, {dec:.3f}), {n} entries")
            return data["entries"]
        except Exception as e:
            logger.debug(f"Catalog cache read failed: {e}")
            return None

    def put_catalog(
        self,
        source: str,
        ra: float,
        dec: float,
        radius: float,
        entries: list[dict[str, Any]],
    ) -> None:
        """Cache a catalog for a region.

        Args:
            source: Data source name (sdss, gaia, mast, ztf).
            ra: Center RA in degrees.
            dec: Center Dec in degrees.
            radius: Radius in arcmin.
            entries: List of serialized CatalogEntry dicts.
        """
        key = self._make_key(source, ra, dec, radius, band="__catalog__")
        cat_path = self.cache_dir / f"{source}_catalog_{key}.json"

        try:
            data = {
                "source": source,
                "ra": ra,
                "dec": dec,
                "radius": radius,
                "n_entries": len(entries),
                "entries": entries,
            }
            cat_path.write_text(json.dumps(data))

            self._index[key] = {
                "path": str(cat_path),
                "source": source,
                "ra": ra,
                "dec": dec,
                "radius": radius,
                "band": "__catalog__",
                "n_entries": len(entries),
            }
            self._save_index()
            logger.info(f"Cached catalog: {source} at ({ra:.3f}, {dec:.3f}), {len(entries)} entries")
        except Exception as e:
            logger.debug(f"Catalog cache write failed: {e}")

    def clear(self) -> int:
        """Clear all cached files. Returns number of files removed."""
        count = 0
        for entry in self._index.values():
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
                count += 1
        self._index.clear()
        self._save_index()
        logger.info(f"Cache cleared: {count} files removed")
        return count

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._index)
