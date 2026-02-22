"""Multi-source data pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from star_pattern.core.config import DataConfig
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.data.base import DataSource
from star_pattern.data.cache import DataCache
from star_pattern.data.sdss import SDSSDataSource
from star_pattern.data.gaia import GaiaDataSource
from star_pattern.data.mast import MASTDataSource
from star_pattern.data.ztf import ZTFDataSource
from star_pattern.utils.logging import get_logger

logger = get_logger("data.pipeline")


SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "sdss": SDSSDataSource,
    "gaia": GaiaDataSource,
    "mast": MASTDataSource,
    "ztf": ZTFDataSource,
}


class DataPipeline:
    """Orchestrate data acquisition from multiple surveys."""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self.cache = DataCache(self.config.cache_dir)
        self._sources: dict[str, DataSource] = {}
        self._init_sources()

    def _init_sources(self) -> None:
        """Initialize configured data sources."""
        for source_name in self.config.sources:
            if source_name in SOURCE_REGISTRY:
                src_class = SOURCE_REGISTRY[source_name]
                try:
                    if source_name in ("sdss", "gaia", "mast", "ztf"):
                        src = src_class(cache=self.cache)
                    else:
                        src = src_class()
                    if src.is_available():
                        self._sources[source_name] = src
                        logger.info(f"Data source available: {source_name}")
                    else:
                        logger.warning(f"Data source not available: {source_name}")
                except Exception as e:
                    logger.warning(f"Failed to init {source_name}: {e}")
            else:
                logger.warning(f"Unknown data source: {source_name}")

    @property
    def available_sources(self) -> list[str]:
        return list(self._sources.keys())

    def fetch_region(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
        sources: list[str] | None = None,
        include_catalog: bool = True,
    ) -> RegionData:
        """Fetch data from all configured sources for a region.

        Args:
            region: Sky region to query.
            bands: Specific bands (source-dependent).
            sources: Subset of sources to use (None = all).
            include_catalog: Whether to fetch catalogs.

        Returns:
            Merged RegionData from all sources.
        """
        active = sources or list(self._sources.keys())
        merged = RegionData(region=region)

        for source_name in active:
            if source_name not in self._sources:
                continue

            source = self._sources[source_name]
            logger.info(f"Fetching from {source_name}...")

            try:
                data = source.fetch_region(
                    region,
                    bands=bands,
                    include_catalog=include_catalog,
                )

                # Merge images (prefix with source name if needed)
                for band, img in data.images.items():
                    key = f"{source_name}_{band}" if band in merged.images else band
                    merged.images[key] = img

                # Merge catalogs
                merged.catalogs.update(data.catalogs)

            except Exception as e:
                logger.error(f"Failed to fetch from {source_name}: {e}")

        logger.info(
            f"Region data: {len(merged.images)} images, "
            f"{sum(len(c) for c in merged.catalogs.values())} catalog entries"
        )
        return merged

    def fetch_batch(
        self,
        regions: list[SkyRegion],
        **kwargs: Any,
    ) -> list[RegionData]:
        """Fetch data for multiple regions."""
        results = []
        for i, region in enumerate(regions):
            logger.info(f"Fetching region {i + 1}/{len(regions)}: {region}")
            results.append(self.fetch_region(region, **kwargs))
        return results
