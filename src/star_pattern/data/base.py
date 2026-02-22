"""Abstract base class for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.catalog import StarCatalog


class DataSource(ABC):
    """Abstract interface for astronomical data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of the data source (e.g., 'sdss', 'gaia')."""
        ...

    @property
    @abstractmethod
    def available_bands(self) -> list[str]:
        """List of available photometric bands."""
        ...

    @abstractmethod
    def fetch_images(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
    ) -> dict[str, FITSImage]:
        """Fetch images for a sky region.

        Args:
            region: The sky region to query.
            bands: Specific bands to fetch. None means default bands.

        Returns:
            Dictionary mapping band name to FITSImage.
        """
        ...

    @abstractmethod
    def fetch_catalog(
        self,
        region: SkyRegion,
        max_results: int = 10000,
    ) -> StarCatalog:
        """Fetch catalog data for a sky region.

        Args:
            region: The sky region to query.
            max_results: Maximum number of catalog entries.

        Returns:
            StarCatalog with entries in the region.
        """
        ...

    def fetch_region(
        self,
        region: SkyRegion,
        bands: list[str] | None = None,
        include_catalog: bool = True,
        max_catalog: int = 10000,
    ) -> RegionData:
        """Fetch all data for a region (images + catalog).

        Args:
            region: The sky region to query.
            bands: Specific bands to fetch.
            include_catalog: Whether to also fetch catalog data.
            max_catalog: Max catalog entries.

        Returns:
            RegionData with images and optionally catalogs.
        """
        images = self.fetch_images(region, bands=bands)
        catalogs = {}
        if include_catalog:
            catalogs[self.name] = self.fetch_catalog(region, max_results=max_catalog)
        return RegionData(region=region, images=images, catalogs=catalogs)

    def is_available(self) -> bool:
        """Check if this data source is accessible."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
