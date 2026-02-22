"""Wide-field pipeline: tiling, multi-source fetch, and mosaicking."""

from __future__ import annotations

from typing import Any

from star_pattern.core.config import PipelineConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion, RegionData
from star_pattern.core.tiling import TileGrid
from star_pattern.data.mosaic import Mosaicker
from star_pattern.data.pipeline import DataPipeline
from star_pattern.utils.logging import get_logger

logger = get_logger("data.wide_field")


class WideFieldPipeline:
    """Fetch and mosaic wide-field sky coverage.

    Orchestrates:
    1. TileGrid to decompose a large field into overlapping tiles
    2. DataPipeline to fetch each tile from all configured sources
    3. Mosaicker to stitch overlapping tiles into a single image per band
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.wf = config.wide_field
        self.data_pipeline = DataPipeline(config.data)
        self.mosaicker = Mosaicker(
            pixel_scale_arcsec=self.wf.mosaic_pixel_scale_arcsec,
            combine=self.wf.mosaic_combine,
        )

    def fetch_wide_field(
        self,
        center_ra: float,
        center_dec: float,
        field_radius_arcmin: float,
        bands: list[str] | None = None,
    ) -> RegionData:
        """Fetch a wide-field region by tiling and mosaicking.

        Args:
            center_ra: Center RA in degrees.
            center_dec: Center Dec in degrees.
            field_radius_arcmin: Field radius in arcminutes.
            bands: Specific bands to fetch (None = source defaults).

        Returns:
            RegionData with mosaicked images and merged catalogs.
        """
        from star_pattern.core.catalog import StarCatalog

        grid = TileGrid(
            center_ra=center_ra,
            center_dec=center_dec,
            field_radius_arcmin=field_radius_arcmin,
            tile_radius_arcmin=self.wf.tile_radius_arcmin,
            overlap_fraction=self.wf.overlap_fraction,
            max_tiles=self.wf.max_tiles,
        )

        logger.info(
            f"Wide-field: {len(grid)} tiles for "
            f"{field_radius_arcmin}' field"
        )

        # Fetch all tiles
        all_images: dict[str, list[FITSImage]] = {}
        all_catalogs: dict[str, StarCatalog] = {}

        for i, tile in enumerate(grid.tiles):
            logger.info(
                f"  Tile {i + 1}/{len(grid)}: "
                f"({tile.ra:.3f}, {tile.dec:.3f})"
            )
            try:
                tile_data = self.data_pipeline.fetch_region(
                    tile, bands=bands
                )
                for band, img in tile_data.images.items():
                    all_images.setdefault(band, []).append(img)
                for src, cat in tile_data.catalogs.items():
                    if src in all_catalogs:
                        all_catalogs[src] = all_catalogs[src].merge(cat)
                    else:
                        all_catalogs[src] = cat
            except Exception as e:
                logger.warning(f"  Tile fetch failed: {e}")

        # Mosaic each band
        mosaic_region = SkyRegion(
            ra=center_ra, dec=center_dec, radius=field_radius_arcmin
        )
        result = RegionData(region=mosaic_region)

        for band, images in all_images.items():
            if len(images) == 0:
                continue
            try:
                if len(images) == 1:
                    result.images[band] = images[0]
                else:
                    result.images[band] = self.mosaicker.mosaic(images)
                logger.info(
                    f"  Mosaic {band}: {result.images[band].shape}"
                )
            except Exception as e:
                logger.warning(f"  Mosaic failed for {band}: {e}")
                # Fall back to first image
                result.images[band] = images[0]

        result.catalogs = all_catalogs
        result.metadata["n_tiles"] = len(grid)
        result.metadata["is_mosaic"] = True

        return result
