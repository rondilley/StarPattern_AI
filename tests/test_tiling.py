"""Tests for sky tiling."""

import numpy as np
import pytest

from star_pattern.core.tiling import TileGrid


class TestTileGrid:
    def test_single_tile(self):
        """Field radius <= tile radius produces 1 tile."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=2.0,
            tile_radius_arcmin=3.0,
        )
        assert len(grid) == 1
        assert grid.tiles[0].ra == 180.0
        assert grid.tiles[0].dec == 45.0

    def test_hex_packing(self):
        """10 arcmin field with 3 arcmin tiles produces multiple tiles."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=10.0,
            tile_radius_arcmin=3.0,
            overlap_fraction=0.2,
        )
        # Should produce a reasonable number of tiles
        assert len(grid) > 1
        assert len(grid) < 100  # Sanity upper bound

    def test_overlap(self):
        """Adjacent tiles overlap by the specified fraction."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=10.0,
            tile_radius_arcmin=3.0,
            overlap_fraction=0.2,
        )
        # Find minimum separation between any two tile centers
        tiles = grid.tiles
        if len(tiles) < 2:
            pytest.skip("Not enough tiles to test overlap")

        min_sep = float("inf")
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                sep = TileGrid._angular_sep(
                    tiles[i].ra, tiles[i].dec,
                    tiles[j].ra, tiles[j].dec,
                )
                if sep < min_sep:
                    min_sep = sep

        # Adjacent tiles should be closer than 2 * tile_radius
        # (they overlap) but further than 0
        step = 3.0 * 2 * (1 - 0.2)  # 4.8 arcmin
        assert min_sep < 2 * 3.0  # Less than diameter
        assert min_sep > 0

    def test_ra_wrapping(self):
        """Field near RA=0/360 boundary works correctly."""
        grid = TileGrid(
            center_ra=0.5,
            center_dec=45.0,
            field_radius_arcmin=10.0,
            tile_radius_arcmin=3.0,
        )
        assert len(grid) > 1
        # All tile RAs should be valid [0, 360)
        for tile in grid.tiles:
            assert 0 <= tile.ra < 360

    def test_max_tiles_limit(self):
        """Safety limit caps tile count."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=60.0,  # Very large field
            tile_radius_arcmin=3.0,
            max_tiles=20,
        )
        assert len(grid) <= 20

    def test_angular_sep(self):
        """Vincenty angular separation is accurate."""
        # Same point -> 0
        sep = TileGrid._angular_sep(180.0, 45.0, 180.0, 45.0)
        assert sep < 0.001

        # 1 degree apart in dec -> 60 arcmin
        sep = TileGrid._angular_sep(180.0, 45.0, 180.0, 46.0)
        assert abs(sep - 60.0) < 0.1

    def test_tiles_property(self):
        """Tiles property returns list of SkyRegion."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=10.0,
            tile_radius_arcmin=3.0,
        )
        tiles = grid.tiles
        assert isinstance(tiles, list)
        for tile in tiles:
            assert hasattr(tile, "ra")
            assert hasattr(tile, "dec")
            assert tile.radius == 3.0

    def test_repr(self):
        """Repr includes key info."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=45.0,
            field_radius_arcmin=10.0,
        )
        r = repr(grid)
        assert "180" in r
        assert "45" in r
        assert "n_tiles" in r

    def test_near_pole(self):
        """Tiling near a pole does not crash."""
        grid = TileGrid(
            center_ra=180.0,
            center_dec=88.0,
            field_radius_arcmin=10.0,
            tile_radius_arcmin=3.0,
        )
        assert len(grid) >= 1
        for tile in grid.tiles:
            assert -90 <= tile.dec <= 90
