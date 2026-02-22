"""Image mosaicking: combine overlapping FITS images via reproject."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.utils.logging import get_logger

logger = get_logger("data.mosaic")


class Mosaicker:
    """Combine multiple FITS images into a mosaic using reproject."""

    def __init__(
        self,
        pixel_scale_arcsec: float = 0.4,
        combine: str = "mean",
    ):
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.combine = combine

    def mosaic(self, images: list[FITSImage]) -> FITSImage:
        """Reproject and co-add images into a single mosaic.

        Args:
            images: List of FITSImage objects with valid WCS.

        Returns:
            Combined FITSImage mosaic.

        Raises:
            ImportError: If reproject is not installed.
            ValueError: If no images have valid WCS.
        """
        # Filter images with valid WCS (before importing reproject)
        valid = [img for img in images if img.wcs is not None]
        if not valid:
            raise ValueError("No images with WCS available for mosaicking")

        if len(valid) == 1:
            return valid[0]

        try:
            from reproject import reproject_interp
            from reproject.mosaicking import (
                find_optimal_celestial_wcs,
                reproject_and_coadd,
            )
        except ImportError as e:
            raise ImportError(
                "reproject is required for mosaicking: "
                "pip install reproject>=0.13"
            ) from e

        from astropy.io import fits
        import astropy.units as u

        logger.info(f"Mosaicking {len(valid)} images")

        # Build input list for reproject
        input_data = [(img.data, img.wcs) for img in valid]

        # Find optimal output WCS covering all inputs
        wcs_out, shape_out = find_optimal_celestial_wcs(
            input_data,
            resolution=self.pixel_scale_arcsec / 3600 * u.deg,
        )

        # Reproject and co-add
        array, footprint = reproject_and_coadd(
            input_data,
            wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function=self.combine,
        )

        # NaN-fill regions with no coverage
        array = np.nan_to_num(array, nan=0.0)

        # Build FITS header
        header = fits.Header(wcs_out.to_header())
        header["NCOMBINE"] = len(valid)
        header["COMBMETH"] = self.combine

        logger.info(f"Mosaic complete: {array.shape}")
        return FITSImage(data=array, header=header, wcs=wcs_out)
