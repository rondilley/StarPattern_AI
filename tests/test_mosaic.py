"""Tests for image mosaicking."""

import numpy as np
import pytest

from star_pattern.core.fits_handler import FITSImage
from star_pattern.data.mosaic import Mosaicker


def _make_image_with_wcs(ra_center: float, dec_center: float, size: int = 64):
    """Create a synthetic FITSImage with a valid TAN WCS."""
    from astropy.wcs import WCS
    from astropy.io import fits

    rng = np.random.default_rng(int(ra_center * 1000 + dec_center * 100))
    data = rng.normal(100, 10, (size, size)).astype(np.float32)

    # Add a bright source at center
    yy, xx = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    data += 200 * np.exp(-(xx**2 + yy**2) / (2 * 3**2))

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [size / 2, size / 2]
    wcs.wcs.cdelt = [-0.4 / 3600, 0.4 / 3600]  # 0.4 arcsec/pixel
    wcs.wcs.crval = [ra_center, dec_center]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = fits.Header(wcs.to_header())
    return FITSImage(data=data, header=header, wcs=wcs)


class TestMosaicker:
    def test_single_image_passthrough(self):
        """Mosaicker with 1 image returns it directly."""
        img = _make_image_with_wcs(180.0, 45.0)
        mosaicker = Mosaicker()
        result = mosaicker.mosaic([img])
        assert result is img  # Should be the same object

    def test_no_wcs_error(self):
        """Images without WCS raise ValueError."""
        data = np.random.default_rng(42).normal(100, 10, (64, 64))
        img = FITSImage.from_array(data.astype(np.float32))
        mosaicker = Mosaicker()
        with pytest.raises(ValueError, match="No images with WCS"):
            mosaicker.mosaic([img])

    def test_empty_list_error(self):
        """Empty image list raises ValueError."""
        mosaicker = Mosaicker()
        with pytest.raises(ValueError, match="No images with WCS"):
            mosaicker.mosaic([])

    def test_two_image_mosaic(self):
        """Two overlapping images produce a combined mosaic."""
        try:
            import reproject  # noqa: F401
        except ImportError:
            pytest.skip("reproject not installed")

        # Two images offset by ~10 arcsec
        img1 = _make_image_with_wcs(180.0, 45.0)
        img2 = _make_image_with_wcs(180.003, 45.003)  # ~10.8 arcsec offset

        mosaicker = Mosaicker(pixel_scale_arcsec=0.4)
        result = mosaicker.mosaic([img1, img2])

        # Mosaic should be larger than either input
        assert result.data.ndim == 2
        assert result.wcs is not None
        assert result.data.shape[0] >= 64
        assert result.data.shape[1] >= 64

    def test_mosaic_with_one_wcs_one_without(self):
        """Mixed WCS/no-WCS images: only WCS image used."""
        img_wcs = _make_image_with_wcs(180.0, 45.0)
        data_no_wcs = np.random.default_rng(42).normal(100, 10, (64, 64))
        img_no_wcs = FITSImage.from_array(data_no_wcs.astype(np.float32))

        mosaicker = Mosaicker()
        result = mosaicker.mosaic([img_wcs, img_no_wcs])
        # Should return the one valid image
        assert result is img_wcs
