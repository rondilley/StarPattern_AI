"""Tests for FITS image handling."""

import numpy as np
import pytest

from star_pattern.core.fits_handler import FITSImage


class TestFITSImage:
    def test_from_array(self):
        data = np.random.default_rng(0).normal(0, 1, (64, 64)).astype(np.float32)
        img = FITSImage.from_array(data)
        assert img.shape == (64, 64)
        assert img.wcs is None

    def test_normalize_arcsinh(self, synthetic_image: FITSImage):
        norm = synthetic_image.normalize("arcsinh")
        assert norm.shape == synthetic_image.shape
        assert np.isfinite(norm.data).all()

    def test_normalize_linear(self, synthetic_image: FITSImage):
        norm = synthetic_image.normalize("linear")
        assert norm.data.min() >= 0
        assert norm.data.max() <= 1

    def test_normalize_zscale(self, synthetic_image: FITSImage):
        norm = synthetic_image.normalize("zscale")
        assert np.isfinite(norm.data).all()

    def test_normalize_log(self, synthetic_image: FITSImage):
        norm = synthetic_image.normalize("log")
        assert np.isfinite(norm.data).all()

    def test_normalize_unknown_raises(self, synthetic_image: FITSImage):
        with pytest.raises(ValueError, match="Unknown"):
            synthetic_image.normalize("foobar")

    def test_to_tensor(self, synthetic_image: FITSImage):
        tensor = synthetic_image.to_tensor()
        assert tensor.shape == (1, 256, 256)
        assert tensor.dtype.is_floating_point

    def test_to_rgb(self, synthetic_image: FITSImage):
        rgb = synthetic_image.to_rgb()
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype == np.uint8

    def test_save_and_load(self, synthetic_image: FITSImage, tmp_path):
        path = tmp_path / "test.fits"
        synthetic_image.save(path)
        loaded = FITSImage.from_file(path)
        np.testing.assert_allclose(loaded.data, synthetic_image.data, atol=1e-5)

    def test_handles_nan(self):
        data = np.array([[1, np.nan], [np.inf, 0]], dtype=np.float32)
        img = FITSImage.from_array(data)
        norm = img.normalize("linear")
        assert np.isfinite(norm.data).all()

    def test_repr(self, synthetic_image: FITSImage):
        r = repr(synthetic_image)
        assert "256" in r
        assert "no WCS" in r
