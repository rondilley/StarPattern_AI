"""Tests for wavelet multi-scale analysis."""

import numpy as np
import pytest

from star_pattern.detection.wavelet import (
    WaveletAnalyzer,
    atrous_decompose,
    _atrous_convolve_2d,
)


class TestAtrousConvolution:
    """Test the a-trous convolution kernel."""

    def test_scale0_smoothing(self):
        """Scale 0 should smooth with the base B3 kernel."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (64, 64))
        smoothed = _atrous_convolve_2d(data, scale=0)

        # Should be smoother than original
        assert np.std(smoothed) < np.std(data)

    def test_higher_scale_smoother(self):
        """Higher scales should produce smoother output on large images."""
        rng = np.random.default_rng(42)
        # Use larger image to avoid boundary effects at higher scales
        data = rng.normal(0, 1, (256, 256))
        s0 = _atrous_convolve_2d(data, scale=0)
        s1 = _atrous_convolve_2d(data, scale=1)
        s2 = _atrous_convolve_2d(data, scale=2)

        assert np.std(s1) < np.std(s0)
        assert np.std(s2) < np.std(s1)

    def test_shape_preserved(self):
        """Convolution should preserve image shape."""
        data = np.ones((64, 64))
        for scale in range(4):
            smoothed = _atrous_convolve_2d(data, scale=scale)
            assert smoothed.shape == data.shape


class TestAtrousDecompose:
    """Test the wavelet decomposition."""

    def test_perfect_reconstruction(self):
        """Sum of details + smooth should equal original."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 20, (64, 64))

        details, smooth = atrous_decompose(data, n_scales=4)

        reconstructed = smooth.copy()
        for d in details:
            reconstructed += d

        np.testing.assert_allclose(reconstructed, data, atol=1e-10)

    def test_n_scales(self):
        """Should return the requested number of detail planes."""
        data = np.ones((64, 64))

        for n in [3, 4, 5]:
            details, smooth = atrous_decompose(data, n_scales=n)
            assert len(details) == n

    def test_detail_mean_near_zero(self):
        """Detail planes should have near-zero mean (high-pass filter)."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 20, (64, 64))
        details, _ = atrous_decompose(data, n_scales=4)

        for d in details:
            # Mean should be close to zero (within noise)
            assert abs(np.mean(d)) < 5.0

    def test_point_source_at_fine_scale(self):
        """A point source should have most energy at fine scales."""
        data = np.zeros((64, 64))
        # Add a point source
        data[32, 32] = 1000.0

        details, _ = atrous_decompose(data, n_scales=4)

        energies = [np.sum(d ** 2) for d in details]
        # Fine scale (scale 0) should have the most energy
        assert energies[0] > energies[2]
        assert energies[0] > energies[3]

    def test_extended_source_at_coarse_scale(self):
        """Extended emission should have more energy at coarse scales."""
        y, x = np.mgrid[:128, :128]
        # Large Gaussian (sigma=20)
        data = 100 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 20 ** 2))

        details, _ = atrous_decompose(data, n_scales=5)

        energies = [np.sum(d ** 2) for d in details]
        # Coarser scales should dominate for extended sources
        coarse_energy = sum(energies[2:])
        fine_energy = sum(energies[:2])
        assert coarse_energy > fine_energy


class TestWaveletAnalyzer:
    """Test the WaveletAnalyzer pipeline."""

    def test_analyze_returns_expected_keys(self):
        """Result dict should have all required keys."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 20, (64, 64)).astype(np.float32)
        analyzer = WaveletAnalyzer(n_scales=4)
        result = analyzer.analyze(data)

        assert "wavelet_score" in result
        assert "n_scales" in result
        assert "scale_analysis" in result
        assert "detections" in result
        assert "multiscale_objects" in result
        assert "scale_spectrum" in result
        assert "mean_scale" in result

    def test_n_scales_capped_by_image_size(self):
        """n_scales should be limited by log2(image_size)."""
        data = np.ones((16, 16))
        analyzer = WaveletAnalyzer(n_scales=10)  # Too many for 16x16
        result = analyzer.analyze(data)

        # log2(16) - 2 = 2, so max 2 scales
        assert result["n_scales"] <= 4

    def test_score_with_sources(self):
        """Image with sources should have nonzero wavelet score."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 5, (128, 128)).astype(np.float32)
        # Add several bright sources
        for sy, sx in [(40, 50), (60, 80), (90, 30), (64, 64)]:
            y, x = np.mgrid[:128, :128]
            data += 500 * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 2 ** 2))

        analyzer = WaveletAnalyzer(n_scales=5, significance_threshold=3.0)
        result = analyzer.analyze(data)

        assert result["wavelet_score"] > 0
        assert len(result["detections"]) > 0

    def test_score_on_uniform(self):
        """Uniform image should have low wavelet score."""
        data = np.full((64, 64), 100.0, dtype=np.float32)
        analyzer = WaveletAnalyzer(n_scales=4)
        result = analyzer.analyze(data)

        assert result["wavelet_score"] < 0.2
        assert len(result["detections"]) == 0

    def test_multiscale_object_detection(self):
        """Extended source should be detected as multi-scale object."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 3, (128, 128)).astype(np.float32)
        # Add an extended source (galaxy-like)
        y, x = np.mgrid[:128, :128]
        data += 300 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 10 ** 2))

        analyzer = WaveletAnalyzer(n_scales=5, significance_threshold=2.5)
        result = analyzer.analyze(data)

        # Should detect the source across multiple scales
        n_multi = len(result["multiscale_objects"])
        assert n_multi >= 1, f"Expected multi-scale objects, got {n_multi}"

    def test_scale_spectrum_sums_to_one(self):
        """Scale spectrum should be normalized."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 20, (64, 64)).astype(np.float32)
        analyzer = WaveletAnalyzer(n_scales=4)
        result = analyzer.analyze(data)

        spectrum = result["scale_spectrum"]
        assert abs(sum(spectrum) - 1.0) < 0.01

    def test_noise_estimation_mad(self):
        """MAD noise estimation should be robust to outliers."""
        analyzer = WaveletAnalyzer(noise_estimation="mad")
        # Mostly noise with a few outliers
        rng = np.random.default_rng(42)
        detail = rng.normal(0, 1, (64, 64))
        detail[30, 30] = 100  # outlier

        noise = analyzer._estimate_noise(detail)
        # Should be close to 1.0, not inflated by the outlier
        assert 0.5 < noise < 2.0

    def test_pixel_scale_reported(self):
        """Physical scale should appear in results when pixel_scale given."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 5, (64, 64)).astype(np.float32)
        data[32, 32] += 500  # bright source

        analyzer = WaveletAnalyzer(n_scales=4, significance_threshold=2.0)
        result = analyzer.analyze(data, pixel_scale_arcsec=0.4)

        # Check that scale_arcsec appears in scale_analysis
        for sa in result["scale_analysis"]:
            assert "scale_arcsec" in sa
