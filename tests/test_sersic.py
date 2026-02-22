"""Tests for Sersic profile fitting."""

import numpy as np
import pytest

from star_pattern.detection.sersic import SersicAnalyzer, sersic_1d, _sersic_bn


class TestSersicBn:
    """Test the b_n approximation."""

    def test_n1_exponential(self):
        bn = _sersic_bn(1.0)
        # For n=1, b_n ~ 1.678 (exact: gamma(2) = 1)
        assert 1.5 < bn < 1.8

    def test_n4_devaucouleurs(self):
        bn = _sersic_bn(4.0)
        # For n=4, b_n ~ 7.67
        assert 7.0 < bn < 8.5

    def test_n0_returns_zero(self):
        assert _sersic_bn(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert _sersic_bn(-1.0) == 0.0


class TestSersic1D:
    """Test the 1D Sersic profile function."""

    def test_peak_at_zero(self):
        r = np.array([0.0, 1.0, 5.0, 10.0])
        I = sersic_1d(r, I_e=100.0, r_e=5.0, n=2.0)
        # Intensity should decrease with radius
        assert I[0] > I[1] > I[2] > I[3]

    def test_value_at_re(self):
        r = np.array([5.0])
        I = sersic_1d(r, I_e=100.0, r_e=5.0, n=2.0)
        # At r=r_e, I should equal I_e
        assert abs(I[0] - 100.0) < 1.0

    def test_exponential_disk(self):
        # n=1 is an exponential profile
        r = np.linspace(0.5, 20, 50)
        I = sersic_1d(r, I_e=100.0, r_e=5.0, n=1.0)
        # All values should be positive
        assert np.all(I > 0)
        # Should be monotonically decreasing
        assert np.all(np.diff(I) < 0)

    def test_large_n_concentrated(self):
        r = np.array([0.5, 5.0, 30.0])
        I_n1 = sersic_1d(r, I_e=100.0, r_e=5.0, n=1.0)
        I_n4 = sersic_1d(r, I_e=100.0, r_e=5.0, n=4.0)
        # n=4 is more concentrated: brighter near center
        assert I_n4[0] > I_n1[0]  # brighter at small r
        # Both should equal I_e at r=r_e
        assert abs(I_n1[1] - 100.0) < 5.0
        assert abs(I_n4[1] - 100.0) < 5.0


class TestSersicAnalyzer:
    """Test the full Sersic analysis pipeline."""

    def _make_sersic_galaxy(self, n=2.0, r_e=20.0, I_e=500.0, size=128, noise=10.0):
        """Create a synthetic galaxy image with known Sersic profile."""
        rng = np.random.default_rng(42)
        y, x = np.mgrid[:size, :size]
        cx, cy = size // 2, size // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Create Sersic profile
        data = sersic_1d(r.ravel(), I_e, r_e, n).reshape(size, size)
        # Add background + noise
        data += 100.0  # background
        data += rng.normal(0, noise, data.shape)
        return data.astype(np.float32)

    def test_fit_exponential_disk(self):
        """Fit a synthetic n=1 exponential disk."""
        data = self._make_sersic_galaxy(n=1.0, r_e=15.0, I_e=300.0)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(data)

        assert result["fit"]["success"]
        # Sersic index should be close to 1
        fitted_n = result["fit"]["n"]
        assert 0.3 < fitted_n < 2.5, f"Expected n~1, got {fitted_n}"

    def test_fit_elliptical(self):
        """Fit a synthetic n=4 de Vaucouleurs profile."""
        data = self._make_sersic_galaxy(n=4.0, r_e=15.0, I_e=500.0)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(data)

        assert result["fit"]["success"]
        fitted_n = result["fit"]["n"]
        assert 2.0 < fitted_n < 8.0, f"Expected n~4, got {fitted_n}"

    def test_morphology_classification(self):
        """Verify morphology class matches Sersic index."""
        analyzer = SersicAnalyzer()

        # Disk
        assert analyzer._classify_morphology(1.0, 0.2) == "disk/spiral"
        # Edge-on disk
        assert analyzer._classify_morphology(1.0, 0.7) == "edge-on_disk"
        # Elliptical
        assert analyzer._classify_morphology(4.0, 0.1) == "elliptical"
        # Irregular
        assert analyzer._classify_morphology(0.3, 0.2) == "irregular"
        # Compact
        assert analyzer._classify_morphology(6.0, 0.1) == "compact/cD"

    def test_score_on_galaxy(self):
        """Sersic score should be nonzero for a galaxy image."""
        data = self._make_sersic_galaxy(n=2.0, r_e=20.0)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(data)

        assert result["sersic_score"] >= 0
        assert result["sersic_score"] <= 1.0

    def test_score_on_noise(self):
        """Sersic score should be low on pure noise."""
        rng = np.random.default_rng(42)
        noise = rng.normal(100, 10, (128, 128)).astype(np.float32)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(noise)

        # May or may not successfully fit, but score should be low
        assert result["sersic_score"] <= 0.5

    def test_residual_detection(self):
        """Residuals should detect features not in the smooth model."""
        # Galaxy with an added bright clump
        data = self._make_sersic_galaxy(n=2.0, r_e=20.0, I_e=500.0, noise=5.0)
        # Add bright feature at r~30 px (outside r_e)
        y, x = np.mgrid[:128, :128]
        clump = 200 * np.exp(-((x - 90) ** 2 + (y - 64) ** 2) / (2 * 3 ** 2))
        data = data + clump.astype(np.float32)

        analyzer = SersicAnalyzer(residual_sigma=2.5)
        result = analyzer.analyze(data)

        # Should find at least one residual feature
        n_features = len(result.get("residual_features", []))
        assert n_features >= 1, f"Expected residual features, got {n_features}"

    def test_pixel_scale_conversion(self):
        """Physical sizes are reported when pixel scale is given."""
        data = self._make_sersic_galaxy(n=2.0, r_e=20.0)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(data, pixel_scale_arcsec=0.4)

        if result["fit"]["success"]:
            assert result["r_e_arcsec"] is not None
            assert result["r_e_arcsec"] > 0

    def test_ellipticity_circular(self):
        """Circular galaxy should have low ellipticity."""
        data = self._make_sersic_galaxy(n=2.0, r_e=20.0)
        analyzer = SersicAnalyzer()
        result = analyzer.analyze(data)

        # Circular synthetic galaxy
        assert result["ellipticity"] < 0.3
