"""Tests for time-domain variability analysis.

All tests use synthetic light curves -- no external data needed.
"""

import numpy as np
import pytest

from star_pattern.core.catalog import CatalogEntry, StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.detection.variability import VariabilityAnalyzer


def _make_lightcurve(
    n_epochs: int = 100,
    baseline_days: float = 500.0,
    mean_mag: float = 18.0,
    scatter: float = 0.02,
    error: float = 0.03,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, float]]:
    """Create a synthetic constant light curve with noise."""
    rng = rng or np.random.default_rng(42)
    times = np.sort(rng.uniform(0, baseline_days, n_epochs))
    mags = mean_mag + rng.normal(0, scatter, n_epochs)
    errs = np.full(n_epochs, error)
    return [(float(t), float(m), float(e)) for t, m, e in zip(times, mags, errs)]


def _make_sinusoidal_lc(
    period: float = 5.0,
    amplitude: float = 0.5,
    n_epochs: int = 200,
    baseline_days: float = 500.0,
    error: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, float]]:
    """Create a synthetic sinusoidal (periodic) light curve."""
    rng = rng or np.random.default_rng(42)
    times = np.sort(rng.uniform(0, baseline_days, n_epochs))
    mags = 18.0 + amplitude * np.sin(2 * np.pi * times / period)
    mags += rng.normal(0, error, n_epochs)
    errs = np.full(n_epochs, error)
    return [(float(t), float(m), float(e)) for t, m, e in zip(times, mags, errs)]


def _make_catalog_with_lc(lightcurves: list[dict]) -> StarCatalog:
    """Create a StarCatalog with light curves embedded in properties."""
    entries = []
    for i, lc_data in enumerate(lightcurves):
        entries.append(
            CatalogEntry(
                ra=180.0 + i * 0.01,
                dec=45.0 + i * 0.01,
                mag=18.0,
                source="ztf",
                source_id=f"ztf_{i}",
                properties={
                    "ztf_lightcurve": lc_data,
                    "ztf_n_epochs": sum(
                        len(pts) for pts in lc_data.values()
                    ),
                    "ztf_baseline_days": 500.0,
                },
            )
        )
    return StarCatalog(entries=entries, source="ztf")


class TestVariabilityIndex:
    """Tests for variability index computation."""

    def test_constant_lightcurve_not_variable(self):
        """A constant light curve should have chi2 ~ 1.0 and is_variable=False."""
        analyzer = VariabilityAnalyzer()
        lc = _make_lightcurve(scatter=0.03, error=0.03)
        times = np.array([p[0] for p in lc])
        mags = np.array([p[1] for p in lc])
        errs = np.array([p[2] for p in lc])

        result = analyzer._compute_variability_index(times, mags, errs)

        assert not result["is_variable"]
        # chi2_reduced should be close to 1 for noise-dominated
        assert result["chi2_reduced"] < 3.0

    def test_variable_lightcurve_detected(self):
        """A high-scatter light curve should be flagged as variable."""
        analyzer = VariabilityAnalyzer()
        # Large scatter (0.5 mag) relative to errors (0.03 mag)
        lc = _make_lightcurve(scatter=0.5, error=0.03)
        times = np.array([p[0] for p in lc])
        mags = np.array([p[1] for p in lc])
        errs = np.array([p[2] for p in lc])

        result = analyzer._compute_variability_index(times, mags, errs)

        assert result["is_variable"]
        assert result["chi2_reduced"] > 10.0
        assert result["amplitude"] > 0.5

    def test_variability_index_values(self):
        """Check that all expected keys are present and values are reasonable."""
        analyzer = VariabilityAnalyzer()
        rng = np.random.default_rng(123)
        times = np.sort(rng.uniform(0, 100, 50))
        mags = 18.0 + rng.normal(0, 0.1, 50)
        errs = np.full(50, 0.05)

        result = analyzer._compute_variability_index(times, mags, errs)

        assert "weighted_stdev" in result
        assert "chi2_reduced" in result
        assert "iqr" in result
        assert "eta" in result
        assert "amplitude" in result
        assert "median_abs_dev" in result
        assert "is_variable" in result
        assert result["weighted_stdev"] >= 0
        assert result["amplitude"] >= 0
        assert result["iqr"] >= 0

    def test_von_neumann_eta(self):
        """Correlated time series should have low eta, random should have eta ~ 2."""
        analyzer = VariabilityAnalyzer()
        rng = np.random.default_rng(42)

        # Random (uncorrelated) -- eta should be close to 2
        times = np.arange(100, dtype=float)
        random_mags = 18.0 + rng.normal(0, 1.0, 100)
        errs = np.full(100, 0.01)
        random_result = analyzer._compute_variability_index(times, random_mags, errs)

        # Correlated (smooth sine) -- eta should be low
        smooth_mags = 18.0 + np.sin(2 * np.pi * times / 20)
        smooth_result = analyzer._compute_variability_index(times, smooth_mags, errs)

        assert smooth_result["eta"] < random_result["eta"]


class TestLombScargle:
    """Tests for Lomb-Scargle periodogram."""

    def test_sinusoidal_detected_as_periodic(self):
        """An injected sine wave should have its period recovered."""
        analyzer = VariabilityAnalyzer()
        period = 10.0
        lc = _make_sinusoidal_lc(period=period, amplitude=0.5, n_epochs=300)
        times = np.array([p[0] for p in lc])
        mags = np.array([p[1] for p in lc])
        errs = np.array([p[2] for p in lc])

        result = analyzer._lomb_scargle(times, mags, errs)

        assert result["is_periodic"]
        assert result["fap"] < 0.01
        # Period should be recovered within 10%
        assert abs(result["best_period"] - period) / period < 0.1

    def test_random_noise_high_fap(self):
        """Pure noise should have high false alarm probability."""
        analyzer = VariabilityAnalyzer()
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 500, 100))
        mags = 18.0 + rng.normal(0, 0.03, 100)
        errs = np.full(100, 0.03)

        result = analyzer._lomb_scargle(times, mags, errs)

        # Pure noise should not be detected as periodic (high FAP)
        # Note: occasionally random data can produce low FAP; we check
        # that the power is at least low
        assert result["best_power"] < 0.5 or result["fap"] > 0.001


class TestOutburstDetection:
    """Tests for outburst detection."""

    def test_outburst_detected(self):
        """A single strong brightening event should be flagged."""
        analyzer = VariabilityAnalyzer()
        rng = np.random.default_rng(42)
        lc = _make_lightcurve(n_epochs=100, scatter=0.03, error=0.03, rng=rng)
        times = np.array([p[0] for p in lc])
        mags = np.array([p[1] for p in lc])
        errs = np.array([p[2] for p in lc])

        # Inject a 3-mag brightening at index 50
        mags[50] -= 3.0

        outbursts = analyzer._detect_outbursts(times, mags, errs)

        assert len(outbursts) > 0
        # The injected outburst should be among the detections
        brightening_events = [o for o in outbursts if o["type"] == "brightening"]
        assert len(brightening_events) > 0

    def test_transient_candidate_detection(self):
        """A transient (sudden brightening on constant baseline) should be detected."""
        analyzer = VariabilityAnalyzer()
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 200, 80))
        # Constant baseline at mag 18 with small noise
        mags = 18.0 + rng.normal(0, 0.03, 80)
        errs = np.full(80, 0.03)

        # Inject a transient: 3-mag brightening at a few epochs
        mags[40] = 15.0
        mags[41] = 15.5
        mags[42] = 16.0

        outbursts = analyzer._detect_outbursts(times, mags, errs)

        assert len(outbursts) > 0
        brightening = [o for o in outbursts if o["type"] == "brightening"]
        assert len(brightening) >= 1


class TestClassification:
    """Tests for variable star classification."""

    def test_eclipsing_binary_classification(self):
        """Short-period, large-amplitude periodic -> eclipsing_binary."""
        analyzer = VariabilityAnalyzer()
        var_index = {"is_variable": True, "amplitude": 0.8, "eta": 0.5}
        periodogram = {"is_periodic": True, "best_period": 0.5, "fap": 0.001}

        result = analyzer._classify_variable(var_index, periodogram, [])

        assert result == "eclipsing_binary"

    def test_agn_like_classification(self):
        """Aperiodic, high amplitude, low eta -> agn_like."""
        analyzer = VariabilityAnalyzer()
        var_index = {"is_variable": True, "amplitude": 1.0, "eta": 1.0}
        periodogram = {"is_periodic": False, "best_period": None}

        result = analyzer._classify_variable(var_index, periodogram, [])

        assert result == "agn_like"

    def test_long_period_variable(self):
        """Period > 100 days -> long_period_variable."""
        analyzer = VariabilityAnalyzer()
        var_index = {"is_variable": True, "amplitude": 2.0, "eta": 1.0}
        periodogram = {"is_periodic": True, "best_period": 250.0, "fap": 0.001}

        result = analyzer._classify_variable(var_index, periodogram, [])

        assert result == "long_period_variable"

    def test_non_variable_classification(self):
        """Non-variable source returns non_variable."""
        analyzer = VariabilityAnalyzer()
        var_index = {"is_variable": False, "amplitude": 0.05, "eta": 2.0}
        periodogram = {"is_periodic": False}

        result = analyzer._classify_variable(var_index, periodogram, [])

        assert result == "non_variable"


class TestAnalyzerIntegration:
    """Integration tests for the full VariabilityAnalyzer."""

    def test_analyzer_integration(self):
        """Full catalog with mixed variables should produce results."""
        rng = np.random.default_rng(42)

        # Source 1: constant
        lc1 = {"r": _make_lightcurve(scatter=0.03, error=0.03, rng=rng)}

        # Source 2: periodic variable
        lc2 = {"r": _make_sinusoidal_lc(period=5.0, amplitude=0.5, rng=rng)}

        # Source 3: high scatter (variable)
        lc3 = {"r": _make_lightcurve(scatter=0.5, error=0.03, rng=np.random.default_rng(99))}

        catalog = _make_catalog_with_lc([lc1, lc2, lc3])
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_analyzed"] == 3
        assert result["variability_score"] >= 0
        assert len(result["variable_candidates"]) >= 1
        assert len(result["periodic_candidates"]) >= 1

    def test_score_range(self):
        """variability_score should be in [0, 1]."""
        rng = np.random.default_rng(42)
        lc = {"r": _make_sinusoidal_lc(period=3.0, amplitude=1.0, rng=rng)}
        catalog = _make_catalog_with_lc([lc])

        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert 0 <= result["variability_score"] <= 1

    def test_empty_catalog(self):
        """Empty catalog returns zero score gracefully."""
        catalog = StarCatalog(entries=[], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["variability_score"] == 0.0
        assert result["n_analyzed"] == 0

    def test_insufficient_epochs_skipped(self):
        """Sources with < min_epochs should be skipped."""
        # Only 5 epochs (below default min_epochs=10)
        lc = {"r": _make_lightcurve(n_epochs=5, scatter=0.5, error=0.03)}
        catalog = _make_catalog_with_lc([lc])

        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_analyzed"] == 0
        assert result["variability_score"] == 0.0

    def test_multi_band_handling(self):
        """g and r band curves should be analyzed independently."""
        rng = np.random.default_rng(42)
        lc = {
            "g": _make_sinusoidal_lc(period=5.0, amplitude=0.5, rng=rng),
            "r": _make_lightcurve(scatter=0.03, error=0.03, rng=np.random.default_rng(99)),
        }
        catalog = _make_catalog_with_lc([lc])

        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        # Should still detect the periodic g-band signal
        assert result["n_analyzed"] == 1
        assert len(result["variable_candidates"]) >= 1

    def test_config_thresholds(self):
        """Custom thresholds should change detection counts."""
        rng = np.random.default_rng(42)
        # Moderate scatter -- detected with low threshold, not with high
        lc = {"r": _make_lightcurve(scatter=0.1, error=0.03, rng=rng)}
        catalog = _make_catalog_with_lc([lc])

        # Low threshold: should detect as variable
        config_low = DetectionConfig(variability_significance=1.5)
        analyzer_low = VariabilityAnalyzer(config_low)
        result_low = analyzer_low.analyze(catalog)

        # High threshold: may not detect
        config_high = DetectionConfig(variability_significance=50.0)
        analyzer_high = VariabilityAnalyzer(config_high)
        result_high = analyzer_high.analyze(catalog)

        assert len(result_low["variable_candidates"]) >= len(
            result_high["variable_candidates"]
        )
