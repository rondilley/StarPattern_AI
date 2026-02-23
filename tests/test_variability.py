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


def _make_stats_entry(
    stats: dict,
    source_id: str = "ztf_test",
    ra: float = 180.0,
    dec: float = 45.0,
    include_lightcurve: bool = False,
) -> CatalogEntry:
    """Create a CatalogEntry with ztf_stats (and optionally a light curve)."""
    properties = {"ztf_stats": stats}
    if include_lightcurve:
        # Add a real light curve so the LC path is taken
        properties["ztf_lightcurve"] = {
            "r": _make_lightcurve(scatter=0.5, error=0.03),
        }
        properties["ztf_n_epochs"] = 100
    else:
        properties["ztf_lightcurve"] = {}
        properties["ztf_n_epochs"] = 0
    return CatalogEntry(
        ra=ra, dec=dec, mag=18.0,
        source="ztf", source_id=source_id,
        properties=properties,
    )


class TestStatsBasedAnalysis:
    """Tests for the stats-based variability analysis fallback."""

    def test_analyze_from_ztf_stats(self):
        """Entries with ztf_stats but no light curves produce variable candidates."""
        entries = [
            _make_stats_entry({
                "magrms": 0.3, "medmagerr": 0.02,
                "medianabsdev": 0.2, "vonneumannratio": 1.2,
                "stetsonj": 1.5, "stetsonk": 0.8,
                "skewness": 0.1, "maxslope": 0.5,
                "maxmag": 18.5, "minmag": 17.5,
                "filtercode": "r", "ngoodobs": 100,
            }, source_id="ztf_001"),
            _make_stats_entry({
                "magrms": 0.1, "medmagerr": 0.02,
                "medianabsdev": 0.05, "vonneumannratio": 1.8,
                "stetsonj": 0.5, "stetsonk": 0.4,
                "skewness": 0.0, "maxslope": 0.1,
                "maxmag": 18.2, "minmag": 17.9,
                "filtercode": "r", "ngoodobs": 80,
            }, source_id="ztf_002"),
        ]
        catalog = StarCatalog(entries=entries, source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_analyzed"] == 2
        assert result["variability_score"] > 0
        assert len(result["variable_candidates"]) >= 1
        # No periodic candidates from stats
        assert result["periodic_candidates"] == []

    def test_stats_classification_agn_like(self):
        """High amplitude + low eta from stats -> agn_like."""
        entry = _make_stats_entry({
            "magrms": 0.5, "medmagerr": 0.02,
            "medianabsdev": 0.3, "vonneumannratio": 1.0,
            "stetsonj": None, "stetsonk": None,
            "skewness": 0.0, "maxslope": 0.3,
            "maxmag": 19.0, "minmag": 17.5,  # amplitude = 1.5
            "filtercode": "r", "ngoodobs": 50,
        })
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert len(result["variable_candidates"]) == 1
        assert result["variable_candidates"][0]["classification"] == "agn_like"

    def test_stats_high_maxslope_eruptive(self):
        """maxslope > 1.0 mag/day -> eruptive classification."""
        entry = _make_stats_entry({
            "magrms": 0.4, "medmagerr": 0.02,
            "medianabsdev": 0.2, "vonneumannratio": 1.8,
            "stetsonj": None, "stetsonk": None,
            "skewness": 0.0, "maxslope": 2.5,
            "maxmag": 18.5, "minmag": 18.0,  # amplitude = 0.5, not enough for agn
            "filtercode": "r", "ngoodobs": 60,
        })
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert len(result["variable_candidates"]) == 1
        assert result["variable_candidates"][0]["classification"] == "eruptive"
        assert result["variable_candidates"][0]["n_outbursts"] == 1

    def test_stats_non_variable(self):
        """Low magrms relative to medmagerr -> non_variable."""
        entry = _make_stats_entry({
            "magrms": 0.03, "medmagerr": 0.03,
            "medianabsdev": 0.02, "vonneumannratio": 2.0,
            "stetsonj": 0.1, "stetsonk": 0.1,
            "skewness": 0.0, "maxslope": 0.05,
            "maxmag": 18.05, "minmag": 17.95,
            "filtercode": "r", "ngoodobs": 100,
        })
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_analyzed"] == 1
        # chi2_reduced = (0.03/0.03)^2 = 1.0, below default threshold (3.0)
        assert len(result["variable_candidates"]) == 0

    def test_stats_and_lightcurves_prefers_lightcurves(self):
        """When both stats and light curves exist, light curve path is used."""
        entry = _make_stats_entry(
            {
                "magrms": 0.01, "medmagerr": 0.01,
                "medianabsdev": 0.01, "vonneumannratio": 2.0,
                "stetsonj": 0.0, "stetsonk": 0.0,
                "skewness": 0.0, "maxslope": 0.0,
                "maxmag": 18.01, "minmag": 17.99,
                "filtercode": "r", "ngoodobs": 100,
            },
            include_lightcurve=True,
        )
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        # The LC has scatter=0.5 so it should be detected as variable via
        # the light curve path (not the stats path which shows non-variable)
        assert result["n_analyzed"] == 1
        assert len(result["variable_candidates"]) >= 1

    def test_stats_insufficient_ngoodobs_skipped(self):
        """Sources with ngoodobs < min_epochs should be skipped."""
        entry = _make_stats_entry({
            "magrms": 0.5, "medmagerr": 0.02,
            "medianabsdev": 0.3, "vonneumannratio": 1.0,
            "stetsonj": None, "stetsonk": None,
            "skewness": 0.0, "maxslope": 0.3,
            "maxmag": 19.0, "minmag": 17.5,
            "filtercode": "r", "ngoodobs": 5,  # below default min_epochs=10
        })
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert result["n_analyzed"] == 0
        assert result["variability_score"] == 0.0

    def test_stats_score_range(self):
        """Stats-based variability_score should be in [0, 1]."""
        entry = _make_stats_entry({
            "magrms": 0.5, "medmagerr": 0.01,
            "medianabsdev": 0.3, "vonneumannratio": 0.5,
            "stetsonj": 3.0, "stetsonk": 1.5,
            "skewness": 0.5, "maxslope": 5.0,
            "maxmag": 20.0, "minmag": 16.0,
            "filtercode": "g", "ngoodobs": 200,
        })
        catalog = StarCatalog(entries=[entry], source="ztf")
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(catalog)

        assert 0 <= result["variability_score"] <= 1
