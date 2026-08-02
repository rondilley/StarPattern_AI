"""Tests for evaluation modules."""

import numpy as np
import pytest

from star_pattern.core.fits_handler import FITSImage
from star_pattern.evaluation.metrics import (
    PatternResult,
    anomaly_score_combined,
    detection_significance,
    diversity_score,
    novelty_score,
    signal_to_noise,
)
from star_pattern.evaluation.statistical import (
    anderson_darling_normality,
    bootstrap_confidence,
    ks_test_uniformity,
    multiple_comparison_correction,
    permutation_test,
)
from star_pattern.evaluation.synthetic import SyntheticInjector


class TestMetrics:
    def test_signal_to_noise(self):
        signal = np.array([100, 200, 150])
        snr = signal_to_noise(signal, 10.0)
        assert snr == 20.0

    def test_snr_zero_background(self):
        assert signal_to_noise(np.array([100]), 0.0) == 0.0

    def test_detection_significance(self):
        result = detection_significance(10, 3.0)
        assert result["sigma"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_detection_significance_extreme_stays_json_safe(self):
        """Extreme counts must not serialize as the invalid JSON token Infinity.

        Regression test: 1 - poisson.cdf() underflows to exactly 0.0 past
        roughly 8 sigma, and norm.isf(0.0) is +inf, which json.dumps writes
        as a bare Infinity that no strict parser accepts.
        """
        import json
        import math

        for observed in (50, 500, 10000):
            result = detection_significance(observed, 1.0)
            assert all(math.isfinite(v) for v in result.values())
            json.dumps(result, allow_nan=False)
        # Precision must survive past the 1 - cdf underflow point.
        assert detection_significance(50, 1.0)["sigma"] > 8.0

    def test_anomaly_score_combined(self):
        scores = {"classical": 0.5, "morphology": 0.8}
        combined = anomaly_score_combined(scores)
        assert 0.5 <= combined <= 0.8

    def test_anomaly_score_with_weights(self):
        scores = {"a": 1.0, "b": 0.0}
        weights = {"a": 0.8, "b": 0.2}
        combined = anomaly_score_combined(scores, weights)
        assert combined == pytest.approx(0.8)

    def test_novelty_score(self):
        features = np.array([10, 10, 10])
        ref = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        score = novelty_score(features, ref, method="euclidean")
        assert 0 <= score <= 1

    def test_diversity_score(self):
        # Diverse features
        features = np.array([[0, 0], [10, 10], [0, 10], [10, 0]])
        d = diversity_score(features)
        assert d > 0

    def test_pattern_result(self):
        result = PatternResult(
            region_ra=180.0,
            region_dec=45.0,
            detection_type="lens",
            anomaly_score=0.8,
            significance=0.7,
        )
        assert result.combined_score > 0
        d = result.to_dict()
        assert d["ra"] == 180.0


class TestStatistical:
    def test_bootstrap(self):
        data = np.random.default_rng(42).normal(5, 1, 100)
        result = bootstrap_confidence(data, np.mean)
        assert 4.5 < result["estimate"] < 5.5
        assert result["lower"] < result["upper"]

    def test_ks_uniformity(self):
        uniform = np.random.default_rng(42).uniform(0, 1, 100)
        result = ks_test_uniformity(uniform)
        assert result["p_value"] > 0.05  # Should pass

    def test_anderson_darling(self):
        normal = np.random.default_rng(42).normal(0, 1, 100)
        result = anderson_darling_normality(normal)
        assert result["is_normal"]

    def test_bonferroni_correction(self):
        p_values = [0.01, 0.03, 0.05]
        corrected = multiple_comparison_correction(p_values, "bonferroni")
        assert all(c >= p for c, p in zip(corrected, p_values))
        assert corrected[0] == pytest.approx(0.03)

    def test_fdr_correction(self):
        p_values = [0.01, 0.03, 0.05]
        corrected = multiple_comparison_correction(p_values, "fdr")
        assert all(0 <= c <= 1 for c in corrected)
        # 0.01*3/1=0.03, 0.03*3/2=0.045, 0.05*3/3=0.05; step-up running
        # minimum over the rank sequence leaves 0.045 in place.
        assert corrected == pytest.approx([0.03, 0.045, 0.05])

    def test_fdr_does_not_promote_noise(self):
        """A single strong detection must not drag noise down with it.

        Regression test: the previous implementation ran the step-up
        monotonicity pass over the INPUT sequence instead of the RANK
        sequence, which returned [0.004, 0.004, 0.004, 0.933] here and
        turned two pure-noise detections into 3-sigma discoveries.
        """
        corrected = multiple_comparison_correction([0.9, 0.5, 0.001, 0.7], "fdr")
        assert corrected == pytest.approx([0.9, 0.9, 0.004, 0.9])

    def test_fdr_matches_scipy_reference(self):
        """BH must agree with scipy.stats.false_discovery_control."""
        fdc = pytest.importorskip("scipy.stats", reason="scipy required").false_discovery_control
        rng = np.random.default_rng(20260801)
        for _ in range(100):
            n = int(rng.integers(1, 80))
            ps = rng.random(n)
            if n > 3:
                # Force ties and boundary values into the sample.
                ps[0] = ps[1]
                ps[2] = 0.0
                ps[3] = 1.0
            ours = multiple_comparison_correction(list(ps), "fdr")
            assert ours == pytest.approx(list(fdc(ps)), abs=1e-12)

    def test_fdr_preserves_rank_order(self):
        """Adjusted p-values never reorder the raw p-values, and never shrink."""
        rng = np.random.default_rng(7)
        ps = list(rng.random(40))
        corrected = multiple_comparison_correction(ps, "fdr")
        assert all(c >= p - 1e-12 for c, p in zip(corrected, ps))
        by_raw = sorted(range(len(ps)), key=lambda i: ps[i])
        adjusted_in_rank_order = [corrected[i] for i in by_raw]
        assert adjusted_in_rank_order == sorted(adjusted_in_rank_order)

    def test_fdr_empty_and_singleton(self):
        assert multiple_comparison_correction([], "fdr") == []
        assert multiple_comparison_correction([0.42], "fdr") == pytest.approx([0.42])

    def test_permutation_test(self):
        rng = np.random.default_rng(42)
        group1 = rng.normal(5, 1, 30)
        group2 = rng.normal(3, 1, 30)
        result = permutation_test(group1, group2, n_permutations=500, rng=rng)
        assert result["p_value"] < 0.05  # Groups are different


class TestSyntheticInjector:
    def test_inject_arc(self, synthetic_image: FITSImage):
        injector = SyntheticInjector()
        modified, metadata = injector.inject_arc(synthetic_image)
        assert metadata["type"] == "arc"
        assert modified.data.sum() > synthetic_image.data.sum()

    def test_inject_ring(self, synthetic_image: FITSImage):
        injector = SyntheticInjector()
        modified, metadata = injector.inject_ring(synthetic_image)
        assert metadata["type"] == "ring"

    def test_inject_overdensity(self, synthetic_image: FITSImage):
        injector = SyntheticInjector()
        modified, metadata = injector.inject_overdensity(synthetic_image)
        assert metadata["type"] == "overdensity"
        assert metadata["n_sources"] > 0

    def test_inject_random(self, synthetic_image: FITSImage):
        injector = SyntheticInjector()
        modified, metadata = injector.inject_random(synthetic_image)
        assert metadata["type"] in ("arc", "ring", "overdensity")
