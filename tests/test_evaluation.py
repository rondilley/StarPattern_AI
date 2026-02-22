"""Tests for evaluation modules."""

import numpy as np
import pytest

from star_pattern.evaluation.metrics import (
    signal_to_noise,
    detection_significance,
    anomaly_score_combined,
    novelty_score,
    diversity_score,
    PatternResult,
)
from star_pattern.evaluation.statistical import (
    bootstrap_confidence,
    ks_test_uniformity,
    anderson_darling_normality,
    multiple_comparison_correction,
    permutation_test,
)
from star_pattern.evaluation.synthetic import SyntheticInjector
from star_pattern.core.fits_handler import FITSImage


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
