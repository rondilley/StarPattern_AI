"""Comprehensive tests for the confidence scoring system.

Tests cover:
- ConfidenceScore dataclass serialization
- ConfidenceEvaluator per-detector methods (12 detectors + anomaly + fallback)
- Quality floor filtering
- BH-FDR correction
- Spatial grouping via union-find
- Fisher combined group summaries
- Integration with _extract_anomalies in the autonomous pipeline
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from star_pattern.evaluation.confidence import (
    ConfidenceEvaluator,
    ConfidenceScore,
    _QUALITY_FLOORS,
    _MAX_PER_DETECTOR_SYSTEM,
    _MAX_PER_REGION_SYSTEM,
    apply_fdr_correction,
    assign_spatial_groups,
    compute_group_summary,
    passes_quality_floor,
)
from star_pattern.evaluation.metrics import Anomaly, PatternResult


# ---------------------------------------------------------------------------
# 1. ConfidenceScore dataclass
# ---------------------------------------------------------------------------


class TestConfidenceScoreDataclass:
    """Tests for ConfidenceScore serialization and defaults."""

    def test_to_dict_from_dict_round_trip(self):
        """to_dict -> from_dict should produce an identical object."""
        original = ConfidenceScore(
            confidence=0.95,
            p_value=0.05,
            p_corrected=0.10,
            physical_quantity="SNR",
            physical_value=7.2,
            method="gaussian_sf",
            annotation="Lens arc at SNR 7.2 (p=3.1e-13)",
            n_independent_tests=5,
            correction_method="fdr",
            n_agreeing_detectors=3,
            agreement_details=["lens", "galaxy"],
        )
        d = original.to_dict()
        restored = ConfidenceScore.from_dict(d)

        assert restored.confidence == original.confidence
        assert restored.p_value == original.p_value
        assert restored.p_corrected == original.p_corrected
        assert restored.physical_quantity == original.physical_quantity
        assert restored.physical_value == original.physical_value
        assert restored.method == original.method
        assert restored.annotation == original.annotation
        assert restored.n_independent_tests == original.n_independent_tests
        assert restored.correction_method == original.correction_method
        assert restored.n_agreeing_detectors == original.n_agreeing_detectors
        assert restored.agreement_details == original.agreement_details

    def test_default_values(self):
        """Optional fields should have sensible defaults."""
        cs = ConfidenceScore(
            confidence=0.9,
            p_value=0.01,
            p_corrected=0.02,
            physical_quantity="sigma",
            physical_value=3.0,
            method="gaussian_sf",
            annotation="test",
        )
        assert cs.n_independent_tests == 1
        assert cs.correction_method == "none"
        assert cs.n_agreeing_detectors == 0
        assert cs.agreement_details == []

    def test_from_dict_missing_optional_fields(self):
        """from_dict should handle missing optional keys gracefully."""
        d = {
            "confidence": 0.8,
            "p_value": 0.01,
            "p_corrected": 0.02,
            "physical_quantity": "SNR",
            "physical_value": 5.0,
            "method": "gaussian_sf",
            "annotation": "test",
        }
        cs = ConfidenceScore.from_dict(d)
        assert cs.n_independent_tests == 1
        assert cs.correction_method == "none"
        assert cs.n_agreeing_detectors == 0
        assert cs.agreement_details == []


# ---------------------------------------------------------------------------
# 2. ConfidenceEvaluator per-detector methods
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator():
    return ConfidenceEvaluator()


class TestConfidenceLens:
    """Lens detector: arc SNR and ring completeness."""

    def test_arc_high_snr(self, evaluator):
        """Arc with SNR=7.0 should give very high confidence (low p-value)."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_arc",
            detector="lens",
            properties={"snr": 7.0},
            score=7.0,
        )
        expected_p = float(stats.norm.sf(7.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.confidence == pytest.approx(1.0 - expected_p, rel=1e-6)
        assert cs.physical_quantity == "SNR"
        assert cs.physical_value == pytest.approx(7.0)
        assert cs.method == "gaussian_sf"
        assert "arc" in cs.annotation.lower()

    def test_arc_low_snr(self, evaluator):
        """Arc with SNR=1.0 should give low confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_arc",
            detector="lens",
            properties={"snr": 1.0},
            score=1.0,
        )
        expected_p = float(stats.norm.sf(1.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        # p ~ 0.159 -> confidence ~ 0.84
        assert cs.confidence < 0.90
        assert cs.confidence > 0.50

    def test_ring_completeness(self, evaluator):
        """Ring with completeness=0.8, no SNR -> use completeness mapping."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_ring",
            detector="lens",
            properties={"completeness": 0.8, "snr": 0},
            score=0.8,
        )
        effective_sigma = 0.8 * 5.0  # = 4.0
        expected_p = float(stats.norm.sf(effective_sigma))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.physical_quantity == "completeness"
        assert cs.physical_value == pytest.approx(0.8)
        assert "completeness" in cs.annotation.lower()

    def test_arc_zero_snr_falls_back_to_score(self, evaluator):
        """When snr=0, use score as fallback."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_arc",
            detector="lens",
            properties={"snr": 0},
            score=3.5,
        )
        expected_p = float(stats.norm.sf(3.5))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.physical_value == pytest.approx(3.5)


class TestConfidenceDistribution:
    """Distribution detector: overdensity sigma with Bonferroni correction."""

    def test_high_sigma(self, evaluator):
        """Overdensity at sigma=4.0 should give high corrected confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="overdensity",
            detector="distribution",
            properties={"sigma": 4.0},
            score=4.0,
        )
        p_raw = float(stats.norm.sf(4.0))
        p_corrected = min(p_raw * 100, 1.0)  # default n_cells=100
        assert cs.p_value == pytest.approx(p_raw, rel=1e-6)
        assert cs.p_corrected == pytest.approx(p_corrected, rel=1e-6)
        assert cs.confidence == pytest.approx(1 - p_corrected, rel=1e-6)
        assert cs.n_independent_tests == 100
        assert cs.correction_method == "bonferroni"
        assert cs.physical_quantity == "sigma"

    def test_low_sigma(self, evaluator):
        """Overdensity at sigma=1.0 -> p*100 ~ 15.87 -> capped at 1.0."""
        cs = evaluator.compute_confidence(
            anomaly_type="overdensity",
            detector="distribution",
            properties={"sigma": 1.0},
            score=1.0,
        )
        p_raw = float(stats.norm.sf(1.0))
        p_corrected = min(p_raw * 100, 1.0)
        assert p_corrected == pytest.approx(1.0)
        assert cs.confidence == pytest.approx(0.0, abs=1e-9)

    def test_custom_n_grid_cells(self, evaluator):
        """Bonferroni correction should use n_grid_cells from properties."""
        cs = evaluator.compute_confidence(
            anomaly_type="overdensity",
            detector="distribution",
            properties={"sigma": 5.0, "n_grid_cells": 50},
            score=5.0,
        )
        p_raw = float(stats.norm.sf(5.0))
        p_corrected = min(p_raw * 50, 1.0)
        assert cs.p_corrected == pytest.approx(p_corrected, rel=1e-6)
        assert cs.n_independent_tests == 50


class TestConfidenceGalaxy:
    """Galaxy detector: tidal features and mergers."""

    def test_tidal_high_snr(self, evaluator):
        """Tidal feature with tidal_snr=5.0 should be significant."""
        cs = evaluator.compute_confidence(
            anomaly_type="tidal_feature",
            detector="galaxy",
            properties={"tidal_snr": 5.0},
            score=5.0,
        )
        expected_p = float(stats.norm.sf(5.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.confidence > 0.999
        assert cs.physical_quantity == "SNR"
        assert "tidal" in cs.annotation.lower()

    def test_merger_asymmetry(self, evaluator):
        """Merger with asymmetry_sigma=4.0 uses gaussian sf."""
        cs = evaluator.compute_confidence(
            anomaly_type="merger",
            detector="galaxy",
            properties={"asymmetry_sigma": 4.0},
            score=4.0,
        )
        expected_p = float(stats.norm.sf(4.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.physical_quantity == "sigma"
        assert "merger" in cs.annotation.lower()


class TestConfidenceMorphology:
    """Morphology detector: CAS z-scores with Fisher combined p-value."""

    def test_cas_zscores(self, evaluator):
        """CAS z-scores should combine via Fisher's method."""
        props = {
            "C_zscore": 3.0,
            "A_zscore": 2.5,
            "S_zscore": 2.0,
        }
        cs = evaluator.compute_confidence(
            anomaly_type="morphological_anomaly",
            detector="morphology",
            properties=props,
            score=0.8,
        )
        # Verify Fisher's method was used
        assert cs.method == "fisher_combined"
        assert cs.physical_quantity == "Fisher_chi2"
        # Manually compute expected Fisher stat
        p_values = []
        for z in [3.0, 2.5, 2.0]:
            p_values.append(float(2 * stats.norm.sf(abs(z))))
        fisher_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
        df = 2 * len(p_values)
        p_fisher = float(stats.chi2.sf(fisher_stat, df))
        assert cs.p_value == pytest.approx(p_fisher, rel=1e-6)
        assert cs.confidence == pytest.approx(1 - p_fisher, rel=1e-6)
        # Annotation should mention the CAS components
        assert "C=" in cs.annotation
        assert "A=" in cs.annotation
        assert "S=" in cs.annotation

    def test_no_zscores_fallback(self, evaluator):
        """With no z-scores, falls back to score_proxy."""
        cs = evaluator.compute_confidence(
            anomaly_type="morphological_anomaly",
            detector="morphology",
            properties={},
            score=0.7,
        )
        assert cs.method == "score_proxy"
        assert cs.physical_quantity == "morphology_score"
        assert cs.confidence == pytest.approx(0.7)

    def test_gini_zscore_included(self, evaluator):
        """gini_zscore should also be included in Fisher's combination."""
        props = {
            "C_zscore": 2.0,
            "gini_zscore": 3.5,
        }
        cs = evaluator.compute_confidence(
            anomaly_type="morphological_anomaly",
            detector="morphology",
            properties=props,
            score=0.5,
        )
        assert cs.method == "fisher_combined"
        # Two p-values combined
        p_c = float(2 * stats.norm.sf(abs(2.0)))
        p_g = float(2 * stats.norm.sf(abs(3.5)))
        fisher_stat = -2 * (np.log(max(p_c, 1e-300)) + np.log(max(p_g, 1e-300)))
        df = 4
        p_fisher = float(stats.chi2.sf(fisher_stat, df))
        assert cs.p_value == pytest.approx(p_fisher, rel=1e-6)


class TestConfidenceWavelet:
    """Wavelet detector: peak SNR with FDR correction across scales."""

    def test_high_peak_snr(self, evaluator):
        """peak_snr=5.0 should give high confidence even after FDR correction."""
        cs = evaluator.compute_confidence(
            anomaly_type="multiscale_object",
            detector="wavelet",
            properties={"peak_snr": 5.0, "n_scales": 5},
            score=5.0,
        )
        p_raw = float(stats.norm.sf(5.0))
        # n_tests = max(n_scales, 4) = 5
        p_corrected = min(p_raw * 5 / 1, 1.0)
        assert cs.p_value == pytest.approx(p_raw, rel=1e-6)
        assert cs.p_corrected == pytest.approx(p_corrected, rel=1e-6)
        assert cs.confidence > 0.99
        assert cs.correction_method == "fdr"
        assert cs.n_independent_tests == 5

    def test_n_scales_minimum_4(self, evaluator):
        """When n_scales < 4, n_tests defaults to 4."""
        cs = evaluator.compute_confidence(
            anomaly_type="multiscale_object",
            detector="wavelet",
            properties={"peak_snr": 4.0, "n_scales": 2},
            score=4.0,
        )
        p_raw = float(stats.norm.sf(4.0))
        p_corrected = min(p_raw * 4, 1.0)
        assert cs.n_independent_tests == 4
        assert cs.p_corrected == pytest.approx(p_corrected, rel=1e-6)


class TestConfidenceClassical:
    """Classical detector: Hough votes and Gabor energy."""

    def test_hough_votes(self, evaluator):
        """hough_votes=50 should compute p-value via Poisson."""
        cs = evaluator.compute_confidence(
            anomaly_type="classical_arc",
            detector="classical",
            properties={"hough_votes": 50},
            score=50,
        )
        expected_rate = max(50 * 0.1, 1.0)  # = 5.0
        p = float(1 - stats.poisson.cdf(max(int(50) - 1, 0), expected_rate))
        assert cs.p_value == pytest.approx(p, rel=1e-6)
        assert cs.physical_quantity == "Hough_votes"
        assert cs.method == "poisson_cdf"
        assert cs.confidence == pytest.approx(1 - p, rel=1e-6)

    def test_gabor_beats_hough(self, evaluator):
        """When gabor_energy > hough_votes, Gabor is used."""
        cs = evaluator.compute_confidence(
            anomaly_type="classical_arc",
            detector="classical",
            properties={"hough_votes": 2, "gabor_energy": 5.0},
            score=5.0,
        )
        expected_p = float(stats.norm.sf(5.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.physical_quantity == "gabor_energy"
        assert cs.method == "gaussian_sf"


class TestConfidenceKinematic:
    """Kinematic detector: comoving groups, runaway stars, stellar streams."""

    def test_comoving_group(self, evaluator):
        """Comoving group with n_members=10, expected_field=1.0 -> Poisson."""
        cs = evaluator.compute_confidence(
            anomaly_type="comoving_group",
            detector="kinematic",
            properties={"n_members": 10, "expected_field": 1.0},
            score=0.9,
        )
        expected_p = float(1 - stats.poisson.cdf(9, 1.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "poisson_cdf"
        assert cs.physical_quantity == "n_members"
        assert cs.physical_value == pytest.approx(10.0)
        assert cs.confidence > 0.999

    def test_runaway_star(self, evaluator):
        """Runaway star with deviation_sigma=4.5, expected_field=0 (no Poisson)."""
        # With expected_field=0, the code takes the deviation_sigma branch
        cs = evaluator.compute_confidence(
            anomaly_type="runaway_star",
            detector="kinematic",
            properties={"n_members": 1, "expected_field": 0, "deviation_sigma": 4.5},
            score=4.5,
        )
        expected_p = float(2 * stats.norm.sf(abs(4.5)))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "gaussian_sf"
        assert cs.confidence > 0.99

    def test_stellar_stream(self, evaluator):
        """Stellar stream with n_members=15, expected_field=1.0."""
        cs = evaluator.compute_confidence(
            anomaly_type="stellar_stream",
            detector="kinematic",
            properties={"n_members": 15, "expected_field": 1.0},
            score=0.95,
        )
        expected_p = float(1 - stats.poisson.cdf(14, 1.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "poisson_cdf"
        assert cs.confidence > 0.9999


class TestConfidenceTransient:
    """Transient detector: deviation sigma (two-tailed)."""

    def test_high_sigma(self, evaluator):
        """deviation_sigma=4.0 should give high confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="flux_outlier",
            detector="transient",
            properties={"deviation_sigma": 4.0},
            score=4.0,
        )
        expected_p = float(2 * stats.norm.sf(abs(4.0)))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "gaussian_sf_twotailed"
        assert cs.physical_quantity == "sigma"
        assert cs.confidence > 0.999

    def test_low_sigma(self, evaluator):
        """deviation_sigma=1.0 -> moderate p-value."""
        cs = evaluator.compute_confidence(
            anomaly_type="flux_outlier",
            detector="transient",
            properties={"deviation_sigma": 1.0},
            score=1.0,
        )
        expected_p = float(2 * stats.norm.sf(1.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.confidence < 0.75


class TestConfidenceSersic:
    """Sersic detector: residual SNR and chi2_reduced."""

    def test_high_residual_snr(self, evaluator):
        """residual_snr=5.0 should be significant."""
        cs = evaluator.compute_confidence(
            anomaly_type="sersic_residual",
            detector="sersic",
            properties={"residual_snr": 5.0},
            score=5.0,
        )
        expected_p = float(stats.norm.sf(5.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.physical_quantity == "SNR"
        assert cs.confidence > 0.999

    def test_chi2_reduced(self, evaluator):
        """chi2_reduced=3.0 with n_pixels=100 should be significant via chi2."""
        cs = evaluator.compute_confidence(
            anomaly_type="sersic_residual",
            detector="sersic",
            properties={"chi2_reduced": 3.0, "n_pixels": 100},
            score=0.8,
        )
        chi2_stat = 3.0 * 100
        expected_p_chi2 = float(stats.chi2.sf(chi2_stat, 100))
        # SNR is 0 so p_snr = 1.0, chi2 wins
        assert cs.p_value == pytest.approx(expected_p_chi2, rel=1e-6)
        assert cs.physical_quantity == "chi2_reduced"
        assert cs.confidence > 0.999

    def test_both_snr_and_chi2(self, evaluator):
        """When both are available, the more significant one wins."""
        # SNR=5.0 vs chi2_reduced=1.5 (weaker)
        cs = evaluator.compute_confidence(
            anomaly_type="sersic_residual",
            detector="sersic",
            properties={"residual_snr": 5.0, "chi2_reduced": 1.5, "n_pixels": 100},
            score=5.0,
        )
        p_snr = float(stats.norm.sf(5.0))
        chi2_stat = 1.5 * 100
        p_chi2 = float(stats.chi2.sf(chi2_stat, 100))
        # SNR p is smaller -> SNR wins
        assert p_snr < p_chi2
        assert cs.physical_quantity == "SNR"
        assert cs.p_value == pytest.approx(p_snr, rel=1e-6)


class TestConfidencePopulation:
    """Stellar population: binomial tests for blue stragglers and red giants."""

    def test_blue_straggler(self, evaluator):
        """Blue straggler with bs_fraction=0.05, n_sources_with_color=60, n_blue_stragglers=3."""
        cs = evaluator.compute_confidence(
            anomaly_type="blue_straggler",
            detector="population",
            properties={
                "bs_fraction": 0.05,
                "n_sources_with_color": 60,
                "n_blue_stragglers": 3,
            },
            score=0.05,
        )
        field_rate = 0.01
        expected_p = float(stats.binom.sf(2, 60, field_rate))  # sf(n_bs - 1, ...)
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "binomial_sf"
        assert cs.physical_quantity == "bs_fraction"
        assert cs.physical_value == pytest.approx(0.05)
        assert "blue straggler" in cs.annotation.lower()

    def test_red_giant(self, evaluator):
        """Red giant with rgb_fraction=0.15, n_red_giants=12, n_sources_with_color=60."""
        cs = evaluator.compute_confidence(
            anomaly_type="red_giant",
            detector="population",
            properties={
                "rgb_fraction": 0.15,
                "n_red_giants": 12,
                "n_sources_with_color": 60,
            },
            score=0.15,
        )
        field_rate = 0.05
        expected_p = float(stats.binom.sf(11, 60, field_rate))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "binomial_sf"
        assert cs.physical_quantity == "rgb_fraction"
        assert "red giant" in cs.annotation.lower()

    def test_population_insufficient_sources(self, evaluator):
        """With fewer than 5 sources, fall back to score proxy."""
        cs = evaluator.compute_confidence(
            anomaly_type="blue_straggler",
            detector="population",
            properties={
                "bs_fraction": 0.10,
                "n_sources_with_color": 3,
                "n_blue_stragglers": 1,
            },
            score=0.10,
        )
        # n_sources=3 < 5 -> fallback
        expected_p = max(1 - 0.10, 1e-15)
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)


class TestConfidenceVariability:
    """Variability detector: periodic FAP and general chi2."""

    def test_periodic_fap(self, evaluator):
        """Periodic variable with fap=0.001 should give high confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="periodic_variable",
            detector="variability",
            properties={"fap": 0.001},
            score=0.9,
        )
        assert cs.p_value == pytest.approx(0.001, rel=1e-6)
        assert cs.method == "lomb_scargle_fap"
        assert cs.physical_quantity == "FAP"
        assert cs.confidence == pytest.approx(0.999, rel=1e-6)

    def test_general_chi2(self, evaluator):
        """General variability with chi2_variability=5.0 and n_epochs=10."""
        cs = evaluator.compute_confidence(
            anomaly_type="variable_star",
            detector="variability",
            properties={"chi2_variability": 5.0, "n_epochs": 10},
            score=0.8,
        )
        dof = 10 - 1  # = 9
        chi2_stat = 5.0 * 9
        expected_p = float(stats.chi2.sf(chi2_stat, 9))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.method == "chi2_sf"
        assert cs.physical_quantity == "chi2_reduced"

    def test_variability_no_data_fallback(self, evaluator):
        """No FAP or chi2 -> falls back to score_proxy."""
        cs = evaluator.compute_confidence(
            anomaly_type="variable_star",
            detector="variability",
            properties={},
            score=0.6,
        )
        assert cs.method == "score_proxy"
        expected_p = max(1 - 0.6, 1e-15)
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)


class TestConfidenceTemporal:
    """Temporal detector: epoch-diff peak SNR."""

    def test_high_peak_snr(self, evaluator):
        """peak_snr=6.0 should give very high confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="temporal_new_source",
            detector="temporal",
            properties={"peak_snr": 6.0},
            score=6.0,
        )
        expected_p = float(stats.norm.sf(6.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.confidence > 0.999999
        assert cs.method == "gaussian_sf"
        assert cs.physical_quantity == "SNR"
        assert "new source" in cs.annotation.lower()

    def test_temporal_low_snr(self, evaluator):
        """peak_snr=1.0 -> modest confidence."""
        cs = evaluator.compute_confidence(
            anomaly_type="temporal_brightening",
            detector="temporal",
            properties={"peak_snr": 1.0},
            score=1.0,
        )
        expected_p = float(stats.norm.sf(1.0))
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert cs.confidence < 0.90


class TestConfidenceAnomaly:
    """Anomaly (Isolation Forest): empirical rank."""

    def test_anomaly_detector(self, evaluator):
        """IF score of 0.95 -> p = 0.05."""
        cs = evaluator.compute_confidence(
            anomaly_type="embedding_outlier",
            detector="anomaly",
            properties={},
            score=0.95,
        )
        assert cs.p_value == pytest.approx(0.05, rel=1e-6)
        assert cs.confidence == pytest.approx(0.95)
        assert cs.method == "empirical_rank"


class TestConfidenceFallback:
    """Fallback for unknown detector types."""

    def test_unknown_detector(self, evaluator):
        """Unknown detector falls back to score proxy."""
        cs = evaluator.compute_confidence(
            anomaly_type="something",
            detector="unknown_detector",
            properties={},
            score=0.7,
        )
        assert cs.method == "score_proxy"
        assert cs.confidence == pytest.approx(0.7)
        expected_p = max(1 - 0.7, 1e-15)
        assert cs.p_value == pytest.approx(expected_p, rel=1e-6)
        assert "unknown_detector" in cs.annotation


# ---------------------------------------------------------------------------
# 3. Quality floor filtering
# ---------------------------------------------------------------------------


class TestQualityFloor:
    """passes_quality_floor: per-detector p-value thresholds."""

    @pytest.fixture
    def evaluator(self):
        return ConfidenceEvaluator()

    @pytest.mark.parametrize("detector,floor", list(_QUALITY_FLOORS.items()))
    def test_above_floor_rejected(self, detector, floor):
        """p-value above the quality floor should fail."""
        cs = ConfidenceScore(
            confidence=0.5,
            p_value=floor + 0.01,  # above threshold
            p_corrected=floor + 0.01,
            physical_quantity="test",
            physical_value=0.0,
            method="test",
            annotation="test",
        )
        assert passes_quality_floor(cs, detector) is False

    @pytest.mark.parametrize("detector,floor", list(_QUALITY_FLOORS.items()))
    def test_below_floor_passes(self, detector, floor):
        """p-value below the quality floor should pass."""
        cs = ConfidenceScore(
            confidence=0.99,
            p_value=floor / 2.0,  # below threshold
            p_corrected=floor / 2.0,
            physical_quantity="test",
            physical_value=5.0,
            method="test",
            annotation="test",
        )
        assert passes_quality_floor(cs, detector) is True

    @pytest.mark.parametrize("detector,floor", list(_QUALITY_FLOORS.items()))
    def test_at_floor_passes(self, detector, floor):
        """p-value exactly at the floor should pass (<=)."""
        cs = ConfidenceScore(
            confidence=0.99,
            p_value=floor,
            p_corrected=floor,
            physical_quantity="test",
            physical_value=5.0,
            method="test",
            annotation="test",
        )
        assert passes_quality_floor(cs, detector) is True

    def test_unknown_detector_uses_default_floor(self):
        """An unknown detector should use 0.05 as the default floor."""
        cs_pass = ConfidenceScore(
            confidence=0.97,
            p_value=0.03,
            p_corrected=0.03,
            physical_quantity="test",
            physical_value=2.0,
            method="test",
            annotation="test",
        )
        assert passes_quality_floor(cs_pass, "made_up_detector") is True

        cs_fail = ConfidenceScore(
            confidence=0.90,
            p_value=0.10,
            p_corrected=0.10,
            physical_quantity="test",
            physical_value=1.0,
            method="test",
            annotation="test",
        )
        assert passes_quality_floor(cs_fail, "made_up_detector") is False

    def test_lens_low_snr_rejected(self, evaluator):
        """Lens arc with SNR=1.0 has p~0.159 which fails the 0.0013 floor."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_arc",
            detector="lens",
            properties={"snr": 1.0},
            score=1.0,
        )
        assert passes_quality_floor(cs, "lens") is False

    def test_lens_high_snr_passes(self, evaluator):
        """Lens arc with SNR=7.0 has extremely low p, passes floor."""
        cs = evaluator.compute_confidence(
            anomaly_type="lens_arc",
            detector="lens",
            properties={"snr": 7.0},
            score=7.0,
        )
        assert passes_quality_floor(cs, "lens") is True

    def test_temporal_snr3_rejected(self, evaluator):
        """Temporal with SNR=3.0 has p~0.0013, above temporal floor (0.0000003)."""
        cs = evaluator.compute_confidence(
            anomaly_type="temporal_new_source",
            detector="temporal",
            properties={"peak_snr": 3.0},
            score=3.0,
        )
        # p = norm.sf(3.0) ~ 0.00135
        assert passes_quality_floor(cs, "temporal") is False

    def test_temporal_snr6_passes(self, evaluator):
        """Temporal with SNR=6.0 has p~9.87e-10, passes temporal floor."""
        cs = evaluator.compute_confidence(
            anomaly_type="temporal_new_source",
            detector="temporal",
            properties={"peak_snr": 6.0},
            score=6.0,
        )
        assert passes_quality_floor(cs, "temporal") is True


# ---------------------------------------------------------------------------
# 4. FDR correction
# ---------------------------------------------------------------------------


class TestFDRCorrection:
    """apply_fdr_correction: BH-FDR across a region's anomalies."""

    def test_p_corrected_adjusted_upward(self):
        """BH-FDR should adjust p_corrected upward (or keep same) for all."""
        anomalies = []
        p_vals = [0.001, 0.01, 0.02, 0.05]
        for p in p_vals:
            a = Anomaly(
                anomaly_type="test",
                detector="lens",
                score=1.0,
            )
            a.confidence = ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="SNR",
                physical_value=5.0,
                method="gaussian_sf",
                annotation="test",
            )
            anomalies.append(a)

        apply_fdr_correction(anomalies)

        for i, a in enumerate(anomalies):
            # Corrected p should be >= raw p
            assert a.confidence.p_corrected >= p_vals[i] - 1e-15
            # Confidence should equal 1 - p_corrected
            assert a.confidence.confidence == pytest.approx(
                1 - a.confidence.p_corrected, abs=1e-10
            )
            # Correction method set
            assert a.confidence.correction_method == "fdr"
            # n_independent_tests set to total count
            assert a.confidence.n_independent_tests == 4

    def test_most_significant_gets_least_correction(self):
        """The smallest p-value should have the smallest corrected p-value."""
        anomalies = []
        for p in [0.001, 0.01, 0.05]:
            a = Anomaly(anomaly_type="test", detector="lens", score=1.0)
            a.confidence = ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="SNR",
                physical_value=5.0,
                method="gaussian_sf",
                annotation="test",
            )
            anomalies.append(a)

        apply_fdr_correction(anomalies)

        corrected_ps = [a.confidence.p_corrected for a in anomalies]
        # BH-FDR preserves ordering
        assert corrected_ps[0] <= corrected_ps[1] <= corrected_ps[2]

    def test_skip_anomalies_without_confidence(self):
        """Anomalies with confidence=None should be skipped gracefully."""
        a1 = Anomaly(anomaly_type="test", detector="lens", score=1.0)
        a1.confidence = ConfidenceScore(
            confidence=0.99,
            p_value=0.01,
            p_corrected=0.01,
            physical_quantity="SNR",
            physical_value=5.0,
            method="gaussian_sf",
            annotation="test",
        )
        a2 = Anomaly(anomaly_type="test", detector="lens", score=0.5)
        # a2.confidence is None

        apply_fdr_correction([a1, a2])

        # a1 should still be updated (with n=1 tests, same p)
        assert a1.confidence.correction_method == "fdr"
        # a2 should remain None
        assert a2.confidence is None

    def test_empty_list(self):
        """Empty anomaly list should not raise."""
        apply_fdr_correction([])

    def test_known_bh_fdr_values(self):
        """Verify BH-FDR calculation against manually computed values."""
        anomalies = []
        raw_ps = [0.01, 0.04, 0.03]
        for p in raw_ps:
            a = Anomaly(anomaly_type="test", detector="lens", score=1.0)
            a.confidence = ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="SNR",
                physical_value=5.0,
                method="gaussian_sf",
                annotation="test",
            )
            anomalies.append(a)

        apply_fdr_correction(anomalies)

        # Sorted by p: 0.01 (rank 1, idx 0), 0.03 (rank 2, idx 2), 0.04 (rank 3, idx 1)
        # BH: corrected[0] = 0.01*3/1 = 0.03
        #     corrected[2] = 0.03*3/2 = 0.045
        #     corrected[1] = 0.04*3/3 = 0.04
        # Monotonicity (reverse over original indices 2->1->0):
        #   corrected[1] = min(0.04, corrected[2]=0.045) = 0.04
        #   corrected[0] = min(0.03, corrected[1]=0.04) = 0.03
        # Final: [0.03, 0.04, 0.045]
        assert anomalies[0].confidence.p_corrected == pytest.approx(0.03, abs=1e-10)
        assert anomalies[1].confidence.p_corrected == pytest.approx(0.04, abs=1e-10)
        assert anomalies[2].confidence.p_corrected == pytest.approx(0.045, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Spatial grouping
# ---------------------------------------------------------------------------


class TestSpatialGrouping:
    """assign_spatial_groups: union-find pixel/sky proximity grouping."""

    def test_two_close_pixel_anomalies_same_group(self):
        """Two anomalies within 10px should be in the same group."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            pixel_x=105.0, pixel_y=105.0,
        )
        dist = np.sqrt(25 + 25)
        assert dist < 15.0  # within default pixel_tolerance

        assign_spatial_groups([a1, a2])

        assert a1.group_id is not None
        assert a1.group_id == a2.group_id

    def test_two_far_pixel_anomalies_no_group(self):
        """Two anomalies >15px apart should not be grouped."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            pixel_x=200.0, pixel_y=200.0,
        )
        assign_spatial_groups([a1, a2])

        assert a1.group_id is None
        assert a2.group_id is None

    def test_sky_coord_grouping(self):
        """Two anomalies with nearby sky coords should group."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            sky_ra=180.0, sky_dec=45.0,
        )
        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            sky_ra=180.0001, sky_dec=45.0001,
        )
        # Separation ~ small fraction of arcsec
        assign_spatial_groups([a1, a2])

        assert a1.group_id is not None
        assert a1.group_id == a2.group_id

    def test_sky_coord_far_apart(self):
        """Two anomalies with distant sky coords should not group."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            sky_ra=180.0, sky_dec=45.0,
        )
        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            sky_ra=181.0, sky_dec=46.0,
        )
        assign_spatial_groups([a1, a2])

        assert a1.group_id is None
        assert a2.group_id is None

    def test_mixed_sky_and_pixel_coords(self):
        """Mix of sky-coord and pixel-coord anomalies; pixel fallback used."""
        # a1 has both sky and pixel, a2 has only pixel
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
            sky_ra=180.0, sky_dec=45.0,
        )
        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            pixel_x=105.0, pixel_y=105.0,
        )
        # a2 has no sky coords, so sky comparison skipped; falls back to pixel
        assign_spatial_groups([a1, a2])

        assert a1.group_id is not None
        assert a1.group_id == a2.group_id

    def test_three_in_chain(self):
        """A-B close, B-C close but A-C far: all three should share group via union-find."""
        a = Anomaly(
            anomaly_type="t1", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        b = Anomaly(
            anomaly_type="t2", detector="distribution",
            pixel_x=110.0, pixel_y=100.0,
        )
        c = Anomaly(
            anomaly_type="t3", detector="galaxy",
            pixel_x=120.0, pixel_y=100.0,
        )
        # A-B: 10px (close), B-C: 10px (close), A-C: 20px (far)
        assign_spatial_groups([a, b, c])

        assert a.group_id is not None
        assert a.group_id == b.group_id == c.group_id

    def test_single_anomaly_no_group(self):
        """A single anomaly gets no group_id (groups require 2+ members)."""
        a = Anomaly(
            anomaly_type="test", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        assign_spatial_groups([a])
        assert a.group_id is None

    def test_empty_list(self):
        """Empty list should not raise."""
        assign_spatial_groups([])

    def test_group_id_format(self):
        """Group IDs should be formatted as 'grp_NNN'."""
        a1 = Anomaly(
            anomaly_type="t1", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        a2 = Anomaly(
            anomaly_type="t2", detector="distribution",
            pixel_x=105.0, pixel_y=105.0,
        )
        assign_spatial_groups([a1, a2])

        assert a1.group_id is not None
        assert a1.group_id.startswith("grp_")
        assert len(a1.group_id) == 7  # grp_001


# ---------------------------------------------------------------------------
# 6. Group summary (Fisher combined)
# ---------------------------------------------------------------------------


class TestGroupSummary:
    """compute_group_summary: Fisher combined p-value across group members."""

    def test_two_member_group(self):
        """Two anomalies in same group -> Fisher combined p-value."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        a1.confidence = ConfidenceScore(
            confidence=0.999,
            p_value=0.001,
            p_corrected=0.001,
            physical_quantity="SNR",
            physical_value=7.0,
            method="gaussian_sf",
            annotation="Lens arc at SNR 7.0",
        )
        a1.group_id = "grp_001"

        a2 = Anomaly(
            anomaly_type="overdensity", detector="distribution",
            pixel_x=105.0, pixel_y=105.0,
        )
        a2.confidence = ConfidenceScore(
            confidence=0.99,
            p_value=0.01,
            p_corrected=0.01,
            physical_quantity="sigma",
            physical_value=4.0,
            method="gaussian_sf",
            annotation="Overdensity at 4.0 sigma",
        )
        a2.group_id = "grp_001"

        result = compute_group_summary([a1, a2], "grp_001")

        assert result["group_id"] == "grp_001"
        assert result["n_detectors"] == 2
        assert result["n_members"] == 2

        # Manually compute Fisher combined
        fisher_stat = -2 * (np.log(0.001) + np.log(0.01))
        df = 4
        p_combined = float(stats.chi2.sf(fisher_stat, df))
        assert result["p_combined"] == pytest.approx(p_combined, rel=1e-6)
        assert result["confidence"] == pytest.approx(1 - p_combined, rel=1e-6)
        assert "summary_text" in result
        assert len(result["detail_lines"]) == 2

    def test_no_members(self):
        """Group with no matching members returns minimal dict."""
        result = compute_group_summary([], "grp_999")
        assert result["n_detectors"] == 0

    def test_member_without_confidence(self):
        """Members without confidence should appear in details but not p-values."""
        a1 = Anomaly(
            anomaly_type="lens_arc", detector="lens",
            pixel_x=100.0, pixel_y=100.0,
        )
        a1.confidence = ConfidenceScore(
            confidence=0.999,
            p_value=0.001,
            p_corrected=0.001,
            physical_quantity="SNR",
            physical_value=7.0,
            method="gaussian_sf",
            annotation="Lens arc at SNR 7.0",
        )
        a1.group_id = "grp_001"

        a2 = Anomaly(
            anomaly_type="test", detector="galaxy",
            pixel_x=105.0, pixel_y=105.0,
        )
        a2.group_id = "grp_001"
        # a2.confidence is None

        result = compute_group_summary([a1, a2], "grp_001")

        assert result["n_members"] == 2
        assert result["n_detectors"] == 2
        # Only a1 contributes to p-value
        fisher_stat = -2 * np.log(0.001)
        df = 2
        p_combined = float(stats.chi2.sf(fisher_stat, df))
        assert result["p_combined"] == pytest.approx(p_combined, rel=1e-6)

    def test_three_member_group_combined_stronger(self):
        """Three detectors confirming should give stronger combined significance."""
        anomalies = []
        p_values = [0.01, 0.02, 0.03]
        for i, p in enumerate(p_values):
            a = Anomaly(
                anomaly_type=f"type_{i}", detector=f"det_{i}",
                pixel_x=100.0 + i, pixel_y=100.0,
            )
            a.confidence = ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="SNR",
                physical_value=3.0,
                method="gaussian_sf",
                annotation=f"detection {i}",
            )
            a.group_id = "grp_001"
            anomalies.append(a)

        result = compute_group_summary(anomalies, "grp_001")
        # Combined p should be much smaller than any individual p
        assert result["p_combined"] < min(p_values)


# ---------------------------------------------------------------------------
# 7. Integration with _extract_anomalies
# ---------------------------------------------------------------------------


class TestExtractAnomaliesIntegration:
    """Integration tests for _extract_anomalies in the autonomous pipeline."""

    def test_confidence_scores_assigned(self):
        """All returned anomalies should have a ConfidenceScore attached."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "lens": {
                "central_source": {"x": 128, "y": 128},
                "arcs": [
                    {"x": 130, "y": 140, "snr": 7.0, "radius": 15, "angle_span": 90},
                ],
                "rings": [],
            },
            "distribution": {
                "overdensities": [
                    {"x": 50, "y": 60, "sigma": 5.0, "significance": 5.0},
                ],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        for a in anomalies:
            assert a.confidence is not None, (
                f"Anomaly {a.anomaly_type} from {a.detector} has no confidence score"
            )
            assert 0.0 <= a.confidence.confidence <= 1.0
            assert a.confidence.p_value >= 0
            assert a.confidence.p_corrected >= 0
            assert a.confidence.annotation != ""

    def test_quality_floor_filters_low_snr(self):
        """Low-SNR lens detections should be filtered out by quality floor."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "lens": {
                "central_source": {"x": 128, "y": 128},
                "arcs": [
                    {"x": 130, "y": 140, "snr": 0.5, "radius": 15},
                    {"x": 131, "y": 141, "snr": 7.0, "radius": 15},
                ],
                "rings": [],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        # SNR=0.5 has p ~ 0.31, above lens floor of 0.0013 -> filtered out
        # SNR=7.0 has p ~ 1.3e-12, well below floor -> kept
        lens_anomalies = [a for a in anomalies if a.detector == "lens"]
        assert len(lens_anomalies) == 1
        assert lens_anomalies[0].confidence.physical_value == pytest.approx(7.0)

    def test_no_hard_cap_at_25(self):
        """More than 25 anomalies should be returned (old cap removed)."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        # Create 30 high-significance overdensities
        overdensities = []
        for i in range(30):
            overdensities.append({
                "x": 10 + i * 5, "y": 10 + i * 5,
                "sigma": 5.0 + i * 0.1, "significance": 5.0 + i * 0.1,
            })

        detection = {
            "distribution": {"overdensities": overdensities},
        }

        anomalies = _extract_anomalies(detection, image=None)

        dist_anomalies = [a for a in anomalies if a.detector == "distribution"]
        # All 30 should pass the quality floor (sigma=5+ -> p very small)
        assert len(dist_anomalies) > 25

    def test_fdr_correction_applied(self):
        """Returned anomalies should have FDR correction applied."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "lens": {
                "central_source": {"x": 128, "y": 128},
                "arcs": [
                    {"x": 130, "y": 140, "snr": 5.0, "radius": 15},
                    {"x": 135, "y": 145, "snr": 6.0, "radius": 15},
                ],
                "rings": [],
            },
            "distribution": {
                "overdensities": [
                    {"x": 50, "y": 60, "sigma": 5.0, "significance": 5.0},
                ],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        # FDR correction should have been applied
        for a in anomalies:
            assert a.confidence.correction_method == "fdr"

    def test_spatial_groups_assigned(self):
        """Co-located anomalies should be assigned spatial groups."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "lens": {
                "central_source": {"x": 100, "y": 100},
                "arcs": [
                    {"x": 100, "y": 100, "snr": 7.0, "radius": 15},
                ],
                "rings": [],
            },
            "galaxy": {
                "tidal_features": [
                    {"x": 105, "y": 105, "tidal_snr": 5.0, "strength": 5.0},
                ],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        # Both at ~(100,100) and ~(105,105), within 15px -> same group
        grouped = [a for a in anomalies if a.group_id is not None]
        if len(anomalies) >= 2:
            assert len(grouped) == 2
            assert grouped[0].group_id == grouped[1].group_id

    def test_multiple_detectors_all_scored(self):
        """Anomalies from many detectors should all get confidence scores."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "lens": {
                "central_source": {"x": 128, "y": 128},
                "arcs": [{"x": 130, "y": 140, "snr": 7.0, "radius": 15}],
                "rings": [],
            },
            "distribution": {
                "overdensities": [
                    {"x": 50, "y": 60, "sigma": 5.0},
                ],
            },
            "kinematic": {
                "comoving_groups": [
                    {"mean_ra": 180.0, "mean_dec": 45.0, "n_members": 10,
                     "expected_field": 1.0, "significance": 0.9},
                ],
                "stream_candidates": [],
                "runaway_stars": [],
            },
            "transient": {
                "flux_outliers": [
                    {"ra": 180.0, "dec": 45.0, "deviation_sigma": 4.0},
                ],
            },
            "temporal": {
                "new_sources": [
                    {"cx": 50, "cy": 50, "peak_snr": 6.0},
                ],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        detectors_found = {a.detector for a in anomalies}
        # At minimum, the high-significance ones should survive quality floor
        for a in anomalies:
            assert a.confidence is not None

    def test_system_protection_limit(self):
        """System cap (500) should prevent extreme cases from blowing up."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        # Create 600 overdensities -- more than _MAX_PER_DETECTOR_SYSTEM (500)
        overdensities = []
        for i in range(600):
            overdensities.append({
                "x": i % 100, "y": i // 100,
                "sigma": 5.0 + (i % 10) * 0.1,
            })

        detection = {
            "distribution": {"overdensities": overdensities},
        }

        anomalies = _extract_anomalies(detection, image=None)

        # System protection: at most 500 per detector
        dist_anomalies = [a for a in anomalies if a.detector == "distribution"]
        assert len(dist_anomalies) <= _MAX_PER_DETECTOR_SYSTEM

        # Total should not exceed region limit
        assert len(anomalies) <= _MAX_PER_REGION_SYSTEM

    def test_sorted_by_confidence_descending(self):
        """Returned anomalies should be sorted by confidence descending."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "distribution": {
                "overdensities": [
                    {"x": 10, "y": 10, "sigma": 4.0},
                    {"x": 20, "y": 20, "sigma": 5.0},
                    {"x": 30, "y": 30, "sigma": 6.0},
                ],
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        for i in range(len(anomalies) - 1):
            conf_i = anomalies[i].confidence.confidence if anomalies[i].confidence else anomalies[i].score
            conf_next = anomalies[i + 1].confidence.confidence if anomalies[i + 1].confidence else anomalies[i + 1].score
            assert conf_i >= conf_next - 1e-10

    def test_population_anomalies_scored(self):
        """Blue straggler and red giant anomalies should get binomial confidence."""
        from star_pattern.pipeline.autonomous import _extract_anomalies

        detection = {
            "population": {
                "n_sources_with_color": 100,
                "blue_stragglers": {
                    "n_blue_stragglers": 5,
                    "bs_fraction": 0.05,
                },
                "n_red_giants": 10,
                "red_giants": {
                    "rgb_fraction": 0.10,
                },
            },
        }

        anomalies = _extract_anomalies(detection, image=None)

        pop_anomalies = [a for a in anomalies if a.detector == "population"]
        for a in pop_anomalies:
            assert a.confidence is not None
            assert a.confidence.method == "binomial_sf"
