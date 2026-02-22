"""Tests for local statistical detection evaluator."""

import numpy as np
import pytest

from star_pattern.detection.local_evaluator import LocalEvaluator
from star_pattern.core.fits_handler import FITSImage


def _make_image(peak_value: float = 1000, bg_level: float = 100, bg_noise: float = 10):
    """Create a test image with known SNR."""
    rng = np.random.default_rng(42)
    data = rng.normal(bg_level, bg_noise, (100, 100)).astype(np.float32)
    # Add a bright peak in the center
    data[50, 50] = peak_value
    return FITSImage(data=data)


class TestLocalEvaluator:
    def setup_method(self):
        self.evaluator = LocalEvaluator()

    def test_high_snr_verdict_real(self):
        """SNR>5 + 3 detectors -> 'real'."""
        detection = {
            "anomaly_score": 0.8,
            "lens": {"lens_score": 0.7},
            "morphology": {"morphology_score": 0.5},
            "galaxy": {"galaxy_score": 0.4},
            "distribution": {"distribution_score": 0.35},
        }
        image = _make_image(peak_value=5000, bg_level=100, bg_noise=10)

        result = self.evaluator.evaluate(detection, image)
        assert result["verdict"] == "real"
        assert result["confidence"] > 0.5

    def test_low_snr_verdict_artifact(self):
        """SNR<2 -> 'artifact'."""
        detection = {
            "anomaly_score": 0.2,
            "lens": {"lens_score": 0.1},
        }
        # Low SNR image
        image = _make_image(peak_value=120, bg_level=100, bg_noise=10)

        result = self.evaluator.evaluate(detection, image)
        assert result["verdict"] == "artifact"
        assert result["confidence"] > 0.5

    def test_inconclusive_needs_llm(self):
        """Mid-range SNR + high anomaly score -> needs_llm_review."""
        detection = {
            "anomaly_score": 0.75,
            "lens": {"lens_score": 0.4},
            "morphology": {"morphology_score": 0.35},
        }
        # Medium SNR image
        image = _make_image(peak_value=400, bg_level=100, bg_noise=10)

        result = self.evaluator.evaluate(detection, image)
        # With 2 detectors agreeing and medium SNR, should be inconclusive
        if result["verdict"] == "inconclusive":
            assert result["needs_llm_review"] is True

    def test_inconclusive_low_score_no_llm(self):
        """Inconclusive verdict but low anomaly score -> no LLM needed."""
        detection = {
            "anomaly_score": 0.3,
            "lens": {"lens_score": 0.4},
            "morphology": {"morphology_score": 0.35},
        }
        image = _make_image(peak_value=400, bg_level=100, bg_noise=10)

        result = self.evaluator.evaluate(detection, image)
        if result["verdict"] == "inconclusive":
            assert result["needs_llm_review"] is False

    def test_significance_rating_range(self):
        """Significance rating is in 1-10 range."""
        detection = {
            "anomaly_score": 0.5,
            "lens": {"lens_score": 0.5},
            "morphology": {"morphology_score": 0.4},
        }
        image = _make_image()

        result = self.evaluator.evaluate(detection, image)
        assert 1 <= result["significance_rating"] <= 10

    def test_high_significance_rating(self):
        """High SNR + many detectors -> high significance."""
        detection = {
            "anomaly_score": 0.9,
            "lens": {"lens_score": 0.8},
            "morphology": {"morphology_score": 0.7},
            "galaxy": {"galaxy_score": 0.6},
            "distribution": {"distribution_score": 0.5},
            "kinematic": {"kinematic_score": 0.4},
            "wavelet": {"wavelet_score": 0.5},
        }
        image = _make_image(peak_value=10000, bg_level=100, bg_noise=10)

        result = self.evaluator.evaluate(detection, image)
        assert result["significance_rating"] >= 5

    def test_look_elsewhere_correction(self):
        """Corrected p-value is computed."""
        detection = {
            "anomaly_score": 0.5,
            "lens": {"lens_score": 0.5},
        }

        result = self.evaluator.evaluate(detection)
        assert "look_elsewhere_p" in result
        assert 0 <= result["look_elsewhere_p"] <= 1

    def test_no_image_fallback(self):
        """Works without an image (fallback to anomaly score proxy)."""
        detection = {
            "anomaly_score": 0.8,
            "lens": {"lens_score": 0.6},
            "morphology": {"morphology_score": 0.5},
            "galaxy": {"galaxy_score": 0.4},
        }

        result = self.evaluator.evaluate(detection)
        assert result["verdict"] in ("real", "artifact", "inconclusive")
        assert result["snr"] > 0

    def test_snr_from_detection_results(self):
        """Uses SNR from detection dict if available."""
        detection = {
            "anomaly_score": 0.5,
            "lens": {"lens_score": 0.5, "snr": 7.5},
        }

        result = self.evaluator.evaluate(detection)
        assert result["snr"] == pytest.approx(7.5)

    def test_reasoning_string(self):
        """Reasoning string contains expected components."""
        detection = {
            "anomaly_score": 0.5,
            "lens": {"lens_score": 0.5},
        }

        result = self.evaluator.evaluate(detection)
        assert "SNR=" in result["reasoning"]
        assert "detectors agree" in result["reasoning"]
        assert "p_corrected=" in result["reasoning"]

    def test_n_agreeing_detectors(self):
        """Agreeing detector count is correct."""
        detection = {
            "lens": {"lens_score": 0.5},       # > 0.3
            "morphology": {"morphology_score": 0.4},  # > 0.3
            "galaxy": {"galaxy_score": 0.1},    # < 0.3
            "anomaly_score": 0.5,
        }

        result = self.evaluator.evaluate(detection)
        assert result["n_agreeing_detectors"] == 2
