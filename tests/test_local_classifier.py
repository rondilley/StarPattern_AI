"""Tests for local deterministic detection classifier."""

import pytest

from star_pattern.detection.local_classifier import (
    LocalClassifier,
    DETECTOR_TO_CLASS,
    FOLLOW_UP_TEMPLATES,
)


class TestLocalClassifier:
    def setup_method(self):
        self.classifier = LocalClassifier()

    def test_lens_classification(self):
        """High lens score -> gravitational_lens."""
        detection = {
            "lens": {"lens_score": 0.8},
            "morphology": {"morphology_score": 0.2},
            "anomaly_score": 0.7,
        }
        result = self.classifier.classify(detection)
        assert result["classification"] == "gravitational_lens"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["dominant_detector"] == "lens"

    def test_galaxy_classification(self):
        """High galaxy score -> galaxy_interaction."""
        detection = {
            "galaxy": {"galaxy_score": 0.7},
            "morphology": {"morphology_score": 0.3},
            "anomaly_score": 0.5,
        }
        result = self.classifier.classify(detection)
        assert result["classification"] == "galaxy_interaction"

    def test_kinematic_classification(self):
        """High kinematic score -> kinematic_group."""
        detection = {
            "kinematic": {"kinematic_score": 0.6},
            "distribution": {"distribution_score": 0.1},
            "anomaly_score": 0.4,
        }
        result = self.classifier.classify(detection)
        assert result["classification"] == "kinematic_group"

    def test_ambiguous_needs_llm(self):
        """Close top-2 scores -> needs_llm_review=True."""
        detection = {
            "lens": {"lens_score": 0.55},
            "morphology": {"morphology_score": 0.50},
            "anomaly_score": 0.5,
        }
        result = self.classifier.classify(detection)
        assert result["needs_llm_review"] is True

    def test_clear_detection_no_llm(self):
        """Clear dominant detector -> needs_llm_review=False."""
        detection = {
            "lens": {"lens_score": 0.9},
            "morphology": {"morphology_score": 0.1},
            "anomaly_score": 0.7,
            "cross_matches": [{"name": "known_lens"}],
        }
        result = self.classifier.classify(detection)
        assert result["needs_llm_review"] is False

    def test_novel_needs_llm(self):
        """No cross-matches + high score -> needs_llm_review=True."""
        detection = {
            "lens": {"lens_score": 0.8},
            "morphology": {"morphology_score": 0.1},
            "anomaly_score": 0.7,
            "cross_matches": [],
        }
        result = self.classifier.classify(detection)
        assert result["needs_llm_review"] is True

    def test_novel_low_score_no_llm(self):
        """No cross-matches but low score and clear dominant -> needs_llm_review=False."""
        detection = {
            "lens": {"lens_score": 0.4},
            "morphology": {"morphology_score": 0.1},
            "cross_matches": [],
        }
        result = self.classifier.classify(detection)
        # Low confidence (0.4 < 0.6 threshold) means not novel enough
        # Gap of 0.3 > 0.15 means not ambiguous
        assert result["needs_llm_review"] is False

    def test_rationale_generated(self):
        """Rationale text is present and non-empty."""
        detection = {
            "lens": {"lens_score": 0.7},
            "anomaly_score": 0.5,
        }
        result = self.classifier.classify(detection)
        assert result["rationale"]
        assert len(result["rationale"]) > 10
        assert "lens" in result["rationale"].lower() or "score" in result["rationale"].lower()

    def test_follow_up_templates(self):
        """Follow-up list is populated for known classifications."""
        detection = {
            "lens": {"lens_score": 0.8},
            "anomaly_score": 0.6,
        }
        result = self.classifier.classify(detection)
        assert len(result["follow_up"]) > 0
        assert isinstance(result["follow_up"][0], str)

    def test_empty_detection(self):
        """Empty detection -> unknown classification."""
        result = self.classifier.classify({})
        assert result["classification"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["needs_llm_review"] is False

    def test_all_detector_classes_have_followup(self):
        """Every classification in DETECTOR_TO_CLASS has follow-up templates."""
        for _detector, classification in DETECTOR_TO_CLASS.items():
            assert classification in FOLLOW_UP_TEMPLATES, (
                f"Missing follow-up template for {classification}"
            )

    def test_detector_scores_returned(self):
        """Detector scores are included in result."""
        detection = {
            "lens": {"lens_score": 0.5},
            "galaxy": {"galaxy_score": 0.3},
            "anomaly_score": 0.4,
        }
        result = self.classifier.classify(detection)
        assert "detector_scores" in result
        assert "lens" in result["detector_scores"]
        assert "galaxy" in result["detector_scores"]

    def test_strength_labels_in_rationale(self):
        """Rationale includes strength descriptors."""
        # Strong detection
        detection = {"lens": {"lens_score": 0.85}, "anomaly_score": 0.8}
        result = self.classifier.classify(detection)
        assert "strong" in result["rationale"]

        # Moderate detection
        detection = {"lens": {"lens_score": 0.5}, "anomaly_score": 0.4}
        result = self.classifier.classify(detection)
        assert "moderate" in result["rationale"]

        # Weak detection
        detection = {"lens": {"lens_score": 0.2}, "anomaly_score": 0.2}
        result = self.classifier.classify(detection)
        assert "weak" in result["rationale"]
