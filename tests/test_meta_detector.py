"""Tests for the learned meta-detector."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from star_pattern.detection.meta_detector import MetaDetector, MetaDetectorConfig


def _make_features(n_features=60, rng=None):
    """Create a random feature vector."""
    rng = rng or np.random.default_rng(42)
    return rng.random(n_features)


class TestMetaDetector:
    """Tests for MetaDetector."""

    def test_linear_mode_equals_linear_score(self):
        """With blend_weight=0, meta_score equals linear_score."""
        config = MetaDetectorConfig(blend_weight=0.0)
        meta = MetaDetector(config)

        features = _make_features()
        result = meta.score(features, linear_score=0.65)

        assert abs(result["meta_score"] - 0.65) < 1e-6
        assert result["model_type"] == "linear"
        assert result["blend_weight"] == 0.0

    def test_full_blend_weight(self):
        """With blend_weight=1.0, meta_score uses only learned score."""
        config = MetaDetectorConfig(blend_weight=1.0)
        meta = MetaDetector(config)

        features = _make_features()
        result = meta.score(features, linear_score=0.65)

        # Without any training, learned_score will differ from linear_score
        assert result["blend_weight"] == 1.0
        assert 0.0 <= result["meta_score"] <= 1.0

    def test_add_sample_increments_count(self):
        """Adding samples increments the sample count."""
        meta = MetaDetector()
        assert meta.n_samples == 0

        meta.add_sample(_make_features(), True)
        assert meta.n_samples == 1

        meta.add_sample(_make_features(), False)
        assert meta.n_samples == 2

    def test_retrain_insufficient_data(self):
        """Retrain with too few samples returns insufficient_data."""
        meta = MetaDetector()
        for i in range(5):
            meta.add_sample(_make_features(rng=np.random.default_rng(i)), True)

        result = meta.retrain()
        assert result["model_type"] == "linear"
        assert result["status"] == "insufficient_data"

    def test_gbm_transition_at_50_samples(self):
        """GBM model activates at 50+ labeled samples."""
        config = MetaDetectorConfig(min_samples_gbm=50, min_samples_nn=200)
        meta = MetaDetector(config)

        rng = np.random.default_rng(42)
        for i in range(60):
            features = rng.random(30)
            # Label based on a clear signal in the features
            is_interesting = features[0] > 0.5
            meta.add_sample(features, is_interesting)

        result = meta.retrain()
        assert result["model_type"] == "gbm"
        assert result["n_samples"] == 60
        assert meta.model_type == "gbm"

    def test_nn_transition_at_200_samples(self):
        """Neural net model activates at 200+ labeled samples."""
        pytest.importorskip("torch")

        config = MetaDetectorConfig(min_samples_gbm=50, min_samples_nn=200)
        meta = MetaDetector(config)

        rng = np.random.default_rng(42)
        for i in range(210):
            features = rng.random(20)
            is_interesting = features[0] > 0.5
            meta.add_sample(features, is_interesting)

        result = meta.retrain()
        assert result["model_type"] == "neural_net"
        assert result["n_samples"] == 210

    def test_gbm_scoring_after_retrain(self):
        """After GBM retrain, scoring uses GBM model."""
        config = MetaDetectorConfig(
            blend_weight=1.0, min_samples_gbm=50, min_samples_nn=10000
        )
        meta = MetaDetector(config)

        rng = np.random.default_rng(42)
        for i in range(60):
            features = rng.random(20)
            is_interesting = features[0] > 0.5
            meta.add_sample(features, is_interesting)

        meta.retrain()
        assert meta.model_type == "gbm"

        # Score with a clearly interesting feature vector
        test_features = np.ones(20) * 0.9
        result = meta.score(test_features, linear_score=0.5)
        assert result["model_type"] == "gbm"
        assert 0.0 <= result["meta_score"] <= 1.0

    def test_feature_importance_returned(self):
        """Feature importance is available after GBM retrain."""
        config = MetaDetectorConfig(min_samples_gbm=50, min_samples_nn=10000)
        meta = MetaDetector(config)

        rng = np.random.default_rng(42)
        for i in range(60):
            features = rng.random(20)
            is_interesting = features[0] > 0.5
            meta.add_sample(features, is_interesting)

        meta.retrain()
        importance = meta.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_serialization_roundtrip(self):
        """Save and load preserves state."""
        config = MetaDetectorConfig(min_samples_gbm=50)
        meta = MetaDetector(config)

        rng = np.random.default_rng(42)
        for i in range(60):
            features = rng.random(15)
            is_interesting = features[0] > 0.5
            meta.add_sample(features, is_interesting)

        meta.retrain()

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "meta_state"
            meta.save_state(save_path)

            # Load into fresh instance
            meta2 = MetaDetector(config)
            meta2.load_state(save_path)

            assert meta2.n_samples == meta.n_samples
            assert meta2.model_type == meta.model_type

    def test_empty_feature_vector(self):
        """Scoring with zero-length features does not crash."""
        meta = MetaDetector()
        features = np.array([], dtype=np.float64)
        # Should not raise
        result = meta.score(features, linear_score=0.5)
        assert 0.0 <= result["meta_score"] <= 1.0
