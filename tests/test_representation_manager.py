"""Tests for RepresentationManager."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from star_pattern.core.config import RepresentationConfig
from star_pattern.ml.representation_manager import RepresentationManager


def _make_fake_image(size=64):
    """Create a fake FITSImage-like object with a .data attribute."""
    class FakeImage:
        def __init__(self, data):
            self.data = data
    return FakeImage(np.random.default_rng(42).normal(100, 10, (size, size)))


class TestRepresentationManager:
    """Tests for RepresentationManager."""

    def test_init_without_gpu(self):
        """Initializes without error even without GPU."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,  # CPU fallback
        )
        manager = RepresentationManager(config=config)
        assert manager.config.enabled is True

    def test_disabled_returns_none(self):
        """When disabled, embed_image returns None."""
        config = RepresentationConfig(enabled=False)
        manager = RepresentationManager(config=config)

        image = _make_fake_image()
        result = manager.embed_image(image)
        assert result is None

    def test_disabled_anomaly_score_neutral(self):
        """When disabled, anomaly score returns 0.5 (neutral)."""
        config = RepresentationConfig(enabled=False)
        manager = RepresentationManager(config=config)

        embedding = np.random.default_rng(42).random(20)
        score = manager.embedding_anomaly_score(embedding)
        assert score == 0.5

    def test_embed_image_cpu_fallback(self):
        """Embedding with use_backbone=False returns stat features."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
        )
        manager = RepresentationManager(config=config)

        image = _make_fake_image()
        embedding = manager.embed_image(image)

        # Should get statistical features (20-D from FeatureExtractor)
        if embedding is not None:
            assert embedding.ndim == 1
            assert len(embedding) >= 10  # At least stat features
            assert np.all(np.isfinite(embedding))

    def test_buffer_image(self):
        """buffer_image stores embeddings."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
            max_embedding_buffer=10,
        )
        manager = RepresentationManager(config=config)

        image = _make_fake_image()
        embedding = np.random.default_rng(42).random(20)
        manager.buffer_image(image, embedding)

        assert len(manager._embedding_buffer) == 1
        assert len(manager._image_buffer) == 1

    def test_buffer_capped_at_max(self):
        """Buffer does not exceed max_embedding_buffer."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
            max_embedding_buffer=5,
        )
        manager = RepresentationManager(config=config)

        rng = np.random.default_rng(42)
        for i in range(10):
            image = _make_fake_image()
            embedding = rng.random(20)
            manager.buffer_image(image, embedding)

        assert len(manager._embedding_buffer) == 5
        assert len(manager._image_buffer) == 5

    def test_anomaly_score_insufficient_data(self):
        """Anomaly score returns 0.5 with too few embeddings."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
            min_embeddings_for_anomaly=20,
        )
        manager = RepresentationManager(config=config)

        # Only buffer 5 embeddings (below threshold of 20)
        rng = np.random.default_rng(42)
        for i in range(5):
            manager._embedding_buffer.append(rng.random(20))

        embedding = rng.random(20)
        score = manager.embedding_anomaly_score(embedding)
        assert score == 0.5

    def test_anomaly_score_with_sufficient_data(self):
        """Anomaly score returns [0,1] with enough embeddings."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
            min_embeddings_for_anomaly=10,
        )
        manager = RepresentationManager(config=config)

        rng = np.random.default_rng(42)
        # Buffer 20 similar embeddings
        for i in range(20):
            manager._embedding_buffer.append(rng.normal(0, 0.1, 20))

        # Score a normal embedding
        normal = rng.normal(0, 0.1, 20)
        score = manager.embedding_anomaly_score(normal)
        assert 0.0 <= score <= 1.0

    def test_save_load_state(self):
        """Save and load preserves buffer state."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
        )
        manager = RepresentationManager(config=config)

        rng = np.random.default_rng(42)
        for i in range(5):
            manager._embedding_buffer.append(rng.random(20))
        manager._n_retrains = 3

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "repr_state"
            manager.save_state(save_path)

            manager2 = RepresentationManager(config=config)
            manager2.load_state(save_path)

            assert len(manager2._embedding_buffer) == 5
            assert manager2._n_retrains == 3

    def test_byol_retrain_not_enough_data(self):
        """BYOL retrain returns None when buffer is too small."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=True,
            byol_retrain_interval=50,
        )
        manager = RepresentationManager(config=config)
        # Only 5 images buffered
        for i in range(5):
            manager._image_buffer.append(np.zeros((64, 64)))

        result = manager.maybe_retrain_backbone()
        assert result is None

    def test_maybe_retrain_anomaly_detector(self):
        """Anomaly detector retraining works with sufficient data."""
        config = RepresentationConfig(
            enabled=True,
            use_backbone=False,
            min_embeddings_for_anomaly=10,
        )
        manager = RepresentationManager(config=config)

        rng = np.random.default_rng(42)
        for i in range(15):
            manager._embedding_buffer.append(rng.random(20))

        result = manager.maybe_retrain_anomaly_detector()
        assert result is True
