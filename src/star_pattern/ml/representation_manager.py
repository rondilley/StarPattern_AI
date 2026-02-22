"""Representation learning integration: bridges existing ML infra into the pipeline.

Orchestrates:
- BackboneWrapper (ml/backbone.py) for deep embeddings
- FeatureExtractor (ml/embeddings.py) for statistical + deep features
- SSLPretrainer (ml/ssl_pretrainer.py) for BYOL self-supervised retraining
- EmbeddingAnomalyDetector (detection/anomaly.py) for embedding-based anomaly scoring
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.core.config import RepresentationConfig
from star_pattern.utils.logging import get_logger

logger = get_logger("ml.representation_manager")


class RepresentationManager:
    """Bridges existing ML infrastructure into the autonomous discovery pipeline.

    All heavy components (BackboneWrapper, FeatureExtractor, EmbeddingAnomalyDetector)
    already exist. This class is a thin orchestrator that manages their lifecycle
    and data flow within the pipeline.
    """

    def __init__(
        self,
        config: RepresentationConfig | None = None,
        checkpoint_dir: Path | None = None,
    ):
        self.config = config or RepresentationConfig()
        self._checkpoint_dir = checkpoint_dir
        self._feature_extractor = None
        self._anomaly_detector = None
        self._embedding_buffer: list[np.ndarray] = []
        self._image_buffer: list[np.ndarray] = []
        self._n_retrains = 0
        self._initialized = False

    def _lazy_init(self) -> None:
        """Initialize ML components lazily (avoids importing torch at module level)."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from star_pattern.ml.embeddings import FeatureExtractor

            self._feature_extractor = FeatureExtractor(
                use_backbone=self.config.use_backbone,
                backbone_name=self.config.backbone_name,
            )
            logger.info(
                f"RepresentationManager initialized "
                f"(backbone={self.config.use_backbone}, "
                f"model={self.config.backbone_name})"
            )
        except Exception as e:
            logger.warning(f"FeatureExtractor init failed: {e}")
            # Fall back to stats-only
            try:
                from star_pattern.ml.embeddings import FeatureExtractor

                self._feature_extractor = FeatureExtractor(use_backbone=False)
                logger.info("RepresentationManager: CPU-only mode (stats features)")
            except Exception as e2:
                logger.warning(f"FeatureExtractor fallback also failed: {e2}")

    def embed_image(self, image: Any) -> np.ndarray | None:
        """Extract embedding from a FITSImage using existing FeatureExtractor.

        Args:
            image: FITSImage object with .data attribute.

        Returns:
            1D embedding vector (1280-D + 20-D with backbone, or 20-D without),
            or None on failure.
        """
        if not self.config.enabled:
            return None

        self._lazy_init()

        if self._feature_extractor is None:
            return None

        try:
            data = image.data if hasattr(image, "data") else image
            embedding = self._feature_extractor.extract(data)
            return embedding
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None

    def embedding_anomaly_score(self, embedding: np.ndarray) -> float:
        """Score an embedding for anomaly using existing EmbeddingAnomalyDetector.

        Args:
            embedding: 1D embedding vector.

        Returns:
            Anomaly score in [0, 1] where 1 = most anomalous.
            Returns 0.5 (neutral) if insufficient data or on failure.
        """
        if not self.config.enabled:
            return 0.5

        if len(self._embedding_buffer) < self.config.min_embeddings_for_anomaly:
            return 0.5

        try:
            if self._anomaly_detector is None:
                from star_pattern.detection.anomaly import EmbeddingAnomalyDetector

                self._anomaly_detector = EmbeddingAnomalyDetector(
                    contamination=self.config.embedding_anomaly_contamination,
                )

            # Build reference set from buffer
            ref = np.stack(self._embedding_buffer)
            # Append query to reference for scoring
            all_embeddings = np.vstack([ref, embedding.reshape(1, -1)])

            result = self._anomaly_detector.detect_from_embeddings(all_embeddings)
            scores = result.get("scores", np.array([0.5]))

            # Last score corresponds to query embedding
            return float(scores[-1])
        except Exception as e:
            logger.debug(f"Embedding anomaly scoring failed: {e}")
            return 0.5

    def buffer_image(
        self, image: Any, embedding: np.ndarray
    ) -> None:
        """Buffer an image and its embedding for future BYOL retraining.

        Args:
            image: FITSImage or ndarray.
            embedding: Corresponding embedding vector.
        """
        self._embedding_buffer.append(embedding.copy())

        data = image.data if hasattr(image, "data") else image
        self._image_buffer.append(data.copy())

        # Cap buffer size
        max_buf = self.config.max_embedding_buffer
        if len(self._embedding_buffer) > max_buf:
            self._embedding_buffer = self._embedding_buffer[-max_buf:]
            self._image_buffer = self._image_buffer[-max_buf:]

    def maybe_retrain_backbone(self) -> dict[str, Any] | None:
        """BYOL retrain using existing SSLPretrainer if enough images buffered.

        Should be called during the evolution phase, not the detection hot path.

        Returns:
            Training history dict, or None if not enough data.
        """
        if not self.config.enabled or not self.config.use_backbone:
            return None

        if len(self._image_buffer) < self.config.byol_retrain_interval:
            return None

        try:
            import tempfile
            import os

            # Write buffered images to temp directory for SSLPretrainer
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, img in enumerate(self._image_buffer):
                    np.save(os.path.join(tmp_dir, f"img_{i:04d}.npy"), img)

                from star_pattern.ml.ssl_pretrainer import SSLPretrainer

                pretrainer = SSLPretrainer(
                    data_dir=tmp_dir,
                    epochs=self.config.byol_epochs,
                    batch_size=min(32, len(self._image_buffer)),
                )

                save_path = None
                if self._checkpoint_dir:
                    save_path = str(
                        self._checkpoint_dir / "byol_backbone.pt"
                    )

                history = pretrainer.pretrain(save_path=save_path)

            self._n_retrains += 1
            self._image_buffer.clear()

            logger.info(
                f"BYOL retrain complete (retrain #{self._n_retrains}, "
                f"{len(self._embedding_buffer)} embeddings buffered)"
            )
            return history

        except Exception as e:
            logger.warning(f"BYOL retrain failed: {e}")
            return None

    def maybe_retrain_anomaly_detector(self) -> bool:
        """Re-fit the embedding anomaly detector with current buffer.

        Returns:
            True if retrained, False otherwise.
        """
        if (
            not self.config.enabled
            or len(self._embedding_buffer) < self.config.min_embeddings_for_anomaly
        ):
            return False

        try:
            from star_pattern.detection.anomaly import EmbeddingAnomalyDetector

            self._anomaly_detector = EmbeddingAnomalyDetector(
                contamination=self.config.embedding_anomaly_contamination,
            )
            ref = np.stack(self._embedding_buffer)
            self._anomaly_detector.detect_from_embeddings(ref)
            return True
        except Exception as e:
            logger.debug(f"Anomaly detector retrain failed: {e}")
            return False

    def save_state(self, path: Path | str) -> None:
        """Save representation manager state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "n_retrains": self._n_retrains,
            "n_embeddings": len(self._embedding_buffer),
            "n_images": len(self._image_buffer),
            "config": {
                "enabled": self.config.enabled,
                "backbone_name": self.config.backbone_name,
                "use_backbone": self.config.use_backbone,
            },
        }
        (path / "repr_manager.json").write_text(json.dumps(meta, indent=2))

        if self._embedding_buffer:
            np.save(
                str(path / "embedding_buffer.npy"),
                np.stack(self._embedding_buffer),
            )

    def load_state(self, path: Path | str) -> None:
        """Load representation manager state from disk."""
        path = Path(path)
        if not path.exists():
            return

        meta_path = path / "repr_manager.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._n_retrains = meta.get("n_retrains", 0)

        emb_path = path / "embedding_buffer.npy"
        if emb_path.exists():
            embeddings = np.load(str(emb_path))
            self._embedding_buffer = [
                embeddings[i] for i in range(len(embeddings))
            ]

        logger.info(
            f"RepresentationManager state loaded: "
            f"{len(self._embedding_buffer)} embeddings, "
            f"{self._n_retrains} retrains"
        )
