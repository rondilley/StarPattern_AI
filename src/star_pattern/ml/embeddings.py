"""Feature extraction pipeline combining classical and learned features."""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("ml.embeddings")


class FeatureExtractor:
    """Extract combined feature vectors from astronomical images."""

    def __init__(self, use_backbone: bool = True, backbone_name: str = "efficientnet_b0"):
        self.use_backbone = use_backbone
        self.backbone_name = backbone_name
        self._backbone = None

    def _get_backbone(self):
        if self._backbone is None and self.use_backbone:
            try:
                from star_pattern.ml.backbone import BackboneWrapper

                self._backbone = BackboneWrapper(model_name=self.backbone_name)
            except Exception as e:
                logger.warning(f"Backbone init failed: {e}, using classical features only")
                self.use_backbone = False
        return self._backbone

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single image.

        Combines:
        - Statistical features (mean, std, skew, kurtosis, percentiles)
        - Morphological features (CAS, Gini, M20)
        - Optional: deep backbone embeddings

        Returns:
            1D feature vector.
        """
        features = []

        # Statistical features
        stat_feats = self._statistical_features(image)
        features.append(stat_feats)

        # Morphological features
        morph_feats = self._morphological_features(image)
        features.append(morph_feats)

        # Texture features
        texture_feats = self._texture_features(image)
        features.append(texture_feats)

        # Deep features
        if self.use_backbone:
            try:
                backbone = self._get_backbone()
                if backbone is not None:
                    deep_feats = backbone.embed_image(image)
                    features.append(deep_feats)
            except Exception as e:
                logger.debug(f"Deep feature extraction failed: {e}")

        return np.concatenate(features)

    def extract_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Extract features for a batch of images.

        Returns:
            NxD feature matrix.
        """
        return np.stack([self.extract(img) for img in images])

    @staticmethod
    def _statistical_features(image: np.ndarray) -> np.ndarray:
        """Basic statistical features."""
        from scipy import stats as sp_stats

        data = image.flatten()
        data = data[np.isfinite(data)]

        if len(data) == 0:
            return np.zeros(10)

        return np.array(
            [
                np.mean(data),
                np.std(data),
                float(sp_stats.skew(data)),
                float(sp_stats.kurtosis(data)),
                np.percentile(data, 5),
                np.percentile(data, 25),
                np.median(data),
                np.percentile(data, 75),
                np.percentile(data, 95),
                np.log1p(np.max(data) - np.min(data)),
            ]
        )

    @staticmethod
    def _morphological_features(image: np.ndarray) -> np.ndarray:
        """CAS + Gini + M20 features."""
        from star_pattern.detection.morphology import MorphologyAnalyzer

        analyzer = MorphologyAnalyzer()
        result = analyzer.analyze(image)
        return np.array(
            [
                result.get("concentration", 0),
                result.get("asymmetry", 0),
                result.get("smoothness", 0),
                result.get("gini", 0),
                result.get("m20", 0),
                result.get("ellipticity", 0),
            ]
        )

    @staticmethod
    def _texture_features(image: np.ndarray) -> np.ndarray:
        """Simple texture features from gradient and Laplacian."""
        from scipy import ndimage

        data = image.astype(np.float64)
        data = np.nan_to_num(data)

        # Gradient magnitude
        gy = ndimage.sobel(data, axis=0)
        gx = ndimage.sobel(data, axis=1)
        grad_mag = np.hypot(gx, gy)

        # Laplacian
        lap = ndimage.laplace(data)

        return np.array(
            [
                np.mean(grad_mag),
                np.std(grad_mag),
                np.mean(np.abs(lap)),
                np.std(lap),
            ]
        )
