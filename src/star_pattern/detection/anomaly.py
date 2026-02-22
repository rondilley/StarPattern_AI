"""Anomaly detection using Isolation Forest and embedding-based methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.anomaly")


class AnomalyDetector:
    """Detect anomalous objects using feature-based methods."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._model: IsolationForest | None = None
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        """Fit the anomaly detector on a set of features.

        Args:
            features: NxD feature matrix.
        """
        if features.shape[0] < 10:
            logger.debug(
                f"Too few samples for IsolationForest ({features.shape[0]}), "
                f"using distance-from-mean fallback"
            )
            return

        scaled = self._scaler.fit_transform(features)
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(scaled)
        self._fitted = True
        logger.info(f"Anomaly detector fitted on {features.shape[0]} samples, {features.shape[1]} features")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score samples (lower = more anomalous).

        Returns:
            Array of anomaly scores in [-1, 0] range, normalized to [0, 1]
            where 1 = most anomalous.
        """
        if not self._fitted or self._model is None:
            # If not fitted, score based on distance from mean
            mean = np.mean(features, axis=0, keepdims=True)
            distances = np.linalg.norm(features - mean, axis=1)
            max_d = max(distances.max(), 1e-10)
            return distances / max_d

        scaled = self._scaler.transform(features)
        raw_scores = self._model.score_samples(scaled)
        # Convert: more negative = more anomalous -> normalize to [0, 1]
        min_s = raw_scores.min()
        max_s = raw_scores.max()
        if max_s - min_s < 1e-10:
            return np.zeros(len(raw_scores))
        return (max_s - raw_scores) / (max_s - min_s)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 = anomaly, 1 = normal)."""
        if not self._fitted or self._model is None:
            scores = self.score(features)
            threshold = np.percentile(scores, 100 * (1 - self.contamination))
            return np.where(scores >= threshold, -1, 1)

        scaled = self._scaler.transform(features)
        return self._model.predict(scaled)

    def detect(self, features: np.ndarray) -> dict[str, Any]:
        """Full anomaly detection pipeline.

        Args:
            features: NxD feature matrix.

        Returns:
            Dict with scores, predictions, top anomaly indices.
        """
        if not self._fitted:
            self.fit(features)

        scores = self.score(features)
        predictions = self.predict(features)

        # Top anomalies
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = scores[anomaly_mask]

        # Sort by anomaly score
        sort_idx = np.argsort(anomaly_scores)[::-1]
        top_anomalies = anomaly_indices[sort_idx]

        return {
            "scores": scores,
            "predictions": predictions,
            "n_anomalies": int(anomaly_mask.sum()),
            "top_anomaly_indices": top_anomalies.tolist(),
            "mean_anomaly_score": float(scores[anomaly_mask].mean()) if anomaly_mask.any() else 0.0,
        }


class EmbeddingAnomalyDetector:
    """Anomaly detection on deep learning embeddings."""

    def __init__(self, contamination: float = 0.05):
        self.detector = AnomalyDetector(contamination=contamination)

    def detect_from_embeddings(
        self, embeddings: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Run anomaly detection on embedding vectors.

        Args:
            embeddings: NxD embedding matrix.
            metadata: Optional list of dicts with per-sample info.

        Returns:
            Detection results with anomaly scores and rankings.
        """
        result = self.detector.detect(embeddings)

        if metadata:
            top_indices = result["top_anomaly_indices"]
            result["top_anomaly_metadata"] = [metadata[i] for i in top_indices[:20]]

        return result
