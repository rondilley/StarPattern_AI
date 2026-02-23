"""Anomaly scoring, SNR, and significance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class Anomaly:
    """A single detected anomaly within a sky region."""

    anomaly_type: str  # "lens_arc", "overdensity", "tidal_feature", etc.
    detector: str  # "lens", "distribution", "galaxy", "kinematic", etc.
    pixel_x: float | None = None  # pixel column (None for catalog-based)
    pixel_y: float | None = None  # pixel row
    sky_ra: float | None = None  # WCS-converted RA (or catalog RA)
    sky_dec: float | None = None  # WCS-converted Dec (or catalog Dec)
    score: float = 0.0  # detector-specific score/significance
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "detector": self.detector,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "sky_ra": self.sky_ra,
            "sky_dec": self.sky_dec,
            "score": self.score,
            "properties": self.properties,
        }


def signal_to_noise(signal: np.ndarray, background_rms: float) -> float:
    """Compute signal-to-noise ratio."""
    if background_rms <= 0:
        return 0.0
    return float(np.max(signal) / background_rms)


def detection_significance(
    observed: float, expected: float, n_trials: int = 1
) -> dict[str, float]:
    """Compute detection significance (sigma) and p-value.

    Args:
        observed: Observed count or statistic.
        expected: Expected value under null hypothesis.
        n_trials: Number of independent trials (for look-elsewhere correction).
    """
    from scipy import stats

    if expected <= 0:
        return {"sigma": 0.0, "p_value": 1.0, "corrected_p_value": 1.0}

    # Poisson significance
    p_value = 1 - stats.poisson.cdf(int(observed) - 1, expected)
    corrected_p = min(p_value * n_trials, 1.0)  # Bonferroni

    # Convert to sigma
    sigma = float(stats.norm.isf(min(p_value, 0.5)))

    return {
        "sigma": max(sigma, 0),
        "p_value": float(p_value),
        "corrected_p_value": float(corrected_p),
    }


def anomaly_score_combined(
    detection_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine multiple detection scores into a single anomaly score.

    Args:
        detection_scores: Dict of detector_name -> score [0, 1].
        weights: Optional weights (default: equal).
    """
    if not detection_scores:
        return 0.0

    if weights is None:
        weights = {k: 1.0 / len(detection_scores) for k in detection_scores}

    total_weight = sum(weights.get(k, 0) for k in detection_scores)
    if total_weight == 0:
        return 0.0

    score = sum(
        detection_scores[k] * weights.get(k, 0) for k in detection_scores
    ) / total_weight

    return float(np.clip(score, 0, 1))


def novelty_score(
    features: np.ndarray,
    reference_features: np.ndarray,
    method: str = "mahalanobis",
) -> float:
    """Compute novelty score relative to a reference set.

    Higher = more novel (different from reference).
    """
    if len(reference_features) < 2:
        return 0.5

    if method == "mahalanobis":
        mean = np.mean(reference_features, axis=0)
        cov = np.cov(reference_features.T)
        # Regularize covariance
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
            diff = features - mean
            dist = float(np.sqrt(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            dist = float(np.linalg.norm(features - mean))
    elif method == "euclidean":
        dists = np.linalg.norm(reference_features - features, axis=1)
        dist = float(np.min(dists))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to [0, 1] using sigmoid
    return float(1 / (1 + np.exp(-dist + 3)))


def diversity_score(population_features: np.ndarray) -> float:
    """Compute diversity of a population of findings.

    Higher = more diverse set of discoveries.
    """
    if len(population_features) < 2:
        return 0.0

    from scipy.spatial.distance import pdist

    dists = pdist(population_features)
    if len(dists) == 0:
        return 0.0

    return float(np.mean(dists) / max(np.max(dists), 1e-10))


class PatternResult:
    """A scored, validated pattern detection result."""

    def __init__(
        self,
        region_ra: float,
        region_dec: float,
        detection_type: str,
        anomaly_score: float,
        significance: float = 0.0,
        novelty: float = 0.0,
        details: dict[str, Any] | None = None,
    ):
        self.region_ra = region_ra
        self.region_dec = region_dec
        self.detection_type = detection_type
        self.anomaly_score = anomaly_score
        self.significance = significance
        self.novelty = novelty
        self.details = details or {}
        self.metadata: dict[str, Any] = {}
        self.cross_matches: list[dict[str, Any]] = []
        self.anomalies: list[Anomaly] = []
        self.hypothesis: str | None = None
        self.debate_verdict: str | None = None
        self.consensus_score: float | None = None

    @property
    def combined_score(self) -> float:
        """Weighted combined score."""
        return (
            0.4 * self.anomaly_score
            + 0.3 * self.significance
            + 0.2 * self.novelty
            + 0.1 * (1.0 if not self.cross_matches else 0.5)  # Novel if not known
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ra": self.region_ra,
            "dec": self.region_dec,
            "type": self.detection_type,
            "anomaly_score": self.anomaly_score,
            "significance": self.significance,
            "novelty": self.novelty,
            "combined_score": self.combined_score,
            "cross_matches": self.cross_matches,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "hypothesis": self.hypothesis,
            "debate_verdict": self.debate_verdict,
            "consensus_score": self.consensus_score,
            "metadata": self.metadata,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"PatternResult({self.detection_type}, "
            f"score={self.combined_score:.3f}, "
            f"({self.region_ra:.3f}, {self.region_dec:.3f}))"
        )
