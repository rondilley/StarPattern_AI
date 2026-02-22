"""Active learning with human/LLM-in-the-loop feedback.

Implements a closed-loop learning system:
1. Query: select uncertain detections for labeling
2. Feedback: record human/LLM labels (interesting/not interesting)
3. Retrain: update anomaly detector with labeled examples
4. Adapt: adjust ensemble weights based on detector-label correlations
5. Strategy: accept LLM strategy adjustments to weights and thresholds
6. Repeat: refined model improves future queries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from star_pattern.evaluation.metrics import PatternResult
from star_pattern.utils.logging import get_logger

if TYPE_CHECKING:
    from star_pattern.llm.strategy import StrategyResult

logger = get_logger("pipeline.active_learning")

# Minimum labeled examples before retraining
_MIN_RETRAIN_POSITIVE = 5
_MIN_RETRAIN_NEGATIVE = 5


class ActiveLearner:
    """Active learning loop for refining detection with feedback.

    Closes the learning loop by:
    - Retraining the anomaly detector when enough labeled data accumulates
    - Adjusting ensemble weights to favor detectors correlated with interest
    - Persisting feedback to disk for cross-session learning
    - Adapting query strategy based on feedback density
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        retrain_interval: int = 10,
        persistence_path: Path | None = None,
        meta_detector: Any | None = None,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.retrain_interval = retrain_interval
        self.persistence_path = persistence_path
        self._meta_detector = meta_detector
        self.feedback_history: list[dict[str, Any]] = []
        self._labeled_positive: list[np.ndarray] = []
        self._labeled_negative: list[np.ndarray] = []
        self._retrained_anomaly_detector = None
        self._learned_weights: dict[str, float] | None = None
        self._feedback_since_retrain = 0

        # Load persisted feedback if available
        if persistence_path:
            self._load_feedback(persistence_path)

    def should_query(self, result: PatternResult) -> bool:
        """Determine if we should query for feedback on this result.

        Query when anomaly score is in the uncertain range (not clearly
        interesting or clearly boring). Adapts uncertainty window based
        on accumulated feedback density.
        """
        score = result.anomaly_score

        # Adaptive threshold: narrow the uncertainty band as we get more feedback
        n_feedback = len(self.feedback_history)
        if n_feedback > 20:
            # Tighten uncertainty band as feedback accumulates
            adapt = min(0.15, n_feedback * 0.005)
            lower = self.uncertainty_threshold + adapt
            upper = 1 - self.uncertainty_threshold - adapt
        else:
            lower = self.uncertainty_threshold
            upper = 1 - self.uncertainty_threshold

        return lower < score < upper

    def add_feedback(
        self,
        result: PatternResult,
        is_interesting: bool,
        features: np.ndarray | None = None,
        rich_features: np.ndarray | None = None,
        detector_scores: dict[str, float] | None = None,
        notes: str = "",
    ) -> None:
        """Record human/LLM feedback on a detection.

        Args:
            result: The pattern result being labeled.
            is_interesting: Whether this detection is scientifically interesting.
            features: Optional feature vector for retraining.
            rich_features: Optional rich feature vector for meta-detector.
            detector_scores: Optional per-detector scores for weight learning.
            notes: Optional text notes about the feedback.
        """
        record: dict[str, Any] = {
            "ra": result.region_ra,
            "dec": result.region_dec,
            "type": result.detection_type,
            "anomaly_score": result.anomaly_score,
            "is_interesting": is_interesting,
            "notes": notes,
        }
        if detector_scores:
            record["detector_scores"] = detector_scores

        self.feedback_history.append(record)

        if features is not None:
            if is_interesting:
                self._labeled_positive.append(features)
            else:
                self._labeled_negative.append(features)

        # Feed meta-detector with rich features
        if self._meta_detector is not None and rich_features is not None:
            try:
                self._meta_detector.add_sample(rich_features, is_interesting)
            except Exception as e:
                logger.debug(f"Meta-detector sample add failed: {e}")

        self._feedback_since_retrain += 1

        logger.info(
            f"Feedback recorded: {'interesting' if is_interesting else 'not interesting'} "
            f"(total: {len(self.feedback_history)})"
        )

        # Auto-retrain when enough new feedback accumulates
        if self._feedback_since_retrain >= self.retrain_interval:
            self._try_retrain()
            # Also retrain meta-detector
            if self._meta_detector is not None:
                try:
                    self._meta_detector.retrain()
                except Exception as e:
                    logger.debug(f"Meta-detector retrain failed: {e}")

        # Persist feedback
        if self.persistence_path:
            self._save_feedback(self.persistence_path)

    def get_refined_threshold(self) -> float:
        """Compute a refined threshold based on feedback."""
        if not self.feedback_history:
            return 0.5

        interesting_scores = [
            f["anomaly_score"] for f in self.feedback_history if f["is_interesting"]
        ]
        boring_scores = [
            f["anomaly_score"] for f in self.feedback_history if not f["is_interesting"]
        ]

        if not interesting_scores or not boring_scores:
            return 0.5

        # Threshold = midpoint between mean interesting and mean boring
        mean_int = np.mean(interesting_scores)
        mean_bor = np.mean(boring_scores)
        return float((mean_int + mean_bor) / 2)

    def get_retrained_detector(self) -> Any:
        """Get the retrained anomaly detector, if available.

        Returns None if insufficient feedback for retraining.
        The caller can use this detector in place of the default one.
        """
        if self._retrained_anomaly_detector is None:
            self._try_retrain()
        return self._retrained_anomaly_detector

    def get_learned_weights(self) -> dict[str, float] | None:
        """Get learned ensemble weights based on feedback correlations.

        Returns None if insufficient feedback. Otherwise returns a dict of
        detector_name -> weight that upweights detectors correlated with
        'interesting' labels.
        """
        if self._learned_weights is None:
            self._learn_weights()
        return self._learned_weights

    def _try_retrain(self) -> None:
        """Retrain anomaly detector with accumulated labeled data."""
        n_pos = len(self._labeled_positive)
        n_neg = len(self._labeled_negative)

        if n_pos < _MIN_RETRAIN_POSITIVE or n_neg < _MIN_RETRAIN_NEGATIVE:
            logger.debug(
                f"Insufficient data for retraining: {n_pos} positive, {n_neg} negative"
            )
            return

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            # Build training set: interesting = inliers, not interesting = outliers
            positive = np.array(self._labeled_positive)
            negative = np.array(self._labeled_negative)

            # Contamination estimated from label ratio
            total = n_pos + n_neg
            contamination = min(max(n_neg / total, 0.01), 0.5)

            # Train new Isolation Forest on the positive (interesting) examples
            # with the contamination rate estimated from feedback
            all_features = np.vstack([positive, negative])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(all_features)

            detector = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                random_state=42,
            )
            detector.fit(scaled)

            self._retrained_anomaly_detector = {
                "detector": detector,
                "scaler": scaler,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "contamination": contamination,
            }
            self._feedback_since_retrain = 0

            logger.info(
                f"Retrained anomaly detector with {n_pos} positive, "
                f"{n_neg} negative examples (contamination={contamination:.3f})"
            )

        except Exception as e:
            logger.warning(f"Anomaly detector retraining failed: {e}")

    def _learn_weights(self) -> None:
        """Learn ensemble weights from feedback-detector correlations.

        For each detector, compute the correlation between its score and the
        'interesting' label. Upweight detectors with positive correlation.
        """
        # Need feedback records with detector_scores
        records_with_scores = [
            f for f in self.feedback_history if "detector_scores" in f
        ]

        if len(records_with_scores) < 10:
            return

        # Extract detector names from first record
        detector_names = list(records_with_scores[0]["detector_scores"].keys())

        # Build arrays
        labels = np.array([
            1.0 if f["is_interesting"] else 0.0
            for f in records_with_scores
        ])
        label_mean = labels.mean()
        label_std = max(labels.std(), 1e-6)

        weights = {}
        for name in detector_names:
            scores = np.array([
                f["detector_scores"].get(name, 0.0)
                for f in records_with_scores
            ])
            score_std = max(scores.std(), 1e-6)

            # Pearson correlation
            correlation = float(
                np.mean((scores - scores.mean()) * (labels - label_mean))
                / (score_std * label_std)
            )

            # Transform correlation to weight: higher correlation = higher weight
            # Use softmax-like scaling to ensure all weights positive
            weights[name] = max(0.01, 0.5 + correlation * 0.5)

        # Normalize to sum=1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        self._learned_weights = weights
        logger.info(f"Learned ensemble weights from {len(records_with_scores)} feedback records")

    def score_with_retrained(self, features: np.ndarray) -> float | None:
        """Score features using the retrained anomaly detector.

        Returns anomaly score [0, 1] or None if no retrained detector.
        """
        if self._retrained_anomaly_detector is None:
            return None

        try:
            detector = self._retrained_anomaly_detector["detector"]
            scaler = self._retrained_anomaly_detector["scaler"]

            scaled = scaler.transform(features.reshape(1, -1))
            raw_score = detector.decision_function(scaled)[0]

            # Convert: more negative = more anomalous = higher score
            score = float(np.clip(-raw_score / 2 + 0.5, 0, 1))
            return score

        except Exception as e:
            logger.debug(f"Retrained scoring failed: {e}")
            return None

    def _save_feedback(self, path: Path) -> None:
        """Persist feedback history to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            feedback_file = path / "feedback_history.json"

            # Serialize (skip numpy arrays in the main records)
            serializable = []
            for record in self.feedback_history:
                r = {k: v for k, v in record.items()}
                # Ensure all values are JSON-serializable
                if "detector_scores" in r:
                    r["detector_scores"] = {
                        k: float(v) for k, v in r["detector_scores"].items()
                    }
                serializable.append(r)

            feedback_file.write_text(json.dumps(serializable, indent=2))

            # Save feature vectors separately as numpy
            if self._labeled_positive:
                np.save(
                    str(path / "labeled_positive.npy"),
                    np.array(self._labeled_positive),
                )
            if self._labeled_negative:
                np.save(
                    str(path / "labeled_negative.npy"),
                    np.array(self._labeled_negative),
                )

        except Exception as e:
            logger.debug(f"Failed to persist feedback: {e}")

    def _load_feedback(self, path: Path) -> None:
        """Load persisted feedback from disk."""
        try:
            feedback_file = path / "feedback_history.json"
            if feedback_file.exists():
                records = json.loads(feedback_file.read_text())
                self.feedback_history = records
                logger.info(
                    f"Loaded {len(records)} feedback records from {feedback_file}"
                )

            pos_file = path / "labeled_positive.npy"
            if pos_file.exists():
                pos = np.load(str(pos_file))
                self._labeled_positive = [pos[i] for i in range(len(pos))]

            neg_file = path / "labeled_negative.npy"
            if neg_file.exists():
                neg = np.load(str(neg_file))
                self._labeled_negative = [neg[i] for i in range(len(neg))]

        except Exception as e:
            logger.debug(f"Failed to load persisted feedback: {e}")

    def get_llm_feedback(
        self, result: PatternResult, provider: Any
    ) -> bool:
        """Get automated feedback from an LLM."""
        from star_pattern.llm.prompts import SYSTEM_ASTRONOMER

        prompt = (
            f"Is this astronomical detection scientifically interesting?\n\n"
            f"Type: {result.detection_type}\n"
            f"Score: {result.anomaly_score:.3f}\n"
            f"Location: ({result.region_ra:.3f}, {result.region_dec:.3f})\n"
            f"Cross-matches: {len(result.cross_matches)}\n\n"
            f"Answer with ONLY 'yes' or 'no'."
        )

        response = provider.generate(
            prompt,
            system_prompt=SYSTEM_ASTRONOMER,
            max_tokens=10,
            temperature=0.1,
        )

        is_interesting = "yes" in response.lower()
        self.add_feedback(result, is_interesting, notes=f"LLM: {response.strip()}")
        return is_interesting

    def get_strategy_summary(self) -> dict[str, Any]:
        """Export compact summary for LLM strategy sessions.

        Returns:
            Dict with labeled counts, interesting rate, learned weights,
            false positive patterns, and detector performance.
        """
        n_total = len(self.feedback_history)
        n_interesting = sum(
            1 for f in self.feedback_history if f["is_interesting"]
        )
        n_boring = n_total - n_interesting

        # Detector accuracy: for each detector, what fraction of its
        # high-scoring detections were labeled interesting?
        detector_accuracy = self._detector_accuracy_summary()

        # Most common false positive detection types
        top_fp_types = self._top_false_positive_types()

        return {
            "n_total": n_total,
            "n_interesting": n_interesting,
            "n_boring": n_boring,
            "interesting_rate": n_interesting / max(n_total, 1),
            "learned_weights": self._learned_weights,
            "top_false_positive_types": top_fp_types,
            "detector_accuracy": detector_accuracy,
        }

    def _detector_accuracy_summary(self) -> dict[str, float]:
        """Compute per-detector accuracy from feedback records."""
        records = [
            f for f in self.feedback_history if "detector_scores" in f
        ]
        if len(records) < 5:
            return {}

        detector_names = list(records[0]["detector_scores"].keys())
        accuracy: dict[str, float] = {}

        for name in detector_names:
            # For each detector, count how often high score -> interesting
            high_score_records = [
                r for r in records
                if r["detector_scores"].get(name, 0) > 0.3
            ]
            if not high_score_records:
                continue

            correct = sum(
                1 for r in high_score_records if r["is_interesting"]
            )
            accuracy[name] = correct / len(high_score_records)

        return accuracy

    def _top_false_positive_types(self) -> list[str]:
        """Identify the most common false positive detection types."""
        fp_types: dict[str, int] = {}
        for record in self.feedback_history:
            if not record["is_interesting"]:
                det_type = record.get("type", "unknown")
                fp_types[det_type] = fp_types.get(det_type, 0) + 1

        sorted_types = sorted(fp_types.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_types[:5]]

    def apply_strategy(self, strategy: StrategyResult) -> None:
        """Apply LLM strategy adjustments to active learning.

        Updates ensemble weights and query thresholds based on
        LLM guidance. No token cost -- uses pre-computed strategy.

        Args:
            strategy: StrategyResult from a StrategyAdvisor session.
        """
        if strategy.weight_adjustments:
            self._apply_weight_adjustments(strategy.weight_adjustments)

    def _apply_weight_adjustments(
        self, weight_adjustments: dict[str, float]
    ) -> None:
        """Apply weight adjustments from LLM strategy.

        Blends current learned weights with LLM suggestions using
        70/30 split to avoid over-relying on LLM guidance.
        """
        if self._learned_weights is None:
            # No existing weights: adopt LLM suggestions directly
            self._learned_weights = dict(weight_adjustments)
        else:
            # Blend: 70% current + 30% LLM
            for name, suggested in weight_adjustments.items():
                current = self._learned_weights.get(name, suggested)
                self._learned_weights[name] = 0.7 * current + 0.3 * suggested

        # Normalize
        if self._learned_weights:
            total = sum(self._learned_weights.values())
            if total > 0:
                self._learned_weights = {
                    k: v / total for k, v in self._learned_weights.items()
                }

        logger.info("Applied LLM strategy weight adjustments to active learner")

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics about accumulated feedback."""
        n_total = len(self.feedback_history)
        n_interesting = sum(
            1 for f in self.feedback_history if f["is_interesting"]
        )
        n_boring = n_total - n_interesting

        return {
            "n_total": n_total,
            "n_interesting": n_interesting,
            "n_boring": n_boring,
            "interesting_rate": n_interesting / max(n_total, 1),
            "n_positive_features": len(self._labeled_positive),
            "n_negative_features": len(self._labeled_negative),
            "has_retrained_detector": self._retrained_anomaly_detector is not None,
            "has_learned_weights": self._learned_weights is not None,
            "refined_threshold": self.get_refined_threshold(),
        }
