"""Fitness evaluation for detection genomes.

The fitness function drives evolutionary optimization of detection parameters.
It combines five components:
1. Anomaly: raw detection scores (top-k mean)
2. Significance: consistency of detections above noise
3. Novelty: how different from previous genomes' findings
4. Diversity: variety of pattern types found
5. Recovery: ability to detect injected synthetic patterns (ground truth)

The recovery component uses the SyntheticInjector to create known patterns
(arcs, rings, overdensities) and measures whether the detector recovers them.
This provides a ground-truth fitness signal independent of the natural data.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.config import EvolutionConfig
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.detection.anomaly import AnomalyDetector
from star_pattern.evaluation.metrics import novelty_score, diversity_score
from star_pattern.utils.logging import get_logger

logger = get_logger("discovery.fitness")


class FitnessEvaluator:
    """Evaluate fitness of a detection genome on a set of images.

    Fitness = anomaly*w_a + significance*w_s + novelty*w_n + diversity*w_d + recovery*w_r

    The recovery component injects synthetic patterns into a subset of images
    and measures the detection rate, providing ground-truth fitness signal.
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        use_synthetic_injection: bool = True,
        n_injections: int = 3,
        max_eval_images: int = 8,
        max_eval_seconds: float = 120,
    ):
        self.config = config or EvolutionConfig()
        self.weights = self.config.fitness_weights
        self._all_features: list[np.ndarray] = []
        self.use_synthetic_injection = use_synthetic_injection
        self.n_injections = n_injections
        self.max_eval_images = max_eval_images
        self.max_eval_seconds = max_eval_seconds
        self._injection_rng = np.random.default_rng(42)
        # Cache detection results for elite genomes (same config + same image)
        self._detection_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(
        self,
        genome_config: dict[str, Any],
        images: list[FITSImage],
    ) -> dict[str, float]:
        """Evaluate a genome on a set of images.

        Args:
            genome_config: Detection config from genome.to_detection_config().
            images: List of FITSImage to process.

        Returns:
            Dict with total fitness and component scores.
        """
        if not images:
            return {
                "fitness": 0.0, "anomaly": 0, "significance": 0,
                "novelty": 0, "diversity": 0, "recovery": 0,
            }

        from star_pattern.core.config import DetectionConfig

        # Create detector with genome parameters
        det_config = DetectionConfig.from_genome_dict(genome_config)

        detector = EnsembleDetector(det_config)

        # Subsample images to cap compute cost
        eval_images = images
        if len(images) > self.max_eval_images:
            indices = np.random.default_rng().choice(
                len(images), size=self.max_eval_images, replace=False
            )
            eval_images = [images[i] for i in indices]
            logger.debug(
                f"Subsampled {len(images)} images to {len(eval_images)} for evaluation"
            )

        # Compute config hash for cache lookups
        config_hash = self._config_hash(genome_config)

        # Run detection with time budget (uses cache for repeated configs)
        scores = []
        n_detections = []
        features = []
        eval_start = time.monotonic()

        for i, img in enumerate(eval_images):
            elapsed = time.monotonic() - eval_start
            if elapsed > self.max_eval_seconds:
                logger.debug(
                    f"Time budget exceeded ({elapsed:.1f}s > {self.max_eval_seconds}s) "
                    f"after {i}/{len(eval_images)} images"
                )
                break

            try:
                cache_key = (id(img), config_hash)
                if cache_key in self._detection_cache:
                    result = self._detection_cache[cache_key]
                    self._cache_hits += 1
                else:
                    result = detector.detect(img)
                    self._detection_cache[cache_key] = result
                    self._cache_misses += 1
                scores.append(result.get("anomaly_score", 0))
                n_detections.append(result.get("n_detections", 0))

                # Collect feature vector for this image
                feat = self._extract_feature_vector(result)
                features.append(feat)
            except Exception as e:
                logger.debug(f"Detection failed: {e}")
                scores.append(0)
                n_detections.append(0)

        scores = np.array(scores)
        features_arr = np.array(features) if features else np.empty((0, 12))

        # Component 1: Anomaly score (mean of top-k scores)
        top_k = max(1, len(scores) // 5)
        top_scores = np.sort(scores)[-top_k:]
        anomaly_component = float(np.mean(top_scores))

        # Component 2: Significance (are detections above noise?)
        if len(scores) > 1:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            significance_component = float(mean_score / max(std_score, 0.01))
            significance_component = min(significance_component / 5, 1.0)
        else:
            significance_component = anomaly_component

        # Component 3: Novelty (how different from previous genomes' findings)
        if features_arr.shape[0] > 0 and self._all_features:
            ref = np.array(self._all_features[-20:]) if len(self._all_features) > 20 else np.array(self._all_features)
            if ref.ndim == 1:
                ref = ref.reshape(1, -1)
            mean_feat = np.mean(features_arr, axis=0)
            novelty_component = novelty_score(mean_feat, ref, method="euclidean")
        else:
            novelty_component = 0.5

        # Type-diversity bonus: reward genomes that activate multiple
        # detector types rather than maxing out a single one.
        # Count detectors with non-trivial scores from the feature vectors.
        # Features 2..11 correspond to per-detector scores in the 12-D fallback;
        # for rich features, use the first 12 dimensions.
        if features_arr.shape[0] > 0:
            n_feat_dims = min(12, features_arr.shape[1])
            mean_feats = np.mean(features_arr, axis=0)[:n_feat_dims]
            active_detectors = int(np.sum(mean_feats > 0.05))
            type_fraction = active_detectors / max(n_feat_dims, 1)
            novelty_component = 0.7 * novelty_component + 0.3 * type_fraction

        # Component 4: Diversity (variety of patterns found)
        if features_arr.shape[0] >= 2:
            diversity_component = diversity_score(features_arr)
        else:
            diversity_component = 0.0

        # Component 5: Recovery (synthetic injection test)
        recovery_component = 0.0
        if self.use_synthetic_injection and images:
            recovery_component = self._evaluate_recovery(detector, images)

        # Update feature history
        if features_arr.shape[0] > 0:
            self._all_features.append(np.mean(features_arr, axis=0))

        # Weighted sum
        fitness = (
            self.weights.get("anomaly", 0.35) * anomaly_component
            + self.weights.get("significance", 0.25) * significance_component
            + self.weights.get("novelty", 0.15) * novelty_component
            + self.weights.get("diversity", 0.1) * diversity_component
            + self.weights.get("recovery", 0.15) * recovery_component
        )

        return {
            "fitness": float(fitness),
            "anomaly": float(anomaly_component),
            "significance": float(significance_component),
            "novelty": float(novelty_component),
            "diversity": float(diversity_component),
            "recovery": float(recovery_component),
        }

    def _extract_feature_vector(self, result: dict[str, Any]) -> np.ndarray:
        """Extract a feature vector from detection results.

        Uses rich_features from FeatureFusionExtractor when available,
        falling back to the original 12-D extraction.
        """
        rich = result.get("rich_features")
        if rich is not None:
            return np.asarray(rich, dtype=np.float64)

        return np.array([
            result.get("anomaly_score", 0),
            result.get("n_detections", 0),
            result.get("classical", {}).get("gabor_score", 0),
            result.get("morphology", {}).get("morphology_score", 0),
            result.get("lens", {}).get("lens_score", 0),
            result.get("distribution", {}).get("distribution_score", 0),
            result.get("galaxy", {}).get("galaxy_score", 0),
            result.get("kinematic", {}).get("kinematic_score", 0),
            result.get("transient", {}).get("transient_score", 0),
            result.get("sersic", {}).get("sersic_score", 0),
            result.get("wavelet", {}).get("wavelet_score", 0),
            result.get("population", {}).get("population_score", 0),
        ])

    @staticmethod
    def _config_hash(genome_config: dict[str, Any]) -> str:
        """Compute a stable hash of a genome config dict for cache keying."""
        serialized = json.dumps(genome_config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def clear_detection_cache(self) -> None:
        """Clear the detection cache between generations."""
        if self._cache_hits > 0:
            logger.debug(
                f"Detection cache: {self._cache_hits} hits, "
                f"{self._cache_misses} misses"
            )
        self._detection_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _evaluate_recovery(
        self,
        detector: EnsembleDetector,
        images: list[FITSImage],
    ) -> float:
        """Evaluate recovery rate of injected synthetic patterns.

        Injects known patterns into a subset of images and checks whether
        the ensemble score increases (indicating successful detection).
        This provides a ground-truth fitness signal.

        Returns:
            Recovery fraction [0, 1].
        """
        from star_pattern.evaluation.synthetic import SyntheticInjector

        injector = SyntheticInjector(rng=self._injection_rng)

        n_tests = min(self.n_injections, len(images))
        if n_tests == 0:
            return 0.0

        # Select images for injection (use different ones each time for variety)
        test_indices = self._injection_rng.choice(
            len(images), size=n_tests, replace=False
        )

        recovered = 0
        total = 0

        for idx in test_indices:
            img = images[idx]

            try:
                # Score the original image
                original_result = detector.detect(img)
                original_score = original_result.get("anomaly_score", 0)

                # Inject a synthetic pattern and score
                injected_img, injection_meta = injector.inject_random(img)
                injected_result = detector.detect(injected_img)
                injected_score = injected_result.get("anomaly_score", 0)

                total += 1

                # Recovery: injected score should be higher than original
                # Use a minimum delta to avoid counting noise fluctuations
                delta = injected_score - original_score
                if delta > 0.05:
                    recovered += 1
                elif delta > 0.01:
                    recovered += 0.5  # Partial credit

            except Exception as e:
                logger.debug(f"Injection test failed: {e}")
                total += 1

        return float(recovered / max(total, 1))
