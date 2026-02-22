"""Local statistical evaluation of detection significance.

Replaces the LLM debate for routine verdicts. Only escalates
genuinely novel or high-scoring detections to LLM review.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.core.fits_handler import FITSImage
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.local_evaluator")


class LocalEvaluator:
    """Evaluate detection significance using local statistical tests.

    Replaces the LLM debate (7 LLM calls, ~15,000 tokens) with
    deterministic local computation at zero token cost.
    """

    # SNR thresholds for verdict
    snr_real_threshold: float = 5.0
    snr_artifact_threshold: float = 2.0

    # Minimum agreeing detectors for "real" verdict
    min_agreeing_real: int = 3

    # Score threshold for LLM escalation
    escalation_score_threshold: float = 0.6

    # Detector agreement threshold
    agreement_threshold: float = 0.3

    def evaluate(
        self,
        detection: dict[str, Any],
        image: FITSImage | None = None,
    ) -> dict[str, Any]:
        """Evaluate whether a detection is real or artifact.

        Uses:
        - Signal-to-noise ratio vs background
        - Look-elsewhere correction (Bonferroni)
        - Detector agreement (how many detectors scored > threshold)

        Args:
            detection: Raw detection dict from EnsembleDetector.
            image: Optional FITS image for SNR computation.

        Returns:
            Dict with verdict, confidence, reasoning,
            significance_rating, and needs_llm_review flag.
        """
        snr = self._compute_snr(detection, image)
        n_agreeing = self._count_agreeing_detectors(detection)
        look_elsewhere_p = self._look_elsewhere_correction(detection)

        # Verdict logic
        if snr > self.snr_real_threshold and n_agreeing >= self.min_agreeing_real:
            verdict = "real"
            confidence = min(0.95, snr / 10)
        elif snr < self.snr_artifact_threshold or n_agreeing <= 1:
            verdict = "artifact"
            confidence = 0.8
        else:
            verdict = "inconclusive"
            confidence = 0.5

        # Significance rating (1-10 scale)
        significance = self._rate_significance(snr, n_agreeing, look_elsewhere_p)

        # Escalate to LLM only if inconclusive AND high anomaly score
        anomaly_score = detection.get("anomaly_score", 0)
        needs_llm = (
            verdict == "inconclusive"
            and anomaly_score > self.escalation_score_threshold
        )

        reasoning = (
            f"SNR={snr:.1f}, {n_agreeing} detectors agree, "
            f"p_corrected={look_elsewhere_p:.3f}"
        )

        logger.debug(
            f"Evaluated: verdict={verdict}, confidence={confidence:.2f}, "
            f"significance={significance}/10"
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "significance_rating": significance,
            "needs_llm_review": needs_llm,
            "snr": snr,
            "n_agreeing_detectors": n_agreeing,
            "look_elsewhere_p": look_elsewhere_p,
        }

    def _compute_snr(
        self,
        detection: dict[str, Any],
        image: FITSImage | None,
    ) -> float:
        """Compute signal-to-noise ratio for the detection.

        Uses the image data if available, otherwise falls back to
        the number of agreeing detectors as a proxy.
        """
        # Try to get SNR from detection results directly
        for section_name in ("lens", "wavelet", "sources"):
            section = detection.get(section_name, {})
            if isinstance(section, dict) and "snr" in section:
                return float(section["snr"])

        # Compute from image if available
        if image is not None and image.data is not None:
            data = image.data
            if data.size > 0:
                # Background-subtracted peak SNR
                median = float(np.median(data))
                mad = float(np.median(np.abs(data - median)))
                bg_rms = mad * 1.4826  # MAD to sigma conversion
                if bg_rms > 0:
                    peak = float(np.max(data))
                    return (peak - median) / bg_rms

        # Fallback: use detector agreement count as SNR proxy.
        # Each agreeing detector adds ~1.5 sigma of confidence.
        # This avoids circular dependency on anomaly_score.
        n_agreeing = self._count_agreeing_detectors(detection)
        return n_agreeing * 1.5

    def _count_agreeing_detectors(self, detection: dict[str, Any]) -> int:
        """Count how many detectors scored above the agreement threshold."""
        score_keys = {
            "lens": ("lens", "lens_score"),
            "morphology": ("morphology", "morphology_score"),
            "distribution": ("distribution", "distribution_score"),
            "galaxy": ("galaxy", "galaxy_score"),
            "kinematic": ("kinematic", "kinematic_score"),
            "transient": ("transient", "transient_score"),
            "sersic": ("sersic", "sersic_score"),
            "wavelet": ("wavelet", "wavelet_score"),
            "population": ("population", "population_score"),
            "anomaly": ("anomaly", "anomaly_score"),
            "classical": ("classical", "classical_score"),
            "variability": ("variability", "variability_score"),
        }

        count = 0
        for _detector, (section, key) in score_keys.items():
            section_data = detection.get(section, {})
            if isinstance(section_data, dict):
                score = section_data.get(key, 0.0)
                if score > self.agreement_threshold:
                    count += 1

        return count

    def _look_elsewhere_correction(self, detection: dict[str, Any]) -> float:
        """Apply Bonferroni look-elsewhere correction.

        Adjusts p-value for the number of independent tests (detectors)
        that were run. Conservative correction ensures we only flag
        detections that survive multiple-testing adjustment.
        """
        from star_pattern.evaluation.metrics import detection_significance

        anomaly_score = detection.get("anomaly_score", 0)

        # Number of detectors used = number of independent trials
        n_detectors = 13  # EnsembleDetector uses up to 13

        # Convert anomaly score to a test statistic
        # Higher anomaly score -> more detections expected under null
        # Use Poisson model: observed = score * 10, expected = 1
        observed = max(1, anomaly_score * 10)
        expected = 1.0

        result = detection_significance(
            observed=observed,
            expected=expected,
            n_trials=n_detectors,
        )

        return result.get("corrected_p_value", 1.0)

    def _rate_significance(
        self,
        snr: float,
        n_agreeing: int,
        look_elsewhere_p: float,
    ) -> int:
        """Rate significance on a 1-10 scale.

        Combines SNR, detector agreement, and statistical significance.
        """
        score = 0.0

        # SNR contribution (0-4 points)
        if snr > 10:
            score += 4
        elif snr > 5:
            score += 3
        elif snr > 3:
            score += 2
        elif snr > 2:
            score += 1

        # Detector agreement contribution (0-3 points)
        if n_agreeing >= 5:
            score += 3
        elif n_agreeing >= 3:
            score += 2
        elif n_agreeing >= 2:
            score += 1

        # Statistical significance contribution (0-3 points)
        if look_elsewhere_p < 0.001:
            score += 3
        elif look_elsewhere_p < 0.01:
            score += 2
        elif look_elsewhere_p < 0.05:
            score += 1

        return max(1, min(10, int(score)))
