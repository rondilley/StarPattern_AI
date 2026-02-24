"""Deterministic local classification of detections from detector scores.

Replaces LLM hypothesis generation for routine detections.
Only genuinely novel or ambiguous findings are escalated to the LLM.
"""

from __future__ import annotations

from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.local_classifier")

# Map dominant detector name to scientific classification
DETECTOR_TO_CLASS: dict[str, str] = {
    "lens": "gravitational_lens",
    "morphology": "morphological_anomaly",
    "galaxy": "galaxy_interaction",
    "kinematic": "kinematic_group",
    "sersic": "galaxy_structure",
    "wavelet": "multiscale_source",
    "population": "stellar_population_anomaly",
    "distribution": "spatial_clustering",
    "transient": "transient_candidate",
    "anomaly": "statistical_outlier",
    "classical": "classical_pattern",
    "variability": "variable_star",
    "temporal": "temporal_change",
}

# Follow-up observation templates per classification
FOLLOW_UP_TEMPLATES: dict[str, list[str]] = {
    "gravitational_lens": [
        "High-resolution imaging to resolve arc morphology",
        "Spectroscopy to confirm source/lens redshifts",
        "Multi-band photometry for photometric redshift",
    ],
    "morphological_anomaly": [
        "Deep imaging to trace low surface brightness features",
        "IFU spectroscopy for velocity field mapping",
        "Multi-band color profiles to constrain stellar populations",
    ],
    "galaxy_interaction": [
        "HI 21cm mapping to trace tidal gas",
        "Deep imaging to detect tidal tails and bridges",
        "Spectroscopy to confirm physical association via redshifts",
    ],
    "kinematic_group": [
        "Radial velocity measurements to confirm 3D co-motion",
        "Chemical abundance analysis for common origin",
        "Age dating via isochrone fitting",
    ],
    "galaxy_structure": [
        "Multi-band decomposition to separate bulge and disk",
        "Kinematic mapping to constrain mass distribution",
        "Comparison with simulated galaxy morphologies",
    ],
    "multiscale_source": [
        "Multi-wavelength follow-up to determine SED",
        "Time-series photometry to check variability",
        "High-resolution imaging to resolve substructure",
    ],
    "stellar_population_anomaly": [
        "Spectroscopic confirmation of CMD outliers",
        "Proper motion analysis to confirm membership",
        "Chemical abundance measurements for population tagging",
    ],
    "spatial_clustering": [
        "Deeper imaging to detect fainter cluster members",
        "Radial velocity survey for membership confirmation",
        "X-ray observations to detect intracluster medium",
    ],
    "transient_candidate": [
        "Time-series photometry to characterize light curve",
        "Spectroscopic classification of transient type",
        "Multi-wavelength follow-up (radio, X-ray)",
    ],
    "statistical_outlier": [
        "Independent re-observation to confirm detection",
        "Multi-band photometry to rule out artifacts",
        "Cross-reference with other survey epochs",
    ],
    "classical_pattern": [
        "Multi-band imaging to confirm spatial pattern",
        "Deeper observation to improve signal-to-noise",
        "Comparison with known diffraction/PSF artifacts",
    ],
    "variable_star": [
        "High-cadence photometric monitoring",
        "Spectroscopic classification of variable type",
        "Cross-match with AAVSO and GCVS variable star catalogs",
    ],
    "temporal_change": [
        "Multi-epoch imaging to track evolution of change",
        "Spectroscopy to classify nature of the transient/variable",
        "Cross-reference with known transient databases (TNS, ATel)",
    ],
}

# Rationale templates for each detector type
_RATIONALE_TEMPLATES: dict[str, str] = {
    "lens": (
        "Detection dominated by gravitational lens signatures. "
        "Lens score {score:.2f} indicates {strength} arc/ring features."
    ),
    "morphology": (
        "Detection dominated by morphological anomaly. "
        "CAS/Gini/M20 score {score:.2f} suggests {strength} structural deviation."
    ),
    "galaxy": (
        "Detection dominated by galaxy interaction features. "
        "Galaxy score {score:.2f} indicates {strength} tidal/merger signatures."
    ),
    "kinematic": (
        "Detection dominated by kinematic grouping. "
        "Kinematic score {score:.2f} suggests {strength} co-moving structure."
    ),
    "sersic": (
        "Detection dominated by Sersic profile anomaly. "
        "Profile score {score:.2f} indicates {strength} structural deviation."
    ),
    "wavelet": (
        "Detection dominated by multi-scale wavelet features. "
        "Wavelet score {score:.2f} suggests {strength} scale-dependent structure."
    ),
    "population": (
        "Detection dominated by stellar population anomaly. "
        "CMD score {score:.2f} indicates {strength} population outliers."
    ),
    "distribution": (
        "Detection dominated by spatial distribution anomaly. "
        "Clustering score {score:.2f} suggests {strength} overdensity."
    ),
    "transient": (
        "Detection dominated by transient/variability signal. "
        "Transient score {score:.2f} indicates {strength} astrometric noise."
    ),
    "anomaly": (
        "Detection dominated by statistical anomaly. "
        "Isolation Forest score {score:.2f} suggests {strength} outlier."
    ),
    "classical": (
        "Detection dominated by classical pattern features. "
        "Pattern score {score:.2f} indicates {strength} spatial structure."
    ),
    "variability": (
        "Detection dominated by time-domain variability signal. "
        "Variability score {score:.2f} indicates {strength} photometric variability."
    ),
    "temporal": (
        "Detection dominated by multi-epoch image differencing. "
        "Temporal score {score:.2f} indicates {strength} change between epochs."
    ),
}


class LocalClassifier:
    """Deterministic classification of detections from detector scores.

    Replaces LLM hypothesis generation for routine detections.
    Only genuinely novel or ambiguous findings are escalated to the LLM.
    """

    # Minimum score gap between top-2 detectors before flagging as ambiguous
    ambiguity_threshold: float = 0.15

    # Minimum confidence for flagging novel detections
    novelty_confidence_threshold: float = 0.6

    def classify(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Classify a detection using deterministic rules.

        Args:
            detection: Raw detection dict from EnsembleDetector.

        Returns:
            Dict with classification, confidence, rationale,
            follow_up, and needs_llm_review flag.
        """
        scores = self._extract_scores(detection)

        if not scores:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "rationale": "No detector scores available",
                "follow_up": [],
                "needs_llm_review": False,
            }

        # Dominant detector determines classification
        dominant = max(scores, key=scores.get)
        classification = DETECTOR_TO_CLASS.get(dominant, "unknown")
        confidence = scores[dominant]

        # Check for ambiguity: needs LLM if top-2 scores are close
        sorted_scores = sorted(scores.values(), reverse=True)
        ambiguous = (
            len(sorted_scores) > 1
            and (sorted_scores[0] - sorted_scores[1]) < self.ambiguity_threshold
        )

        # Check for novelty: needs LLM if no cross-matches AND high score
        cross_matches = detection.get("cross_matches", [])
        novel = (
            not cross_matches
            and confidence > self.novelty_confidence_threshold
        )

        needs_llm = ambiguous or novel

        rationale = self._generate_rationale(dominant, scores)
        follow_up = FOLLOW_UP_TEMPLATES.get(classification, [])

        logger.debug(
            f"Classified as {classification} (conf={confidence:.2f}, "
            f"ambiguous={ambiguous}, novel={novel})"
        )

        return {
            "classification": classification,
            "confidence": confidence,
            "rationale": rationale,
            "follow_up": list(follow_up),
            "needs_llm_review": needs_llm,
            "dominant_detector": dominant,
            "detector_scores": scores,
        }

    def _extract_scores(self, detection: dict[str, Any]) -> dict[str, float]:
        """Extract per-detector scores from a detection dict.

        Handles the nested structure from EnsembleDetector output.
        """
        scores: dict[str, float] = {}

        # Standard detector score keys
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
            "temporal": ("temporal", "temporal_score"),
        }

        for detector, (section, key) in score_keys.items():
            section_data = detection.get(section, {})
            if isinstance(section_data, dict):
                score = section_data.get(key, 0.0)
                if score > 0:
                    scores[detector] = float(score)

        # Also check top-level anomaly_score
        if "anomaly" not in scores and "anomaly_score" in detection:
            score = detection["anomaly_score"]
            if score > 0:
                scores["anomaly"] = float(score)

        return scores

    def _generate_rationale(
        self, dominant: str, scores: dict[str, float]
    ) -> str:
        """Generate a deterministic rationale string."""
        score = scores.get(dominant, 0.0)

        if score > 0.7:
            strength = "strong"
        elif score > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        template = _RATIONALE_TEMPLATES.get(dominant)
        if template:
            return template.format(score=score, strength=strength)

        return (
            f"Detection dominated by {dominant} detector "
            f"with score {score:.2f} ({strength})."
        )
