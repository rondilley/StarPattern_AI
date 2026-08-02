"""Unified confidence scoring grounded in per-detector physical measurements.

Replaces arbitrary hard caps with statistically defensible quality floors.

Detections fall into two distinct evidence families, and the two must never
be mixed:

EVIDENCE_TAIL
    The detector supplies a physical measurement with a real null model
    (Gaussian SNR, Poisson counts, binomial rates, chi2 residuals,
    Lomb-Scargle FAP). These carry a genuine p_value and are the only
    members of the region-wide BH-FDR family.

EVIDENCE_HEURISTIC
    The detector supplies a unitless 0-1 score with no null distribution
    behind it (isolation-forest scores, Hough vote counts, ring
    completeness, score fallbacks). These carry a heuristic_score and an
    explicitly None p_value. Reporting `1 - score` in a p_value field
    would manufacture a probability that was never measured, and feeding
    it to FDR alongside real tail probabilities invalidates the
    correction for every detection in the region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from star_pattern.evaluation.statistical import (
    multiple_comparison_correction,
)
from star_pattern.utils.logging import get_logger

logger = get_logger("evaluation.confidence")

# Evidence family tags. See the module docstring.
EVIDENCE_TAIL = "tail_probability"
EVIDENCE_HEURISTIC = "heuristic"

# Legacy `method` tags that identify a pre-separation heuristic score.
_LEGACY_HEURISTIC_METHODS = frozenset({"score_proxy", "empirical_rank"})


def _finite_or_none(value: float | None) -> float | None:
    """Return value, or None when it is not JSON-representable.

    json.dumps writes inf and nan as the bare tokens Infinity and NaN,
    which no strict JSON parser accepts. Report artifacts must stay
    machine-readable.
    """
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass
class ConfidenceScore:
    """Statistical confidence for an anomaly detection.

    Carries the score, its evidence family, and a human-readable
    annotation explaining how it was computed.
    """

    confidence: float  # [0, 1] ranking currency
    p_value: float | None  # Raw p-value; None for the heuristic family
    p_corrected: float | None  # After correction; None for the heuristic family
    physical_quantity: str  # "SNR", "sigma", "FAP", etc.
    physical_value: float  # The raw measurement (SNR=7.2, sigma=4.1)
    method: str  # "gaussian_sf", "fisher_combined", "isolation_forest_score"
    annotation: str  # "Lens arc at SNR 7.2 (p=3.1e-13)"
    evidence_basis: str = EVIDENCE_TAIL
    heuristic_score: float | None = None  # Set only for the heuristic family
    n_independent_tests: int = 1
    correction_method: str = "none"  # "bonferroni", "fdr"
    n_agreeing_detectors: int = 0
    agreement_details: list[str] = field(default_factory=list)

    @property
    def is_tail(self) -> bool:
        """True when this score rests on a real null distribution."""
        return self.evidence_basis == EVIDENCE_TAIL

    @classmethod
    def heuristic(
        cls,
        score: float,
        physical_quantity: str,
        physical_value: float,
        method: str,
        annotation: str,
    ) -> ConfidenceScore:
        """Build a score for a detector with no null distribution.

        The annotation must describe a score, never a probability.
        """
        bounded = float(min(max(score, 0.0), 1.0))
        return cls(
            confidence=bounded,
            p_value=None,
            p_corrected=None,
            physical_quantity=physical_quantity,
            physical_value=float(physical_value),
            method=method,
            annotation=annotation,
            evidence_basis=EVIDENCE_HEURISTIC,
            heuristic_score=bounded,
            correction_method="none",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": _finite_or_none(self.confidence),
            "p_value": _finite_or_none(self.p_value),
            "p_corrected": _finite_or_none(self.p_corrected),
            "physical_quantity": self.physical_quantity,
            "physical_value": _finite_or_none(self.physical_value),
            "method": self.method,
            "annotation": self.annotation,
            "evidence_basis": self.evidence_basis,
            "heuristic_score": _finite_or_none(self.heuristic_score),
            "n_independent_tests": self.n_independent_tests,
            "correction_method": self.correction_method,
            "n_agreeing_detectors": self.n_agreeing_detectors,
            "agreement_details": self.agreement_details,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConfidenceScore:
        # Reports written before the family split carry no evidence_basis.
        # Their heuristic entries are identifiable by their method tag.
        basis = d.get("evidence_basis")
        if basis is None:
            basis = (
                EVIDENCE_HEURISTIC
                if d.get("method") in _LEGACY_HEURISTIC_METHODS
                else EVIDENCE_TAIL
            )
        heuristic_score = d.get("heuristic_score")
        if basis == EVIDENCE_HEURISTIC and heuristic_score is None:
            heuristic_score = d.get("confidence")
        return cls(
            confidence=d["confidence"],
            p_value=d.get("p_value"),
            p_corrected=d.get("p_corrected"),
            physical_quantity=d["physical_quantity"],
            physical_value=d["physical_value"],
            method=d["method"],
            annotation=d["annotation"],
            evidence_basis=basis,
            heuristic_score=heuristic_score,
            n_independent_tests=d.get("n_independent_tests", 1),
            correction_method=d.get("correction_method", "none"),
            n_agreeing_detectors=d.get("n_agreeing_detectors", 0),
            agreement_details=d.get("agreement_details", []),
        )


# Per-detector quality floor thresholds (minimum p_corrected to keep)
_QUALITY_FLOORS: dict[str, float] = {
    "lens": 0.0013,  # SNR >= 3
    "distribution": 0.0013,  # sigma >= 3 after correction
    "galaxy": 0.0013,  # Any sub-feature SNR >= 3
    "morphology": 0.05,  # Fisher p < 0.05
    "wavelet": 0.05,  # Corrected p < 0.05 at any scale
    "classical": 0.01,  # Any method p < 0.01
    "kinematic": 0.01,  # p < 0.01
    "transient": 0.0013,  # sigma >= 3
    "sersic": 0.01,  # SNR >= 3 or chi2 p < 0.01
    "population": 0.05,  # 95% CI lower bound > field rate
    "variability": 0.01,  # FAP < 0.01 or chi2 p < 0.01
    "temporal": 0.0000003,  # SNR >= 5 (existing threshold)
    "anomaly": 0.05,  # Top 5th percentile
}

# Triage cutoffs for the heuristic family, on the detector's own 0-1
# score scale.
#
# These are NOT significance thresholds and must never be described as
# such. A heuristic score has no null distribution behind it, so no
# threshold on it carries a false-positive rate. They exist to decide
# which uncalibrated candidates are worth a look, and each value is set
# to reproduce the effective selectivity the detector had before the
# evidence families were split:
#
#   lens        old ring path was norm.sf(completeness * 5) <= 0.0013,
#               i.e. completeness >= 0.6
#   classical   old Hough path derived its own Poisson null from the
#               vote count, so it admitted essentially every arc with a
#               handful of votes; 0.5 keeps it a candidate-generation
#               channel without pretending to significance
#   others      old fallback was p = 1 - score <= floor, i.e.
#               score >= 1 - floor
#
# Calibrating these against blank-sky fields would let the affected
# detectors rejoin the tail family with real p-values. Until then the
# reports must show them as scores.
_HEURISTIC_SCORE_FLOORS: dict[str, float] = {
    "lens": 0.6,
    "classical": 0.5,
    "galaxy": 0.9,
    "morphology": 0.95,
    "kinematic": 0.99,
    "sersic": 0.99,
    "population": 0.95,
    "variability": 0.99,
    "anomaly": 0.95,
}
_DEFAULT_HEURISTIC_SCORE_FLOOR = 0.95

# System protection limits (OOM/bug guards)
_MAX_PER_DETECTOR_SYSTEM = 500
_MAX_PER_REGION_SYSTEM = 500

# Detectors whose primary confidence path uses stats.norm.sf (batchable)
_NORM_SF_DETECTORS = frozenset(
    {
        "lens",
        "distribution",
        "galaxy",
        "wavelet",
        "transient",
        "temporal",
    }
)


def _extract_norm_sf_input(a: Any) -> tuple[float, str] | None:
    """Extract the value to pass to norm.sf and a mode tag for post-processing.

    Returns (value, mode) if this anomaly can be batched via norm.sf,
    or None if it needs individual treatment (complex branching).
    """
    det = a.detector
    atype = a.anomaly_type
    props = a.properties
    score = a.score

    if det == "lens":
        if atype == "lens_ring":
            completeness = props.get("completeness", 0)
            snr = props.get("snr", 0)
            if completeness > 0 and snr <= 0:
                # Ring completeness is a geometric fraction, not a sigma.
                # Route to the individual heuristic path.
                return None
        snr = props.get("snr", 0)
        if snr <= 0:
            snr = score
        return (max(snr, 0), "lens_snr")

    if det == "distribution":
        sigma = props.get("sigma", props.get("significance", score))
        if sigma <= 0:
            sigma = score
        return (max(sigma, 0), "distribution")

    if det == "galaxy":
        # Only the explicit sigma/SNR keys are in real sigma units. The
        # "asymmetry" and "strength" keys are 0-1 detector scores, and
        # norm.sf() of a 0-1 score is not a tail probability.
        if atype == "merger":
            sigma = props.get("asymmetry_sigma")
            if sigma is None or sigma <= 0:
                return None
            return (max(sigma, 0), "galaxy_merger")
        snr = props.get("tidal_snr")
        if snr is None or snr <= 0:
            return None
        return (max(snr, 0), "galaxy_tidal")

    if det == "wavelet":
        peak_snr = props.get(
            "peak_snr",
            props.get("peak_significance", props.get("max_significance", score)),
        )
        if peak_snr <= 0:
            peak_snr = score
        return (max(peak_snr, 0), "wavelet")

    if det == "transient":
        sigma = props.get("deviation_sigma", props.get("deviation", score))
        if sigma <= 0:
            sigma = score
        return (abs(sigma), "transient")

    if det == "temporal":
        snr = props.get("peak_snr", score)
        if snr <= 0:
            snr = score
        return (max(snr, 0), "temporal")

    return None


def _build_norm_sf_confidence(
    a: Any,
    p_raw: float,
    mode: str,
) -> ConfidenceScore:
    """Build a ConfidenceScore from a batched norm.sf result.

    Each mode mirrors the corresponding _confidence_* method's output
    but skips the redundant norm.sf call (already computed in batch).
    """
    props = a.properties
    score = a.score
    atype = a.anomaly_type

    if mode == "lens_snr":
        snr = props.get("snr", 0)
        if snr <= 0:
            snr = score
        label = "arc" if "arc" in atype else "ring"
        return ConfidenceScore(
            confidence=1 - p_raw,
            p_value=p_raw,
            p_corrected=p_raw,
            physical_quantity="SNR",
            physical_value=float(snr),
            method="gaussian_sf",
            annotation=f"Lens {label} at SNR {snr:.1f} (p={p_raw:.2e})",
        )

    if mode == "distribution":
        sigma = props.get("sigma", props.get("significance", score))
        if sigma <= 0:
            sigma = score
        n_cells = props.get("n_grid_cells", 100)
        p_corrected = min(p_raw * n_cells, 1.0)
        return ConfidenceScore(
            confidence=1 - p_corrected,
            p_value=p_raw,
            p_corrected=p_corrected,
            physical_quantity="sigma",
            physical_value=float(sigma),
            method="gaussian_sf",
            annotation=(
                f"Overdensity at {sigma:.1f} sigma "
                f"(p={p_corrected:.2e}, Bonferroni {n_cells} cells)"
            ),
            n_independent_tests=n_cells,
            correction_method="bonferroni",
        )

    if mode == "galaxy_merger":
        sigma = props.get("asymmetry_sigma", props.get("asymmetry", score))
        if sigma <= 0:
            sigma = score
        return ConfidenceScore(
            confidence=1 - p_raw,
            p_value=p_raw,
            p_corrected=p_raw,
            physical_quantity="sigma",
            physical_value=float(sigma),
            method="gaussian_sf",
            annotation=f"Merger asymmetry {sigma:.1f} sigma (p={p_raw:.2e})",
        )

    if mode == "galaxy_tidal":
        snr = props.get("tidal_snr", props.get("strength", score))
        if snr <= 0:
            snr = score
        return ConfidenceScore(
            confidence=1 - p_raw,
            p_value=p_raw,
            p_corrected=p_raw,
            physical_quantity="SNR",
            physical_value=float(snr),
            method="gaussian_sf",
            annotation=f"Tidal feature at SNR {snr:.1f} (p={p_raw:.2e})",
        )

    if mode == "wavelet":
        peak_snr = props.get(
            "peak_snr",
            props.get("peak_significance", props.get("max_significance", score)),
        )
        n_scales = props.get("n_scales", 1)
        if peak_snr <= 0:
            peak_snr = score
        n_tests = max(n_scales, 4)
        p_corrected = min(p_raw * n_tests, 1.0)
        return ConfidenceScore(
            confidence=1 - p_corrected,
            p_value=p_raw,
            p_corrected=p_corrected,
            physical_quantity="SNR",
            physical_value=float(peak_snr),
            method="gaussian_sf",
            annotation=(
                f"Wavelet {n_scales}-scale detection at SNR {peak_snr:.1f} "
                f"(p_Bonf={p_corrected:.2e})"
            ),
            n_independent_tests=n_tests,
            correction_method="bonferroni",
        )

    if mode == "transient":
        sigma = props.get("deviation_sigma", props.get("deviation", score))
        if sigma <= 0:
            sigma = score
        p = 2 * p_raw  # two-tailed
        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity="sigma",
            physical_value=float(sigma),
            method="gaussian_sf_twotailed",
            annotation=f"Transient outlier at {sigma:.1f} sigma (p={p:.2e})",
        )

    # mode == "temporal"
    snr = props.get("peak_snr", score)
    if snr <= 0:
        snr = score
    change_type = atype.replace("temporal_", "").replace("_", " ")
    return ConfidenceScore(
        confidence=1 - p_raw,
        p_value=p_raw,
        p_corrected=p_raw,
        physical_quantity="SNR",
        physical_value=float(snr),
        method="gaussian_sf",
        annotation=f"Temporal {change_type} at SNR {snr:.1f} (p={p_raw:.2e})",
    )


class ConfidenceEvaluator:
    """Compute per-detector confidence scores for anomalies.

    Each method maps a detector's raw physical measurement to a
    p-value using the appropriate null-hypothesis distribution.
    """

    def __init__(self) -> None:
        self._method_map = {
            "lens": self._confidence_lens,
            "distribution": self._confidence_distribution,
            "galaxy": self._confidence_galaxy,
            "morphology": self._confidence_morphology,
            "wavelet": self._confidence_wavelet,
            "classical": self._confidence_classical,
            "kinematic": self._confidence_kinematic,
            "transient": self._confidence_transient,
            "sersic": self._confidence_sersic,
            "population": self._confidence_population,
            "variability": self._confidence_variability,
            "temporal": self._confidence_temporal,
            "anomaly": self._confidence_anomaly,
        }

    def compute_confidence(
        self,
        anomaly_type: str,
        detector: str,
        properties: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Dispatch to the appropriate per-detector confidence method."""
        method = self._method_map.get(detector)
        if method is None:
            return self._confidence_fallback(anomaly_type, detector, properties, score)
        return method(anomaly_type, properties, score)

    def score_anomalies_batch(self, anomalies: list[Any]) -> None:
        """Score anomalies in-place, batching norm.sf calls for speed.

        Anomalies whose detector uses norm.sf as the primary path get
        batched into a single vectorized call. Others fall back to
        individual compute_confidence calls.
        """
        if not anomalies:
            return

        batch_items: list[tuple[int, float, str]] = []  # (index, value, mode)
        individual: list[int] = []

        for i, a in enumerate(anomalies):
            if a.detector in _NORM_SF_DETECTORS:
                extracted = _extract_norm_sf_input(a)
                if extracted is not None:
                    batch_items.append((i, extracted[0], extracted[1]))
                    continue
            individual.append(i)

        # Single vectorized norm.sf call for all batchable anomalies
        if batch_items:
            values = np.array([v for _, v, _ in batch_items])
            p_values = stats.norm.sf(values)
            for (idx, _value, mode), p_raw in zip(batch_items, p_values):
                anomalies[idx].confidence = _build_norm_sf_confidence(
                    anomalies[idx],
                    float(p_raw),
                    mode,
                )

        # Individual calls for non-batchable detectors
        for idx in individual:
            a = anomalies[idx]
            a.confidence = self.compute_confidence(
                a.anomaly_type,
                a.detector,
                a.properties,
                a.score,
            )

    def _confidence_lens(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Lens: arc/ring SNR -> p = norm.sf(snr).

        Ring completeness is a geometric fraction with no null model, so
        it reports as a heuristic score rather than a tail probability.
        """
        snr = props.get("snr", 0)
        completeness = props.get("completeness", 0)
        label = "arc" if "arc" in anomaly_type else "ring"

        if anomaly_type == "lens_ring" and completeness > 0 and snr <= 0:
            return ConfidenceScore.heuristic(
                score=float(completeness),
                physical_quantity="completeness",
                physical_value=float(completeness),
                method="ring_completeness",
                annotation=(
                    f"Lens ring completeness {completeness:.0%} " f"(heuristic, uncalibrated)"
                ),
            )

        if snr <= 0:
            snr = score
        p = float(stats.norm.sf(max(snr, 0)))
        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity="SNR",
            physical_value=float(snr),
            method="gaussian_sf",
            annotation=f"Lens {label} at SNR {snr:.1f} (p={p:.2e})",
        )

    def _confidence_distribution(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Distribution: overdensity sigma -> norm.sf(sigma) + Bonferroni for grid cells."""
        sigma = props.get("sigma", props.get("significance", score))
        if sigma <= 0:
            sigma = score
        # Typical KDE grid: ~100 cells
        n_cells = props.get("n_grid_cells", 100)
        p_raw = float(stats.norm.sf(max(sigma, 0)))
        p_corrected = min(p_raw * n_cells, 1.0)
        return ConfidenceScore(
            confidence=1 - p_corrected,
            p_value=p_raw,
            p_corrected=p_corrected,
            physical_quantity="sigma",
            physical_value=float(sigma),
            method="gaussian_sf",
            annotation=f"Overdensity at {sigma:.1f} sigma (p={p_corrected:.2e}, Bonferroni {n_cells} cells)",
            n_independent_tests=n_cells,
            correction_method="bonferroni",
        )

    def _confidence_galaxy(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Galaxy: tidal feature SNR or merger asymmetry sigma.

        Only asymmetry_sigma and tidal_snr are in sigma units. The
        "asymmetry" and "strength" keys are 0-1 detector scores, so they
        report as heuristics rather than passing through norm.sf.
        """
        if anomaly_type == "merger":
            sigma = props.get("asymmetry_sigma")
            if sigma is not None and sigma > 0:
                p = float(stats.norm.sf(max(sigma, 0)))
                return ConfidenceScore(
                    confidence=1 - p,
                    p_value=p,
                    p_corrected=p,
                    physical_quantity="sigma",
                    physical_value=float(sigma),
                    method="gaussian_sf",
                    annotation=f"Merger asymmetry {sigma:.1f} sigma (p={p:.2e})",
                )
            asym = props.get("asymmetry", score)
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="asymmetry",
                physical_value=float(asym),
                method="score_proxy",
                annotation=(f"Merger asymmetry {asym:.2f} (heuristic, uncalibrated)"),
            )

        # Tidal features
        snr = props.get("tidal_snr")
        if snr is not None and snr > 0:
            p = float(stats.norm.sf(max(snr, 0)))
            return ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="SNR",
                physical_value=float(snr),
                method="gaussian_sf",
                annotation=f"Tidal feature at SNR {snr:.1f} (p={p:.2e})",
            )
        strength = props.get("strength", score)
        return ConfidenceScore.heuristic(
            score=float(score),
            physical_quantity="strength",
            physical_value=float(strength),
            method="score_proxy",
            annotation=f"Tidal feature strength {strength:.2f} (heuristic, uncalibrated)",
        )

    def _confidence_morphology(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Morphology: CAS z-scores -> Fisher's combined p on individual p-values."""
        z_keys = ["C_zscore", "A_zscore", "S_zscore", "gini_zscore"]
        p_values = []
        details = []
        for key in z_keys:
            z = props.get(key, 0)
            if z != 0:
                p_i = float(2 * stats.norm.sf(abs(z)))  # two-tailed
                p_values.append(p_i)
                details.append(f"{key.split('_')[0]}={z:.1f}sigma")

        if not p_values:
            # No CAS z-scores available: the raw score has no null model.
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="morphology_score",
                physical_value=float(score),
                method="score_proxy",
                annotation=f"Morphology score {score:.2f} (heuristic, uncalibrated)",
            )

        # Fisher's method: -2 * sum(ln(p_i)) ~ chi2(2k)
        p_arr = np.asarray(p_values)
        fisher_stat = float(-2 * np.log(np.clip(p_arr, 1e-300, None)).sum())
        df = 2 * len(p_values)
        p_fisher = float(stats.chi2.sf(fisher_stat, df))

        return ConfidenceScore(
            confidence=1 - p_fisher,
            p_value=p_fisher,
            p_corrected=p_fisher,
            physical_quantity="Fisher_chi2",
            physical_value=float(fisher_stat),
            method="fisher_combined",
            annotation=f"Morphology CAS: {', '.join(details)} (Fisher p={p_fisher:.2e})",
        )

    def _confidence_wavelet(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Wavelet: per-scale SNR -> BH-FDR across scales."""
        peak_snr = props.get(
            "peak_snr", props.get("peak_significance", props.get("max_significance", score))
        )
        n_scales = props.get("n_scales", 1)
        if peak_snr <= 0:
            peak_snr = score

        p_raw = float(stats.norm.sf(max(peak_snr, 0)))
        # Bonferroni across scales (typically 4-6 scales). This multiplies
        # by the test count, which is Bonferroni and not BH-FDR.
        n_tests = max(n_scales, 4)
        p_corrected = min(p_raw * n_tests, 1.0)

        return ConfidenceScore(
            confidence=1 - p_corrected,
            p_value=p_raw,
            p_corrected=p_corrected,
            physical_quantity="SNR",
            physical_value=float(peak_snr),
            method="gaussian_sf",
            annotation=f"Wavelet {n_scales}-scale detection at SNR {peak_snr:.1f} (p_Bonf={p_corrected:.2e})",
            n_independent_tests=n_tests,
            correction_method="bonferroni",
        )

    def _confidence_classical(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Classical: Hough vote count or Gabor filter energy.

        Both signals are heuristics. The Hough "null rate" was derived
        from the observation itself (expected = votes * 0.1), which makes
        the resulting number a monotone transform of the vote count
        rather than a probability under any data-independent null. Gabor
        energy is a filter response, not a z-score. Neither may enter the
        FDR family until the detector supplies a measured background rate.
        """
        votes = props.get("hough_votes", props.get("votes", props.get("strength", 0)))
        gabor = props.get("gabor_energy", 0)

        if votes > gabor:
            quantity = "Hough_votes"
            value = float(votes)
            ann = f"Classical arc: {votes:.0f} Hough votes (heuristic, uncalibrated)"
            method = "hough_vote_count"
        elif gabor > 0:
            quantity = "gabor_energy"
            value = float(gabor)
            ann = f"Classical Gabor energy {gabor:.2f} (heuristic, uncalibrated)"
            method = "gabor_energy"
        else:
            quantity = "score"
            value = float(score)
            ann = f"Classical score {score:.2f} (heuristic, uncalibrated)"
            method = "score_proxy"

        return ConfidenceScore.heuristic(
            score=float(score),
            physical_quantity=quantity,
            physical_value=value,
            method=method,
            annotation=ann,
        )

    def _confidence_kinematic(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """ProperMotion: group membership count vs expected field rate."""
        n_members = props.get("n_members", 0)
        expected_field = props.get("expected_field", 1.0)

        if n_members > 0 and expected_field > 0:
            # sf() rather than 1 - cdf(): the latter underflows to exactly
            # 0.0 for rich groups and reports false certainty.
            p = float(stats.poisson.sf(max(int(n_members) - 1, 0), expected_field))
            ann = (
                f"{anomaly_type.replace('_', ' ').capitalize()}: "
                f"{n_members} members vs {expected_field:.1f} expected (p={p:.2e})"
            )
            quantity = "n_members"
            value = float(n_members)
            method = "poisson_sf"
        elif n_members > 0 and "deviation_sigma" in props:
            sigma = props["deviation_sigma"]
            p = float(2 * stats.norm.sf(abs(max(sigma, 0))))
            ann = f"{anomaly_type.replace('_', ' ').capitalize()} at {sigma:.1f} sigma (p={p:.2e})"
            quantity = "sigma"
            value = float(sigma)
            method = "gaussian_sf_twotailed"
        else:
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="score",
                physical_value=float(score),
                method="score_proxy",
                annotation=f"Kinematic score {score:.2f} (heuristic, uncalibrated)",
            )

        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity=quantity,
            physical_value=value,
            method=method,
            annotation=ann,
        )

    def _confidence_transient(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Transient: deviation sigma -> 2 * norm.sf(|sigma|) two-tailed."""
        sigma = props.get("deviation_sigma", props.get("deviation", score))
        if sigma <= 0:
            sigma = score
        p = float(2 * stats.norm.sf(abs(sigma)))
        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity="sigma",
            physical_value=float(sigma),
            method="gaussian_sf_twotailed",
            annotation=f"Transient outlier at {sigma:.1f} sigma (p={p:.2e})",
        )

    def _confidence_sersic(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Sersic: residual peak SNR or chi2 of fit."""
        snr = props.get("residual_snr", props.get("peak_snr", 0))
        chi2_red = props.get("chi2_reduced", 0)

        if snr > 0:
            p_snr = float(stats.norm.sf(max(snr, 0)))
        else:
            p_snr = 1.0

        if chi2_red > 1:
            # Approximate: high chi2_reduced means poor fit = real structure
            # Use chi2 sf with moderate dof
            dof = props.get("n_pixels", 100)
            chi2_stat = chi2_red * dof
            p_chi2 = float(stats.chi2.sf(chi2_stat, dof))
        else:
            p_chi2 = 1.0

        # Use the more significant measure
        if p_snr < p_chi2:
            p = p_snr
            quantity = "SNR"
            value = float(snr)
            ann = f"Sersic residual at SNR {snr:.1f} (p={p:.2e})"
        else:
            p = p_chi2
            quantity = "chi2_reduced"
            value = float(chi2_red)
            ann = f"Sersic fit chi2_red={chi2_red:.2f} (p={p:.2e})"

        if p >= 1.0:
            # Neither the residual SNR nor the chi2 fit gave a usable
            # tail probability, so the raw score has no null model.
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="score",
                physical_value=float(score),
                method="score_proxy",
                annotation=f"Sersic score {score:.2f} (heuristic, uncalibrated)",
            )

        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity=quantity,
            physical_value=value,
            method="gaussian_sf" if "SNR" in quantity else "chi2_sf",
            annotation=ann,
        )

    def _confidence_population(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """StellarPopulation: binomial test on observed vs field rate."""
        n_sources = props.get("n_sources_with_color", 0)

        if anomaly_type == "red_giant":
            rgb_fraction = props.get("rgb_fraction", 0)
            n_rg = props.get("n_red_giants", 0)
            if n_rg > 0 and n_sources >= 5:
                field_rate = 0.05  # typical field RGB fraction
                p = float(stats.binom.sf(max(n_rg - 1, 0), n_sources, field_rate))
                ann = (
                    f"Red giant fraction {rgb_fraction:.3f} "
                    f"({n_rg}/{n_sources} vs {field_rate:.2f} expected, p={p:.2e})"
                )
            else:
                return ConfidenceScore.heuristic(
                    score=float(score),
                    physical_quantity="rgb_fraction",
                    physical_value=float(rgb_fraction),
                    method="score_proxy",
                    annotation=(f"Population score {score:.3f} (heuristic, uncalibrated)"),
                )
            return ConfidenceScore(
                confidence=1 - p,
                p_value=p,
                p_corrected=p,
                physical_quantity="rgb_fraction",
                physical_value=float(rgb_fraction),
                method="binomial_sf",
                annotation=ann,
            )

        # Blue straggler (default)
        bs_fraction = props.get("bs_fraction", 0)
        if n_sources == 0:
            n_sources = props.get("n_blue_stragglers", 0)

        if bs_fraction > 0 and n_sources >= 5:
            field_rate = 0.01
            n_bs = props.get("n_blue_stragglers", max(1, int(bs_fraction * n_sources)))
            p = float(stats.binom.sf(max(n_bs - 1, 0), n_sources, field_rate))
            ann = (
                f"Blue straggler fraction {bs_fraction:.3f} "
                f"({n_bs}/{n_sources} vs {field_rate:.2f} expected, p={p:.2e})"
            )
        else:
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="bs_fraction",
                physical_value=float(bs_fraction),
                method="score_proxy",
                annotation=f"Population score {score:.3f} (heuristic, uncalibrated)",
            )

        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity="bs_fraction",
            physical_value=float(bs_fraction),
            method="binomial_sf",
            annotation=ann,
        )

    def _confidence_variability(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Variability: FAP for periodic, chi2 for general variability."""
        fap = props.get("fap", None)
        chi2 = props.get("chi2_variability", props.get("chi2_reduced", 0))

        if anomaly_type == "periodic_variable" and fap is not None:
            p = float(fap)
            ann = f"Periodic variable: FAP={fap:.2e}"
            quantity = "FAP"
            value = float(fap)
            method = "lomb_scargle_fap"
        elif chi2 > 1:
            # chi2 test for excess variance
            dof = max(int(props.get("n_epochs", 10)) - 1, 1)
            chi2_stat = chi2 * dof
            p = float(stats.chi2.sf(chi2_stat, dof))
            ann = f"Variable: chi2_red={chi2:.2f} (p={p:.2e})"
            quantity = "chi2_reduced"
            value = float(chi2)
            method = "chi2_sf"
        else:
            return ConfidenceScore.heuristic(
                score=float(score),
                physical_quantity="score",
                physical_value=float(score),
                method="score_proxy",
                annotation=f"Variability score {score:.2f} (heuristic, uncalibrated)",
            )

        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity=quantity,
            physical_value=value,
            method=method,
            annotation=ann,
        )

    def _confidence_temporal(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Temporal: epoch-diff peak SNR -> norm.sf(snr)."""
        snr = props.get("peak_snr", score)
        if snr <= 0:
            snr = score
        p = float(stats.norm.sf(max(snr, 0)))
        change_type = anomaly_type.replace("temporal_", "").replace("_", " ")
        return ConfidenceScore(
            confidence=1 - p,
            p_value=p,
            p_corrected=p,
            physical_quantity="SNR",
            physical_value=float(snr),
            method="gaussian_sf",
            annotation=f"Temporal {change_type} at SNR {snr:.1f} (p={p:.2e})",
        )

    def _confidence_anomaly(
        self,
        anomaly_type: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Anomaly (Isolation Forest): normalized outlier score.

        The score is min-max normalized within the current batch, not
        ranked against a calibrated null population, so 1 - score is not
        an empirical p-value and must not enter the FDR family.
        """
        return ConfidenceScore.heuristic(
            score=float(score),
            physical_quantity="IF_score",
            physical_value=float(score),
            method="isolation_forest_score",
            annotation=(
                f"Isolation forest score {score:.3f} "
                f"(heuristic, batch-normalized, uncalibrated)"
            ),
        )

    def _confidence_fallback(
        self,
        anomaly_type: str,
        detector: str,
        props: dict[str, Any],
        score: float,
    ) -> ConfidenceScore:
        """Fallback for unknown detectors: report the raw score honestly."""
        return ConfidenceScore.heuristic(
            score=float(score),
            physical_quantity="score",
            physical_value=float(score),
            method="score_proxy",
            annotation=(
                f"{detector}/{anomaly_type} score {score:.3f} " f"(heuristic, uncalibrated)"
            ),
        )


def passes_quality_floor(
    confidence: ConfidenceScore,
    detector: str,
) -> bool:
    """Check if an anomaly passes its detector's quality floor.

    Tail-family detections pass when the raw p-value is at or below the
    detector's significance threshold. Heuristic detections have no
    p-value, so they are triaged against _HEURISTIC_SCORE_FLOORS instead.
    Reusing a p-value floor as a score cutoff would be a category error:
    the lens floor of 0.0013 encodes "3 sigma", not "score >= 0.9987".
    """
    if confidence.evidence_basis == EVIDENCE_HEURISTIC:
        floor = _HEURISTIC_SCORE_FLOORS.get(detector, _DEFAULT_HEURISTIC_SCORE_FLOOR)
        return (confidence.heuristic_score or 0.0) >= floor
    return confidence.p_value is not None and confidence.p_value <= _QUALITY_FLOORS.get(
        detector, 0.05
    )


def apply_fdr_correction(
    anomalies: list[Any],
) -> None:
    """Apply BH-FDR correction across the tail-probability family.

    Updates each member's confidence.p_corrected and confidence.confidence
    in-place. Anomalies without confidence scores are skipped.

    Heuristic detections are never members. A detector score is not a
    p-value, and including one would both corrupt the ranks of the real
    p-values and give the heuristic a corrected p-value it never earned.
    """
    # Collect p-values from tail-family anomalies only.
    indexed: list[tuple[int, float]] = []
    for i, a in enumerate(anomalies):
        if (
            a.confidence is not None
            and a.confidence.evidence_basis == EVIDENCE_TAIL
            and a.confidence.p_value is not None
        ):
            indexed.append((i, a.confidence.p_value))

    if not indexed:
        return

    indices, p_values = zip(*indexed)
    corrected = multiple_comparison_correction(list(p_values), method="fdr")

    for idx, p_corr in zip(indices, corrected):
        a = anomalies[idx]
        a.confidence.p_corrected = p_corr
        a.confidence.confidence = 1 - p_corr
        a.confidence.correction_method = "fdr"
        a.confidence.n_independent_tests = len(indexed)


def assign_spatial_groups(
    anomalies: list[Any],
    sky_tolerance_arcsec: float = 5.0,
    pixel_tolerance: float = 15.0,
) -> None:
    """Group spatially co-located anomalies by proximity.

    Assigns matching group_id strings (e.g. "grp_001") to anomalies
    within the tolerance radius. Each anomaly retains its own
    ConfidenceScore; the group_id links them for reporting.
    """
    n = len(anomalies)
    if n == 0:
        return

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build coordinate arrays for vectorized pairwise distance
    sky_ra = np.array([a.sky_ra if a.sky_ra is not None else np.nan for a in anomalies])
    sky_dec = np.array([a.sky_dec if a.sky_dec is not None else np.nan for a in anomalies])
    px_x = np.array([a.pixel_x if a.pixel_x is not None else np.nan for a in anomalies])
    px_y = np.array([a.pixel_y if a.pixel_y is not None else np.nan for a in anomalies])

    # Sky-coord pairs: vectorized pairwise separation
    has_sky = ~(np.isnan(sky_ra) | np.isnan(sky_dec))
    sky_grouped = np.zeros(n, dtype=bool)
    if has_sky.sum() >= 2:
        idx_sky = np.where(has_sky)[0]
        ra_s = sky_ra[idx_sky]
        dec_s = sky_dec[idx_sky]
        cos_dec = np.cos(np.radians(dec_s))
        # Pairwise distances via broadcasting
        dra = (ra_s[:, None] - ra_s[None, :]) * cos_dec[:, None] * 3600
        ddec = (dec_s[:, None] - dec_s[None, :]) * 3600
        sep = np.sqrt(dra**2 + ddec**2)
        # Upper triangle: pairs within tolerance
        mask_upper = np.triu(np.ones((len(idx_sky), len(idx_sky)), dtype=bool), k=1)
        ii, jj = np.where((sep <= sky_tolerance_arcsec) & mask_upper)
        for a_idx, b_idx in zip(ii, jj):
            union(idx_sky[a_idx], idx_sky[b_idx])
            sky_grouped[idx_sky[a_idx]] = True
            sky_grouped[idx_sky[b_idx]] = True

    # Pixel-coord pairs for anomalies not already grouped via sky
    has_px = ~(np.isnan(px_x) | np.isnan(px_y))
    if has_px.sum() >= 2:
        idx_px = np.where(has_px)[0]
        x_p = px_x[idx_px]
        y_p = px_y[idx_px]
        dx = x_p[:, None] - x_p[None, :]
        dy = y_p[:, None] - y_p[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        mask_upper = np.triu(np.ones((len(idx_px), len(idx_px)), dtype=bool), k=1)
        ii, jj = np.where((dist <= pixel_tolerance) & mask_upper)
        for a_idx, b_idx in zip(ii, jj):
            union(idx_px[a_idx], idx_px[b_idx])

    # Assign group IDs to groups with 2+ members
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    grp_num = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        grp_num += 1
        gid = f"grp_{grp_num:03d}"
        for idx in members:
            anomalies[idx].group_id = gid


def compute_group_summary_from_members(
    members: list[Any],
    group_id: str,
) -> dict[str, Any]:
    """Compute Fisher's combined p-value for a pre-filtered group.

    Takes the members list directly (already filtered by group_id)
    to avoid O(n) scan of the full anomaly list per group.
    """
    if not members:
        return {"group_id": group_id, "n_detectors": 0}

    p_values = []
    detail_lines = []
    detectors_seen = set()
    n_heuristic = 0

    for a in members:
        detectors_seen.add(a.detector)
        if a.confidence is None:
            detail_lines.append(f"  - {a.detector}: {a.anomaly_type} (no confidence)")
            continue
        if a.confidence.evidence_basis == EVIDENCE_TAIL and a.confidence.p_value is not None:
            # Fisher's method combines p-values. Only real tail
            # probabilities qualify; a heuristic score would inflate the
            # combined significance with a number that measures nothing.
            p_values.append(a.confidence.p_value)
            detail_lines.append(f"  - {a.detector}: {a.confidence.annotation}")
        else:
            n_heuristic += 1
            detail_lines.append(f"  - {a.detector}: {a.confidence.annotation} [not combined]")

    # Fisher's combined test
    if p_values:
        p_arr = np.asarray(p_values)
        fisher_stat = float(-2 * np.log(np.clip(p_arr, 1e-300, None)).sum())
        df = 2 * len(p_values)
        p_combined = float(stats.chi2.sf(fisher_stat, df))
    else:
        p_combined = 1.0

    header = (
        f"Group {group_id} -- {len(detectors_seen)} detectors, "
        f"Fisher combined p={p_combined:.2e} over {len(p_values)} p-values:"
    )
    if n_heuristic:
        header += f" ({n_heuristic} heuristic member(s) not combined)"

    return {
        "group_id": group_id,
        "n_detectors": len(detectors_seen),
        "n_members": len(members),
        "n_combined_members": len(p_values),
        "n_heuristic_members": n_heuristic,
        "p_combined": p_combined,
        "confidence": 1 - p_combined,
        "summary_text": "\n".join([header] + detail_lines),
        "detail_lines": detail_lines,
    }


def compute_group_summary(
    anomalies: list[Any],
    group_id: str,
) -> dict[str, Any]:
    """Compute Fisher's combined p-value for a spatial group.

    Convenience wrapper that filters members from the full list.
    Prefer compute_group_summary_from_members when processing
    multiple groups to avoid repeated O(n) scans.
    """
    members = [a for a in anomalies if a.group_id == group_id]
    return compute_group_summary_from_members(members, group_id)
