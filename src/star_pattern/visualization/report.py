"""Markdown report generation for discovery results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from star_pattern.evaluation.metrics import Anomaly, PatternResult
from star_pattern.utils.logging import get_logger
from star_pattern.visualization.mosaic import (
    _assign_finding_numbers,
    _extract_sub_detections,
)

logger = get_logger("visualization.report")

# Canonical detector names used in _detector_summary_table
_DETECTOR_NAMES = [
    "classical", "source_extractor", "morphology", "anomaly", "lens",
    "distribution", "galaxy", "kinematic", "transient", "sersic",
    "wavelet", "population", "variability", "temporal",
]


def _confidence_label(rating: int) -> str:
    """Map significance rating (1-10) to a confidence label."""
    if rating >= 8:
        return "High"
    if rating >= 5:
        return "Moderate"
    if rating >= 3:
        return "Low"
    return "Very Low"


# Map snake_case classifications to readable display names
_CLASSIFICATION_DISPLAY: dict[str, str] = {
    "gravitational_lens": "Gravitational lens candidate",
    "morphological_anomaly": "Morphological anomaly",
    "galaxy_interaction": "Galaxy interaction",
    "kinematic_group": "Kinematic group",
    "galaxy_structure": "Galaxy structure anomaly",
    "multiscale_source": "Multi-scale source",
    "stellar_population_anomaly": "Stellar population anomaly",
    "spatial_clustering": "Spatial clustering",
    "transient_candidate": "Transient candidate",
    "statistical_outlier": "Statistical outlier",
    "classical_pattern": "Classical pattern",
    "variable_star": "Variable star candidate",
    "temporal_change": "Temporal change",
}

# Common SIMBAD object type codes -> readable descriptions
_SIMBAD_TYPES: dict[str, str] = {
    "*": "Star",
    "**": "Double star",
    "*iC": "Star in cluster",
    "*iN": "Star in nebula",
    "V*": "Variable star",
    "Psr": "Pulsar",
    "WD*": "White dwarf",
    "Em*": "Emission-line star",
    "Be*": "Be star",
    "BS*": "Blue straggler",
    "RG*": "Red giant",
    "s*r": "S star",
    "HB*": "Horizontal branch star",
    "C*": "Carbon star",
    "G": "Galaxy",
    "GiC": "Galaxy in cluster",
    "GiG": "Galaxy in group",
    "GiP": "Galaxy in pair",
    "BiC": "Brightest galaxy in cluster",
    "AGN": "Active galactic nucleus",
    "SyG": "Seyfert galaxy",
    "Sy1": "Seyfert 1 galaxy",
    "Sy2": "Seyfert 2 galaxy",
    "LIN": "LINER",
    "QSO": "Quasar",
    "Bla": "Blazar",
    "BLL": "BL Lac object",
    "GrG": "Group of galaxies",
    "ClG": "Cluster of galaxies",
    "PCG": "Proto-cluster of galaxies",
    "SCG": "Supercluster of galaxies",
    "IG": "Interacting galaxies",
    "Cl*": "Star cluster",
    "GlC": "Globular cluster",
    "OpC": "Open cluster",
    "As*": "Stellar association",
    "MGr": "Moving group",
    "Neb": "Nebula",
    "HII": "HII region",
    "PN": "Planetary nebula",
    "SNR": "Supernova remnant",
    "ISM": "Interstellar medium",
    "UvS": "UV source",
    "IrS": "Infrared source",
    "rG": "Radio galaxy",
    "mR": "Metric radio source",
    "cm": "Centimetric radio source",
    "mm": "Millimetric radio source",
    "smm": "Sub-millimetric source",
    "HI": "HI (21cm) source",
    "X": "X-ray source",
    "gam": "Gamma-ray source",
    "gLS": "Gravitational lens system",
    "LeI": "Lensed image",
    "LeG": "Lensed galaxy",
    "LeQ": "Lensed quasar",
    "SN": "Supernova",
    "No*": "Nova",
    "PM*": "High proper-motion star",
    "mul": "Composite object",
    "err": "Not an object (error)",
    "PoG": "Part of a galaxy",
    "PoC": "Part of a cloud",
    "reg": "Region defined in the sky",
    "?": "Object of unknown nature",
    "Rad": "Radio source",
    "MoC": "Molecular cloud",
    "DNe": "Dark nebula",
    "RNe": "Reflection nebula",
    "bub": "Bubble",
    "EmO": "Emission object",
    "YSO": "Young stellar object",
    "pr*": "Pre-main sequence star",
    "TT*": "T Tauri star",
    "Ce*": "Cepheid",
    "RR*": "RR Lyrae star",
    "Mi*": "Mira variable",
    "LP*": "Long-period variable",
    "Er*": "Eruptive variable",
    "Fl*": "Flare star",
    "Or*": "Orion variable",
    "RS*": "RS CVn variable",
    "Ro*": "BY Dra variable",
    "a2*": "alpha2 CVn variable",
    "gD*": "gamma Dor variable",
    "dS*": "delta Sct variable",
    "bC*": "beta Cep variable",
    "El*": "Eclipsing binary",
    "EB*": "Eclipsing binary",
    "SB*": "Spectroscopic binary",
    "AB*": "Astrometric binary",
    "CV*": "Cataclysmic variable",
    "XB*": "X-ray binary",
    "LXB": "Low-mass X-ray binary",
    "HXB": "High-mass X-ray binary",
    "WR*": "Wolf-Rayet star",
    "BH": "Black hole",
    "NS": "Neutron star",
}

# Physical descriptions for sub-detection types
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "wavelet features": "structures at multiple spatial scales (compact sources, extended emission, filaments)",
    "multiscale objects": "objects detected consistently across multiple wavelet scales, suggesting real structure",
    "lens arcs": "curved features consistent with gravitational lensing of background sources",
    "lens rings": "ring-like features around a central mass, possibly Einstein rings",
    "tidal features": "distorted morphology suggesting gravitational interaction between galaxies",
    "merger nuclei": "multiple brightness peaks suggesting an ongoing galaxy merger",
    "overdensities": "regions with significantly more sources than the local background",
    "classical arcs": "arc-shaped patterns detected via Hough transform",
    "Sersic residuals": "features remaining after subtracting the best-fit Sersic profile (hidden substructure)",
    "comoving groups": "clusters of stars sharing similar proper motions (possible physical association)",
    "stream candidates": "elongated chains of co-moving stars (possible tidal stream or dissolved cluster)",
    "runaway stars": "stars with unusually high proper motion relative to neighbors",
    "transient outliers": "sources with anomalous flux, color, or astrometric properties",
    "variable candidates": "sources showing significant brightness variation over time",
    "periodic candidates": "sources with periodic brightness variation (eclipsing binaries, pulsators)",
    "blue stragglers": "stars bluer/brighter than the main-sequence turnoff (possible mass transfer or merger)",
    "red giants": "evolved stars on the red giant branch",
    "new sources (temporal)": "sources appearing in later epochs that were absent in the reference",
    "disappeared sources": "sources present in the reference but absent in later epochs",
    "brightenings": "sources showing increased brightness across multiple epochs",
    "fadings": "sources showing decreased brightness across multiple epochs",
    "moving objects": "sources whose position shifts between epochs (asteroids, high-PM stars)",
}


def _classification_display_name(cls: str) -> str:
    """Map snake_case classification to a readable display name."""
    if cls in _CLASSIFICATION_DISPLAY:
        return _CLASSIFICATION_DISPLAY[cls]
    # Fallback: replace underscores with spaces, capitalize first word
    return cls.replace("_", " ").capitalize()


def _decode_simbad_type(code: str) -> str:
    """Decode a SIMBAD object type code to a readable name.

    Returns "code (description)" if known, or just the code if not.
    """
    desc = _SIMBAD_TYPES.get(code, "")
    if desc:
        return f"{desc} ({code})"
    return code


def _describe_detected_features(details: dict) -> list[str]:
    """Build an itemized bullet list of all detected features.

    Returns markdown lines. Includes source counts, each non-zero
    sub-detection category with count, and a physical description
    of what each feature type means.
    """
    if not details:
        return []

    counts = _extract_sub_detections(details)
    if not counts:
        return []

    lines: list[str] = []
    n_sources = counts.pop("sources", 0)
    if n_sources:
        lines.append(f"- **{n_sources} extracted sources**")

    total_anomalous = sum(counts.values())
    if total_anomalous:
        lines.append(f"- **{total_anomalous} anomalous sub-detections:**")
        for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            desc = _FEATURE_DESCRIPTIONS.get(name, "")
            if desc:
                lines.append(f"  - {count} {name} -- {desc}")
            else:
                lines.append(f"  - {count} {name}")

    return lines


def _interpret_metric(metric: str, value: float) -> str:
    """Map a metric name and value to a plain-English interpretation."""
    if metric == "snr":
        if value >= 10:
            return "Strong signal"
        if value >= 5:
            return "Moderate signal"
        if value >= 3:
            return "Weak signal"
        return "Below noise threshold"
    if metric == "n_agreeing_detectors":
        n = int(value)
        if n >= 8:
            return "Strong agreement"
        if n >= 5:
            return "Moderate agreement"
        if n >= 3:
            return "Weak agreement"
        return "Minimal agreement"
    if metric == "p_value":
        if value < 0.001:
            return "Highly significant"
        if value < 0.01:
            return "Significant"
        if value < 0.05:
            return "Marginally significant"
        return "Not significant"
    if metric == "confidence":
        if value >= 0.8:
            return "High confidence"
        if value >= 0.5:
            return "Moderate confidence"
        if value >= 0.3:
            return "Low confidence"
        return "Very low confidence"
    return ""


def _detector_summary_table(findings: list[PatternResult]) -> list[str]:
    """Generate a markdown table summarizing per-detector activity.

    Columns: Detector | Active in N findings | Total sub-detections |
             Mean score | Max score
    """
    if not findings:
        return []

    # Collect per-detector stats
    stats: dict[str, dict[str, Any]] = {}
    for name in _DETECTOR_NAMES:
        stats[name] = {
            "active_count": 0,
            "sub_det_total": 0,
            "scores": [],
        }

    for f in findings:
        det_scores = f.metadata.get("local_classification", {}).get(
            "detector_scores", {},
        )
        sub_dets = _extract_sub_detections(f.details)

        for name in _DETECTOR_NAMES:
            score = det_scores.get(name, 0.0)
            if score > 0.1:
                stats[name]["active_count"] += 1
                stats[name]["scores"].append(score)

        # Map sub-detection keys to detectors
        det_sub_map = {
            "wavelet features": "wavelet",
            "multiscale objects": "wavelet",
            "lens arcs": "lens",
            "lens rings": "lens",
            "tidal features": "galaxy",
            "merger nuclei": "galaxy",
            "overdensities": "distribution",
            "classical arcs": "classical",
            "Sersic residuals": "sersic",
            "comoving groups": "kinematic",
            "stream candidates": "kinematic",
            "runaway stars": "kinematic",
            "transient outliers": "transient",
            "variable candidates": "variability",
            "periodic candidates": "variability",
            "blue stragglers": "population",
            "red giants": "population",
            "sources": "source_extractor",
            "new sources (temporal)": "temporal",
            "disappeared sources": "temporal",
            "brightenings": "temporal",
            "fadings": "temporal",
            "moving objects": "temporal",
        }
        for key, count in sub_dets.items():
            det = det_sub_map.get(key)
            if det and det in stats:
                stats[det]["sub_det_total"] += count

    # Build table (only include detectors that were active)
    lines = []
    lines.append(
        "| Detector | Active in | Sub-detections | Mean score | Max score |"
    )
    lines.append(
        "|----------|-----------|----------------|------------|-----------|"
    )
    for name in _DETECTOR_NAMES:
        s = stats[name]
        if s["active_count"] == 0 and s["sub_det_total"] == 0:
            continue
        scores = s["scores"]
        mean_s = f"{np.mean(scores):.3f}" if scores else "--"
        max_s = f"{np.max(scores):.3f}" if scores else "--"
        lines.append(
            f"| {name} | {s['active_count']} | "
            f"{s['sub_det_total']} | {mean_s} | {max_s} |"
        )
    lines.append("")
    return lines


_ANOMALY_DISPLAY_NAMES: dict[str, str] = {
    "lens_arc": "Lens arc",
    "lens_ring": "Lens ring",
    "overdensity": "Overdensity",
    "multiscale_object": "Multi-scale object",
    "tidal_feature": "Tidal feature",
    "merger": "Merger",
    "classical_arc": "Classical arc",
    "sersic_residual": "Sersic residual",
    "comoving_group": "Co-moving group",
    "stellar_stream": "Stellar stream",
    "runaway_star": "Runaway star",
    "flux_outlier": "Flux outlier",
    "variable_star": "Variable star",
    "periodic_variable": "Periodic variable",
    "blue_straggler": "Blue straggler",
    "red_giant": "Red giant",
    "temporal_new_source": "New source (temporal)",
    "temporal_disappeared": "Disappeared source",
    "temporal_brightening": "Brightening",
    "temporal_fading": "Fading",
    "temporal_moving": "Moving object",
}


def _format_anomaly_location(a: Anomaly) -> str:
    """Format anomaly location as RA, Dec string or pixel coords."""
    if a.sky_ra is not None and a.sky_dec is not None:
        return f"{a.sky_ra:.4f}, {a.sky_dec:.4f}"
    if a.pixel_x is not None and a.pixel_y is not None:
        return f"px ({a.pixel_x:.0f}, {a.pixel_y:.0f})"
    return "--"


def _format_anomaly_score(a: Anomaly) -> str:
    """Format anomaly score with appropriate units.

    Uses raw_score (pre-normalization) from properties when it has
    physically meaningful units (SNR, sigma). Falls back to normalized
    score for detectors whose raw values are arbitrary scales.
    """
    raw = a.properties.get("raw_score", a.score)
    if raw == 0 and a.score == 0:
        return "--"

    if a.anomaly_type == "overdensity":
        return f"{raw:.1f} sigma"
    if a.anomaly_type == "lens_arc":
        snr = a.properties.get("snr", raw)
        return f"SNR {snr:.1f}" if snr > 0 else "--"
    if a.anomaly_type == "lens_ring":
        return f"{raw:.0%}"
    if a.anomaly_type == "multiscale_object":
        n = a.properties.get("n_scales", 0)
        return f"{n} scales" if n > 0 else "--"
    # Tidal features and sersic residuals have raw scores on arbitrary
    # pixel-value scales -- show normalized relative score instead.
    if a.anomaly_type in ("tidal_feature", "sersic_residual"):
        return f"{a.score:.2f}"
    if a.anomaly_type in ("runaway_star", "flux_outlier"):
        return f"{raw:.1f} sigma"
    if a.anomaly_type in ("variable_star", "periodic_variable"):
        return f"{raw:.2f}"
    if a.anomaly_type.startswith("temporal_"):
        snr = a.properties.get("peak_snr", raw)
        return f"SNR {snr:.1f}" if snr > 0 else "--"
    if a.score > 0:
        return f"{a.score:.2f}"
    return "--"


def _format_anomaly_props(a: Anomaly) -> str:
    """Format key anomaly properties as a compact string."""
    parts: list[str] = []
    p = a.properties
    if "radius" in p:
        parts.append(f"r={p['radius']}px")
    if "angle_span" in p:
        parts.append(f"{p['angle_span']:.0f} deg span")
    if "n_sources" in p:
        parts.append(f"{p['n_sources']} sources")
    if "radius_px" in p:
        parts.append(f"r={p['radius_px']}px")
    if "n_members" in p:
        parts.append(f"{p['n_members']} members")
    if "mean_pmra" in p and "mean_pmdec" in p:
        parts.append(f"PM=({p['mean_pmra']:.1f}, {p['mean_pmdec']:.1f}) mas/yr")
    if "pm_total" in p:
        parts.append(f"PM={p['pm_total']:.1f} mas/yr")
    if "period" in p:
        parts.append(f"P={p['period']:.2f}d")
    if "classification" in p:
        parts.append(str(p["classification"]))
    if "length" in p:
        parts.append(f"len={p['length']}px")
    if "orientation" in p:
        import math
        deg = math.degrees(p["orientation"]) % 360
        parts.append(f"{deg:.0f} deg")
    if "extent_px" in p:
        parts.append(f"extent={p['extent_px']}px")
    if "area_px" in p:
        parts.append(f"area={p['area_px']}px")
    if "area" in p and "area_px" not in p:
        parts.append(f"area={p['area']}px")
    if "peak_snr" in p:
        parts.append(f"SNR={abs(p['peak_snr']):.1f}")
    if "dist_in_re" in p:
        parts.append(f"{p['dist_in_re']:.1f} Re")
    if "n_scales" in p and a.anomaly_type != "multiscale_object":
        # Avoid duplicating "N scales" when it's already in the score column
        parts.append(f"{p['n_scales']} scales")
    if "min_scale" in p:
        parts.append(f"scales {p['min_scale']}-{p.get('max_scale', '?')}")
    if "n_nuclei" in p:
        parts.append(f"{p['n_nuclei']} nuclei")
    if "asymmetry" in p and "n_nuclei" not in p:
        parts.append(f"asym={p['asymmetry']:.2f}")
    if "n_epochs_detected" in p and a.anomaly_type.startswith("temporal_"):
        parts.append(f"{p['n_epochs_detected']} epochs")
    return ", ".join(parts) if parts else "--"


def _format_confidence_str(a: Anomaly) -> str:
    """Format confidence score for table display."""
    if a.confidence is None:
        return "--"
    conf = a.confidence
    return f"{conf.confidence:.3f} (p={conf.p_corrected:.1e})"


def _format_anomaly_table(anomalies: list[Anomaly]) -> list[str]:
    """Format anomalies as a markdown table."""
    if not anomalies:
        return []

    # Check if any anomaly has confidence
    has_confidence = any(a.confidence is not None for a in anomalies)
    has_groups = any(a.group_id is not None for a in anomalies)

    lines: list[str] = []
    lines.append(f"#### Detected anomalies ({len(anomalies)} total)")
    lines.append("")

    if has_confidence:
        header = "| # | Type | Location (RA, Dec) | Detector | Score | Confidence | Key properties |"
        sep = "|---|------|--------------------|----------|-------|------------|----------------|"
    else:
        header = "| # | Type | Location (RA, Dec) | Detector | Score | Key properties |"
        sep = "|---|------|--------------------|----------|-------|----------------|"
    lines.append(header)
    lines.append(sep)

    for i, a in enumerate(anomalies, 1):
        display = _ANOMALY_DISPLAY_NAMES.get(a.anomaly_type, a.anomaly_type.replace("_", " ").capitalize())
        loc = _format_anomaly_location(a)
        score_str = _format_anomaly_score(a)
        props = _format_anomaly_props(a)
        grp = f" [{a.group_id}]" if a.group_id else ""
        if has_confidence:
            conf_str = _format_confidence_str(a)
            lines.append(
                f"| A{i}{grp} | {display} | {loc} | {a.detector} "
                f"| {score_str} | {conf_str} | {props} |"
            )
        else:
            lines.append(
                f"| A{i}{grp} | {display} | {loc} | {a.detector} | {score_str} | {props} |"
            )

    lines.append("")

    # Per-anomaly evidence breakdown (annotation from confidence)
    annotations = [
        (i, a) for i, a in enumerate(anomalies, 1)
        if a.confidence is not None and a.confidence.annotation
    ]
    if annotations:
        lines.append("#### Evidence breakdown")
        lines.append("")
        lines.append("| # | Statistical basis |")
        lines.append("|---|-------------------|")
        for i, a in annotations:
            grp = f" [{a.group_id}]" if a.group_id else ""
            lines.append(f"| A{i}{grp} | {a.confidence.annotation} |")
        lines.append("")

    # Group summaries (pre-build index to avoid O(g*n) repeated scans)
    if has_groups:
        from star_pattern.evaluation.confidence import compute_group_summary_from_members

        groups_by_id: dict[str, list] = {}
        for a in anomalies:
            if a.group_id:
                groups_by_id.setdefault(a.group_id, []).append(a)
        for gid, members in groups_by_id.items():
            summary = compute_group_summary_from_members(members, gid)
            if summary.get("n_members", 0) >= 2:
                lines.append(f"**{summary.get('summary_text', '')}**")
                lines.append("")

    return lines


def _format_finding(
    finding_number: int,
    category_tag: str,
    f: PatternResult,
) -> list[str]:
    """Format a single finding as markdown lines.

    Each finding answers: What was found? How? Does it match known things?
    What's the confidence?
    """
    classification = f.metadata.get("local_classification", {})
    evaluation = f.metadata.get("local_evaluation", {})
    sig_rating = evaluation.get("significance_rating", 0)
    confidence = _confidence_label(sig_rating)
    verdict = evaluation.get("verdict", f.debate_verdict or "unknown")
    snr = evaluation.get("snr", 0)
    n_agree = evaluation.get("n_agreeing_detectors", 0)
    p_value = evaluation.get("look_elsewhere_p", 1.0)
    cls_confidence = classification.get("confidence", 0)
    cls_type = classification.get("classification", f.detection_type)
    display_name = _classification_display_name(cls_type)

    det_scores = classification.get("detector_scores", {})
    active_detectors = {k: v for k, v in det_scores.items() if v > 0.1}
    dominant = max(active_detectors, key=active_detectors.get) if active_detectors else ""

    lines: list[str] = []

    # Header
    lines.append(f"### Finding F{finding_number}: {display_name}")
    lines.append("")
    lines.append(
        f"**Location:** RA={f.region_ra:.4f}, Dec={f.region_dec:.4f} | "
        f"**Confidence:** {confidence} ({sig_rating}/10) | "
        f"**Assessment:** {verdict.capitalize()}"
    )
    lines.append("")

    # What was detected
    lines.append("#### What was detected")
    lines.append("")
    rationale = classification.get("rationale") or f.hypothesis
    if rationale:
        lines.append(rationale)
    else:
        lines.append(f"Classified as {display_name.lower()} based on detector scores.")
    lines.append("")
    feature_lines = _describe_detected_features(f.details)
    if feature_lines:
        lines.extend(feature_lines)
        lines.append("")

    # How it was found
    lines.append("#### How it was found")
    lines.append("")
    if dominant:
        lines.append(
            f"Primary detection by the **{dominant}** detector "
            f"(score {active_detectors[dominant]:.2f})."
        )
        lines.append("")

        # What each active detector contributed
        sub_counts = _extract_sub_detections(f.details)
        det_sub_map = {
            "wavelet": ["wavelet features", "multiscale objects"],
            "lens": ["lens arcs", "lens rings"],
            "galaxy": ["tidal features", "merger nuclei"],
            "distribution": ["overdensities"],
            "classical": ["classical arcs"],
            "sersic": ["Sersic residuals"],
            "kinematic": ["comoving groups", "stream candidates", "runaway stars"],
            "transient": ["transient outliers"],
            "variability": ["variable candidates", "periodic candidates"],
            "population": ["blue stragglers", "red giants"],
            "temporal": [
                "new sources (temporal)", "disappeared sources",
                "brightenings", "fadings", "moving objects",
            ],
        }
        for det_name, score in sorted(
            active_detectors.items(), key=lambda x: x[1], reverse=True,
        ):
            feature_keys = det_sub_map.get(det_name, [])
            found = [
                f"{sub_counts[k]} {k}" for k in feature_keys
                if sub_counts.get(k, 0) > 0
            ]
            if found:
                lines.append(
                    f"- **{det_name}** ({score:.2f}): found {', '.join(found)}"
                )
            elif det_name in ("morphology", "anomaly", "source_extractor"):
                # These detectors don't produce named sub-detections
                lines.append(f"- **{det_name}** ({score:.2f}): signal detected")
            else:
                lines.append(f"- **{det_name}** ({score:.2f}): scored above threshold")
    else:
        lines.append("No individual detector scored above threshold.")
    lines.append("")

    # Per-anomaly table
    if f.anomalies:
        lines.extend(_format_anomaly_table(f.anomalies))

    # Catalog cross-reference
    lines.append("#### Catalog cross-reference")
    lines.append("")
    if f.cross_matches:
        lines.append(f"Matches {len(f.cross_matches)} known object(s):")
        lines.append("")
        for m in f.cross_matches:
            name = m.get("name", "N/A")
            otype = m.get("object_type", "")
            decoded = _decode_simbad_type(otype) if otype else "unknown type"
            sep = m.get("separation_arcsec")
            sep_str = f", separation {sep:.1f} arcsec" if sep is not None else ""
            catalog = m.get("catalog", "")
            cat_str = f" via {catalog}" if catalog else ""
            lines.append(f"- **{name}** -- {decoded}{sep_str}{cat_str}")
        if category_tag == "KNOWN":
            lines.append("")
            lines.append(
                "These matches confirm the pipeline detected real astronomical "
                "objects at this location."
            )
    else:
        lines.append(
            "No matches found in SIMBAD, NED, or TNS catalogs. "
            "This object may be previously uncatalogued and warrants follow-up observation."
        )
    lines.append("")

    # Evidence table
    lines.append("#### Evidence")
    lines.append("")
    lines.append("| Measure | Value | Interpretation |")
    lines.append("|---------|-------|----------------|")
    lines.append(
        f"| SNR | {snr:.1f} | {_interpret_metric('snr', snr)} |"
    )
    lines.append(
        f"| Agreeing detectors | {n_agree}/13 | "
        f"{_interpret_metric('n_agreeing_detectors', n_agree)} |"
    )
    if p_value < 1.0:
        lines.append(
            f"| Look-elsewhere p-value | {p_value:.2e} | "
            f"{_interpret_metric('p_value', p_value)} |"
        )
    if cls_confidence > 0:
        lines.append(
            f"| Classification confidence | {cls_confidence:.0%} | "
            f"{_interpret_metric('confidence', cls_confidence)} |"
        )
    lines.append("")

    # Suggested follow-up (new discoveries only)
    if category_tag == "NEW":
        follow_up = classification.get("follow_up", [])
        if follow_up:
            lines.append("#### Suggested follow-up")
            lines.append("")
            for step in follow_up:
                lines.append(f"- {step}")
            lines.append("")

    return lines


class DiscoveryReport:
    """Generate reports from discovery results."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a markdown discovery report.

        Sections:
        1. Header -- run name, cycles, regions
        2. Summary -- narrative paragraph, simple counts
        3. Findings -- all new + known findings with full format (F1, F2, ...)
        4. Appendix: Low-Confidence Detections -- compact table
        5. Technical Details -- per-detector summary + type breakdown
        6. Parameter Evolution
        """
        report_path = self.output_dir / "report.md"
        meta = run_metadata or {}
        lines: list[str] = []

        # Assign finding numbers for consistent F1, F2, ... across report
        numbered = _assign_finding_numbers(findings)
        new_findings = [(n, tag, f) for n, tag, f in numbered if tag == "NEW"]
        known_findings = [(n, tag, f) for n, tag, f in numbered if tag == "KNOWN"]
        low_findings = [(n, tag, f) for n, tag, f in numbered if tag == "LOW"]

        # --- Header ---
        lines.append("# Star Pattern AI -- Discovery Report")
        lines.append("")
        if meta:
            lines.append(
                f"**Run:** {meta.get('run_name', 'unknown')} | "
                f"**Cycles:** {meta.get('n_cycles', 'N/A')} | "
                f"**Regions searched:** {meta.get('n_regions', 'N/A')}"
            )
            token_usage = meta.get("token_usage", {})
            if token_usage:
                lines.append(
                    f"**LLM tokens used:** {token_usage.get('total_tokens', 0):,}"
                )
            lines.append("")

        # --- Summary ---
        lines.append("## Summary")
        lines.append("")

        n_total = len(findings)

        if n_total == 0:
            lines.append("No findings were detected in this run.")
            lines.append("")
        else:
            n_follow = len(new_findings)
            n_known = len(known_findings)
            n_low = len(low_findings)
            n_regions = meta.get("n_regions", "?")

            # Opening context
            lines.append(
                f"Searched {n_regions} sky region(s), producing "
                f"{n_total} finding(s)."
            )
            lines.append("")

            # Explicit new-vs-known framing
            if n_follow > 0:
                top_new = new_findings[0]
                top_name = _classification_display_name(
                    top_new[2].metadata.get("local_classification", {}).get(
                        "classification", top_new[2].detection_type,
                    ),
                )
                lines.append(
                    f"**{n_follow} potentially new discovery(ies)** "
                    f"with no catalog match -- these warrant follow-up. "
                    f"The strongest is F{top_new[0]} ({top_name.lower()})."
                )
            else:
                lines.append(
                    "**No new discoveries** in this run. "
                    "All findings matched known catalog objects."
                )

            if n_known > 0:
                lines.append(
                    f"**{n_known} known object(s) confirmed** -- "
                    f"the pipeline correctly identified catalogued objects, "
                    f"validating detection accuracy."
                )

            if n_low > 0:
                lines.append(
                    f"**{n_low} low-confidence detection(s)** "
                    f"assessed as likely artifacts or below significance threshold."
                )
            lines.append("")

            # Anomaly-level summary across all findings
            all_anomalies = [a for f in findings for a in f.anomalies]
            if all_anomalies:
                type_counts: dict[str, int] = {}
                for a in all_anomalies:
                    display = _ANOMALY_DISPLAY_NAMES.get(
                        a.anomaly_type,
                        a.anomaly_type.replace("_", " "),
                    )
                    type_counts[display] = type_counts.get(display, 0) + 1

                n_anom = len(all_anomalies)
                n_with_coords = sum(
                    1 for a in all_anomalies
                    if a.sky_ra is not None
                )
                n_no_coords = n_anom - n_with_coords
                type_list = ", ".join(
                    f"{count} {name.lower()}" + ("s" if count > 1 else "")
                    for name, count in sorted(
                        type_counts.items(), key=lambda x: -x[1],
                    )
                )
                lines.append(
                    f"**Anomaly summary:** {n_anom} individual anomalies "
                    f"across {n_total} finding(s): {type_list}."
                )
                lines.append("")

        # --- Visual Overview ---
        # Image files are generated by generate_full_report in the same
        # directory as this markdown file; embed them with relative paths.
        lines.append("## Visual Overview")
        lines.append("")
        lines.append("![Discovery Mosaic](mosaic.png)")
        lines.append("")
        lines.append(
            "*Top anomalies with cutouts centered on each detection. "
            "Overview panel shows the full field with anomaly locations marked.*"
        )
        lines.append("")

        # --- Findings (new + known, with full format) ---
        main_findings = new_findings + known_findings
        if main_findings:
            lines.append("## Findings")
            lines.append("")
            for finding_num, tag, f in main_findings:
                lines.extend(_format_finding(finding_num, tag, f))

        # --- Appendix: Low-Confidence Detections ---
        if low_findings:
            lines.append("## Appendix: Low-Confidence Detections")
            lines.append("")
            lines.append(
                f"{len(low_findings)} detection(s) evaluated as artifacts "
                "or below confidence threshold."
            )
            lines.append("")

            lines.append(
                "| Finding | Type | RA | Dec | Score | Assessment | Confidence |"
            )
            lines.append(
                "|---------|------|-----|-----|-------|------------|------------|"
            )
            for finding_num, _tag, f in low_findings:
                ev = f.metadata.get("local_evaluation", {})
                verdict = ev.get("verdict", "?")
                rating = ev.get("significance_rating", 0)
                cls = f.metadata.get("local_classification", {})
                dtype = cls.get("classification", f.detection_type)
                display = _classification_display_name(dtype)
                lines.append(
                    f"| F{finding_num} | {display} | {f.region_ra:.3f} "
                    f"| {f.region_dec:.3f} | {f.anomaly_score:.3f} "
                    f"| {verdict.capitalize()} | {_confidence_label(rating)} "
                    f"({rating}/10) |"
                )
            lines.append("")

        # --- Technical Details ---
        if findings:
            lines.append("## Technical Details")
            lines.append("")
            lines.append("### Score Distribution")
            lines.append("")
            lines.append("![Anomaly Score Distribution](scores_histogram.png)")
            lines.append("")

            # Per-detector summary
            det_table = _detector_summary_table(findings)
            if det_table:
                lines.append("### Per-Detector Summary")
                lines.append("")
                lines.extend(det_table)

            # Detection type breakdown
            lines.append("### Detection Type Breakdown")
            lines.append("")

            by_type: dict[str, list[PatternResult]] = {}
            for f in findings:
                cls = f.metadata.get("local_classification", {})
                dtype = cls.get("classification", f.detection_type)
                by_type.setdefault(dtype, []).append(f)

            lines.append("| Type | Count | Mean Score | Max Score | Real | Artifacts |")
            lines.append("|------|-------|------------|-----------|------|-----------|")
            for dtype, group in sorted(by_type.items(), key=lambda x: -len(x[1])):
                scores = [f.anomaly_score for f in group]
                n_real_type = sum(
                    1 for f in group
                    if f.metadata.get("local_evaluation", {}).get("verdict") == "real"
                )
                n_artifact = sum(
                    1 for f in group
                    if f.metadata.get("local_evaluation", {}).get("verdict") == "artifact"
                )
                lines.append(
                    f"| {_classification_display_name(dtype)} | {len(group)} "
                    f"| {np.mean(scores):.4f} | {np.max(scores):.4f} "
                    f"| {n_real_type} | {n_artifact} |"
                )
            lines.append("")

        # --- Parameter Evolution ---
        evolution = meta.get("evolution_history", [])
        if evolution:
            lines.append("## Parameter Evolution")
            lines.append("")
            lines.append(
                f"Detection parameters were evolved {len(evolution)} time(s) "
                "via genetic algorithm."
            )
            lines.append("")
            lines.append("| Cycle | Fitness |")
            lines.append("|-------|---------|")
            for entry in evolution:
                lines.append(
                    f"| {entry['cycle']} | {entry['fitness']:.4f} |"
                )
            lines.append("")

        report_path.write_text("\n".join(lines))
        logger.info(f"Report saved to {report_path}")
        return report_path

    def generate_json_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a JSON report."""
        report_path = self.output_dir / "report.json"
        data = {
            "metadata": run_metadata or {},
            "n_findings": len(findings),
            "findings": [f.to_dict() for f in findings],
        }
        report_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"JSON report saved to {report_path}")
        return report_path

    def generate_full_report(
        self,
        findings: list[PatternResult],
        run_metadata: dict[str, Any] | None = None,
        images: list[Any] | None = None,
    ) -> dict[str, Path]:
        """Generate all report formats.

        Each step is fault-isolated so that a failure in one
        (e.g. mosaic) does not prevent the others (markdown, JSON).
        """
        import traceback

        paths: dict[str, Path] = {}

        # Markdown report
        try:
            paths["markdown"] = self.generate_markdown_report(findings, run_metadata)
        except Exception as e:
            logger.warning(f"Markdown report failed: {e}\n{traceback.format_exc()}")

        # JSON report
        try:
            paths["json"] = self.generate_json_report(findings, run_metadata)
        except Exception as e:
            logger.warning(f"JSON report failed: {e}\n{traceback.format_exc()}")

        # Generate plots
        try:
            from star_pattern.visualization.mosaic import (
                create_discovery_mosaic,
                create_score_histogram,
            )
            import matplotlib.pyplot as plt

            mosaic = create_discovery_mosaic(findings, images=images)
            mosaic_path = self.output_dir / "mosaic.png"
            mosaic.savefig(str(mosaic_path), dpi=150, bbox_inches="tight")
            paths["mosaic"] = mosaic_path
            plt.close(mosaic)

            hist = create_score_histogram(findings)
            hist_path = self.output_dir / "scores_histogram.png"
            hist.savefig(str(hist_path), dpi=150, bbox_inches="tight")
            paths["histogram"] = hist_path
            plt.close(hist)

        except Exception as e:
            logger.warning(f"Plot generation failed: {e}\n{traceback.format_exc()}")

        return paths
