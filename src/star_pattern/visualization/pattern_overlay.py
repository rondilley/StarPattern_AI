"""Overlay detections on FITS images."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Arc, FancyArrowPatch

from star_pattern.core.fits_handler import FITSImage
from star_pattern.utils.logging import get_logger

logger = get_logger("visualization.overlay")


def overlay_sources(
    image: FITSImage,
    sources: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay extracted sources on an image."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    positions = np.array(sources.get("positions", []))
    if positions.ndim != 2:
        positions = np.empty((0, 2))
    if len(positions) > 0:
        star_mask = np.array(
            sources.get("star_mask", np.ones(len(positions), dtype=bool)),
            dtype=bool,
        )
        stars = positions[star_mask]
        galaxies = positions[~star_mask]

        if len(stars) > 0:
            ax.scatter(stars[:, 0], stars[:, 1], s=20, marker="*",
                       facecolors="none", edgecolors="cyan", linewidth=0.5, label="Stars")
        if len(galaxies) > 0:
            ax.scatter(galaxies[:, 0], galaxies[:, 1], s=30, marker="o",
                       facecolors="none", edgecolors="lime", linewidth=0.5, label="Galaxies")

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Source extraction: {len(positions)} objects")
    plt.tight_layout()
    return fig


def overlay_lens_detection(
    image: FITSImage,
    lens_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay lens detection results (arcs, rings) on an image."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    central = lens_result.get("central_source", {})
    if central:
        ax.plot(central["x"], central["y"], "r+", markersize=15, markeredgewidth=2)

    # Draw arc detections
    for arc in lens_result.get("arcs", []):
        circle = Circle(
            (central.get("x", 128), central.get("y", 128)),
            arc["radius"],
            fill=False,
            edgecolor="yellow",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(circle)

    # Draw ring detections
    for ring in lens_result.get("rings", []):
        circle = Circle(
            (central.get("x", 128), central.get("y", 128)),
            ring["radius"],
            fill=False,
            edgecolor="lime" if ring.get("is_complete_ring") else "orange",
            linewidth=2,
        )
        ax.add_patch(circle)

    score = lens_result.get("lens_score", 0)
    candidate = "LENS CANDIDATE" if lens_result.get("is_candidate") else ""
    ax.set_title(f"Lens Detection (score={score:.3f}) {candidate}")

    plt.tight_layout()
    return fig


def overlay_distribution(
    image: FITSImage,
    dist_result: dict[str, Any],
    positions: np.ndarray | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay distribution analysis results."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r", alpha=0.7)

    # Plot source positions
    if positions is not None:
        positions = np.asarray(positions)
    if positions is not None and positions.ndim == 2 and len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], s=5, c="cyan", alpha=0.5)

    # Highlight overdensities
    for od in dist_result.get("overdensities", []):
        circle = Circle(
            (od["x"], od["y"]),
            20,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(circle)
        ax.text(
            od["x"], od["y"] + 25,
            f"{od['sigma']:.1f}σ",
            color="red",
            fontsize=8,
            ha="center",
        )

    ce = dist_result.get("clark_evans_r", 1.0)
    ax.set_title(f"Distribution (Clark-Evans={ce:.2f})")

    plt.tight_layout()
    return fig


def overlay_kinematic_groups(
    kinematic_result: dict[str, Any],
    positions: np.ndarray | None = None,
    pm_data: np.ndarray | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay proper motion vectors and co-moving groups.

    Args:
        kinematic_result: Output from ProperMotionAnalyzer.analyze().
        positions: Nx2 array of (ra, dec) positions.
        pm_data: Nx2 array of (pmra, pmdec) proper motions.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with PM vectors and group annotations.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot field stars with PM arrows
    if positions is not None and pm_data is not None and len(positions) > 0:
        # Scale arrows for visibility
        pm_mag = np.sqrt(pm_data[:, 0]**2 + pm_data[:, 1]**2)
        scale = 0.01 if np.max(pm_mag) > 0 else 1.0

        ax.quiver(
            positions[:, 0], positions[:, 1],
            pm_data[:, 0] * scale, pm_data[:, 1] * scale,
            pm_mag, cmap="viridis", alpha=0.5, scale=1, scale_units="xy",
        )

    # Color co-moving groups
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
    for i, group in enumerate(kinematic_result.get("comoving_groups", [])):
        color = colors[i % len(colors)]
        ax.plot(
            group["mean_ra"], group["mean_dec"],
            "o", color=color, markersize=10, markeredgewidth=2,
            markerfacecolor="none",
        )
        ax.annotate(
            f"G{i+1} (n={group['n_members']})",
            (group["mean_ra"], group["mean_dec"]),
            textcoords="offset points", xytext=(10, 10),
            fontsize=8, color=color,
        )

    # Mark runaway stars
    for star in kinematic_result.get("runaway_stars", []):
        ax.plot(star["ra"], star["dec"], "x", color="red", markersize=12, markeredgewidth=2)

    n_groups = len(kinematic_result.get("comoving_groups", []))
    n_runaways = len(kinematic_result.get("runaway_stars", []))
    score = kinematic_result.get("kinematic_score", 0)
    ax.set_title(f"Kinematics (score={score:.3f}, groups={n_groups}, runaways={n_runaways})")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")

    plt.tight_layout()
    return fig


def overlay_galaxy_features(
    image: FITSImage,
    galaxy_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay galaxy detection results (tidal features, mergers).

    Args:
        image: FITSImage to use as background.
        galaxy_result: Output from GalaxyDetector.detect().
        figsize: Figure size.

    Returns:
        Matplotlib Figure with galaxy feature annotations.
    """
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    # Mark tidal features
    for feat in galaxy_result.get("tidal_features", []):
        circle = Circle(
            (feat["x"], feat["y"]),
            max(10, feat.get("area", 100) ** 0.5),
            fill=False,
            edgecolor="magenta",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.text(
            feat["x"], feat["y"] + 15,
            "tidal",
            color="magenta",
            fontsize=7,
            ha="center",
        )

    # Mark merger candidates
    for merger in galaxy_result.get("merger_candidates", []):
        n1 = merger["nucleus_1"]
        n2 = merger["nucleus_2"]
        ax.plot(n1["x"], n1["y"], "+", color="red", markersize=12, markeredgewidth=2)
        ax.plot(n2["x"], n2["y"], "+", color="red", markersize=12, markeredgewidth=2)
        ax.plot(
            [n1["x"], n2["x"]], [n1["y"], n2["y"]],
            "--", color="red", linewidth=1, alpha=0.7,
        )
        mid_x = (n1["x"] + n2["x"]) / 2
        mid_y = (n1["y"] + n2["y"]) / 2
        ax.text(
            mid_x, mid_y + 10,
            f"merger (A={merger['asymmetry']:.2f})",
            color="red",
            fontsize=7,
            ha="center",
        )

    score = galaxy_result.get("galaxy_score", 0)
    n_tidal = len(galaxy_result.get("tidal_features", []))
    n_mergers = len(galaxy_result.get("merger_candidates", []))
    ax.set_title(f"Galaxy Features (score={score:.3f}, tidal={n_tidal}, mergers={n_mergers})")

    plt.tight_layout()
    return fig


def overlay_classical_detection(
    image: FITSImage,
    classical_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay classical pattern detection (Hough arcs, Gabor/FFT scores)."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    for arc in classical_result.get("arcs", []):
        cx = arc.get("center_x", arc.get("x", 0))
        cy = arc.get("center_y", arc.get("y", 0))
        r = arc.get("radius", 20)
        circle = Circle(
            (cx, cy), r,
            fill=False, edgecolor="#ff6600", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)
        strength = arc.get("strength", 0)
        if strength > 0:
            ax.text(
                cx, cy + r + 5, f"{strength:.2f}",
                color="#ff6600", fontsize=7, ha="center",
            )

    gabor = classical_result.get("gabor_score", 0)
    fft = classical_result.get("fft_score", 0)
    arc_score = classical_result.get("arc_score", 0)
    n_arcs = classical_result.get("n_arcs", len(classical_result.get("arcs", [])))
    ax.set_title(
        f"Classical: Gabor={gabor:.3f} FFT={fft:.3f} "
        f"Arc={arc_score:.3f} ({n_arcs} arcs)"
    )

    plt.tight_layout()
    return fig


def overlay_morphology(
    image: FITSImage,
    morphology_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay morphology CAS/Gini/M20 values on an image."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    # Build text block with morphology values
    lines = []
    for key, label in [
        ("concentration", "C"),
        ("asymmetry", "A"),
        ("smoothness", "S"),
        ("gini", "Gini"),
        ("m20", "M20"),
    ]:
        val = morphology_result.get(key)
        if val is not None:
            lines.append(f"{label} = {val:.3f}")

    if lines:
        text = "\n".join(lines)
        ax.text(
            0.03, 0.97, text,
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6),
            color="white",
        )

    score = morphology_result.get("morphology_score", 0)
    ax.set_title(f"Morphology (score={score:.3f})")

    plt.tight_layout()
    return fig


def overlay_sersic_analysis(
    image: FITSImage,
    sersic_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay Sersic analysis: residual features and radial profile inset."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    for feat in sersic_result.get("residual_features", []):
        radius = max(5, feat.get("area_px", 50) ** 0.5)
        circle = Circle(
            (feat["x"], feat["y"]), radius,
            fill=False, edgecolor="white", linewidth=1, linestyle=":",
        )
        ax.add_patch(circle)

    # Radial profile inset (minimal, top-right corner)
    rp = sersic_result.get("radial_profile", {})
    if rp.get("radii") and rp.get("intensity"):
        inset = ax.inset_axes([0.78, 0.78, 0.18, 0.16])
        inset.plot(rp["radii"], rp["intensity"], "c-", linewidth=0.8)
        inset.set_xlabel("r", fontsize=5, labelpad=1)
        inset.set_ylabel("I", fontsize=5, labelpad=1)
        inset.tick_params(labelsize=4, pad=1)
        inset.set_title("Radial", fontsize=5, pad=2)
        inset.patch.set_alpha(0.7)

    sersic_n = sersic_result.get("sersic_n", 0)
    r_e = sersic_result.get("r_e", 0)
    morph_class = sersic_result.get("morphology_class", "?")
    score = sersic_result.get("sersic_score", 0)
    ax.set_title(f"Sersic (score={score:.3f}, n={sersic_n:.1f}, r_e={r_e:.1f}, {morph_class})")

    plt.tight_layout()
    return fig


def overlay_wavelet_detection(
    image: FITSImage,
    wavelet_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay wavelet multi-scale detections with scale-colored circles."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    scale_colors = ["#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"]
    for det in wavelet_result.get("detections", []):
        scale_idx = det.get("scale", 0)
        color = scale_colors[min(scale_idx, len(scale_colors) - 1)]
        radius = max(3, det.get("area_px", 25) ** 0.5)
        circle = Circle(
            (det["x"], det["y"]), radius,
            fill=False, edgecolor=color, linewidth=1, alpha=0.7,
        )
        ax.add_patch(circle)

    for ms in wavelet_result.get("multiscale_objects", []):
        ax.plot(
            ms["x"], ms["y"], "s",
            color="white", markersize=8, markeredgewidth=1.5,
            markerfacecolor="none",
        )

    # Scale spectrum inset bar chart (minimal, top-right corner)
    spectrum = wavelet_result.get("scale_spectrum", [])
    if spectrum:
        inset = ax.inset_axes([0.82, 0.80, 0.15, 0.14])
        bar_colors = [
            scale_colors[min(i, len(scale_colors) - 1)]
            for i in range(len(spectrum))
        ]
        inset.bar(range(len(spectrum)), spectrum, color=bar_colors, alpha=0.8)
        inset.set_xticks(range(len(spectrum)))
        inset.set_xticklabels([str(i) for i in range(len(spectrum))], fontsize=4)
        inset.tick_params(axis="y", labelsize=4, pad=1)
        inset.tick_params(axis="x", pad=1)
        inset.set_title("Scale", fontsize=5, pad=2)
        inset.patch.set_alpha(0.7)

    score = wavelet_result.get("wavelet_score", 0)
    n_det = len(wavelet_result.get("detections", []))
    n_ms = len(wavelet_result.get("multiscale_objects", []))
    ax.set_title(f"Wavelet (score={score:.3f}, {n_det} det, {n_ms} multi-scale)")

    plt.tight_layout()
    return fig


def overlay_transient_detection(
    image: FITSImage,
    transient_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay transient detection: astrometric/photometric/parallax outliers on RA/Dec."""
    fig, ax = plt.subplots(figsize=figsize)

    for outlier in transient_result.get("astrometric_outliers", []):
        ax.plot(
            outlier["ra"], outlier["dec"],
            "^", color="red", markersize=8, alpha=0.8,
            label="Astrometric",
        )
    for outlier in transient_result.get("photometric_outliers", []):
        ax.plot(
            outlier["ra"], outlier["dec"],
            "v", color="orange", markersize=8, alpha=0.8,
            label="Photometric",
        )
    for outlier in transient_result.get("parallax_anomalies", []):
        ax.plot(
            outlier["ra"], outlier["dec"],
            "d", color="yellow", markersize=8, alpha=0.8,
            label="Parallax",
        )

    # De-duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    score = transient_result.get("transient_score", 0)
    ax.set_title(f"Transient Detection (score={score:.3f})")

    plt.tight_layout()
    return fig


def overlay_variability(
    image: FITSImage,
    variability_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay variability analysis: variable/periodic/transient candidates on RA/Dec."""
    fig, ax = plt.subplots(figsize=figsize)

    for vc in variability_result.get("variable_candidates", []):
        ax.plot(
            vc["ra"], vc["dec"],
            "s", color="cyan", markersize=6, markerfacecolor="none",
            markeredgewidth=1, label="Variable",
        )
    for pc in variability_result.get("periodic_candidates", []):
        ax.plot(
            pc["ra"], pc["dec"],
            "o", color="lime", markersize=7, markerfacecolor="none",
            markeredgewidth=1.5, label="Periodic",
        )
    for tc in variability_result.get("transient_candidates", []):
        ax.plot(
            tc["ra"], tc["dec"],
            "*", color="red", markersize=10, label="Transient",
        )

    # De-duplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    score = variability_result.get("variability_score", 0)
    n_var = len(variability_result.get("variable_candidates", []))
    n_per = len(variability_result.get("periodic_candidates", []))
    n_trans = len(variability_result.get("transient_candidates", []))
    ax.set_title(
        f"Variability (score={score:.3f}, var={n_var}, per={n_per}, trans={n_trans})"
    )

    plt.tight_layout()
    return fig


def overlay_population_cmd(
    image: FITSImage,
    population_result: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay stellar population CMD: peaks, blue stragglers, red giants."""
    fig, ax = plt.subplots(figsize=figsize)

    # CMD peaks
    peaks = population_result.get("cmd_peaks", [])
    if peaks:
        colors_cmd = [p.get("color", 0) for p in peaks]
        mags = [p.get("mag", 0) for p in peaks]
        ax.scatter(
            colors_cmd, mags, s=30, c="blue", alpha=0.6,
            label=f"CMD peaks ({len(peaks)})",
        )

    # Blue straggler candidates
    bs_cands = population_result.get("blue_straggler_candidates", [])
    if bs_cands:
        bs_colors = [c.get("color", 0) for c in bs_cands]
        bs_mags = [c.get("mag", 0) for c in bs_cands]
        ax.scatter(
            bs_colors, bs_mags, s=25, c="cyan", marker="^",
            label=f"Blue str. ({len(bs_cands)})",
        )

    # Red giant candidates
    rg_cands = population_result.get("red_giant_candidates", [])
    if rg_cands:
        rg_colors = [c.get("color", 0) for c in rg_cands]
        rg_mags = [c.get("mag", 0) for c in rg_cands]
        ax.scatter(
            rg_colors, rg_mags, s=25, c="red", marker="v",
            label=f"RGB ({len(rg_cands)})",
        )

    # Turnoff point
    turnoff = population_result.get("turnoff", {})
    if turnoff.get("color") is not None and turnoff.get("mag") is not None:
        ax.plot(
            turnoff["color"], turnoff["mag"],
            "P", color="white", markersize=12, markeredgecolor="black",
            markeredgewidth=1.5, label="Turnoff",
        )

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Color (BP-RP)")
    ax.set_ylabel("Mag (G)")
    score = population_result.get("population_score", 0)
    ax.set_title(f"Stellar Population CMD (score={score:.3f})")

    plt.tight_layout()
    return fig


def overlay_anomaly_scores(
    image: FITSImage,
    detection: dict[str, Any],
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Overlay anomaly score bar chart on the image."""
    fig, ax = plt.subplots(figsize=figsize)

    norm = image.normalize("zscale")
    ax.imshow(norm.data, origin="lower", cmap="gray_r")

    # Inset bar chart of detector scores
    detector_names, det_scores = _get_detector_scores(detection)
    inset = ax.inset_axes([0.02, 0.02, 0.4, 0.55])
    bar_colors = [
        "#e74c3c" if s > 0.3 else "#f39c12" if s > 0.1 else "#95a5a6"
        for s in det_scores
    ]
    inset.barh(detector_names, det_scores, color=bar_colors)
    inset.set_xlim(0, max(max(det_scores) * 1.2, 0.5) if det_scores else 0.5)
    inset.tick_params(axis="y", labelsize=6)
    inset.tick_params(axis="x", labelsize=6)
    inset.set_title("Detector scores", fontsize=7)
    inset.patch.set_alpha(0.8)

    ensemble = detection.get("anomaly_score", 0)
    ax.set_title(f"Anomaly Overview (ensemble={ensemble:.4f})")

    plt.tight_layout()
    return fig


def _get_detector_scores(
    detection: dict[str, Any],
) -> tuple[list[str], list[float]]:
    """Extract detector names and scores from ensemble results."""
    detector_names = [
        "classical", "morphology", "anomaly", "lens", "distribution",
        "galaxy", "kinematic", "transient", "sersic", "wavelet",
        "population", "variability",
    ]
    score_keys = {
        "classical": ("classical", "gabor_score"),
        "morphology": ("morphology", "morphology_score"),
        "anomaly": ("anomaly_score",),
        "lens": ("lens", "lens_score"),
        "distribution": ("distribution", "distribution_score"),
        "galaxy": ("galaxy", "galaxy_score"),
        "kinematic": ("kinematic", "kinematic_score"),
        "transient": ("transient", "transient_score"),
        "sersic": ("sersic", "sersic_score"),
        "wavelet": ("wavelet", "wavelet_score"),
        "population": ("population", "population_score"),
        "variability": ("variability", "variability_score"),
    }
    det_scores = []
    for name in detector_names:
        keys = score_keys[name]
        if len(keys) == 1:
            det_scores.append(float(detection.get(keys[0], 0)))
        else:
            det_scores.append(
                float(detection.get(keys[0], {}).get(keys[1], 0))
            )
    return detector_names, det_scores


def create_annotated_summary(
    image: FITSImage,
    detection: dict[str, Any],
    ra: float = 0.0,
    dec: float = 0.0,
    cycle: int = 0,
    elapsed_minutes: float = 0.0,
    classification: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    evolution_history: list[dict[str, Any]] | None = None,
    figsize: tuple[int, int] = (24, 20),
) -> Figure:
    """Create a 4x3 multi-panel annotated summary image for a region.

    Layout:
      Row 0: [Composite Detection] [Galaxy+Morphology] [Lens+Sersic]
      Row 1: [Distribution+Sources] [Wavelet Multi-scale] [Classical Arcs]
      Row 2: [Kinematic+PM]         [Transient+Variab.]  [Stellar Pop CMD]
      Row 3: [Detector Scores]      [Evolution+Learning] [Classification]

    Args:
        image: FITSImage to visualize.
        detection: Full ensemble detection result dict (with spatial data).
        ra: Region RA in degrees.
        dec: Region Dec in degrees.
        cycle: Current pipeline cycle number.
        elapsed_minutes: Minutes elapsed since pipeline start.
        classification: LocalClassifier result dict.
        evaluation: LocalEvaluator result dict.
        evolution_history: List of evolution cycle dicts with fitness data.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with all 12 panels.
    """
    fig, axes = plt.subplots(4, 3, figsize=figsize)

    norm = image.normalize("zscale")
    norm_data = norm.data

    sources = detection.get("sources", {})
    positions = np.array(sources.get("positions", []))
    if len(positions) == 0:
        positions = np.empty((0, 2))

    # ---- Row 0, Col 0: Composite Detection Map ----
    ax = axes[0, 0]
    ax.imshow(norm_data, origin="lower", cmap="gray_r")
    # Sources
    if len(positions) > 0:
        star_mask = np.array(
            sources.get("star_mask", [True] * len(positions))
        )
        stars = np.array(positions)[star_mask]
        galaxies_pos = np.array(positions)[~star_mask]
        if len(stars) > 0:
            ax.scatter(
                np.array(stars)[:, 0], np.array(stars)[:, 1],
                s=12, marker="*", facecolors="none", edgecolors="cyan",
                linewidth=0.5, label=f"Stars ({len(stars)})",
            )
        if len(galaxies_pos) > 0:
            ax.scatter(
                np.array(galaxies_pos)[:, 0],
                np.array(galaxies_pos)[:, 1],
                s=18, marker="o", facecolors="none", edgecolors="lime",
                linewidth=0.5, label=f"Galaxies ({len(galaxies_pos)})",
            )
    # Overdensities
    dist = detection.get("distribution", {})
    for od in dist.get("overdensities", []):
        circle = Circle(
            (od["x"], od["y"]), 20,
            fill=False, edgecolor="red", linewidth=1.5,
        )
        ax.add_patch(circle)
    # Lens arcs
    lens = detection.get("lens", {})
    central = lens.get("central_source", {})
    for arc_item in lens.get("arcs", []):
        circle = Circle(
            (central.get("x", 128), central.get("y", 128)),
            arc_item["radius"],
            fill=False, edgecolor="yellow", linewidth=1, linestyle="--",
            alpha=0.6,
        )
        ax.add_patch(circle)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.set_title(f"Composite ({len(positions)} sources, {detection.get('n_detections', 0)} det)")

    # ---- Row 0, Col 1: Galaxy + Morphology ----
    ax = axes[0, 1]
    ax.imshow(norm_data, origin="lower", cmap="gray_r")
    galaxy = detection.get("galaxy", {})
    for feat in galaxy.get("tidal_features", []):
        circle = Circle(
            (feat["x"], feat["y"]),
            max(10, feat.get("area", 100) ** 0.5),
            fill=False, edgecolor="magenta", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)
        ax.text(
            feat["x"], feat["y"] + 15, "tidal",
            color="magenta", fontsize=6, ha="center",
        )
    for merger in galaxy.get("merger_candidates", []):
        n1, n2 = merger["nucleus_1"], merger["nucleus_2"]
        ax.plot(n1["x"], n1["y"], "+", color="red", markersize=10, markeredgewidth=2)
        ax.plot(n2["x"], n2["y"], "+", color="red", markersize=10, markeredgewidth=2)
        ax.plot([n1["x"], n2["x"]], [n1["y"], n2["y"]], "--", color="red", alpha=0.7)
    morphology = detection.get("morphology", {})
    cas_str = ""
    if morphology.get("concentration"):
        cas_str = (
            f" C={morphology['concentration']:.2f}"
            f" A={morphology['asymmetry']:.2f}"
            f" S={morphology['smoothness']:.2f}"
        )
    ax.set_title(
        f"Galaxy={galaxy.get('galaxy_score', 0):.3f}"
        f" Morph={morphology.get('morphology_score', 0):.3f}{cas_str}"
    )

    # ---- Row 0, Col 2: Lens + Sersic ----
    ax = axes[0, 2]
    ax.imshow(norm_data, origin="lower", cmap="gray_r")
    if central:
        ax.plot(central["x"], central["y"], "r+", markersize=12, markeredgewidth=2)
    for arc_item in lens.get("arcs", []):
        circle = Circle(
            (central.get("x", 128), central.get("y", 128)),
            arc_item["radius"],
            fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)
    for ring in lens.get("rings", []):
        circle = Circle(
            (central.get("x", 128), central.get("y", 128)),
            ring["radius"],
            fill=False,
            edgecolor="lime" if ring.get("is_complete_ring") else "orange",
            linewidth=2,
        )
        ax.add_patch(circle)
    # Sersic residual features
    sersic = detection.get("sersic", {})
    for feat in sersic.get("residual_features", []):
        circle = Circle(
            (feat["x"], feat["y"]),
            max(5, feat.get("area_px", 50) ** 0.5),
            fill=False, edgecolor="white", linewidth=1, linestyle=":",
        )
        ax.add_patch(circle)
    # Radial profile inset
    rp = sersic.get("radial_profile", {})
    if rp.get("radii") and rp.get("intensity"):
        inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        inset.plot(rp["radii"], rp["intensity"], "c-", linewidth=1)
        inset.set_xlabel("r (px)", fontsize=5)
        inset.set_ylabel("I", fontsize=5)
        inset.tick_params(labelsize=5)
        inset.set_title("Sersic profile", fontsize=6)
    lens_score = lens.get("lens_score", 0)
    sersic_n = sersic.get("sersic_n", 0)
    candidate_str = " CANDIDATE" if lens.get("is_candidate") else ""
    ax.set_title(
        f"Lens={lens_score:.3f}{candidate_str} Sersic n={sersic_n:.1f}"
    )

    # ---- Row 1, Col 0: Distribution + Sources ----
    ax = axes[1, 0]
    ax.imshow(norm_data, origin="lower", cmap="gray_r", alpha=0.7)
    if len(positions) > 0:
        ax.scatter(
            np.array(positions)[:, 0], np.array(positions)[:, 1],
            s=3, c="cyan", alpha=0.4,
        )
    for od in dist.get("overdensities", []):
        circle = Circle(
            (od["x"], od["y"]), 20,
            fill=False, edgecolor="red", linewidth=2,
        )
        ax.add_patch(circle)
        ax.text(
            od["x"], od["y"] + 25,
            f"{od['sigma']:.1f}sig",
            color="red", fontsize=7, ha="center",
        )
    ce = dist.get("clark_evans_r", 1.0)
    ax.set_title(
        f"Distribution (CE={ce:.2f}, "
        f"{dist.get('n_overdensities', len(dist.get('overdensities', [])))} OD)"
    )

    # ---- Row 1, Col 1: Wavelet Multi-scale ----
    ax = axes[1, 1]
    ax.imshow(norm_data, origin="lower", cmap="gray_r", alpha=0.7)
    wavelet = detection.get("wavelet", {})
    scale_colors = ["#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"]
    for det_item in wavelet.get("detections", []):
        scale_idx = det_item.get("scale", 0)
        color = scale_colors[min(scale_idx, len(scale_colors) - 1)]
        radius = max(3, det_item.get("area_px", 25) ** 0.5)
        circle = Circle(
            (det_item["x"], det_item["y"]), radius,
            fill=False, edgecolor=color, linewidth=1, alpha=0.7,
        )
        ax.add_patch(circle)
    for ms in wavelet.get("multiscale_objects", []):
        ax.plot(
            ms["x"], ms["y"], "s",
            color="white", markersize=8, markeredgewidth=1.5,
            markerfacecolor="none",
        )
    wav_score = wavelet.get("wavelet_score", 0)
    n_det = len(wavelet.get("detections", []))
    n_ms = len(wavelet.get("multiscale_objects", []))
    ax.set_title(f"Wavelet={wav_score:.3f} ({n_det} det, {n_ms} multi)")

    # ---- Row 1, Col 2: Classical Patterns (Hough arcs) ----
    ax = axes[1, 2]
    ax.imshow(norm_data, origin="lower", cmap="gray_r")
    classical = detection.get("classical", {})
    for arc_item in classical.get("arcs", []):
        cx = arc_item.get("center_x", arc_item.get("x", 0))
        cy = arc_item.get("center_y", arc_item.get("y", 0))
        r = arc_item.get("radius", 20)
        circle = Circle(
            (cx, cy), r,
            fill=False, edgecolor="#ff6600", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)
        strength = arc_item.get("strength", 0)
        if strength > 0:
            ax.text(
                cx, cy + r + 5, f"{strength:.2f}",
                color="#ff6600", fontsize=6, ha="center",
            )
    gabor = classical.get("gabor_score", 0)
    fft = classical.get("fft_score", 0)
    arc_score = classical.get("arc_score", 0)
    ax.set_title(
        f"Classical: Gabor={gabor:.3f} FFT={fft:.3f} "
        f"Arc={arc_score:.3f} ({classical.get('n_arcs', 0)})"
    )

    # ---- Row 2, Col 0: Kinematic + Proper Motion ----
    ax = axes[2, 0]
    kinematic = detection.get("kinematic", {})
    if kinematic.get("no_catalog"):
        ax.text(
            0.5, 0.5, "No catalog data\n(kinematic analysis requires catalog)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Kinematic: no catalog")
    else:
        group_colors = [
            "red", "blue", "green", "orange", "purple", "brown", "pink",
        ]
        for i, group in enumerate(kinematic.get("comoving_groups", [])):
            color = group_colors[i % len(group_colors)]
            ax.plot(
                group["mean_ra"], group["mean_dec"],
                "o", color=color, markersize=10, markeredgewidth=2,
                markerfacecolor="none",
            )
            ax.annotate(
                f"G{i+1} (n={group['n_members']})",
                (group["mean_ra"], group["mean_dec"]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=7, color=color,
            )
        for stream in kinematic.get("stream_candidates", []):
            ax.plot(
                stream["mean_ra"], stream["mean_dec"],
                "D", color="cyan", markersize=8, markeredgewidth=1.5,
                markerfacecolor="none",
            )
        for star in kinematic.get("runaway_stars", []):
            ax.plot(
                star["ra"], star["dec"],
                "x", color="red", markersize=10, markeredgewidth=2,
            )
        ax.set_xlabel("RA (deg)", fontsize=7)
        ax.set_ylabel("Dec (deg)", fontsize=7)
        n_g = len(kinematic.get("comoving_groups", []))
        n_s = len(kinematic.get("stream_candidates", []))
        n_r = len(kinematic.get("runaway_stars", []))
        ax.set_title(
            f"Kinematic={kinematic.get('kinematic_score', 0):.3f} "
            f"(G={n_g} S={n_s} R={n_r})"
        )

    # ---- Row 2, Col 1: Transient + Variability ----
    ax = axes[2, 1]
    transient = detection.get("transient", {})
    variability = detection.get("variability", {})
    has_transient = not transient.get("no_catalog")
    has_variability = not variability.get("no_catalog")
    if not has_transient and not has_variability:
        ax.text(
            0.5, 0.5,
            "No catalog data\n(transient/variability requires catalog)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Transient + Variability: no catalog")
    else:
        # Plot outlier positions on RA/Dec
        for outlier in transient.get("astrometric_outliers", []):
            ax.plot(
                outlier["ra"], outlier["dec"],
                "^", color="red", markersize=7, alpha=0.8,
            )
        for outlier in transient.get("photometric_outliers", []):
            ax.plot(
                outlier["ra"], outlier["dec"],
                "v", color="orange", markersize=7, alpha=0.8,
            )
        for outlier in transient.get("parallax_anomalies", []):
            ax.plot(
                outlier["ra"], outlier["dec"],
                "d", color="yellow", markersize=7, alpha=0.8,
            )
        for vc in variability.get("variable_candidates", []):
            ax.plot(
                vc["ra"], vc["dec"],
                "s", color="cyan", markersize=5, markerfacecolor="none",
                markeredgewidth=1, alpha=0.7,
            )
        for pc in variability.get("periodic_candidates", []):
            ax.plot(
                pc["ra"], pc["dec"],
                "o", color="lime", markersize=6, markerfacecolor="none",
                markeredgewidth=1.5,
            )
        ax.set_xlabel("RA (deg)", fontsize=7)
        ax.set_ylabel("Dec (deg)", fontsize=7)
        t_score = transient.get("transient_score", 0)
        v_score = variability.get("variability_score", 0)
        ax.set_title(f"Transient={t_score:.3f} Variability={v_score:.3f}")

    # ---- Row 2, Col 2: Stellar Population CMD ----
    ax = axes[2, 2]
    population = detection.get("population", {})
    if population.get("no_catalog"):
        ax.text(
            0.5, 0.5, "No catalog data\n(CMD requires photometric catalog)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Stellar Populations: no catalog")
    else:
        # Plot CMD peaks
        peaks = population.get("cmd_peaks", [])
        if peaks:
            colors_cmd = [p.get("color", 0) for p in peaks]
            mags = [p.get("mag", 0) for p in peaks]
            ax.scatter(
                colors_cmd, mags, s=30, c="blue", alpha=0.6,
                label=f"Peaks ({len(peaks)})",
            )
        # Blue stragglers
        bs_cands = population.get("blue_straggler_candidates", [])
        if bs_cands:
            bs_colors = [c.get("color", 0) for c in bs_cands]
            bs_mags = [c.get("mag", 0) for c in bs_cands]
            ax.scatter(
                bs_colors, bs_mags, s=25, c="cyan", marker="^",
                label=f"Blue str. ({len(bs_cands)})",
            )
        # Red giants
        rg_cands = population.get("red_giant_candidates", [])
        if rg_cands:
            rg_colors = [c.get("color", 0) for c in rg_cands]
            rg_mags = [c.get("mag", 0) for c in rg_cands]
            ax.scatter(
                rg_colors, rg_mags, s=25, c="red", marker="v",
                label=f"RGB ({len(rg_cands)})",
            )
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
        ax.invert_yaxis()
        ax.set_xlabel("Color (BP-RP)", fontsize=7)
        ax.set_ylabel("Mag (G)", fontsize=7)
        pop_score = population.get("population_score", 0)
        n_bs = population.get("n_blue_stragglers", 0)
        n_rg = population.get("n_red_giants", 0)
        ax.set_title(f"Population={pop_score:.3f} (BS={n_bs} RGB={n_rg})")

    # ---- Row 3, Col 0: Detector Scores bar chart ----
    ax = axes[3, 0]
    detector_names, det_scores = _get_detector_scores(detection)
    bar_colors = [
        "#e74c3c" if s > 0.3 else "#f39c12" if s > 0.1 else "#95a5a6"
        for s in det_scores
    ]
    ax.barh(detector_names, det_scores, color=bar_colors)
    ax.set_xlim(0, max(max(det_scores) * 1.2, 0.5) if det_scores else 0.5)
    ax.set_title(f"Ensemble={detection.get('anomaly_score', 0):.4f}")
    ax.tick_params(axis="y", labelsize=7)

    # ---- Row 3, Col 1: Evolution + Learning trend ----
    ax = axes[3, 1]
    if evolution_history:
        cycles = [h.get("cycle", i) for i, h in enumerate(evolution_history)]
        fitnesses = [h.get("fitness", 0) for h in evolution_history]
        ax.plot(cycles, fitnesses, "o-", color="#2ecc71", linewidth=1.5, markersize=4)
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("Best fitness", fontsize=8)
        ax.set_title(f"Evolution ({len(evolution_history)} runs, best={max(fitnesses):.4f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, "No evolution data yet",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Evolution + Learning")

    # ---- Row 3, Col 2: Classification + Info text ----
    ax = axes[3, 2]
    ax.axis("off")

    scored = [
        (name, det_scores[i])
        for i, name in enumerate(detector_names)
        if det_scores[i] > 0.05
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_str = (
        ", ".join(f"{n}={s:.3f}" for n, s in scored[:5])
        if scored else "none above 0.05"
    )

    info_lines = [
        f"RA: {ra:.4f}  Dec: {dec:.4f}",
        f"Cycle: {cycle}  Elapsed: {elapsed_minutes:.1f} min",
        f"Ensemble score: {detection.get('anomaly_score', 0):.4f}",
        f"N sources: {len(positions)}",
        f"N detections: {detection.get('n_detections', 0)}",
        "",
        f"Top: {top_str}",
    ]

    if classification:
        info_lines.extend([
            "",
            f"Class: {classification.get('classification', '?')}",
            f"Confidence: {classification.get('confidence', 0):.2f}",
            f"Rationale: {classification.get('rationale', '')[:60]}",
        ])

    if evaluation:
        info_lines.extend([
            "",
            f"Verdict: {evaluation.get('verdict', '?')}",
            f"Significance: {evaluation.get('significance_rating', '?')}",
        ])

    info_text = "\n".join(info_lines)
    ax.text(
        0.05, 0.95, info_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
    )

    fig.suptitle(
        f"Region RA={ra:.4f} Dec={dec:.4f} | Cycle {cycle} "
        f"| Score={detection.get('anomaly_score', 0):.4f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def create_evolution_summary(
    generation_histories: list[list[dict[str, Any]]],
    evolution_history: list[dict[str, Any]],
    figsize: tuple[int, int] = (14, 10),
) -> Figure:
    """Create a 2x2 evolution tracking summary.

    Panels:
      [0,0] Per-generation fitness curves for each evolution cycle
      [0,1] Best fitness trend across evolution cycles
      [1,0] Mutation rate adaptation over generations
      [1,1] Fitness component breakdown (stacked bars)

    Args:
        generation_histories: List of per-cycle generation histories.
            Each is a list of dicts with keys: generation, best_fitness,
            mean_fitness, mutation_rate.
        evolution_history: List of evolution cycle summaries with
            cycle, fitness, components keys.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with evolution tracking panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    cycle_colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e",
    ]

    # ---- [0,0]: Per-generation fitness curves ----
    ax = axes[0, 0]
    for i, gen_hist in enumerate(generation_histories):
        if not gen_hist:
            continue
        color = cycle_colors[i % len(cycle_colors)]
        gens = [g.get("generation", j) for j, g in enumerate(gen_hist)]
        best = [g.get("best_fitness", 0) for g in gen_hist]
        mean = [g.get("mean_fitness", 0) for g in gen_hist]
        label = f"Cycle {evolution_history[i].get('cycle', i) if i < len(evolution_history) else i}"
        ax.plot(gens, best, "-", color=color, linewidth=1.5, label=label)
        ax.plot(gens, mean, "--", color=color, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Fitness", fontsize=9)
    ax.set_title("Per-generation fitness (solid=best, dashed=mean)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # ---- [0,1]: Best fitness trend across cycles ----
    ax = axes[0, 1]
    if evolution_history:
        cycles = [h.get("cycle", i) for i, h in enumerate(evolution_history)]
        fitnesses = [h.get("fitness", 0) for h in evolution_history]
        ax.plot(
            cycles, fitnesses, "o-", color="#2ecc71",
            linewidth=2, markersize=6,
        )
        ax.fill_between(cycles, fitnesses, alpha=0.15, color="#2ecc71")
        ax.set_xlabel("Pipeline cycle", fontsize=9)
        ax.set_ylabel("Best fitness", fontsize=9)
        ax.set_title(f"Best fitness trend (peak={max(fitnesses):.4f})")
    else:
        ax.text(
            0.5, 0.5, "No evolution data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Best fitness trend")
    ax.grid(True, alpha=0.3)

    # ---- [1,0]: Mutation rate adaptation ----
    ax = axes[1, 0]
    has_mutation_data = False
    for i, gen_hist in enumerate(generation_histories):
        if not gen_hist:
            continue
        rates = [g.get("mutation_rate", None) for g in gen_hist]
        if any(r is not None for r in rates):
            has_mutation_data = True
            color = cycle_colors[i % len(cycle_colors)]
            gens = [g.get("generation", j) for j, g in enumerate(gen_hist)]
            valid_rates = [r if r is not None else 0 for r in rates]
            label = f"Cycle {evolution_history[i].get('cycle', i) if i < len(evolution_history) else i}"
            ax.plot(gens, valid_rates, "-", color=color, linewidth=1.5, label=label)
    if not has_mutation_data:
        ax.text(
            0.5, 0.5, "No mutation rate data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Mutation rate", fontsize=9)
    ax.set_title("Mutation rate adaptation")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # ---- [1,1]: Fitness component breakdown ----
    ax = axes[1, 1]
    component_names = [
        "anomaly", "significance", "novelty", "diversity", "recovery",
    ]
    component_colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    ]
    entries_with_components = [
        h for h in evolution_history
        if h.get("components")
    ]
    if entries_with_components:
        x_labels = [
            str(h.get("cycle", i))
            for i, h in enumerate(entries_with_components)
        ]
        x_pos = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        for j, comp_name in enumerate(component_names):
            values = [
                h["components"].get(comp_name, 0)
                for h in entries_with_components
            ]
            ax.bar(
                x_pos, values, bottom=bottom, width=0.6,
                color=component_colors[j % len(component_colors)],
                label=comp_name,
            )
            bottom += np.array(values)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Pipeline cycle", fontsize=9)
        ax.set_ylabel("Fitness contribution", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.set_title("Fitness component breakdown")
    else:
        ax.text(
            0.5, 0.5, "No component data",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
        )
        ax.set_title("Fitness component breakdown")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Evolution Summary ({len(evolution_history)} cycles)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
