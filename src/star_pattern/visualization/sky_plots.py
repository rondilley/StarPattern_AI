"""WCS-aware sky region plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.sky_region import SkyRegion
from star_pattern.utils.logging import get_logger

logger = get_logger("visualization.sky_plots")


def plot_fits_image(
    image: FITSImage,
    title: str = "",
    cmap: str = "gray_r",
    normalize: str = "zscale",
    figsize: tuple[int, int] = (8, 8),
) -> Figure:
    """Plot a FITS image with optional WCS axes."""
    norm = image.normalize(normalize)

    fig = plt.figure(figsize=figsize)

    if image.wcs is not None:
        ax = fig.add_subplot(111, projection=image.wcs)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    else:
        ax = fig.add_subplot(111)

    ax.imshow(norm.data, origin="lower", cmap=cmap)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_sky_coverage(
    regions: list[SkyRegion],
    findings: list[dict[str, Any]] | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot sky coverage of searched regions and findings."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    # Searched regions
    if regions:
        ras = np.array([r.ra for r in regions])
        decs = np.array([r.dec for r in regions])
        # Convert to radians, shift RA to [-pi, pi]
        ra_rad = np.radians(ras - 180)
        dec_rad = np.radians(decs)
        ax.scatter(ra_rad, dec_rad, s=10, c="gray", alpha=0.3, label="Searched")

    # Findings
    if findings:
        f_ras = np.array([f.get("ra", 0) for f in findings])
        f_decs = np.array([f.get("dec", 0) for f in findings])
        f_scores = np.array([f.get("anomaly_score", 0) for f in findings])

        f_ra_rad = np.radians(f_ras - 180)
        f_dec_rad = np.radians(f_decs)

        scatter = ax.scatter(
            f_ra_rad, f_dec_rad,
            s=30 + 100 * f_scores,
            c=f_scores,
            cmap="hot",
            alpha=0.8,
            label="Findings",
            edgecolors="white",
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Anomaly Score", shrink=0.6)

    ax.set_title("Sky Coverage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()
    return fig


def plot_region_summary(
    region: SkyRegion,
    detection: dict[str, Any],
    image: FITSImage | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """Plot a summary card for a single region detection."""
    n_panels = 3 if image is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    if image is not None:
        norm = image.normalize("zscale")
        axes[0].imshow(norm.data, origin="lower", cmap="gray_r")
        axes[0].set_title(f"({region.ra:.2f}, {region.dec:.2f})")
        panel_offset = 1
    else:
        panel_offset = 0

    # Score bar chart
    scores = {
        "classical": detection.get("classical", {}).get("gabor_score", 0),
        "morphology": detection.get("morphology", {}).get("morphology_score", 0),
        "lens": detection.get("lens", {}).get("lens_score", 0),
        "distribution": detection.get("distribution", {}).get("distribution_score", 0),
    }
    ax_bar = axes[panel_offset]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    ax_bar.barh(list(scores.keys()), list(scores.values()), color=colors)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title(f"Scores (ensemble: {detection.get('anomaly_score', 0):.3f})")

    # Text summary
    ax_text = axes[panel_offset + 1]
    ax_text.axis("off")
    text = (
        f"RA: {region.ra:.4f}\n"
        f"Dec: {region.dec:.4f}\n"
        f"Sources: {detection.get('sources', {}).get('n_sources', 'N/A')}\n"
        f"Detections: {detection.get('n_detections', 0)}\n"
        f"Anomaly: {detection.get('anomaly_score', 0):.4f}\n"
    )
    if detection.get("lens", {}).get("is_candidate"):
        text += "LENS CANDIDATE"
    ax_text.text(0.1, 0.5, text, transform=ax_text.transAxes, fontsize=10,
                 verticalalignment="center", fontfamily="monospace")

    plt.tight_layout()
    return fig
