"""Discovery mosaic: visual summary of findings."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from star_pattern.core.fits_handler import FITSImage
from star_pattern.evaluation.metrics import PatternResult
from star_pattern.utils.logging import get_logger

logger = get_logger("visualization.mosaic")


def create_discovery_mosaic(
    findings: list[PatternResult],
    images: list[FITSImage | None] | None = None,
    max_panels: int = 25,
    figsize: tuple[int, int] = (20, 20),
) -> Figure:
    """Create a mosaic of discovery findings.

    Color coding:
    - GREEN border: Interesting (anomaly_score > 0.5)
    - YELLOW border: Known object (has cross-matches)
    - GRAY border: Not interesting (low score)
    """
    # Sort by score
    sorted_findings = sorted(findings, key=lambda f: f.anomaly_score, reverse=True)
    n_show = min(len(sorted_findings), max_panels)

    if n_show == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No findings to display", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx >= n_show:
            ax.axis("off")
            continue

        finding = sorted_findings[idx]

        # Determine border color based on detection type and score
        det_type = finding.detection_type
        type_colors = {
            "galaxy": "#E040FB",     # Purple: galaxy interactions
            "kinematic": "#00BCD4",  # Teal: kinematic detections
            "transient": "#FF5722",  # Deep orange: transients
        }

        if finding.anomaly_score > 0.5 and not finding.cross_matches:
            border_color = type_colors.get(det_type, "#4CAF50")  # Type-specific or green
            status = "INTERESTING"
        elif finding.cross_matches:
            border_color = "#FFC107"  # Yellow: known object
            status = "KNOWN"
        else:
            border_color = "#9E9E9E"  # Gray: not interesting
            status = "LOW"

        # Plot image if available
        if images and idx < len(images) and images[idx] is not None:
            norm = images[idx].normalize("zscale")
            ax.imshow(norm.data, origin="lower", cmap="gray_r")
        else:
            # Placeholder
            ax.text(0.5, 0.5, f"({finding.region_ra:.1f},\n{finding.region_dec:.1f})",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)

        # Apply colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Title with score
        ax.set_title(
            f"{finding.detection_type[:5]} {finding.anomaly_score:.3f} [{status}]",
            fontsize=8,
            color=border_color,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Discovery Mosaic: {n_show} findings "
        f"(green=novel, purple=galaxy, teal=kinematic, orange=transient, "
        f"yellow=known, gray=low)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def create_score_histogram(findings: list[PatternResult]) -> Figure:
    """Create histogram of anomaly scores."""
    if not findings:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No findings", ha="center", va="center")
        return fig

    scores = [f.anomaly_score for f in findings]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=30, edgecolor="black", alpha=0.7, color="#2196F3")
    ax.axvline(0.5, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Anomaly Scores (N={len(findings)})")
    ax.legend()

    plt.tight_layout()
    return fig
