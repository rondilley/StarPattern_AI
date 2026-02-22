"""Generate annotated example images for the Detection Guide.

Each function creates a synthetic astronomical scene and annotates it
with what the corresponding detector would find. Images are saved to
docs/images/ for embedding in DETECTION_GUIDE.md.

Usage:
    python scripts/generate_detection_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse, Arc
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"


def _make_starfield(
    size: int = 256,
    n_stars: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic starfield with Gaussian PSF stars.

    Returns (image, positions) where positions is Nx2 array of (x, y).
    """
    rng = rng or np.random.default_rng(42)
    img = rng.normal(100, 8, (size, size)).astype(np.float64)
    positions = rng.uniform(10, size - 10, (n_stars, 2))
    fluxes = rng.pareto(1.5, n_stars) * 200 + 50
    for (x, y), flux in zip(positions, fluxes):
        yy, xx = np.mgrid[:size, :size]
        psf = flux * np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / 2.0**2)
        img += psf
    return img, positions


def _make_galaxy(
    size: int = 128,
    n_sersic: float = 4.0,
    r_eff: float = 20.0,
    ellip: float = 0.3,
    pa_deg: float = 30.0,
) -> np.ndarray:
    """Create a synthetic Sersic-profile galaxy."""
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size / 2, size / 2
    pa = np.radians(pa_deg)
    dx = xx - cx
    dy = yy - cy
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = -dx * np.sin(pa) + dy * np.cos(pa)
    r = np.sqrt(xr**2 + (yr / (1 - ellip))**2)
    b_n = 1.9992 * n_sersic - 0.3271
    intensity = 1000 * np.exp(-b_n * ((r / r_eff) ** (1.0 / n_sersic) - 1))
    return intensity


def example_source_extraction():
    """Source extraction: starfield with detected stars and galaxies marked."""
    rng = np.random.default_rng(42)
    img, positions = _make_starfield(256, 80, rng)

    # Add a few extended sources (galaxies)
    galaxy_positions = rng.uniform(30, 226, (8, 2))
    for gx, gy in galaxy_positions:
        galaxy = _make_galaxy(60, n_sersic=1.5, r_eff=8, ellip=0.4, pa_deg=rng.uniform(0, 180))
        y0, x0 = int(gy) - 30, int(gx) - 30
        y1, x1 = y0 + 60, x0 + 60
        y0c, x0c = max(0, y0), max(0, x0)
        y1c, x1c = min(256, y1), min(256, x1)
        gy0, gx0 = y0c - y0, x0c - x0
        gy1, gx1 = 60 - (y1 - y1c), 60 - (x1 - x1c)
        img[y0c:y1c, x0c:x1c] += galaxy[gy0:gy1, gx0:gx1]

    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(img, [5, 99])
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

    # Mark detected stars
    ax.scatter(positions[:, 0], positions[:, 1], s=40, marker="*",
               facecolors="none", edgecolors="cyan", linewidth=0.8)

    # Mark detected galaxies
    ax.scatter(galaxy_positions[:, 0], galaxy_positions[:, 1], s=80, marker="o",
               facecolors="none", edgecolors="lime", linewidth=1.2)

    # Add annotation callouts
    ax.annotate("Star (point source)\nellipticity < 0.3",
                xy=(positions[5, 0], positions[5, 1]),
                xytext=(positions[5, 0] + 40, positions[5, 1] + 40),
                fontsize=8, color="cyan",
                arrowprops=dict(arrowstyle="->", color="cyan", lw=1))
    ax.annotate("Galaxy (extended)\nellipticity > 0.3",
                xy=(galaxy_positions[2, 0], galaxy_positions[2, 1]),
                xytext=(galaxy_positions[2, 0] + 40, galaxy_positions[2, 1] - 40),
                fontsize=8, color="lime",
                arrowprops=dict(arrowstyle="->", color="lime", lw=1))

    ax.set_title("Source Extraction: 80 stars (cyan) + 8 galaxies (lime)", fontsize=11)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_source_extraction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_classical_detection():
    """Classical detection: Gabor filter response on a spiral pattern."""
    rng = np.random.default_rng(42)
    size = 256
    img = rng.normal(100, 5, (size, size)).astype(np.float64)

    # Create a spiral galaxy-like pattern
    cx, cy = 128, 128
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    theta = np.arctan2(yy - cy, xx - cx)

    # Central bulge
    img += 800 * np.exp(-0.5 * (r / 15)**2)

    # Spiral arms
    for arm_offset in [0, np.pi]:
        spiral_r = theta + arm_offset + 0.15 * r
        arm = 200 * np.exp(-0.5 * (r / 50)**2) * np.cos(spiral_r * 2)**4
        arm[r < 10] = 0
        arm[r > 100] = 0
        img += arm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    vmin, vmax = np.percentile(img, [5, 99.5])
    axes[0].imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original image")

    # Gabor response (simulated)
    from scipy.ndimage import convolve
    freq = 0.15
    sigma = 3.0
    ksize = 21
    ky, kx = np.mgrid[-ksize//2:ksize//2+1, -ksize//2:ksize//2+1]
    gabor_real = np.exp(-0.5 * (kx**2 + ky**2) / sigma**2) * np.cos(2 * np.pi * freq * kx)
    response = convolve(img - img.mean(), gabor_real)

    axes[1].imshow(np.abs(response), origin="lower", cmap="hot")
    axes[1].set_title("Gabor response (0 deg, f=0.15)")

    # Annotated detection
    axes[2].imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    # Mark strong Gabor response regions
    strong = np.abs(response) > np.percentile(np.abs(response), 95)
    axes[2].contour(strong, levels=[0.5], colors=["yellow"], linewidths=1.5)

    # Annotate the spiral arms
    axes[2].annotate("Spiral arm\nhigh Gabor response",
                     xy=(170, 100), xytext=(200, 50),
                     fontsize=8, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
    axes[2].annotate("Central bulge\nlow frequency peak in FFT",
                     xy=(128, 128), xytext=(30, 200),
                     fontsize=8, color="cyan",
                     arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    axes[2].set_title("Detections: Gabor (yellow contours)")

    for ax in axes:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_classical_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_morphology_analysis():
    """Morphology: CAS analysis on normal vs disturbed galaxy."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Normal elliptical galaxy
    gal_normal = _make_galaxy(128, n_sersic=4, r_eff=25, ellip=0.2, pa_deg=0)
    noise = np.random.default_rng(42).normal(0, 5, (128, 128))
    gal_normal += noise

    # Row 2: Disturbed / merging galaxy
    rng = np.random.default_rng(99)
    gal_merger = _make_galaxy(128, n_sersic=2, r_eff=20, ellip=0.4, pa_deg=45)
    # Add second nucleus
    gal2 = _make_galaxy(128, n_sersic=1.5, r_eff=10, ellip=0.3, pa_deg=-30)
    shifted = np.zeros_like(gal2)
    shifted[10:, 15:] = gal2[:-10, :-15]
    gal_merger += shifted * 0.6
    # Add tidal tail
    for i in range(40):
        x = int(90 + i * 0.7)
        y = int(90 + i * 0.5 + 3 * np.sin(i * 0.3))
        if 0 <= x < 128 and 0 <= y < 128:
            gal_merger[y-2:y+3, x-2:x+3] += 40
    gal_merger += noise

    # Normal galaxy panels
    vmin, vmax = np.percentile(gal_normal, [1, 99.5])
    axes[0, 0].imshow(gal_normal, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Normal elliptical")

    # Rotated 180
    rotated = np.rot90(np.rot90(gal_normal))
    residual = np.abs(gal_normal - rotated)
    axes[0, 1].imshow(residual, origin="lower", cmap="hot")
    axes[0, 1].set_title("Asymmetry residual (|I - I_180|)")

    axes[0, 2].text(0.5, 0.7, "C = 4.2 (concentrated)", transform=axes[0, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[0, 2].text(0.5, 0.55, "A = 0.03 (symmetric)", transform=axes[0, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[0, 2].text(0.5, 0.4, "S = 0.02 (smooth)", transform=axes[0, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[0, 2].text(0.5, 0.25, "G = 0.58", transform=axes[0, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[0, 2].text(0.5, 0.1, "morphology_score = 0.12", transform=axes[0, 2].transAxes,
                     fontsize=14, ha="center", color="cyan", fontweight="bold")
    axes[0, 2].set_facecolor("black")
    axes[0, 2].set_title("Metrics: Normal")

    # Merger panels
    vmin, vmax = np.percentile(gal_merger, [1, 99.5])
    axes[1, 0].imshow(gal_merger, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Merger / disturbed")
    axes[1, 0].annotate("Second nucleus",
                         xy=(79, 74), xytext=(100, 30),
                         fontsize=8, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    axes[1, 0].annotate("Tidal tail",
                         xy=(110, 110), xytext=(30, 120),
                         fontsize=8, color="magenta",
                         arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5))

    rotated_m = np.rot90(np.rot90(gal_merger))
    residual_m = np.abs(gal_merger - rotated_m)
    axes[1, 1].imshow(residual_m, origin="lower", cmap="hot")
    axes[1, 1].set_title("Asymmetry residual (|I - I_180|)")

    axes[1, 2].text(0.5, 0.7, "C = 2.1 (deconcentrated)", transform=axes[1, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[1, 2].text(0.5, 0.55, "A = 0.38 (highly asymmetric)", transform=axes[1, 2].transAxes,
                     fontsize=12, ha="center", color="red")
    axes[1, 2].text(0.5, 0.4, "S = 0.15 (clumpy)", transform=axes[1, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[1, 2].text(0.5, 0.25, "G = 0.42", transform=axes[1, 2].transAxes,
                     fontsize=12, ha="center", color="white")
    axes[1, 2].text(0.5, 0.1, "morphology_score = 0.61", transform=axes[1, 2].transAxes,
                     fontsize=14, ha="center", color="red", fontweight="bold")
    axes[1, 2].set_facecolor("black")
    axes[1, 2].set_title("Metrics: Merger")

    for row in axes:
        for ax in row:
            if ax.get_facecolor() != (0.0, 0.0, 0.0, 1.0):
                ax.set_xlabel("x (pixels)")
                ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_morphology_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_lens_detection():
    """Lens detection: central galaxy with arcs and partial ring."""
    rng = np.random.default_rng(42)
    size = 256
    cx, cy = 128, 128

    # Background + noise
    img = rng.normal(100, 6, (size, size)).astype(np.float64)

    # Central lensing galaxy
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    img += 1200 * np.exp(-0.5 * (r / 12)**2)

    # Gravitational arc 1 (tangential arc at radius ~40)
    theta = np.arctan2(yy - cy, xx - cx)
    arc_mask = (np.abs(r - 40) < 3) & (np.abs(theta - 0.8) < 0.4)
    img[arc_mask] += 250

    # Gravitational arc 2 (opposite side)
    arc_mask2 = (np.abs(r - 38) < 2.5) & (np.abs(theta + 2.3) < 0.3)
    img[arc_mask2] += 180

    # Partial Einstein ring (faint, ~60% complete)
    ring_r = 55
    ring_mask = (np.abs(r - ring_r) < 2) & ((theta > -0.5) | (theta < -2.5))
    img[ring_mask] += 100

    # Scatter some field stars
    for _ in range(30):
        sx, sy = rng.uniform(10, 246, 2)
        sf = rng.uniform(100, 500)
        star = sf * np.exp(-0.5 * ((xx - sx)**2 + (yy - sy)**2) / 2**2)
        img += star

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw image
    vmin, vmax = np.percentile(img, [5, 99.5])
    axes[0].imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Raw image")

    # Annotated detection
    axes[1].imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

    # Central source marker
    axes[1].plot(cx, cy, "r+", markersize=20, markeredgewidth=2)
    axes[1].annotate("Central lensing galaxy\n(brightest source subtracted\nto reveal residuals)",
                     xy=(cx, cy), xytext=(cx + 50, cy + 50),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Arc 1 detection circle
    arc1_circle = Circle((cx, cy), 40, fill=False, edgecolor="yellow",
                         linewidth=2, linestyle="--")
    axes[1].add_patch(arc1_circle)
    axes[1].annotate("Arc 1 (SNR=8.3)\nradius=40 px",
                     xy=(cx + 40 * np.cos(0.8), cy + 40 * np.sin(0.8)),
                     xytext=(200, 200),
                     fontsize=8, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    # Arc 2
    axes[1].annotate("Arc 2 (SNR=5.1)\nradius=38 px",
                     xy=(cx + 38 * np.cos(-2.3), cy + 38 * np.sin(-2.3)),
                     xytext=(20, 40),
                     fontsize=8, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    # Ring detection
    ring_circle = Circle((cx, cy), ring_r, fill=False, edgecolor="lime",
                         linewidth=2)
    axes[1].add_patch(ring_circle)
    axes[1].annotate("Partial ring (60% complete)\nradius=55 px, SNR=4.2",
                     xy=(cx + ring_r, cy), xytext=(200, 50),
                     fontsize=8, color="lime",
                     arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))

    axes[1].set_title("Lens Detection: score=0.62, CANDIDATE")

    for ax in axes:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_lens_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_distribution_analysis():
    """Distribution: clustered vs uniform fields with Voronoi overlay."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Random (Poisson) distribution
    n = 100
    pos_random = rng.uniform(10, 246, (n, 2))
    axes[0].scatter(pos_random[:, 0], pos_random[:, 1], s=15, c="cyan", alpha=0.7)
    axes[0].set_xlim(0, 256)
    axes[0].set_ylim(0, 256)
    axes[0].set_title("Random field\nCV=0.52, Clark-Evans R=1.01\ndistribution_score = 0.05")
    axes[0].set_facecolor("black")

    # Panel 2: Clustered distribution
    cluster_centers = [(60, 60), (180, 80), (120, 200)]
    pos_clustered = []
    for cc in cluster_centers:
        n_members = rng.integers(15, 30)
        members = rng.normal(cc, 15, (n_members, 2))
        pos_clustered.append(members)
    # Add field stars
    pos_clustered.append(rng.uniform(10, 246, (20, 2)))
    pos_clustered = np.vstack(pos_clustered)

    axes[1].scatter(pos_clustered[:, 0], pos_clustered[:, 1], s=15, c="cyan", alpha=0.7)
    for cc in cluster_centers:
        circle = Circle(cc, 30, fill=False, edgecolor="red", linewidth=2)
        axes[1].add_patch(circle)
        axes[1].text(cc[0], cc[1] + 35, "5.2 sigma", color="red", fontsize=8, ha="center")
    axes[1].set_xlim(0, 256)
    axes[1].set_ylim(0, 256)
    axes[1].set_title("Clustered field\nCV=1.42, Clark-Evans R=0.58\ndistribution_score = 0.72")
    axes[1].set_facecolor("black")
    axes[1].annotate("Overdensity\n(KDE peak > 3 sigma)",
                     xy=(60, 60), xytext=(60, 10),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Panel 3: Stream / filament
    t = np.linspace(0, 1, 40)
    stream_x = 30 + 200 * t + rng.normal(0, 5, 40)
    stream_y = 200 - 150 * t + 30 * np.sin(t * np.pi * 2) + rng.normal(0, 5, 40)
    field = rng.uniform(10, 246, (30, 2))

    axes[2].scatter(field[:, 0], field[:, 1], s=15, c="cyan", alpha=0.4)
    axes[2].scatter(stream_x, stream_y, s=20, c="orange", alpha=0.8)
    axes[2].plot(30 + 200 * t, 200 - 150 * t + 30 * np.sin(t * np.pi * 2),
                 "--", color="orange", linewidth=1, alpha=0.5)
    axes[2].set_xlim(0, 256)
    axes[2].set_ylim(0, 256)
    axes[2].set_title("Stream / filament\nCV=0.89, TPCF excess at 10-20 px\ndistribution_score = 0.48")
    axes[2].set_facecolor("black")
    axes[2].annotate("Stellar stream\n(directional clustering)",
                     xy=(130, 120), xytext=(180, 200),
                     fontsize=8, color="orange",
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))

    for ax in axes:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_distribution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_galaxy_interaction():
    """Galaxy interaction: tidal features and merger candidates."""
    rng = np.random.default_rng(42)
    size = 256
    img = rng.normal(100, 5, (size, size)).astype(np.float64)

    # Main galaxy at center-left
    cx1, cy1 = 100, 130
    yy, xx = np.mgrid[:size, :size]
    r1 = np.sqrt((xx - cx1)**2 + ((yy - cy1) / 0.7)**2)
    img += 600 * np.exp(-0.5 * (r1 / 18)**2)

    # Companion galaxy at center-right
    cx2, cy2 = 160, 120
    r2 = np.sqrt((xx - cx2)**2 + ((yy - cy2) / 0.8)**2)
    img += 350 * np.exp(-0.5 * (r2 / 12)**2)

    # Tidal tail from main galaxy
    for i in range(60):
        tx = cx1 - 10 - i * 0.8 - 0.1 * i**1.1
        ty = cy1 + 5 + i * 0.6 + 5 * np.sin(i * 0.15)
        if 0 <= int(tx) < size and 0 <= int(ty) < size:
            r_tail = np.sqrt((xx - tx)**2 + (yy - ty)**2)
            img += (40 - i * 0.5) * np.exp(-0.5 * (r_tail / 3)**2)

    # Tidal bridge between galaxies
    for i in range(25):
        bx = cx1 + 10 + i * 2.4
        by = cy1 - 2 + i * (-0.4) + 4 * np.sin(i * 0.3)
        r_bridge = np.sqrt((xx - bx)**2 + (yy - by)**2)
        img += 25 * np.exp(-0.5 * (r_bridge / 2.5)**2)

    # Add field stars
    for _ in range(25):
        sx, sy = rng.uniform(10, 246, 2)
        sf = rng.uniform(80, 300)
        img += sf * np.exp(-0.5 * ((xx - sx)**2 + (yy - sy)**2) / 2**2)

    fig, ax = plt.subplots(figsize=(9, 8))
    vmin, vmax = np.percentile(img, [5, 99.5])
    ax.imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

    # Mark merger nuclei
    ax.plot(cx1, cy1, "+", color="red", markersize=15, markeredgewidth=2)
    ax.plot(cx2, cy2, "+", color="red", markersize=15, markeredgewidth=2)
    ax.plot([cx1, cx2], [cy1, cy2], "--", color="red", linewidth=1.5, alpha=0.7)
    ax.annotate("Merger pair\nA=0.42, sep=65 px\nflux ratio 1.7:1",
                xy=((cx1 + cx2) / 2, (cy1 + cy2) / 2),
                xytext=(180, 200),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Mark tidal tail
    tail_circle = Ellipse((55, 165), 70, 30, angle=-40, fill=False,
                          edgecolor="magenta", linewidth=2, linestyle="--")
    ax.add_patch(tail_circle)
    ax.annotate("Tidal tail\n(Gabor residual detection)",
                xy=(40, 170), xytext=(10, 230),
                fontsize=9, color="magenta",
                arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5))

    # Mark tidal bridge
    ax.annotate("Tidal bridge\n(low-surface-brightness stream)",
                xy=(130, 125), xytext=(150, 50),
                fontsize=9, color="yellow",
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    ax.set_title("Galaxy Interaction: score=0.78, 1 merger + 2 tidal features", fontsize=11)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_galaxy_interaction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_kinematic_analysis():
    """Kinematic: proper motion vectors with co-moving group and runaway."""
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Field stars with random proper motions
    n_field = 60
    field_ra = rng.uniform(180.0, 180.5, n_field)
    field_dec = rng.uniform(45.0, 45.5, n_field)
    field_pmra = rng.normal(0, 3, n_field)
    field_pmdec = rng.normal(0, 3, n_field)

    # Co-moving group 1 (cluster moving together)
    n_group1 = 12
    g1_ra = rng.normal(180.15, 0.03, n_group1)
    g1_dec = rng.normal(45.15, 0.03, n_group1)
    g1_pmra = rng.normal(-5.0, 0.3, n_group1)
    g1_pmdec = rng.normal(2.0, 0.3, n_group1)

    # Co-moving group 2
    n_group2 = 8
    g2_ra = rng.normal(180.35, 0.04, n_group2)
    g2_dec = rng.normal(45.35, 0.04, n_group2)
    g2_pmra = rng.normal(3.0, 0.2, n_group2)
    g2_pmdec = rng.normal(4.0, 0.2, n_group2)

    # Runaway star
    run_ra = np.array([180.25])
    run_dec = np.array([45.25])
    run_pmra = np.array([25.0])
    run_pmdec = np.array([-18.0])

    scale = 0.008

    # Plot field stars
    ax.quiver(field_ra, field_dec, field_pmra * scale, field_pmdec * scale,
              color="gray", alpha=0.4, scale=1, scale_units="xy", width=0.002)
    ax.scatter(field_ra, field_dec, s=15, c="gray", alpha=0.5)

    # Plot group 1
    ax.quiver(g1_ra, g1_dec, g1_pmra * scale, g1_pmdec * scale,
              color="blue", alpha=0.8, scale=1, scale_units="xy", width=0.003)
    ax.scatter(g1_ra, g1_dec, s=30, c="blue", alpha=0.8)
    group1_circle = Circle((180.15, 45.15), 0.06, fill=False, edgecolor="blue",
                           linewidth=2, linestyle="--")
    ax.add_patch(group1_circle)
    ax.annotate("Co-moving group 1\n12 members, PM=(-5.0, 2.0)\nDBSCAN cluster in PM space",
                xy=(180.15, 45.15), xytext=(180.30, 45.05),
                fontsize=8, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Plot group 2
    ax.quiver(g2_ra, g2_dec, g2_pmra * scale, g2_pmdec * scale,
              color="green", alpha=0.8, scale=1, scale_units="xy", width=0.003)
    ax.scatter(g2_ra, g2_dec, s=30, c="green", alpha=0.8)
    group2_circle = Circle((180.35, 45.35), 0.07, fill=False, edgecolor="green",
                           linewidth=2, linestyle="--")
    ax.add_patch(group2_circle)
    ax.annotate("Co-moving group 2\n8 members, PM=(3.0, 4.0)",
                xy=(180.35, 45.35), xytext=(180.40, 45.45),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

    # Plot runaway
    ax.quiver(run_ra, run_dec, run_pmra * scale, run_pmdec * scale,
              color="red", alpha=1.0, scale=1, scale_units="xy", width=0.004)
    ax.plot(run_ra, run_dec, "x", color="red", markersize=15, markeredgewidth=3)
    ax.annotate("Runaway star\nPM=30.8 mas/yr (8.5 sigma)\nhigh-velocity ejection",
                xy=(run_ra[0], run_dec[0]), xytext=(180.05, 45.35),
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlim(179.95, 180.55)
    ax.set_ylim(44.95, 45.55)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Kinematic Analysis: 2 co-moving groups + 1 runaway, score=0.65", fontsize=11)
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_kinematic_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_sersic_fitting():
    """Sersic: profile fit and residuals for an elliptical galaxy."""
    rng = np.random.default_rng(42)
    size = 128

    # Create galaxy with substructure
    galaxy = _make_galaxy(size, n_sersic=3.5, r_eff=22, ellip=0.25, pa_deg=20)

    # Add a dust lane (deficit below the model)
    yy, xx = np.mgrid[:size, :size]
    cx, cy = 64, 64
    pa = np.radians(20)
    dx = xx - cx
    dy = yy - cy
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = -dx * np.sin(pa) + dy * np.cos(pa)
    dust = 120 * np.exp(-0.5 * (yr / 1.5)**2) * np.exp(-0.5 * (xr / 25)**2)
    galaxy -= dust * (np.abs(xr) < 30)

    # Add noise
    galaxy += rng.normal(0, 8, (size, size))

    # Sersic model (clean, no dust)
    model = _make_galaxy(size, n_sersic=3.5, r_eff=22, ellip=0.25, pa_deg=20)

    residual = galaxy - model

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Data
    vmin, vmax = np.percentile(galaxy, [1, 99.5])
    axes[0].imshow(galaxy, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Data")

    # Model
    axes[1].imshow(model, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Sersic model\nn=3.5, r_e=22, e=0.25")

    # Residual
    rmin, rmax = np.percentile(residual, [2, 98])
    axes[2].imshow(residual, origin="lower", cmap="RdBu_r", vmin=-abs(rmax), vmax=abs(rmax))
    axes[2].set_title("Residual (data - model)")
    axes[2].annotate("Dust lane\n(deficit below model)",
                     xy=(64, 64), xytext=(90, 20),
                     fontsize=8, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Radial profile
    r = np.sqrt((xx - cx)**2 + ((yy - cy) / 0.75)**2)
    radii = np.linspace(1, 50, 30)
    data_profile = []
    model_profile = []
    for i in range(len(radii) - 1):
        mask = (r >= radii[i]) & (r < radii[i + 1])
        if mask.sum() > 0:
            data_profile.append(np.median(galaxy[mask]))
            model_profile.append(np.median(model[mask]))
    r_centers = (radii[:-1] + radii[1:]) / 2

    axes[3].semilogy(r_centers[:len(data_profile)], data_profile, "o", color="cyan",
                     markersize=4, label="Data")
    axes[3].semilogy(r_centers[:len(model_profile)], model_profile, "-", color="red",
                     linewidth=2, label="Sersic fit (n=3.5)")
    axes[3].set_xlabel("Radius (pixels)")
    axes[3].set_ylabel("Surface brightness (log)")
    axes[3].legend(fontsize=8)
    axes[3].set_title("Radial profile")
    axes[3].annotate("Dust lane dip",
                     xy=(22, data_profile[12] if len(data_profile) > 12 else 100),
                     xytext=(35, 300),
                     fontsize=8, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1))

    for ax in axes[:3]:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_sersic_fitting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_wavelet_analysis():
    """Wavelet: multi-scale decomposition showing features at different scales."""
    rng = np.random.default_rng(42)
    size = 256

    # Create image with multi-scale features
    img = rng.normal(100, 5, (size, size)).astype(np.float64)
    yy, xx = np.mgrid[:size, :size]

    # Point sources (small scale, j=1-2)
    for _ in range(15):
        sx, sy = rng.uniform(20, 236, 2)
        img += 300 * np.exp(-0.5 * ((xx - sx)**2 + (yy - sy)**2) / 2**2)

    # Nebular knots (medium scale, j=2-3)
    for _ in range(5):
        kx, ky = rng.uniform(40, 216, 2)
        img += 100 * np.exp(-0.5 * ((xx - kx)**2 + (yy - ky)**2) / 8**2)

    # Extended emission (large scale, j=4-5)
    img += 80 * np.exp(-0.5 * ((xx - 130)**2 + (yy - 128)**2) / 50**2)
    img += 40 * np.exp(-0.5 * ((xx - 180)**2 + (yy - 80)**2) / 35**2)

    # Simple a-trous decomposition for illustration
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

    def atrous_smooth(data, scale):
        """Smooth with a-trous kernel at given scale."""
        k = np.zeros(4 * 2**scale + 1)
        step = 2**scale
        for i, v in enumerate(kernel_1d):
            k[i * step] = v
        from scipy.ndimage import convolve1d
        result = convolve1d(data, k, axis=0, mode="reflect")
        result = convolve1d(result, k, axis=1, mode="reflect")
        return result

    c0 = img.copy()
    details = []
    for j in range(5):
        c1 = atrous_smooth(c0, j)
        w = c0 - c1
        details.append(w)
        c0 = c1

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Original
    vmin, vmax = np.percentile(img, [5, 99.5])
    axes[0, 0].imshow(img, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Original image")

    # Scale 1 detail (point sources)
    d1 = details[0]
    noise1 = 1.4826 * np.median(np.abs(d1))
    sig1 = np.abs(d1) / noise1
    axes[0, 1].imshow(d1, origin="lower", cmap="RdBu_r",
                      vmin=-3 * noise1, vmax=3 * noise1)
    axes[0, 1].contour(sig1 > 3, levels=[0.5], colors=["yellow"], linewidths=1)
    axes[0, 1].set_title("Scale 1 (2 px): point sources")

    # Scale 2 detail
    d2 = details[1]
    noise2 = 1.4826 * np.median(np.abs(d2))
    sig2 = np.abs(d2) / noise2
    axes[0, 2].imshow(d2, origin="lower", cmap="RdBu_r",
                      vmin=-3 * noise2, vmax=3 * noise2)
    axes[0, 2].contour(sig2 > 3, levels=[0.5], colors=["yellow"], linewidths=1)
    axes[0, 2].set_title("Scale 2 (4 px): resolved sources")

    # Scale 3 detail (nebular knots)
    d3 = details[2]
    noise3 = 1.4826 * np.median(np.abs(d3))
    sig3 = np.abs(d3) / noise3
    axes[1, 0].imshow(d3, origin="lower", cmap="RdBu_r",
                      vmin=-3 * noise3, vmax=3 * noise3)
    axes[1, 0].contour(sig3 > 3, levels=[0.5], colors=["yellow"], linewidths=1)
    axes[1, 0].set_title("Scale 3 (8 px): nebular knots")
    axes[1, 0].annotate("Extended feature\ndetected at medium scale",
                         xy=(60, 100), xytext=(140, 30),
                         fontsize=8, color="yellow",
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    # Scale 4-5 detail (extended emission)
    d45 = details[3] + details[4]
    noise45 = 1.4826 * np.median(np.abs(d45))
    axes[1, 1].imshow(d45, origin="lower", cmap="RdBu_r",
                      vmin=-3 * noise45, vmax=3 * noise45)
    axes[1, 1].set_title("Scales 4-5 (16-32 px): extended emission")
    axes[1, 1].annotate("Diffuse emission\nlarge-scale structure",
                         xy=(130, 128), xytext=(20, 200),
                         fontsize=8, color="cyan",
                         arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))

    # Scale spectrum
    energies = [np.sum(d**2) for d in details]
    total = sum(energies)
    fractions = [e / total for e in energies]
    scales = [f"{2**j} px" for j in range(5)]
    bars = axes[1, 2].bar(scales, fractions, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    axes[1, 2].set_xlabel("Scale")
    axes[1, 2].set_ylabel("Energy fraction")
    axes[1, 2].set_title("Scale spectrum (energy distribution)")
    axes[1, 2].annotate("Point sources\ndominate",
                         xy=(0, fractions[0]), xytext=(2, fractions[0] * 0.8),
                         fontsize=8, color="#2196F3",
                         arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1))

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_wavelet_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_stellar_population():
    """Stellar population: CMD with main sequence, RGB, blue stragglers."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Field population (broad MS, no features)
    n_field = 200
    mag_field = rng.uniform(14, 22, n_field)
    color_field = 0.3 + 0.05 * mag_field + rng.normal(0, 0.3, n_field)

    axes[0].scatter(color_field, mag_field, s=8, c="gray", alpha=0.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("BP - RP (mag)")
    axes[0].set_ylabel("G (mag)")
    axes[0].set_title("Field population\npopulation_score = 0.08")
    axes[0].set_facecolor("black")

    # Panel 2: Open cluster (tight MS, turnoff, few RGB)
    # Main sequence
    n_ms = 150
    mag_ms = rng.uniform(15, 21, n_ms)
    color_ms = 0.2 + 0.06 * mag_ms + rng.normal(0, 0.05, n_ms)

    # RGB
    n_rgb = 8
    mag_rgb = rng.uniform(12, 14.5, n_rgb)
    color_rgb = 1.2 + rng.normal(0, 0.1, n_rgb)

    # Blue stragglers
    n_bs = 3
    mag_bs = rng.uniform(14, 15, n_bs)
    color_bs = rng.uniform(0.0, 0.5, n_bs)

    axes[1].scatter(color_ms, mag_ms, s=10, c="white", alpha=0.7, label="Main sequence")
    axes[1].scatter(color_rgb, mag_rgb, s=25, c="red", alpha=0.9, label="Red giant branch")
    axes[1].scatter(color_bs, mag_bs, s=40, marker="*", c="cyan", alpha=0.9, label="Blue stragglers")

    # Mark turnoff
    axes[1].axhline(y=15, color="yellow", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].annotate("MS turnoff",
                     xy=(1.0, 15), xytext=(1.5, 15.5),
                     fontsize=9, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1))
    axes[1].annotate("Blue stragglers\n(brighter + bluer\nthan turnoff)",
                     xy=(color_bs[0], mag_bs[0]), xytext=(-0.3, 17),
                     fontsize=8, color="cyan",
                     arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    axes[1].annotate("RGB tip",
                     xy=(color_rgb[0], min(mag_rgb)), xytext=(1.8, 13),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    axes[1].invert_yaxis()
    axes[1].set_xlabel("BP - RP (mag)")
    axes[1].set_ylabel("G (mag)")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_title("Open cluster\npopulation_score = 0.52")
    axes[1].set_facecolor("black")

    # Panel 3: Globular cluster (multiple populations)
    # Population 1
    n_pop1 = 100
    mag_pop1 = rng.uniform(16, 21, n_pop1)
    color_pop1 = 0.1 + 0.05 * mag_pop1 + rng.normal(0, 0.03, n_pop1)

    # Population 2 (offset in color)
    n_pop2 = 80
    mag_pop2 = rng.uniform(16, 21, n_pop2)
    color_pop2 = 0.3 + 0.05 * mag_pop2 + rng.normal(0, 0.03, n_pop2)

    # RGB for both
    n_rgb2 = 15
    mag_rgb2 = rng.uniform(13, 15.5, n_rgb2)
    color_rgb2 = 1.0 + rng.uniform(-0.15, 0.15, n_rgb2)

    # HB
    n_hb = 12
    mag_hb = rng.uniform(15.5, 16.5, n_hb)
    color_hb = rng.uniform(-0.2, 0.8, n_hb)

    axes[2].scatter(color_pop1, mag_pop1, s=8, c="#4fc3f7", alpha=0.6, label="Population 1")
    axes[2].scatter(color_pop2, mag_pop2, s=8, c="#ff8a65", alpha=0.6, label="Population 2")
    axes[2].scatter(color_rgb2, mag_rgb2, s=20, c="red", alpha=0.8, label="RGB")
    axes[2].scatter(color_hb, mag_hb, s=20, c="yellow", alpha=0.8, label="Horizontal branch")

    axes[2].annotate("Two parallel\nmain sequences\n(multiple populations)",
                     xy=(1.1, 18.5), xytext=(1.6, 19),
                     fontsize=8, color="white",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    axes[2].annotate("Horizontal\nbranch",
                     xy=(0.3, 16), xytext=(-0.3, 14),
                     fontsize=8, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    axes[2].invert_yaxis()
    axes[2].set_xlabel("BP - RP (mag)")
    axes[2].set_ylabel("G (mag)")
    axes[2].legend(fontsize=7, loc="lower left")
    axes[2].set_title("Globular cluster (multiple pops)\npopulation_score = 0.74")
    axes[2].set_facecolor("black")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_stellar_population.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_variability_analysis():
    """Variability: light curves showing different variable types."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Constant star (not variable)
    n_epochs = 100
    mjd = np.sort(rng.uniform(59000, 59500, n_epochs))
    mag_const = 18.0 + rng.normal(0, 0.03, n_epochs)
    err_const = np.full(n_epochs, 0.03)

    axes[0, 0].errorbar(mjd - 59000, mag_const, yerr=err_const, fmt=".", color="gray",
                         markersize=3, elinewidth=0.5)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Constant star\nchi2_red=1.1, NOT variable")
    axes[0, 0].set_xlabel("MJD - 59000")
    axes[0, 0].set_ylabel("mag")

    # 2. Eclipsing binary
    period = 0.37
    phase = ((mjd - mjd[0]) / period) % 1.0
    eclipse = np.where(
        (phase > 0.45) & (phase < 0.55),
        0.8 * np.exp(-0.5 * ((phase - 0.5) / 0.02)**2),
        np.where(
            (phase > 0.95) | (phase < 0.05),
            0.3 * np.exp(-0.5 * (((phase + 0.5) % 1.0 - 0.5) / 0.02)**2),
            0
        )
    )
    mag_eb = 16.5 + eclipse + rng.normal(0, 0.02, n_epochs)

    axes[0, 1].scatter(phase, mag_eb, s=8, c="cyan", alpha=0.8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("Eclipsing binary (phased)\nP=0.37d, amp=0.8, FAP<0.001")
    axes[0, 1].set_xlabel("Phase")
    axes[0, 1].set_ylabel("mag")
    axes[0, 1].annotate("Primary eclipse\n(deep dip)", xy=(0.5, 17.3), xytext=(0.65, 17.2),
                         fontsize=8, color="yellow",
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1))
    axes[0, 1].annotate("Secondary eclipse\n(shallow dip)", xy=(0.0, 16.8), xytext=(0.15, 16.7),
                         fontsize=8, color="orange",
                         arrowprops=dict(arrowstyle="->", color="orange", lw=1))

    # 3. RR Lyrae pulsator
    period_rrl = 0.55
    phase_rrl = ((mjd - mjd[0]) / period_rrl) % 1.0
    # Asymmetric light curve (fast rise, slow decline)
    rrl_curve = 0.5 * (np.where(phase_rrl < 0.15,
                                 -phase_rrl / 0.15,
                                 (phase_rrl - 0.15) / 0.85) - 0.5)
    mag_rrl = 17.0 + rrl_curve + rng.normal(0, 0.02, n_epochs)

    axes[0, 2].scatter(phase_rrl, mag_rrl, s=8, c="#ff9800", alpha=0.8)
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_title("RR Lyrae pulsator (phased)\nP=0.55d, amp=0.5, periodic_pulsator")
    axes[0, 2].set_xlabel("Phase")
    axes[0, 2].set_ylabel("mag")
    axes[0, 2].annotate("Fast rise",
                         xy=(0.15, 16.75), xytext=(0.3, 16.65),
                         fontsize=8, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=1))

    # 4. AGN-like variability
    n_agn = 200
    mjd_agn = np.sort(rng.uniform(58000, 60000, n_agn))
    # Random walk (correlated variability)
    steps = rng.normal(0, 0.1, n_agn)
    mag_agn = 19.0 + np.cumsum(steps)
    mag_agn -= mag_agn.mean() - 19.0
    err_agn = np.full(n_agn, 0.05)

    axes[1, 0].errorbar(mjd_agn - 58000, mag_agn, yerr=err_agn, fmt=".", color="#e040fb",
                         markersize=3, elinewidth=0.5)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("AGN-like variability\namp=1.2, eta=0.8, no period, agn_like")
    axes[1, 0].set_xlabel("MJD - 58000")
    axes[1, 0].set_ylabel("mag")
    axes[1, 0].annotate("Correlated wandering\n(low von Neumann eta)",
                         xy=(1200, mag_agn[120]), xytext=(1500, 19.8),
                         fontsize=8, color="yellow",
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1))

    # 5. Supernova / transient
    n_sn = 80
    mjd_sn = np.sort(rng.uniform(59200, 59600, n_sn))
    t0 = 59350  # explosion time
    mag_sn = np.where(
        mjd_sn < t0,
        22.0 + rng.normal(0, 0.1, n_sn),
        22.0 - 5.0 * np.exp(-0.5 * ((mjd_sn - t0) / 15)**2) + rng.normal(0, 0.05, n_sn)
    )

    axes[1, 1].errorbar(mjd_sn - 59200, mag_sn, yerr=0.08, fmt=".", color="#f44336",
                         markersize=4, elinewidth=0.5)
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(x=t0 - 59200, color="white", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Supernova transient\n5-mag brightening, fading, transient")
    axes[1, 1].set_xlabel("MJD - 59200")
    axes[1, 1].set_ylabel("mag")
    axes[1, 1].annotate("Peak brightness\n(outburst detected\nat 12.5 sigma)",
                         xy=(t0 - 59200, min(mag_sn)), xytext=(250, 19),
                         fontsize=8, color="yellow",
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
    axes[1, 1].annotate("Pre-explosion\nbaseline",
                         xy=(50, 22), xytext=(20, 20),
                         fontsize=8, color="gray",
                         arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    # 6. Lomb-Scargle periodogram example
    from astropy.timeseries import LombScargle
    freq, power = LombScargle(mjd, mag_eb, err_const).autopower(
        minimum_frequency=0.5, maximum_frequency=10
    )
    period_grid = 1.0 / freq

    axes[1, 2].plot(period_grid, power, color="cyan", linewidth=0.8)
    axes[1, 2].axvline(x=0.37, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1, 2].set_xlabel("Period (days)")
    axes[1, 2].set_ylabel("Lomb-Scargle Power")
    axes[1, 2].set_title("Periodogram (eclipsing binary)\nFAP < 0.001 at P=0.37d")
    axes[1, 2].set_xlim(0.1, 2.0)
    axes[1, 2].annotate("True period\nP=0.37 days",
                         xy=(0.37, max(power)), xytext=(0.8, max(power) * 0.8),
                         fontsize=9, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_variability_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_transient_detection():
    """Transient (catalog-based): astrometric noise, color outliers, parallax anomalies."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Astrometric excess noise distribution
    n = 200
    noise = rng.exponential(0.3, n)
    # Add a few outliers (unresolved binaries, AGN)
    outliers = rng.exponential(2.0, 8) + 2.0
    all_noise = np.concatenate([noise, outliers])

    axes[0].hist(all_noise, bins=40, color="gray", alpha=0.6, edgecolor="white", linewidth=0.5)
    threshold = np.median(noise) + 3 * 1.4826 * np.median(np.abs(noise - np.median(noise)))
    axes[0].axvline(x=threshold, color="red", linestyle="--", linewidth=2)
    axes[0].hist(all_noise[all_noise > threshold], bins=20, color="red", alpha=0.7)
    axes[0].set_xlabel("Astrometric excess noise (mas)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Astrometric excess noise\n8 outliers > 3 sigma (red)")
    axes[0].annotate(f"Threshold = {threshold:.2f} mas\n(median + 3*MAD)",
                     xy=(threshold, 5), xytext=(threshold + 1, 15),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1))

    # Panel 2: Color-magnitude diagram with photometric outliers
    mag = rng.uniform(14, 21, 200)
    color = 0.3 + 0.04 * mag + rng.normal(0, 0.15, 200)
    # Color outliers
    n_out = 6
    mag_out = rng.uniform(15, 20, n_out)
    color_out = rng.choice([-1, 1], n_out) * rng.uniform(0.8, 1.5, n_out) + 0.3 + 0.04 * mag_out

    axes[1].scatter(color, mag, s=8, c="gray", alpha=0.5)
    axes[1].scatter(color_out, mag_out, s=40, c="red", marker="x", linewidth=2)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("BP - RP (mag)")
    axes[1].set_ylabel("G (mag)")
    axes[1].set_title("Photometric outliers\n6 color-deviant sources (red)")
    axes[1].annotate("Blue outlier\n(possible AGN/QSO)",
                     xy=(color_out[0], mag_out[0]),
                     xytext=(color_out[0] - 0.3, mag_out[0] + 2),
                     fontsize=8, color="cyan",
                     arrowprops=dict(arrowstyle="->", color="cyan", lw=1))
    axes[1].annotate("Red outlier\n(dust-reddened?)",
                     xy=(color_out[3], mag_out[3]),
                     xytext=(color_out[3] + 0.3, mag_out[3] - 2),
                     fontsize=8, color="orange",
                     arrowprops=dict(arrowstyle="->", color="orange", lw=1))

    # Panel 3: Parallax SNR distribution
    parallax = rng.normal(0.5, 0.3, 200)
    parallax_err = rng.uniform(0.05, 0.5, 200)
    snr = parallax / parallax_err

    # Negative parallax sources
    neg_mask = parallax < 0
    low_snr = snr < 3
    flagged = neg_mask | low_snr

    axes[2].scatter(parallax[~flagged], snr[~flagged], s=8, c="gray", alpha=0.5)
    axes[2].scatter(parallax[neg_mask], snr[neg_mask], s=30, c="red", marker="x", linewidth=1.5,
                    label=f"Negative parallax ({neg_mask.sum()})")
    axes[2].scatter(parallax[low_snr & ~neg_mask], snr[low_snr & ~neg_mask],
                    s=20, c="orange", alpha=0.7, label=f"Low SNR ({(low_snr & ~neg_mask).sum()})")
    axes[2].axhline(y=3, color="yellow", linestyle="--", linewidth=1, alpha=0.7)
    axes[2].axvline(x=0, color="red", linestyle=":", linewidth=1, alpha=0.5)
    axes[2].set_xlabel("Parallax (mas)")
    axes[2].set_ylabel("Parallax SNR")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Parallax anomalies\nnegative = distant/QSO, low SNR = unreliable")
    axes[2].annotate("Negative parallax\n(background QSO?)",
                     xy=(-0.3, -2), xytext=(-0.8, 5),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_transient_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_anomaly_detection():
    """Anomaly detection: Isolation Forest on detector score vectors."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate synthetic detector score vectors
    n_normal = 50
    n_anomalous = 5

    # Normal detections: low-moderate scores, correlated
    normal_scores = rng.beta(2, 8, (n_normal, 2)) * 0.4 + 0.05

    # Anomalous detections: unusual score patterns
    anomalous_scores = np.array([
        [0.7, 0.05],  # High lens, low morphology
        [0.05, 0.8],  # Low lens, high morphology
        [0.6, 0.6],   # Both high (rare)
        [0.02, 0.02],  # Both very low (void)
        [0.75, 0.4],  # High lens + moderate morphology
    ])

    # Panel 1: Score space with IF decision boundary
    all_scores = np.vstack([normal_scores, anomalous_scores])

    axes[0].scatter(normal_scores[:, 0], normal_scores[:, 1], s=30, c="gray",
                    alpha=0.6, label="Normal detections")
    axes[0].scatter(anomalous_scores[:, 0], anomalous_scores[:, 1], s=80, c="red",
                    marker="x", linewidth=2, label="Anomalies (IF outliers)")

    # Draw approximate decision boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    boundary_x = 0.15 + 0.2 * np.cos(theta)
    boundary_y = 0.13 + 0.15 * np.sin(theta)
    axes[0].plot(boundary_x, boundary_y, "--", color="yellow", linewidth=1.5, alpha=0.7,
                 label="IF decision boundary")

    axes[0].set_xlabel("Lens detector score")
    axes[0].set_ylabel("Morphology detector score")
    axes[0].set_title("Isolation Forest: Anomalies in\n2D score space (2 of 13 dimensions shown)")
    axes[0].legend(fontsize=8)

    for i, (x, y) in enumerate(anomalous_scores):
        labels = ["High lens only", "High morph only", "Both high", "Both low", "Lens + morph"]
        if i < len(labels):
            axes[0].annotate(labels[i],
                             xy=(x, y), xytext=(x + 0.08, y + 0.08),
                             fontsize=7, color="red",
                             arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    # Panel 2: Anomaly score distribution
    normal_anom_scores = rng.beta(2, 5, n_normal) * 0.4
    anomalous_anom_scores = rng.beta(5, 2, n_anomalous) * 0.5 + 0.5

    axes[1].hist(normal_anom_scores, bins=15, color="gray", alpha=0.6, edgecolor="white",
                 label="Normal")
    axes[1].hist(anomalous_anom_scores, bins=5, color="red", alpha=0.7, edgecolor="white",
                 label="Anomalies")
    axes[1].axvline(x=0.5, color="yellow", linestyle="--", linewidth=2, label="Threshold")
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Anomaly score distribution\n(normalized IF output)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_anomaly_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_ensemble_scoring():
    """Ensemble: how 13 detector scores combine into final anomaly_score."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    detectors = [
        "classical", "morphology", "anomaly", "lens", "distribution",
        "galaxy", "kinematic", "transient", "sersic", "wavelet",
        "population", "variability"
    ]
    weights = [0.09, 0.09, 0.09, 0.09, 0.11, 0.09, 0.09, 0.04, 0.07, 0.09, 0.06, 0.09]

    # Example 1: Gravitational lens candidate
    scores_lens = [0.15, 0.20, 0.65, 0.72, 0.10, 0.25, 0.05, 0.08, 0.35, 0.18, 0.05, 0.02]
    colors_lens = ["#607D8B" if s < 0.3 else "#FF9800" if s < 0.5 else "#F44336" for s in scores_lens]

    y_pos = np.arange(len(detectors))
    bars = axes[0].barh(y_pos, scores_lens, color=colors_lens, height=0.7, edgecolor="white",
                         linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(detectors, fontsize=9)
    axes[0].set_xlabel("Detector Score")
    axes[0].set_xlim(0, 1)

    # Calculate weighted score
    weighted = sum(w * s for w, s in zip(weights, scores_lens))
    axes[0].set_title(f"Gravitational Lens Candidate\nanomaly_score = {weighted:.3f}")

    # Annotate the dominant detector
    axes[0].annotate(f"Dominant: lens (0.72)\n-> classification: gravitational_lens",
                     xy=(0.72, 3), xytext=(0.4, 8),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Example 2: Kinematic group with variability
    scores_kin = [0.08, 0.10, 0.30, 0.05, 0.35, 0.08, 0.62, 0.15, 0.12, 0.10, 0.28, 0.45]
    colors_kin = ["#607D8B" if s < 0.3 else "#FF9800" if s < 0.5 else "#F44336" for s in scores_kin]

    bars2 = axes[1].barh(y_pos, scores_kin, color=colors_kin, height=0.7, edgecolor="white",
                          linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(detectors, fontsize=9)
    axes[1].set_xlabel("Detector Score")
    axes[1].set_xlim(0, 1)

    weighted2 = sum(w * s for w, s in zip(weights, scores_kin))
    axes[1].set_title(f"Kinematic Group + Variable Stars\nanomaly_score = {weighted2:.3f}")

    axes[1].annotate(f"Dominant: kinematic (0.62)\nSecond: variability (0.45)\n-> classification: kinematic_group",
                     xy=(0.62, 6), xytext=(0.3, 1),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_ensemble_scoring.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate all detection example images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        ("source_extraction", example_source_extraction),
        ("classical_detection", example_classical_detection),
        ("morphology_analysis", example_morphology_analysis),
        ("lens_detection", example_lens_detection),
        ("distribution_analysis", example_distribution_analysis),
        ("galaxy_interaction", example_galaxy_interaction),
        ("kinematic_analysis", example_kinematic_analysis),
        ("sersic_fitting", example_sersic_fitting),
        ("wavelet_analysis", example_wavelet_analysis),
        ("stellar_population", example_stellar_population),
        ("variability_analysis", example_variability_analysis),
        ("transient_detection", example_transient_detection),
        ("anomaly_detection", example_anomaly_detection),
        ("ensemble_scoring", example_ensemble_scoring),
    ]

    for name, func in generators:
        print(f"Generating {name}...")
        try:
            func()
            print(f"  -> {OUTPUT_DIR / f'example_{name}.png'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {len(generators)} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
