"""Discovery mosaic: visual summary of findings."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from star_pattern.core.fits_handler import FITSImage
from star_pattern.evaluation.metrics import PatternResult
from star_pattern.utils.logging import get_logger

logger = get_logger("visualization.mosaic")


def _load_fits_from_metadata(finding: PatternResult) -> FITSImage | None:
    """Try to load a FITS image for a finding from disk.

    Attempts two strategies:
    1. Direct path: metadata["fits_path"] saved during the pipeline run.
    2. Cache lookup: search the data cache by RA/Dec for older runs
       that don't have fits_path.

    Returns None if no image can be loaded.
    """
    from pathlib import Path

    # Strategy 1: direct path from metadata
    fits_path = finding.metadata.get("fits_path")
    if fits_path:
        p = Path(fits_path)
        if p.exists():
            try:
                return FITSImage.from_file(p)
            except Exception as e:
                logger.debug(f"Failed to load FITS from {p}: {e}")

    # Strategy 2: look up in data cache by RA/Dec.
    # Only attempt if metadata shows images were saved during the run
    # (prevents spurious cache hits in tests or image-free runs).
    if not finding.metadata.get("image_paths"):
        return None

    ra = getattr(finding, "region_ra", None)
    dec = getattr(finding, "region_dec", None)
    if ra is None or dec is None:
        return None

    try:
        from star_pattern.data.cache import DataCache
        cache = DataCache()
        # Try common bands in priority order
        for band in ("r", "i", "g", "z"):
            cached = cache.get_path("sdss", ra, dec, 3.0, band=band)
            if cached is not None:
                return FITSImage.from_file(cached)
    except Exception as e:
        logger.debug(f"Cache lookup failed for ({ra:.3f}, {dec:.3f}): {e}")

    return None


def _extract_sub_detections(details: dict) -> dict[str, int]:
    """Extract per-detector object counts from a finding's details dict.

    Returns only non-zero entries. Keys are human-readable detector names.
    """
    if not details:
        return {}

    counts: dict[str, int] = {}

    # Sources (catalog extractions)
    sources = details.get("sources", {})
    n_src = sources.get("n_sources", 0)
    if n_src == 0:
        positions = sources.get("positions", [])
        if hasattr(positions, "__len__"):
            n_src = len(positions)
    if n_src:
        counts["sources"] = n_src

    # Wavelet detections and multiscale objects
    wavelet = details.get("wavelet", {})
    n_wav = len(wavelet.get("detections", []))
    if n_wav:
        counts["wavelet features"] = n_wav
    n_ms = len(wavelet.get("multiscale_objects", []))
    if n_ms:
        counts["multiscale objects"] = n_ms

    # Lens arcs and rings
    lens = details.get("lens", {})
    n_arcs = len(lens.get("arcs", []))
    if n_arcs:
        counts["lens arcs"] = n_arcs
    n_rings = len(lens.get("rings", []))
    if n_rings:
        counts["lens rings"] = n_rings

    # Galaxy tidal features and merger nuclei
    galaxy = details.get("galaxy", {})
    n_tidal = len(galaxy.get("tidal_features", []))
    if n_tidal:
        counts["tidal features"] = n_tidal
    n_nuclei = len(galaxy.get("merger_nuclei", []))
    if n_nuclei:
        counts["merger nuclei"] = n_nuclei

    # Distribution overdensities
    dist = details.get("distribution", {})
    n_od = len(dist.get("overdensities", []))
    if n_od:
        counts["overdensities"] = n_od

    # Classical arcs
    classical = details.get("classical", {})
    n_classical = classical.get("n_arcs", 0)
    if n_classical == 0:
        n_classical = len(classical.get("arcs", []))
    if n_classical == 0:
        n_classical = len(classical.get("hough_arcs", []))
    if n_classical:
        counts["classical arcs"] = n_classical

    # Sersic residual features
    sersic = details.get("sersic", {})
    n_sersic = len(sersic.get("residual_features", []))
    if n_sersic:
        counts["Sersic residuals"] = n_sersic

    # Kinematic: comoving groups, stream candidates, runaway stars
    kinematic = details.get("kinematic", {})
    n_comov = len(kinematic.get("comoving_groups", []))
    if n_comov:
        counts["comoving groups"] = n_comov
    n_stream = len(kinematic.get("stream_candidates", []))
    if n_stream:
        counts["stream candidates"] = n_stream
    n_runaway = len(kinematic.get("runaway_stars", []))
    if n_runaway:
        counts["runaway stars"] = n_runaway

    # Transient outliers (sum all outlier lists)
    transient = details.get("transient", {})
    n_trans = (
        len(transient.get("flux_outliers", []))
        + len(transient.get("color_outliers", []))
        + len(transient.get("proper_motion_outliers", []))
        + len(transient.get("parallax_outliers", []))
    )
    if n_trans:
        counts["transient outliers"] = n_trans

    # Variability candidates
    variability = details.get("variability", {})
    n_var = len(variability.get("variable_candidates", []))
    if n_var:
        counts["variable candidates"] = n_var
    n_per = len(variability.get("periodic_candidates", []))
    if n_per:
        counts["periodic candidates"] = n_per

    # Stellar population counts
    population = details.get("population", {})
    n_bs = population.get("n_blue_stragglers", 0)
    if n_bs:
        counts["blue stragglers"] = n_bs
    n_rg = population.get("n_red_giants", 0)
    if n_rg:
        counts["red giants"] = n_rg

    # Temporal (multi-epoch image differencing)
    temporal = details.get("temporal", {})
    n_new = temporal.get("n_new_sources", 0)
    if n_new:
        counts["new sources (temporal)"] = n_new
    n_dis = temporal.get("n_disappeared", 0)
    if n_dis:
        counts["disappeared sources"] = n_dis
    n_brt = temporal.get("n_brightenings", 0)
    if n_brt:
        counts["brightenings"] = n_brt
    n_fad = temporal.get("n_fadings", 0)
    if n_fad:
        counts["fadings"] = n_fad
    n_mov = temporal.get("n_moving", 0)
    if n_mov:
        counts["moving objects"] = n_mov

    return counts


def _total_sub_detections(details: dict) -> int:
    """Total anomalous sub-detections (excludes catalog sources)."""
    counts = _extract_sub_detections(details)
    return sum(v for k, v in counts.items() if k != "sources")


def _categorize_findings(
    findings: list[PatternResult],
) -> tuple[list[PatternResult], list[PatternResult], list[PatternResult]]:
    """Split findings into (new, known, low_confidence).

    - low_confidence: verdict == 'artifact' or significance_rating <= 2
    - known: has cross_matches
    - new: everything else

    Each list sorted by anomaly_score descending.
    """
    new: list[PatternResult] = []
    known: list[PatternResult] = []
    low_conf: list[PatternResult] = []

    for f in findings:
        eval_data = f.metadata.get("local_evaluation", {})
        verdict = eval_data.get("verdict", f.debate_verdict or "")
        sig_rating = eval_data.get("significance_rating", 0)

        if verdict == "artifact" or sig_rating <= 2:
            low_conf.append(f)
        elif f.cross_matches:
            known.append(f)
        else:
            new.append(f)

    new.sort(key=lambda f: f.anomaly_score, reverse=True)
    known.sort(key=lambda f: f.anomaly_score, reverse=True)
    low_conf.sort(key=lambda f: f.anomaly_score, reverse=True)

    return new, known, low_conf


def _assign_finding_numbers(
    findings: list[PatternResult],
) -> list[tuple[int, str, PatternResult]]:
    """Assign sequential finding numbers across categories.

    Returns list of (finding_number, category_tag, finding) tuples.
    Order: new discoveries first, then known, then low-confidence,
    each sorted by anomaly_score descending.

    Category tags: "NEW", "KNOWN", "LOW".
    """
    new, known, low_conf = _categorize_findings(findings)

    numbered: list[tuple[int, str, PatternResult]] = []
    n = 1
    for f in new:
        numbered.append((n, "NEW", f))
        n += 1
    for f in known:
        numbered.append((n, "KNOWN", f))
        n += 1
    for f in low_conf:
        numbered.append((n, "LOW", f))
        n += 1

    return numbered


def _draw_panel_overlays(ax: plt.Axes, finding: PatternResult) -> None:
    """Draw lightweight detection overlays on a mosaic panel.

    Reads detection data from finding.details and draws the same markers
    as the individual overlay functions, but sized for small panels.
    """
    detection = finding.details
    if not detection:
        return

    # Sources: stars (cyan *) and galaxies (lime o)
    sources = detection.get("sources", {})
    positions = np.array(sources.get("positions", []))
    if positions.ndim == 2 and len(positions) > 0:
        star_mask = np.array(
            sources.get("star_mask", np.ones(len(positions), dtype=bool)),
            dtype=bool,
        )
        stars = positions[star_mask]
        galaxies = positions[~star_mask]
        if len(stars) > 0:
            ax.scatter(
                stars[:, 0], stars[:, 1], s=8, marker="*",
                facecolors="none", edgecolors="cyan", linewidth=0.4,
                zorder=5,
            )
        if len(galaxies) > 0:
            ax.scatter(
                galaxies[:, 0], galaxies[:, 1], s=12, marker="o",
                facecolors="none", edgecolors="lime", linewidth=0.4,
                zorder=5,
            )

    # Lens: central source cross + arc/ring circles
    lens = detection.get("lens", {})
    if lens.get("lens_score", 0) > 0.15:
        central = lens.get("central_source", {})
        cx, cy = central.get("x", 128), central.get("y", 128)
        if central:
            ax.plot(cx, cy, "r+", markersize=8, markeredgewidth=1.5, zorder=6)
        for arc in lens.get("arcs", []):
            circle = Circle(
                (cx, cy), arc["radius"],
                fill=False, edgecolor="yellow", linewidth=0.8,
                linestyle="--", alpha=0.6, zorder=6,
            )
            ax.add_patch(circle)
        for ring in lens.get("rings", []):
            color = "lime" if ring.get("is_complete_ring") else "orange"
            circle = Circle(
                (cx, cy), ring["radius"],
                fill=False, edgecolor=color, linewidth=1.2, zorder=6,
            )
            ax.add_patch(circle)

    # Galaxy: tidal features (magenta) + merger nuclei (red +)
    galaxy = detection.get("galaxy", {})
    if galaxy.get("galaxy_score", 0) > 0.15:
        for feat in galaxy.get("tidal_features", []):
            radius = max(4, feat.get("extent_px", 20))
            circle = Circle(
                (feat["x"], feat["y"]), radius,
                fill=False, edgecolor="magenta", linewidth=0.8,
                linestyle="--", zorder=6,
            )
            ax.add_patch(circle)
        nuclei = galaxy.get("merger_nuclei", [])
        for n in nuclei:
            ax.plot(n["x"], n["y"], "r+", markersize=6, markeredgewidth=1, zorder=6)

    # Distribution: overdensity circles (red)
    dist = detection.get("distribution", {})
    if dist.get("distribution_score", 0) > 0.15:
        for od in dist.get("overdensities", []):
            radius = max(3, od.get("radius_px", 10))
            circle = Circle(
                (od["x"], od["y"]), radius,
                fill=False, edgecolor="red", linewidth=0.8, zorder=6,
            )
            ax.add_patch(circle)

    # Classical: Hough arc circles (orange)
    classical = detection.get("classical", {})
    if classical.get("gabor_score", 0) > 0.15 or classical.get("arc_score", 0) > 0.15:
        for arc in classical.get("hough_arcs", []):
            circle = Circle(
                (arc["cx"], arc["cy"]), arc["radius"],
                fill=False, edgecolor="orange", linewidth=0.8,
                linestyle="--", zorder=6,
            )
            ax.add_patch(circle)

    # Wavelet: scale-colored circles
    wavelet = detection.get("wavelet", {})
    if wavelet.get("wavelet_score", 0) > 0.15:
        scale_colors = ["#00ffff", "#00ff00", "#ffff00", "#ff8800", "#ff0000"]
        for det in wavelet.get("detections", []):
            scale_idx = det.get("scale", 0)
            color = scale_colors[min(scale_idx, len(scale_colors) - 1)]
            radius = max(2, det.get("area_px", 25) ** 0.5)
            circle = Circle(
                (det["x"], det["y"]), radius,
                fill=False, edgecolor=color, linewidth=0.6, alpha=0.7,
                zorder=5,
            )
            ax.add_patch(circle)
        for ms in wavelet.get("multiscale_objects", []):
            ax.plot(
                ms["x"], ms["y"], "s", color="white", markersize=4,
                markeredgewidth=0.8, markerfacecolor="none", zorder=6,
            )

    # Sersic: residual features (white dotted circles)
    sersic = detection.get("sersic", {})
    if sersic.get("sersic_score", 0) > 0.15:
        for feat in sersic.get("residual_features", []):
            radius = max(3, feat.get("area_px", 50) ** 0.5)
            circle = Circle(
                (feat["x"], feat["y"]), radius,
                fill=False, edgecolor="white", linewidth=0.6,
                linestyle=":", zorder=5,
            )
            ax.add_patch(circle)

    # Transient: outlier markers
    transient = detection.get("transient", {})
    if transient.get("transient_score", 0) > 0.15:
        for out in transient.get("flux_outliers", []):
            ax.plot(
                out["x"], out["y"], "^", color="#FF5722",
                markersize=5, zorder=6,
            )
        for out in transient.get("color_outliers", []):
            ax.plot(
                out["x"], out["y"], "v", color="#FF9800",
                markersize=4, zorder=6,
            )


def _has_source_at_center(
    image: FITSImage, px: float, py: float, min_snr: float = 2.0,
) -> bool:
    """Check if there is a detectable source at the given pixel location.

    Compares the center pixel value against the local background
    (median of a surrounding annulus). Returns True if the center
    is at least min_snr standard deviations above the local median.
    """
    h, w = image.data.shape[:2]
    cx, cy = int(px), int(py)
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return False

    # Sample a small box around center (5x5) and a larger annulus (30x30)
    r_inner = 3
    r_outer = 30
    x0 = max(0, cx - r_outer)
    x1 = min(w, cx + r_outer)
    y0 = max(0, cy - r_outer)
    y1 = min(h, cy + r_outer)

    region = image.data[y0:y1, x0:x1]
    if region.size < 10:
        return False

    local_med = float(np.median(region))
    local_std = float(np.std(region))
    if local_std < 1e-10:
        return False

    # Average of 5x5 around center
    cx0 = max(0, cx - r_inner)
    cx1 = min(w, cx + r_inner + 1)
    cy0 = max(0, cy - r_inner)
    cy1 = min(h, cy + r_inner + 1)
    center_val = float(np.mean(image.data[cy0:cy1, cx0:cx1]))

    return abs(center_val - local_med) / local_std >= min_snr


def _collect_top_anomalies(
    findings: list[PatternResult],
    images: list[FITSImage | None] | None,
    max_anomalies: int,
) -> list[tuple[int, str, PatternResult, FITSImage | None, Any]]:
    """Collect top anomalies across all findings for mosaic display.

    Returns list of (finding_num, category_tag, finding, image, anomaly)
    tuples sorted by anomaly score descending, capped at max_anomalies.
    Filters out anomalies with no detectable source at their pixel
    location to avoid blank cutout panels.
    """
    from star_pattern.evaluation.metrics import Anomaly

    padded_images = images if images else [None] * len(findings)
    image_map: dict[int, FITSImage | None] = {}
    for f, img in zip(findings, padded_images):
        if img is None:
            # Fallback: reload from FITS cache path stored in metadata
            img = _load_fits_from_metadata(f)
        image_map[id(f)] = img

    numbered = _assign_finding_numbers(findings)

    # Gather all anomalies with their parent finding context
    all_items: list[tuple[int, str, PatternResult, FITSImage | None, Anomaly]] = []
    for finding_num, tag, finding in numbered:
        img = image_map.get(id(finding))
        for anomaly in finding.anomalies:
            all_items.append((finding_num, tag, finding, img, anomaly))

    # Sort by confidence (descending), fall back to score
    def _sort_key(item: tuple) -> float:
        anomaly = item[4]
        if anomaly.confidence is not None:
            return anomaly.confidence.confidence
        return anomaly.score

    all_items.sort(key=_sort_key, reverse=True)

    # Extended feature types: tidal features, sersic residuals, and
    # multiscale objects are diffuse structures that won't pass a
    # point-source signal check. Always include them.
    _EXTENDED_TYPES = {
        "tidal_feature", "sersic_residual", "multiscale_object",
        "overdensity", "merger",
    }

    # Filter: skip point-source anomalies whose cutout center has no
    # visible source. Extended features bypass this check.
    selected: list[tuple[int, str, PatternResult, FITSImage | None, Anomaly]] = []
    for item in all_items:
        if len(selected) >= max_anomalies:
            break
        finding_num, tag, finding, img, anomaly = item

        # Catalog-only anomalies (no image): always include
        if img is None:
            selected.append(item)
            continue

        # Extended features: always include (they're diffuse, not point sources)
        if anomaly.anomaly_type in _EXTENDED_TYPES:
            selected.append(item)
            continue

        px, py = _resolve_pixel_coords(anomaly, img)
        if px is None or py is None:
            # No renderable position -- include as text panel
            selected.append(item)
            continue

        # Point-source anomalies: check for actual signal at center
        if _has_source_at_center(img, px, py):
            selected.append(item)
        else:
            logger.debug(
                f"Skipping blank anomaly {anomaly.anomaly_type} at "
                f"px=({px:.0f},{py:.0f}): no source above background"
            )

    return selected


def _sky_to_pixel(
    ra: float, dec: float, image: FITSImage,
) -> tuple[float | None, float | None]:
    """Convert RA/Dec to pixel coords on a specific image via WCS.

    Returns (None, None) if conversion fails or position falls outside image.
    """
    if image.wcs is None:
        return None, None
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        px, py = image.wcs.world_to_pixel(coord)
        px, py = float(px), float(py)
        h, w = image.data.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            return px, py
    except Exception:
        pass
    return None, None


def _resolve_pixel_coords(
    anomaly: Any, image: FITSImage | None,
) -> tuple[float | None, float | None]:
    """Resolve pixel coordinates for an anomaly on a given image.

    1. If the anomaly has pixel_x/y, return those (assumes caller
       verified they are valid for this image).
    2. Otherwise try WCS conversion from sky_ra/dec.
    """
    if anomaly.pixel_x is not None and anomaly.pixel_y is not None:
        return anomaly.pixel_x, anomaly.pixel_y

    if (
        image is not None
        and anomaly.sky_ra is not None
        and anomaly.sky_dec is not None
    ):
        return _sky_to_pixel(anomaly.sky_ra, anomaly.sky_dec, image)

    return None, None


def _draw_anomaly_cutout(
    ax: plt.Axes,
    image: FITSImage | None,
    anomaly: Any,
) -> None:
    """Draw a cutout centered on a single anomaly.

    For catalog-based anomalies (no pixel coords), tries WCS conversion
    from RA/Dec to show a real image cutout instead of a blank text panel.

    Cutout radius scales with image size: ~3% of the smaller image
    dimension, clamped to [30, 200] pixels.
    """
    px, py = _resolve_pixel_coords(anomaly, image)

    if image is None or px is None or py is None:
        ax.set_facecolor("#1a1a1a")
        # Text info panel for anomalies with no renderable location
        info_parts = [anomaly.anomaly_type.replace("_", " ").capitalize()]
        if anomaly.sky_ra is not None:
            info_parts.append(f"RA={anomaly.sky_ra:.4f}")
            info_parts.append(f"Dec={anomaly.sky_dec:.4f}")
        for k, v in list(anomaly.properties.items())[:4]:
            info_parts.append(f"{k}={v}")
        ax.text(
            0.5, 0.5, "\n".join(info_parts),
            ha="center", va="center", fontsize=7, color="white",
            transform=ax.transAxes,
        )
        return

    # Extract cutout from image
    h, w = image.data.shape[:2]
    cx = int(px)
    cy = int(py)

    # Determine cutout radius based on feature type.
    # Extended features (tidal tails, sersic residuals) need a larger
    # cutout to show the full diffuse structure.
    feature_area = anomaly.properties.get("area", anomaly.properties.get("area_px", 0))
    is_extended = anomaly.anomaly_type in (
        "tidal_feature", "sersic_residual", "multiscale_object",
        "overdensity", "merger",
    )

    if is_extended and feature_area > 100:
        # Size cutout to ~1.5x the feature's characteristic radius
        feature_radius = int(np.sqrt(feature_area) * 0.75)
        r = max(60, min(400, feature_radius))
    else:
        # Default: ~3% of smaller image dimension
        r = max(30, min(200, int(min(h, w) * 0.03)))

    if is_extended:
        # Extended feature centroids (gradient blobs, residuals) often
        # land in faint sky between the bright source and the background.
        # Recenter the cutout on the brightest pixel within a search
        # radius so the visualization shows the source generating the
        # feature, with the gradient structure visible around it.
        search_r = r
        sx0 = max(0, cx - search_r)
        sx1 = min(w, cx + search_r)
        sy0 = max(0, cy - search_r)
        sy1 = min(h, cy + search_r)
        search_region = image.data[sy0:sy1, sx0:sx1]
        if search_region.size > 0:
            peak_idx = np.unravel_index(
                np.nanargmax(search_region), search_region.shape,
            )
            new_cy = sy0 + int(peak_idx[0])
            new_cx = sx0 + int(peak_idx[1])
            # Only shift if the peak is significantly brighter than
            # the original center (avoid jitter on uniform regions).
            peak_val = image.data[new_cy, new_cx]
            orig_val = image.data[
                min(cy, h - 1), min(cx, w - 1)
            ]
            bg_std = float(np.nanstd(search_region))
            if bg_std > 0 and (peak_val - orig_val) / bg_std > 2.0:
                cx, cy = new_cx, new_cy

    # Clamp to image bounds
    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)

    if x1 <= x0 or y1 <= y0:
        ax.set_facecolor("#1a1a1a")
        return

    cutout = image.data[y0:y1, x0:x1]
    try:
        from astropy.visualization import ZScaleInterval

        if is_extended:
            # For diffuse features, use a more aggressive stretch:
            # ZScale with wider contrast to reveal low surface brightness.
            interval = ZScaleInterval(contrast=0.15)
        else:
            interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(cutout.flatten())
        ax.imshow(
            cutout, origin="lower", cmap="gray_r",
            vmin=vmin, vmax=vmax,
        )
    except Exception:
        ax.imshow(cutout, origin="lower", cmap="gray_r")

    # Clip axes to cutout bounds so overlays don't extend into white padding
    cut_h, cut_w = cutout.shape[:2]
    ax.set_xlim(-0.5, cut_w - 0.5)
    ax.set_ylim(-0.5, cut_h - 0.5)

    # Draw crosshair at the original detection position within cutout.
    # For recentered extended features, this marks where the gradient/
    # residual was detected, while the cutout view is centered on the
    # nearest bright source for context.
    det_x = int(px) - x0
    det_y = int(py) - y0
    ax.axhline(det_y, color="cyan", linewidth=0.5, alpha=0.4, zorder=5)
    ax.axvline(det_x, color="cyan", linewidth=0.5, alpha=0.4, zorder=5)
    local_x = det_x
    local_y = det_y

    # Draw a circle around the anomaly, capped to fit within cutout.
    # For extended features, derive circle radius from feature area.
    if is_extended and feature_area > 100:
        radius = int(np.sqrt(feature_area / np.pi))
    else:
        radius = anomaly.properties.get(
            "radius", anomaly.properties.get("radius_px", 8),
        )
    if isinstance(radius, (int, float)) and radius > 0:
        # Cap radius so circle stays within cutout
        max_radius = min(
            local_x, local_y,
            cut_w - local_x, cut_h - local_y,
            r * 0.8,
        )
        draw_radius = min(radius, max(max_radius, 3))
        circle = Circle(
            (local_x, local_y), draw_radius,
            fill=False, edgecolor="yellow", linewidth=1.0,
            linestyle="--", zorder=6,
        )
        ax.add_patch(circle)


def _draw_overview_panel(
    ax: plt.Axes,
    findings: list[PatternResult],
    images: list[FITSImage | None] | None,
    anomaly_items: list[tuple[int, str, PatternResult, FITSImage | None, Any]],
) -> None:
    """Draw overview panel showing full image with numbered anomaly positions."""
    # Use the first available image (try fallback from cache if needed)
    padded_images = images if images else [None] * len(findings)
    overview_image = None
    for i, img in enumerate(padded_images):
        if img is not None:
            overview_image = img
            break
    if overview_image is None:
        # Fallback: try loading from FITS cache paths in metadata
        for f in findings:
            loaded = _load_fits_from_metadata(f)
            if loaded is not None:
                overview_image = loaded
                break

    if overview_image is not None:
        try:
            norm = overview_image.normalize("zscale")
            ax.imshow(norm.data, origin="lower", cmap="gray_r")
        except Exception:
            ax.set_facecolor("#1a1a1a")
    else:
        ax.set_facecolor("#1a1a1a")

    # Plot numbered markers for each anomaly.
    # Always use WCS conversion to the overview image -- anomaly pixel coords
    # are relative to their parent image which may differ from the overview.
    for idx, (finding_num, tag, finding, img, anomaly) in enumerate(anomaly_items):
        px, py = None, None
        if overview_image is not None and anomaly.sky_ra is not None:
            px, py = _sky_to_pixel(
                anomaly.sky_ra, anomaly.sky_dec, overview_image,
            )
        if px is not None and py is not None:
            ax.plot(
                px, py,
                "o", markersize=8, markerfacecolor="none",
                markeredgecolor="yellow", markeredgewidth=1.0, zorder=6,
            )
            ax.text(
                px + 3, py + 3,
                f"A{idx + 1}",
                fontsize=5, color="yellow", fontweight="bold", zorder=7,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02, 0.97, "Overview",
        ha="left", va="top", fontsize=8, fontweight="bold",
        color="white", transform=ax.transAxes,
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2),
        zorder=11,
    )


# Short display names for anomaly types (mosaic panels)
_ANOMALY_DISPLAY_MAP: dict[str, str] = {
    "lens_arc": "Lens arc",
    "lens_ring": "Lens ring",
    "overdensity": "Overdensity",
    "multiscale_object": "Multi-scale",
    "tidal_feature": "Tidal feat.",
    "merger": "Merger",
    "classical_arc": "Classical arc",
    "sersic_residual": "Sersic resid.",
    "comoving_group": "Co-moving grp",
    "stellar_stream": "Stream",
    "runaway_star": "Runaway star",
    "flux_outlier": "Flux outlier",
    "variable_star": "Variable star",
    "periodic_variable": "Periodic var.",
    "blue_straggler": "Blue strag.",
    "red_giant": "Red giant",
}


def create_discovery_mosaic(
    findings: list[PatternResult],
    images: list[FITSImage | None] | None = None,
    max_panels: int = 100,
) -> Figure:
    """Create an anomaly-centric mosaic of discovery findings.

    Shows cutouts centered on anomalies passing quality floors (sorted
    by confidence descending, falling back to score). Panel count is
    dynamic: all quality-passing anomalies up to max_panels (100 for
    system protection). Uses compact 6-column layout when >30 panels.
    Falls back to full-field finding panels when no per-anomaly data
    is available.
    """
    if not findings:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5, 0.5, "No findings to display",
            ha="center", va="center", fontsize=14,
        )
        ax.axis("off")
        return fig

    # Check if any findings have anomalies
    has_anomalies = any(f.anomalies for f in findings)

    if not has_anomalies:
        # Fall back to per-finding panels (legacy behavior)
        return _create_finding_mosaic(findings, images, max_panels)

    # Collect top anomalies across all findings
    anomaly_items = _collect_top_anomalies(
        findings, images, max_anomalies=max_panels - 1,
    )

    if not anomaly_items:
        return _create_finding_mosaic(findings, images, max_panels)

    # Layout: 1 overview panel + N anomaly cutout panels
    # Compact layout (6 cols, smaller panels) when >30 panels
    n_cutouts = len(anomaly_items)
    n_panels = n_cutouts + 1  # +1 for overview
    compact = n_cutouts > 30
    n_cols = min(6 if compact else 4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    panel_size = 3.5 if compact else 5.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    # Panel 0: overview
    _draw_overview_panel(axes[0, 0], findings, images, anomaly_items)

    # Remaining panels: anomaly cutouts
    for idx, (finding_num, tag, finding, img, anomaly) in enumerate(anomaly_items):
        panel_idx = idx + 1  # skip overview
        r = panel_idx // n_cols
        c = panel_idx % n_cols
        ax = axes[r, c]

        _draw_anomaly_cutout(ax, img, anomaly)

        # Thin border
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.5)
            spine.set_zorder(10)

        # Label: anomaly number + type + confidence
        atype = _ANOMALY_DISPLAY_MAP.get(
            anomaly.anomaly_type,
            anomaly.anomaly_type.replace("_", " ").capitalize(),
        )
        conf_str = ""
        if anomaly.confidence is not None:
            conf_str = f"\nconf={anomaly.confidence.confidence:.3f}"
        label_size = 5 if compact else 6
        ax.text(
            0.02, 0.97,
            f"A{idx + 1}: {atype}{conf_str}\n(F{finding_num} [{tag}])",
            ha="left", va="top", fontsize=label_size, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2),
            zorder=11,
        )

        # Coordinates
        if anomaly.sky_ra is not None:
            ax.text(
                0.98, 0.97,
                f"RA={anomaly.sky_ra:.4f}\nDec={anomaly.sky_dec:.4f}",
                ha="right", va="top", fontsize=5,
                color="white", alpha=0.8, transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
                zorder=11,
            )

        # Bottom: score + key props
        score_str = f"score={anomaly.score:.2f}" if anomaly.score else ""
        props_parts = []
        for k in ("n_members", "period", "radius", "n_sources"):
            if k in anomaly.properties:
                props_parts.append(f"{k}={anomaly.properties[k]}")
        bottom = ", ".join(filter(None, [score_str] + props_parts[:2]))
        if bottom:
            ax.text(
                0.5, 0.02, bottom,
                ha="center", va="bottom", fontsize=5.5,
                color="white", transform=ax.transAxes,
                bbox=dict(
                    facecolor="black", alpha=0.7,
                    edgecolor="none", pad=2,
                    boxstyle="round,pad=0.3",
                ),
                zorder=11,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused cells
    for idx in range(n_panels, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    total_anomalies = sum(len(f.anomalies) for f in findings)
    fig.suptitle(
        f"Discovery Mosaic -- {n_cutouts} top anomalies "
        f"(of {total_anomalies} total across {len(findings)} findings)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.97, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.04, wspace=0.04)
    return fig


def _create_finding_mosaic(
    findings: list[PatternResult],
    images: list[FITSImage | None] | None = None,
    max_panels: int = 25,
) -> Figure:
    """Legacy per-finding mosaic (used when no anomaly data is available).

    Panels are ordered by finding number (new first, then known, then
    low-confidence). Each panel has a finding-number label top-left
    ("F1 [NEW]") and coordinates top-right. Detection marker overlays
    are preserved.
    """
    # Build finding -> image mapping before reordering
    padded_images = images if images else [None] * len(findings)
    image_map: dict[int, FITSImage | None] = {}
    for f, img in zip(findings, padded_images):
        if img is None:
            img = _load_fits_from_metadata(f)
        image_map[id(f)] = img

    # Assign finding numbers
    numbered = _assign_finding_numbers(findings)
    numbered = numbered[:max_panels]

    n_panels = len(numbered)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    panel_size = 6.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    for idx, (finding_num, tag, finding) in enumerate(numbered):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        # Plot image
        img = image_map.get(id(finding))
        if img is not None:
            try:
                norm = img.normalize("zscale")
                ax.imshow(norm.data, origin="lower", cmap="gray_r")
            except Exception:
                ax.set_facecolor("#1a1a1a")
        else:
            ax.set_facecolor("#1a1a1a")
            ax.text(
                0.5, 0.5,
                f"RA={finding.region_ra:.2f}\nDec={finding.region_dec:.2f}",
                ha="center", va="center", fontsize=8, color="white",
                transform=ax.transAxes,
            )

        # Draw detection overlays
        _draw_panel_overlays(ax, finding)

        # Thin dark gray border (0.5px)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(0.5)
            spine.set_zorder(10)

        # Build classification and key info
        classification = finding.metadata.get("local_classification", {})
        evaluation = finding.metadata.get("local_evaluation", {})
        cls_type = classification.get("classification", finding.detection_type)
        display_map = {
            "gravitational_lens": "Grav. lens",
            "morphological_anomaly": "Morph. anomaly",
            "galaxy_interaction": "Galaxy interaction",
            "kinematic_group": "Kinematic group",
            "galaxy_structure": "Galaxy structure",
            "multiscale_source": "Multi-scale source",
            "stellar_population_anomaly": "Stellar pop.",
            "spatial_clustering": "Spatial cluster",
            "transient_candidate": "Transient",
            "statistical_outlier": "Stat. outlier",
            "classical_pattern": "Classical pattern",
            "variable_star": "Variable star",
        }
        display_name = display_map.get(cls_type, cls_type.replace("_", " ").capitalize())
        sig_rating = evaluation.get("significance_rating", 0)
        verdict = evaluation.get("verdict", "?")

        # Top-left: finding number + classification
        top_left = f"F{finding_num} [{tag}]\n{display_name}"
        ax.text(
            0.02, 0.97, top_left,
            ha="left", va="top", fontsize=7, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2),
            zorder=11,
        )

        # Top-right: coordinates
        ax.text(
            0.98, 0.97,
            f"RA={finding.region_ra:.2f}\nDec={finding.region_dec:.2f}",
            ha="right", va="top", fontsize=5,
            color="white", alpha=0.8, transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            zorder=11,
        )

        # Bottom: key features and cross-match info
        bottom_parts = []
        bottom_parts.append(f"{verdict} ({sig_rating}/10)")
        sub_counts = _extract_sub_detections(finding.details)
        anomalous = {k: v for k, v in sub_counts.items() if k != "sources"}
        if anomalous:
            top_feats = sorted(anomalous.items(), key=lambda x: x[1], reverse=True)[:3]
            bottom_parts.append(", ".join(f"{v} {k}" for k, v in top_feats))
        if finding.cross_matches:
            n_matches = len(finding.cross_matches)
            first_name = finding.cross_matches[0].get("name", "")
            if n_matches == 1:
                bottom_parts.append(f"Match: {first_name}")
            else:
                bottom_parts.append(f"{n_matches} matches incl. {first_name}")

        ax.text(
            0.5, 0.02, "\n".join(bottom_parts),
            ha="center", va="bottom", fontsize=5.5,
            color="white", transform=ax.transAxes,
            bbox=dict(
                facecolor="black", alpha=0.7,
                edgecolor="none", pad=2,
                boxstyle="round,pad=0.3",
            ),
            zorder=11,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused cells
    for idx in range(n_panels, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    first_num = numbered[0][0]
    last_num = numbered[-1][0]
    fig.suptitle(
        f"Discovery Mosaic -- {n_panels} findings (F{first_num}-F{last_num})",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.97, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.04, wspace=0.04)
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
