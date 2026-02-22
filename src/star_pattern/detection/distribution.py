"""Stellar distribution analysis: Voronoi, persistence homology, 2-pt correlation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy import stats

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.distribution")


class DistributionAnalyzer:
    """Analyze spatial distribution of sources for structure detection."""

    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap

    def analyze(self, positions: np.ndarray, boundary: tuple[float, float] | None = None) -> dict[str, Any]:
        """Full distribution analysis on source positions.

        Args:
            positions: Nx2 array of (x, y) positions.
            boundary: Optional (width, height) of the field.

        Returns:
            Dict with Voronoi stats, 2-pt correlation, overdensity flags.
        """
        if len(positions) < 4:
            return {"n_sources": len(positions), "error": "Too few sources"}

        results: dict[str, Any] = {"n_sources": len(positions)}

        # Voronoi tessellation
        voronoi_stats = self._voronoi_analysis(positions, boundary)
        results.update(voronoi_stats)

        # Two-point correlation
        tpcf = self._two_point_correlation(positions, boundary)
        results.update(tpcf)

        # Nearest-neighbor distribution
        nn_stats = self._nearest_neighbor(positions)
        results.update(nn_stats)

        # Overdensity detection
        overdensities = self._detect_overdensities(positions, boundary)
        results["overdensities"] = overdensities

        # Overall distribution score (higher = more structured/anomalous)
        results["distribution_score"] = self._compute_score(results)

        return results

    def _voronoi_analysis(
        self, positions: np.ndarray, boundary: tuple[float, float] | None = None
    ) -> dict[str, Any]:
        """Analyze Voronoi cell area distribution."""
        try:
            vor = Voronoi(positions)
        except Exception as e:
            logger.debug(f"Voronoi failed: {e}")
            return {"voronoi_error": str(e)}

        # Compute finite cell areas
        areas = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            vertices = vor.vertices[region]
            # Shoelace formula
            n = len(vertices)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += vertices[i, 0] * vertices[j, 1]
                area -= vertices[j, 0] * vertices[i, 1]
            areas.append(abs(area) / 2)

        if not areas:
            return {"voronoi_n_cells": 0}

        areas = np.array(areas)
        median_area = float(np.median(areas))
        mean_area = float(np.mean(areas))

        # Coefficient of variation (higher = more clustered)
        cv = float(np.std(areas) / mean_area) if mean_area > 0 else 0

        # Compare to Poisson expectation (CV ~= 0.53 for random)
        poisson_cv = 0.53
        clustering_excess = max(cv - poisson_cv, 0)

        return {
            "voronoi_n_cells": len(areas),
            "voronoi_median_area": median_area,
            "voronoi_mean_area": mean_area,
            "voronoi_cv": cv,
            "voronoi_clustering_excess": clustering_excess,
        }

    def _two_point_correlation(
        self, positions: np.ndarray, boundary: tuple[float, float] | None = None
    ) -> dict[str, Any]:
        """Estimate 2-point correlation using the Landy-Szalay estimator.

        Uses the Landy-Szalay estimator: xi(r) = (DD - 2*DR + RR) / RR,
        which provides boundary-corrected correlation by comparing data-data,
        data-random, and random-random pair counts.
        """
        n = len(positions)
        if n < 10:
            return {"tpcf_error": "Too few points"}

        from scipy.spatial.distance import pdist, cdist

        # Data-Data pair distances
        dd_dists = pdist(positions)

        # Determine field boundary
        if boundary:
            x_min, y_min = 0.0, 0.0
            x_max, y_max = boundary[0], boundary[1]
        else:
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            x_min, y_min = mins[0], mins[1]
            x_max, y_max = maxs[0], maxs[1]

        # Generate random catalog (2x the data count for better statistics)
        rng = np.random.default_rng(42)
        n_random = n * 2
        random_positions = np.column_stack([
            rng.uniform(x_min, x_max, n_random),
            rng.uniform(y_min, y_max, n_random),
        ])

        # Random-Random pair distances
        rr_dists = pdist(random_positions)

        # Data-Random cross-distances
        dr_dists = cdist(positions, random_positions).ravel()

        # Bin distances
        max_dist = np.percentile(dd_dists, 95)
        bins = np.linspace(0, max_dist, 20)
        dd_hist, _ = np.histogram(dd_dists, bins=bins)
        rr_hist, _ = np.histogram(rr_dists, bins=bins)
        dr_hist, _ = np.histogram(dr_dists, bins=bins)

        # Normalize pair counts by total pairs in each category
        n_dd_total = n * (n - 1) / 2
        n_rr_total = n_random * (n_random - 1) / 2
        n_dr_total = n * n_random

        with np.errstate(divide="ignore", invalid="ignore"):
            dd_norm = dd_hist / max(n_dd_total, 1)
            rr_norm = rr_hist / max(n_rr_total, 1)
            dr_norm = dr_hist / max(n_dr_total, 1)

            # Landy-Szalay: xi(r) = (DD - 2*DR + RR) / RR
            correlation = np.where(
                rr_norm > 0,
                (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm,
                0.0,
            )

        # Summary: excess clustering at small scales
        small_scale = correlation[: len(correlation) // 3]
        clustering_amplitude = float(np.mean(small_scale)) if len(small_scale) > 0 else 0

        return {
            "tpcf_bins": bins.tolist(),
            "tpcf_correlation": correlation.tolist(),
            "tpcf_clustering_amplitude": clustering_amplitude,
        }

    def _nearest_neighbor(self, positions: np.ndarray) -> dict[str, Any]:
        """Nearest-neighbor distance statistics."""
        from scipy.spatial import cKDTree

        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)  # k=2: self + nearest
        nn_dists = dists[:, 1]  # Skip self-distance

        # Clark-Evans statistic: ratio of mean NN distance to expected random
        mean_nn = float(np.mean(nn_dists))
        n = len(positions)
        area = (positions.max(0) - positions.min(0)).prod()
        density = n / max(area, 1)
        expected_nn = 0.5 / np.sqrt(max(density, 1e-10))

        clark_evans = mean_nn / max(expected_nn, 1e-10)
        # < 1: clustered, = 1: random, > 1: uniform/regular

        return {
            "nn_mean_distance": mean_nn,
            "nn_std_distance": float(np.std(nn_dists)),
            "clark_evans_r": clark_evans,
            "nn_distribution": "clustered" if clark_evans < 0.8 else "random" if clark_evans < 1.2 else "uniform",
        }

    def _detect_overdensities(
        self, positions: np.ndarray, boundary: tuple[float, float] | None = None
    ) -> list[dict[str, Any]]:
        """Detect local overdensities using kernel density estimation."""
        from scipy.stats import gaussian_kde

        if len(positions) < 10:
            return []

        try:
            kde = gaussian_kde(positions.T)
        except Exception:
            return []

        # Evaluate on grid
        mins = positions.min(0)
        maxs = positions.max(0)
        x_grid = np.linspace(mins[0], maxs[0], 50)
        y_grid = np.linspace(mins[1], maxs[1], 50)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

        density = kde(grid_points).reshape(xx.shape)
        mean_density = np.mean(density)
        std_density = np.std(density)

        # Find significant overdensities (>3 sigma)
        threshold = mean_density + 3 * std_density
        overdense = density > threshold

        overdensities = []
        if overdense.any():
            from scipy.ndimage import label

            labeled, n_features = label(overdense)
            for i in range(1, n_features + 1):
                mask = labeled == i
                peak_idx = np.unravel_index(
                    np.argmax(density * mask), density.shape
                )
                overdensities.append(
                    {
                        "x": float(x_grid[peak_idx[1]]),
                        "y": float(y_grid[peak_idx[0]]),
                        "peak_density": float(density[peak_idx]),
                        "sigma": float((density[peak_idx] - mean_density) / max(std_density, 1e-10)),
                        "n_pixels": int(mask.sum()),
                    }
                )

        overdensities.sort(key=lambda o: o["sigma"], reverse=True)
        return overdensities[:10]

    def _compute_score(self, results: dict[str, Any]) -> float:
        """Compute overall distribution anomaly score [0, 1]."""
        score = 0.0

        # Voronoi clustering excess
        cv_excess = results.get("voronoi_clustering_excess", 0)
        score += min(cv_excess / 0.5, 0.3)

        # 2-pt correlation
        clustering = results.get("tpcf_clustering_amplitude", 0)
        score += min(abs(clustering) / 2, 0.3)

        # Clark-Evans departure from random
        ce = results.get("clark_evans_r", 1.0)
        score += min(abs(ce - 1.0) / 0.5, 0.2)

        # Overdensity significance
        overdensities = results.get("overdensities", [])
        if overdensities:
            max_sigma = overdensities[0].get("sigma", 0)
            score += min(max_sigma / 10, 0.2)

        return float(np.clip(score, 0, 1))
