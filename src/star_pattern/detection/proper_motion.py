"""Proper motion / kinematic analysis of stellar catalog data."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from star_pattern.core.catalog import StarCatalog
from star_pattern.core.config import DetectionConfig
from star_pattern.utils.logging import get_logger

logger = get_logger("detection.proper_motion")


class ProperMotionAnalyzer:
    """Analyze Gaia proper motion data for kinematic structures.

    Detects:
    - Co-moving groups (DBSCAN in pmra/pmdec space)
    - Stellar streams (linear structures in position+PM 4D space)
    - Runaway stars (high-PM outliers)
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.pm_min = self.config.kinematic_pm_min
        self.cluster_eps = self.config.kinematic_cluster_eps
        self.cluster_min = self.config.kinematic_cluster_min
        self.stream_min_length = self.config.kinematic_stream_min_length

    def analyze(self, catalog: StarCatalog) -> dict[str, Any]:
        """Run kinematic analysis on catalog entries with proper motion data.

        Args:
            catalog: StarCatalog with entries containing pmra/pmdec in properties.

        Returns:
            Dict with kinematic_score and detected structures.
        """
        # Extract entries with proper motion data
        pm_entries = []
        for entry in catalog.entries:
            pmra = entry.properties.get("pmra")
            pmdec = entry.properties.get("pmdec")
            if pmra is not None and pmdec is not None:
                try:
                    pmra_f = float(pmra)
                    pmdec_f = float(pmdec)
                    if np.isfinite(pmra_f) and np.isfinite(pmdec_f):
                        pm_entries.append(entry)
                except (ValueError, TypeError):
                    continue

        results: dict[str, Any] = {
            "n_pm_sources": len(pm_entries),
        }

        if len(pm_entries) < 5:
            results["kinematic_score"] = 0.0
            results["comoving_groups"] = []
            results["stream_candidates"] = []
            results["runaway_stars"] = []
            logger.info("Too few proper motion sources for kinematic analysis")
            return results

        # Build arrays
        positions = np.array([[e.ra, e.dec] for e in pm_entries])
        pm_data = np.array([
            [float(e.properties["pmra"]), float(e.properties["pmdec"])]
            for e in pm_entries
        ])

        try:
            comoving = self._find_comoving_groups(pm_data, pm_entries)
            results["comoving_groups"] = comoving
        except Exception as e:
            logger.warning(f"Co-moving group detection failed: {e}")
            results["comoving_groups"] = []

        try:
            streams = self._detect_streams(positions, pm_data, pm_entries)
            results["stream_candidates"] = streams
        except Exception as e:
            logger.warning(f"Stream detection failed: {e}")
            results["stream_candidates"] = []

        try:
            runaways = self._find_runaways(pm_data, pm_entries)
            results["runaway_stars"] = runaways
        except Exception as e:
            logger.warning(f"Runaway detection failed: {e}")
            results["runaway_stars"] = []

        # Composite score
        n_groups = len(results["comoving_groups"])
        n_streams = len(results["stream_candidates"])
        n_runaways = len(results["runaway_stars"])

        kinematic_score = float(np.clip(
            0.4 * min(n_groups, 3) / 3.0
            + 0.35 * min(n_streams, 2) / 2.0
            + 0.25 * min(n_runaways, 5) / 5.0,
            0.0, 1.0,
        ))

        results["kinematic_score"] = kinematic_score
        results["n_detections"] = n_groups + n_streams + n_runaways

        logger.info(
            f"Kinematic analysis: score={kinematic_score:.3f}, "
            f"groups={n_groups}, streams={n_streams}, runaways={n_runaways}"
        )
        return results

    def _find_comoving_groups(
        self,
        pm_data: np.ndarray,
        entries: list[Any],
    ) -> list[dict[str, Any]]:
        """Find co-moving groups via DBSCAN clustering in proper motion space.

        Stars sharing similar proper motions may belong to the same
        open cluster, moving group, or dissolving association.
        """
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min)
        labels = clustering.fit_predict(pm_data)

        groups = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # noise

        for label in sorted(unique_labels):
            mask = labels == label
            member_count = int(np.sum(mask))
            group_pm = pm_data[mask]
            member_entries = [entries[i] for i in range(len(entries)) if mask[i]]

            mean_pmra = float(np.mean(group_pm[:, 0]))
            mean_pmdec = float(np.mean(group_pm[:, 1]))
            std_pmra = float(np.std(group_pm[:, 0]))
            std_pmdec = float(np.std(group_pm[:, 1]))

            # Mean position
            mean_ra = float(np.mean([e.ra for e in member_entries]))
            mean_dec = float(np.mean([e.dec for e in member_entries]))

            groups.append({
                "type": "comoving_group",
                "n_members": member_count,
                "mean_pmra": mean_pmra,
                "mean_pmdec": mean_pmdec,
                "std_pmra": std_pmra,
                "std_pmdec": std_pmdec,
                "mean_ra": mean_ra,
                "mean_dec": mean_dec,
                "pm_total": float(np.sqrt(mean_pmra**2 + mean_pmdec**2)),
                "member_ids": [e.source_id for e in member_entries[:20]],
            })

        return groups

    def _detect_streams(
        self,
        positions: np.ndarray,
        pm_data: np.ndarray,
        entries: list[Any],
    ) -> list[dict[str, Any]]:
        """Detect stellar streams as linear structures in 4D (ra, dec, pmra, pmdec).

        Uses RANSAC to find linear alignments: stream members share
        a common motion direction and lie along a spatial line.
        """
        if len(positions) < self.stream_min_length:
            return []

        streams = []

        # Normalize features to comparable scales
        pos_std = np.std(positions, axis=0)
        pm_std = np.std(pm_data, axis=0)

        pos_std = np.where(pos_std > 1e-10, pos_std, 1.0)
        pm_std = np.where(pm_std > 1e-10, pm_std, 1.0)

        pos_norm = positions / pos_std
        pm_norm = pm_data / pm_std

        # Combine into 4D feature space
        features_4d = np.hstack([pos_norm, pm_norm])

        # Simple RANSAC-like approach: sample pairs, find inliers
        rng = np.random.default_rng(42)
        n_points = len(features_4d)
        best_inliers = None
        best_count = 0
        n_iterations = min(100, n_points * (n_points - 1) // 2)

        residual_threshold = 0.5  # in normalized units

        for _ in range(n_iterations):
            idx = rng.choice(n_points, size=2, replace=False)
            p1, p2 = features_4d[idx[0]], features_4d[idx[1]]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            direction = direction / length

            # Project all points onto the line
            diff = features_4d - p1
            projections = diff @ direction
            residuals = diff - np.outer(projections, direction)
            distances = np.linalg.norm(residuals, axis=1)

            inliers = distances < residual_threshold
            count = int(np.sum(inliers))

            if count >= self.stream_min_length and count > best_count:
                best_count = count
                best_inliers = inliers.copy()

        if best_inliers is not None and best_count >= self.stream_min_length:
            member_entries = [entries[i] for i in range(len(entries)) if best_inliers[i]]
            member_pm = pm_data[best_inliers]
            member_pos = positions[best_inliers]

            streams.append({
                "type": "stream",
                "n_members": best_count,
                "mean_ra": float(np.mean(member_pos[:, 0])),
                "mean_dec": float(np.mean(member_pos[:, 1])),
                "mean_pmra": float(np.mean(member_pm[:, 0])),
                "mean_pmdec": float(np.mean(member_pm[:, 1])),
                "extent_deg": float(
                    np.sqrt(np.ptp(member_pos[:, 0])**2 + np.ptp(member_pos[:, 1])**2)
                ),
                "member_ids": [e.source_id for e in member_entries[:20]],
            })

        return streams

    def _find_runaways(
        self,
        pm_data: np.ndarray,
        entries: list[Any],
    ) -> list[dict[str, Any]]:
        """Flag stars with proper motion far exceeding the local median.

        Runaway stars or hypervelocity star candidates have proper motions
        >3 sigma from the field population.
        """
        runaways = []

        pm_total = np.sqrt(pm_data[:, 0]**2 + pm_data[:, 1]**2)
        median_pm = np.median(pm_total)
        mad = np.median(np.abs(pm_total - median_pm))
        # Robust sigma estimate
        sigma = mad * 1.4826 if mad > 0 else np.std(pm_total)

        if sigma < 1e-10:
            return runaways

        for i, entry in enumerate(entries):
            pm = pm_total[i]
            deviation = (pm - median_pm) / sigma

            if deviation > 3.0 and pm > self.pm_min:
                parallax = entry.properties.get("parallax")
                runaways.append({
                    "type": "runaway",
                    "source_id": entry.source_id,
                    "ra": entry.ra,
                    "dec": entry.dec,
                    "pmra": float(pm_data[i, 0]),
                    "pmdec": float(pm_data[i, 1]),
                    "pm_total": float(pm),
                    "deviation_sigma": float(deviation),
                    "parallax": float(parallax) if parallax is not None else None,
                })

        return runaways
