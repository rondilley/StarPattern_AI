"""Cross-detector feature fusion: extract rich feature vectors from ensemble results.

Extracts ~55-70 dimensional feature vectors from the full detection result dict,
capturing detailed per-detector outputs that the scalar ensemble score discards.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("detection.feature_fusion")


# Feature schema: (section_key, field_path, default_value)
# Each entry defines one feature dimension.
_FEATURE_SCHEMA: list[tuple[str, str, float]] = [
    # Sources (5)
    ("sources", "n_sources", 0),
    ("sources", "mean_flux", 0),
    ("sources", "flux_std", 0),
    ("sources", "spatial_concentration", 0),
    ("sources", "ellipticity_mean", 0),
    # Classical (6)
    ("classical", "gabor_score", 0),
    ("classical", "fft_score", 0),
    ("classical", "arc_score", 0),
    ("classical", "n_arcs", 0),
    ("classical", "dominant_frequency", 0),
    ("classical", "dominant_orientation", 0),
    # Morphology (6)
    ("morphology", "concentration", 0),
    ("morphology", "asymmetry", 0),
    ("morphology", "smoothness", 0),
    ("morphology", "gini", 0),
    ("morphology", "m20", 0),
    ("morphology", "morphology_score", 0),
    # Lens (5)
    ("lens", "lens_score", 0),
    ("lens", "n_arcs", 0),
    ("lens", "n_rings", 0),
    ("lens", "arc_coverage", 0),
    ("lens", "is_candidate", 0),
    # Distribution (4)
    ("distribution", "distribution_score", 0),
    ("distribution", "voronoi_cv", 0),
    ("distribution", "clark_evans_r", 1.0),
    ("distribution", "n_overdensities", 0),
    # Galaxy (4)
    ("galaxy", "galaxy_score", 0),
    ("galaxy", "n_tidal", 0),
    ("galaxy", "n_mergers", 0),
    ("galaxy", "n_color_outliers", 0),
    # Kinematic (4)
    ("kinematic", "kinematic_score", 0),
    ("kinematic", "n_comoving_groups", 0),
    ("kinematic", "n_streams", 0),
    ("kinematic", "n_runaways", 0),
    # Transient (4)
    ("transient", "transient_score", 0),
    ("transient", "n_astrometric", 0),
    ("transient", "n_photometric", 0),
    ("transient", "n_parallax", 0),
    # Sersic (5)
    ("sersic", "sersic_score", 0),
    ("sersic", "sersic_n", 0),
    ("sersic", "r_e", 0),
    ("sersic", "ellipticity", 0),
    ("sersic", "n_residual_features", 0),
    # Wavelet (4)
    ("wavelet", "wavelet_score", 0),
    ("wavelet", "n_detections", 0),
    ("wavelet", "n_multiscale", 0),
    ("wavelet", "mean_scale", 0),
    # Population (4)
    ("population", "population_score", 0),
    ("population", "n_blue_stragglers", 0),
    ("population", "n_red_giants", 0),
    ("population", "multiple_populations", 0),
    # Variability (4)
    ("variability", "variability_score", 0),
    ("variability", "n_variables", 0),
    ("variability", "n_periodic", 0),
    ("variability", "n_transients", 0),
    # Temporal (6)
    ("temporal", "temporal_score", 0),
    ("temporal", "n_new_sources", 0),
    ("temporal", "n_disappeared", 0),
    ("temporal", "n_brightenings", 0),
    ("temporal", "n_moving", 0),
    ("temporal", "baseline_days", 0),
    # Anomaly detector (1)
    ("anomaly", "anomaly_score", 0),
    # Ensemble-level (1)
    ("_top", "anomaly_score", 0),
    # Embedding anomaly (injected by RepresentationManager, Phase 3)
    ("_top", "embedding_anomaly_score", 0.5),
    # Composed pipeline score (injected by ComposedPipeline, Phase 4)
    ("_top", "composed_score", 0),
]


def _compute_source_derived(sources: dict[str, Any]) -> dict[str, float]:
    """Compute derived source features from raw source extraction data."""
    derived: dict[str, float] = {}

    fluxes = sources.get("fluxes")
    if fluxes is not None and len(fluxes) > 0:
        fluxes_arr = np.asarray(fluxes, dtype=np.float64)
        fluxes_arr = fluxes_arr[np.isfinite(fluxes_arr)]
        if len(fluxes_arr) > 0:
            derived["mean_flux"] = float(np.mean(fluxes_arr))
            derived["flux_std"] = float(np.std(fluxes_arr))

    ellipticity = sources.get("ellipticity")
    if ellipticity is not None and len(ellipticity) > 0:
        ell_arr = np.asarray(ellipticity, dtype=np.float64)
        ell_arr = ell_arr[np.isfinite(ell_arr)]
        if len(ell_arr) > 0:
            derived["ellipticity_mean"] = float(np.mean(ell_arr))

    positions = sources.get("positions")
    if positions is not None and len(positions) > 0:
        pos_arr = np.asarray(positions, dtype=np.float64)
        if pos_arr.ndim == 2 and pos_arr.shape[0] >= 3:
            centroid = np.mean(pos_arr, axis=0)
            dists = np.linalg.norm(pos_arr - centroid, axis=1)
            mean_dist = np.mean(dists)
            if mean_dist > 0:
                derived["spatial_concentration"] = float(
                    np.std(dists) / mean_dist
                )

    return derived


class FeatureFusionExtractor:
    """Extract rich feature vectors from ensemble detection results.

    Produces a fixed-length feature vector (~60D) from the full detection
    result dict, capturing per-detector outputs that the scalar ensemble
    score would otherwise discard.
    """

    def __init__(self) -> None:
        self._schema = list(_FEATURE_SCHEMA)
        self._feature_names: list[str] = []
        for section, field, _ in self._schema:
            if section == "_top":
                self._feature_names.append(field)
            else:
                self._feature_names.append(f"{section}.{field}")

    @property
    def n_features(self) -> int:
        return len(self._schema)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def extract(self, detection: dict[str, Any]) -> np.ndarray:
        """Extract feature vector from a single detection result dict.

        Args:
            detection: Result dict from EnsembleDetector.detect().

        Returns:
            1D float64 array of shape (n_features,).
        """
        # Pre-compute derived source features
        sources = detection.get("sources", {})
        if isinstance(sources, dict) and "error" not in sources:
            source_derived = _compute_source_derived(sources)
        else:
            source_derived = {}

        features = np.zeros(self.n_features, dtype=np.float64)

        for i, (section, field, default) in enumerate(self._schema):
            if section == "_top":
                val = detection.get(field, default)
            elif section == "sources" and field in source_derived:
                val = source_derived[field]
            else:
                section_data = detection.get(section, {})
                if isinstance(section_data, dict) and "error" not in section_data:
                    val = section_data.get(field, default)
                else:
                    val = default

            # Convert booleans
            if isinstance(val, bool):
                val = float(val)

            try:
                features[i] = float(val)
            except (TypeError, ValueError):
                features[i] = float(default)

        # Replace any NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def extract_batch(self, detections: list[dict[str, Any]]) -> np.ndarray:
        """Extract feature vectors from a batch of detection results.

        Args:
            detections: List of result dicts from EnsembleDetector.detect().

        Returns:
            2D float64 array of shape (N, n_features).
        """
        if not detections:
            return np.empty((0, self.n_features), dtype=np.float64)

        return np.stack([self.extract(d) for d in detections])
