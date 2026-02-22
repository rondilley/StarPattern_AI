"""Tests for cross-detector feature fusion."""

import numpy as np
import pytest

from star_pattern.detection.feature_fusion import FeatureFusionExtractor


def _make_detection_dict(**overrides):
    """Build a realistic detection dict with all sections."""
    detection = {
        "shape": [256, 256],
        "pixel_scale_arcsec": 0.4,
        "anomaly_score": 0.42,
        "n_detections": 3,
        "sources": {
            "n_sources": 25,
            "background_rms": 0.01,
            "positions": [[50, 60], [100, 120], [200, 180]],
            "fluxes": [100.0, 250.0, 75.0],
            "ellipticity": [0.1, 0.3, 0.2],
        },
        "classical": {
            "gabor_score": 0.35,
            "fft_score": 0.2,
            "arc_score": 0.1,
            "n_arcs": 1,
        },
        "morphology": {
            "concentration": 0.6,
            "asymmetry": 0.15,
            "smoothness": 0.08,
            "gini": 0.55,
            "m20": -1.8,
            "morphology_score": 0.4,
        },
        "lens": {
            "lens_score": 0.05,
            "n_arcs": 0,
            "n_rings": 0,
            "is_candidate": False,
        },
        "distribution": {
            "distribution_score": 0.3,
            "voronoi_cv": 0.8,
            "clark_evans_r": 0.9,
            "n_overdensities": 2,
        },
        "galaxy": {
            "galaxy_score": 0.2,
            "n_tidal": 0,
            "n_mergers": 1,
            "n_color_outliers": 3,
        },
        "kinematic": {
            "kinematic_score": 0.15,
            "n_comoving_groups": 1,
            "n_streams": 0,
            "n_runaways": 2,
        },
        "transient": {
            "transient_score": 0.1,
            "n_astrometric": 1,
            "n_photometric": 0,
            "n_parallax": 0,
        },
        "sersic": {
            "sersic_score": 0.5,
            "sersic_n": 2.5,
            "r_e": 15.0,
            "ellipticity": 0.2,
            "n_residual_features": 3,
        },
        "wavelet": {
            "wavelet_score": 0.4,
            "n_detections": 5,
            "n_multiscale": 2,
            "mean_scale": 3.0,
        },
        "population": {
            "population_score": 0.25,
            "n_blue_stragglers": 2,
            "n_red_giants": 8,
            "multiple_populations": True,
        },
        "variability": {
            "variability_score": 0.3,
            "n_variables": 4,
            "n_periodic": 1,
            "n_transients": 0,
        },
        "anomaly": {
            "anomaly_score": 0.35,
            "n_anomalies": 2,
        },
    }
    detection.update(overrides)
    return detection


class TestFeatureFusionExtractor:
    """Tests for FeatureFusionExtractor."""

    def test_extract_correct_dimensionality(self):
        """Feature vector has expected number of dimensions."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        features = extractor.extract(detection)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert features.shape[0] == extractor.n_features
        assert features.dtype == np.float64

    def test_feature_names_match_count(self):
        """Feature names list matches n_features."""
        extractor = FeatureFusionExtractor()
        assert len(extractor.feature_names) == extractor.n_features

    def test_extract_missing_sections_defaults_to_zero(self):
        """Missing detector sections produce zero features (not errors)."""
        extractor = FeatureFusionExtractor()
        # Minimal detection dict with only anomaly_score
        detection = {"anomaly_score": 0.5}
        features = extractor.extract(detection)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == extractor.n_features
        assert np.all(np.isfinite(features))

    def test_extract_error_sections_defaults_to_zero(self):
        """Sections with 'error' key produce defaults."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        detection["morphology"] = {"error": "some failure"}
        detection["sersic"] = {"error": "sersic failed"}

        features = extractor.extract(detection)
        assert np.all(np.isfinite(features))

    def test_batch_extraction(self):
        """Batch extraction produces correct shape."""
        extractor = FeatureFusionExtractor()
        detections = [_make_detection_dict() for _ in range(5)]
        batch = extractor.extract_batch(detections)

        assert batch.shape == (5, extractor.n_features)
        assert batch.dtype == np.float64

    def test_batch_empty(self):
        """Empty batch returns empty array with correct feature count."""
        extractor = FeatureFusionExtractor()
        batch = extractor.extract_batch([])
        assert batch.shape == (0, extractor.n_features)

    def test_no_nan_or_inf(self):
        """Output never contains NaN or Inf."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        # Inject NaN and Inf into a section
        detection["classical"]["gabor_score"] = float("nan")
        detection["morphology"]["concentration"] = float("inf")

        features = extractor.extract(detection)
        assert np.all(np.isfinite(features))

    def test_source_derived_features(self):
        """Derived features (mean_flux, ellipticity_mean, etc.) are extracted."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        features = extractor.extract(detection)

        # Find the mean_flux feature index
        names = extractor.feature_names
        flux_idx = names.index("sources.mean_flux")
        assert features[flux_idx] > 0  # We set fluxes=[100, 250, 75]

    def test_boolean_conversion(self):
        """Boolean values (is_candidate, multiple_populations) convert to float."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        features = extractor.extract(detection)

        names = extractor.feature_names
        mp_idx = names.index("population.multiple_populations")
        assert features[mp_idx] == 1.0  # True -> 1.0

    def test_embedding_anomaly_score_feature(self):
        """Embedding anomaly score from Phase 3 is extracted when present."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        detection["embedding_anomaly_score"] = 0.85

        features = extractor.extract(detection)
        names = extractor.feature_names
        idx = names.index("embedding_anomaly_score")
        assert abs(features[idx] - 0.85) < 1e-6

    def test_composed_score_feature(self):
        """Composed score from Phase 4 is extracted when present."""
        extractor = FeatureFusionExtractor()
        detection = _make_detection_dict()
        detection["composed_score"] = 0.7

        features = extractor.extract(detection)
        names = extractor.feature_names
        idx = names.index("composed_score")
        assert abs(features[idx] - 0.7) < 1e-6
