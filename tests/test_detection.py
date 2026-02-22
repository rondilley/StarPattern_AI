"""Tests for detection modules."""

import numpy as np
import pytest

from star_pattern.core.fits_handler import FITSImage
from star_pattern.core.config import DetectionConfig
from star_pattern.detection.classical import GaborFilterBank, FFTAnalyzer, ClassicalDetector
from star_pattern.detection.source_extraction import SourceExtractor
from star_pattern.detection.morphology import MorphologyAnalyzer
from star_pattern.detection.anomaly import AnomalyDetector
from star_pattern.detection.lens_detector import LensDetector
from star_pattern.detection.distribution import DistributionAnalyzer
from star_pattern.detection.ensemble import EnsembleDetector


class TestGaborFilterBank:
    def test_build_kernels(self):
        bank = GaborFilterBank(frequencies=[0.1, 0.2], n_orientations=4)
        assert len(bank._kernels) == 8  # 2 freqs * 4 orientations

    def test_apply(self, synthetic_image: FITSImage):
        bank = GaborFilterBank(frequencies=[0.1], n_orientations=4)
        result = bank.apply(synthetic_image.data)
        assert "max_response" in result
        assert result["max_response"].shape == synthetic_image.shape
        assert "dominant_orientation" in result


class TestFFTAnalyzer:
    def test_power_spectrum(self, synthetic_image: FITSImage):
        fft = FFTAnalyzer()
        ps = fft.power_spectrum(synthetic_image.data)
        assert ps.shape == synthetic_image.shape
        assert (ps >= 0).all()

    def test_analyze(self, synthetic_image: FITSImage):
        fft = FFTAnalyzer()
        result = fft.analyze(synthetic_image.data)
        assert "dominant_frequency" in result
        assert "total_power" in result


class TestSourceExtractor:
    def test_extract(self, synthetic_image: FITSImage):
        ext = SourceExtractor(threshold=3.0)
        result = ext.extract(synthetic_image.data)
        assert result["n_sources"] > 0
        assert len(result["positions"]) == result["n_sources"]

    def test_source_density(self, synthetic_image: FITSImage):
        ext = SourceExtractor()
        density = ext.source_density(synthetic_image.data, grid_size=4)
        assert density.shape == (4, 4)


class TestMorphologyAnalyzer:
    def test_analyze(self, synthetic_image: FITSImage):
        morph = MorphologyAnalyzer()
        result = morph.analyze(synthetic_image.data)
        assert "concentration" in result
        assert "asymmetry" in result
        assert "gini" in result
        assert "morphology_score" in result
        assert 0 <= result["morphology_score"] <= 1


class TestAnomalyDetector:
    def test_detect(self):
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (100, 10))
        # Add anomalies
        features[0] = rng.normal(5, 0.5, 10)
        features[1] = rng.normal(-5, 0.5, 10)

        detector = AnomalyDetector(contamination=0.05)
        result = detector.detect(features)
        assert result["n_anomalies"] > 0
        # Injected anomalies should score high
        assert result["scores"][0] > 0.5 or result["scores"][1] > 0.5


class TestLensDetector:
    def test_detect_on_arc(self, synthetic_image_with_arc: FITSImage):
        detector = LensDetector(snr_threshold=2.0)
        result = detector.detect(synthetic_image_with_arc.data)
        assert "lens_score" in result
        assert "arcs" in result
        assert "rings" in result

    def test_detect_on_noise(self):
        noise = np.random.default_rng(0).normal(100, 10, (128, 128)).astype(np.float32)
        detector = LensDetector()
        result = detector.detect(noise)
        assert result["lens_score"] < 0.5


class TestDistributionAnalyzer:
    def test_analyze_random(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (200, 2))
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(positions, boundary=(100, 100))
        assert "voronoi_cv" in result
        assert "clark_evans_r" in result
        assert "distribution_score" in result

    def test_analyze_clustered(self):
        rng = np.random.default_rng(42)
        # Create clustered distribution
        cluster1 = rng.normal([25, 25], 3, (50, 2))
        cluster2 = rng.normal([75, 75], 3, (50, 2))
        positions = np.vstack([cluster1, cluster2])
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(positions, boundary=(100, 100))
        # Should detect clustering
        assert result["clark_evans_r"] < 1.0  # Clustered

    def test_too_few_sources(self):
        positions = np.array([[1, 2], [3, 4]])
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze(positions)
        assert "error" in result


class TestEnsembleDetector:
    def test_detect(self, synthetic_image: FITSImage):
        config = DetectionConfig()
        detector = EnsembleDetector(config)
        result = detector.detect(synthetic_image)
        assert "anomaly_score" in result
        assert 0 <= result["anomaly_score"] <= 1
        assert "n_detections" in result

    def test_detect_with_catalog(self, synthetic_image: FITSImage, sample_catalog):
        config = DetectionConfig()
        detector = EnsembleDetector(config)
        result = detector.detect(synthetic_image, catalog=sample_catalog)
        assert "anomaly_score" in result
        assert "galaxy" in result
        assert "kinematic" in result
        assert "transient" in result

    def test_detect_batch(self, synthetic_image: FITSImage):
        config = DetectionConfig()
        detector = EnsembleDetector(config)
        results = detector.detect_batch([synthetic_image, synthetic_image])
        assert len(results) == 2
