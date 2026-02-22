"""Pattern detection modules."""

from star_pattern.detection.classical import ClassicalDetector, GaborFilterBank
from star_pattern.detection.source_extraction import SourceExtractor
from star_pattern.detection.morphology import MorphologyAnalyzer
from star_pattern.detection.anomaly import AnomalyDetector, EmbeddingAnomalyDetector
from star_pattern.detection.lens_detector import LensDetector
from star_pattern.detection.distribution import DistributionAnalyzer
from star_pattern.detection.galaxy_detector import GalaxyDetector
from star_pattern.detection.proper_motion import ProperMotionAnalyzer
from star_pattern.detection.transient import TransientDetector
from star_pattern.detection.sersic import SersicAnalyzer
from star_pattern.detection.wavelet import WaveletAnalyzer
from star_pattern.detection.stellar_population import StellarPopulationAnalyzer
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.detection.feature_fusion import FeatureFusionExtractor
from star_pattern.detection.meta_detector import MetaDetector, MetaDetectorConfig
from star_pattern.detection.compositional import (
    ComposedPipeline,
    ComposedPipelineScorer,
    OperationRegistry,
)

__all__ = [
    "ClassicalDetector",
    "GaborFilterBank",
    "SourceExtractor",
    "MorphologyAnalyzer",
    "AnomalyDetector",
    "EmbeddingAnomalyDetector",
    "LensDetector",
    "DistributionAnalyzer",
    "GalaxyDetector",
    "ProperMotionAnalyzer",
    "TransientDetector",
    "SersicAnalyzer",
    "WaveletAnalyzer",
    "StellarPopulationAnalyzer",
    "EnsembleDetector",
    "FeatureFusionExtractor",
    "MetaDetector",
    "MetaDetectorConfig",
    "ComposedPipeline",
    "ComposedPipelineScorer",
    "OperationRegistry",
]
