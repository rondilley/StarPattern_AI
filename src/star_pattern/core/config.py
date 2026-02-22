"""All configuration dataclasses for the pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Configuration for data acquisition."""

    sources: list[str] = field(default_factory=lambda: ["sdss", "gaia", "mast", "ztf"])
    cache_dir: str = "output/cache"
    default_radius_arcmin: float = 3.0
    min_galactic_latitude: float = 20.0
    max_concurrent_downloads: int = 4


@dataclass
class DetectionConfig:
    """Configuration for pattern detection."""

    source_extraction_threshold: float = 3.0
    gabor_frequencies: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    gabor_orientations: int = 8
    anomaly_contamination: float = 0.05

    # Galaxy detection
    galaxy_tidal_threshold: float = 0.3
    galaxy_color_sigma: float = 2.5
    galaxy_asymmetry_threshold: float = 0.3

    # Kinematic detection (proper motion)
    kinematic_pm_min: float = 5.0
    kinematic_cluster_eps: float = 2.0
    kinematic_cluster_min: int = 5
    kinematic_stream_min_length: int = 8

    # Transient/variability detection
    transient_noise_threshold: float = 3.0
    transient_parallax_snr: float = 5.0

    # Sersic profile fitting
    sersic_max_radius_frac: float = 0.8
    sersic_residual_sigma: float = 3.0

    # Wavelet analysis
    wavelet_n_scales: int = 5
    wavelet_significance: float = 3.0

    # Stellar population / CMD
    population_ms_width: float = 0.3
    population_blue_straggler_offset: float = 0.3

    # Variability / time-domain detection
    variability_min_epochs: int = 10
    variability_significance: float = 3.0
    variability_period_min: float = 0.1    # days
    variability_period_max: float = 500.0  # days

    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {
            "classical": 0.09,
            "morphology": 0.09,
            "anomaly": 0.09,
            "lens": 0.09,
            "distribution": 0.11,
            "galaxy": 0.09,
            "kinematic": 0.09,
            "transient": 0.04,
            "sersic": 0.07,
            "wavelet": 0.09,
            "population": 0.06,
            "variability": 0.09,
        }
    )

    @classmethod
    def from_genome_dict(cls, d: dict[str, Any]) -> DetectionConfig:
        """Create DetectionConfig from genome.to_detection_config() output.

        Handles both old (48-gene) and new (54-gene) genome dicts
        via .get() defaults for new sections (meta, representation,
        compositional).
        """
        se = d.get("source_extraction", {})
        gabor = d.get("gabor", {})
        anomaly = d.get("anomaly", {})
        galaxy = d.get("galaxy", {})
        kinematic = d.get("kinematic", {})
        transient = d.get("transient", {})
        sersic = d.get("sersic", {})
        wavelet = d.get("wavelet", {})
        population = d.get("population", {})
        variability = d.get("variability", {})
        return cls(
            source_extraction_threshold=se.get("threshold", 3.0),
            gabor_frequencies=gabor.get("frequencies", [0.05, 0.1, 0.2, 0.4]),
            gabor_orientations=gabor.get("n_orientations", 8),
            anomaly_contamination=anomaly.get("contamination", 0.05),
            galaxy_tidal_threshold=galaxy.get("tidal_threshold", 0.3),
            galaxy_color_sigma=galaxy.get("color_sigma", 2.5),
            galaxy_asymmetry_threshold=galaxy.get("asymmetry_threshold", 0.3),
            kinematic_pm_min=kinematic.get("pm_min", 5.0),
            kinematic_cluster_eps=kinematic.get("cluster_eps", 2.0),
            kinematic_cluster_min=kinematic.get("cluster_min", 5),
            kinematic_stream_min_length=kinematic.get("stream_min_length", 8),
            transient_noise_threshold=transient.get("noise_threshold", 3.0),
            transient_parallax_snr=transient.get("parallax_snr", 5.0),
            sersic_max_radius_frac=sersic.get("max_radius_frac", 0.8),
            sersic_residual_sigma=sersic.get("residual_sigma", 3.0),
            wavelet_n_scales=wavelet.get("n_scales", 5),
            wavelet_significance=wavelet.get("significance", 3.0),
            population_ms_width=population.get("ms_width", 0.3),
            population_blue_straggler_offset=population.get("blue_straggler_offset", 0.3),
            variability_min_epochs=variability.get("min_epochs", 10),
            variability_significance=variability.get("significance", 3.0),
            variability_period_min=variability.get("period_min", 0.1),
            variability_period_max=variability.get("period_max", 500.0),
            ensemble_weights=d.get("ensemble_weights", {}),
        )


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary search."""

    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elite_count: int = 2
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "anomaly": 0.35,
            "significance": 0.25,
            "novelty": 0.15,
            "diversity": 0.1,
            "recovery": 0.15,
        }
    )


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""

    key_dir: str = "."
    max_tokens: int = 2048
    temperature: float = 0.7
    debate_rounds: int = 3
    consensus_min_providers: int = 2
    token_budget: int = 500_000  # Per-session token budget
    strategy_interval: int = 25  # Cycles between LLM strategy sessions
    max_debate_tokens: int = 5_000  # Cap debate at this budget


@dataclass
class WideFieldConfig:
    """Configuration for wide-field sky coverage."""

    tile_radius_arcmin: float = 3.0
    overlap_fraction: float = 0.2
    max_tiles: int = 500
    mosaic_pixel_scale_arcsec: float = 0.4
    mosaic_combine: str = "mean"
    fetch_all_sources: bool = True
    max_mast_observations: int = 10


@dataclass
class SurveyConfig:
    """Configuration for HEALPix grid survey mode."""

    nside: int = 64
    min_galactic_lat: float = 20.0
    radius_arcmin: float = 3.0
    order: str = "galactic_latitude"
    state_file: str = "survey_state.json"


@dataclass
class MetaDetectorConfig:
    """Configuration for the learned meta-detector."""

    enabled: bool = True
    blend_weight: float = 0.0
    min_samples_gbm: int = 50
    min_samples_nn: int = 200
    gbm_n_estimators: int = 100
    gbm_max_depth: int = 3
    nn_hidden: list[int] = field(default_factory=lambda: [64, 32])


@dataclass
class RepresentationConfig:
    """Configuration for representation learning integration."""

    enabled: bool = True
    backbone_name: str = "efficientnet_b0"
    use_backbone: bool = True
    byol_retrain_interval: int = 50
    byol_epochs: int = 10
    embedding_anomaly_contamination: float = 0.05
    min_embeddings_for_anomaly: int = 20
    max_embedding_buffer: int = 500


@dataclass
class CompositionalConfig:
    """Configuration for compositional detection pipelines."""

    enabled: bool = True
    n_pipelines: int = 10
    min_ops: int = 2
    max_ops: int = 5
    evolve_generations: int = 10
    evolve_population: int = 15


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    wide_field: WideFieldConfig = field(default_factory=WideFieldConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    meta: MetaDetectorConfig = field(default_factory=MetaDetectorConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    compositional: CompositionalConfig = field(default_factory=CompositionalConfig)
    output_dir: str = "output/runs"
    checkpoint_interval: int = 10
    max_cycles: int = 1000
    batch_size: int = 10
    evolve_interval: int = 25
    evolve_generations: int = 5
    evolve_population: int = 15
    evolve_max_seconds: int = 600
    evolve_workers: int = 4

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a JSON file."""
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        """Create config from a dictionary."""
        data = DataConfig(**d.get("data", {}))
        detection = DetectionConfig(**d.get("detection", {}))
        evolution = EvolutionConfig(**d.get("evolution", {}))
        llm = LLMConfig(**d.get("llm", {}))
        wide_field = WideFieldConfig(**d.get("wide_field", {}))
        survey = SurveyConfig(**d.get("survey", {}))
        meta = MetaDetectorConfig(**d.get("meta", {}))
        representation = RepresentationConfig(**d.get("representation", {}))
        compositional = CompositionalConfig(**d.get("compositional", {}))
        pipeline = d.get("pipeline", {})
        return cls(
            data=data,
            detection=detection,
            evolution=evolution,
            llm=llm,
            wide_field=wide_field,
            survey=survey,
            meta=meta,
            representation=representation,
            compositional=compositional,
            output_dir=pipeline.get("output_dir", "output/runs"),
            checkpoint_interval=pipeline.get("checkpoint_interval", 10),
            max_cycles=pipeline.get("max_cycles", 1000),
            batch_size=pipeline.get("batch_size", 10),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        from dataclasses import asdict

        return asdict(self)
