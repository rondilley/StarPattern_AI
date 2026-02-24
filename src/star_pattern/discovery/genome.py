"""DetectionGenome: evolvable detection parameters (~22 params)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from star_pattern.utils.logging import get_logger

logger = get_logger("discovery.genome")


@dataclass
class GeneRange:
    """Range and metadata for a single gene."""

    name: str
    min_val: float
    max_val: float
    dtype: str = "float"  # float, int, bool
    description: str = ""

    def clip(self, value: float) -> float:
        return np.clip(value, self.min_val, self.max_val)

    def random(self, rng: np.random.Generator) -> float:
        if self.dtype == "bool":
            return float(rng.random() > 0.5)
        elif self.dtype == "int":
            return float(rng.integers(int(self.min_val), int(self.max_val) + 1))
        return float(rng.uniform(self.min_val, self.max_val))


# All evolvable detection parameters
GENE_DEFINITIONS: list[GeneRange] = [
    # Source extraction
    GeneRange("source_threshold", 1.5, 10.0, "float", "Source detection threshold (sigma)"),
    GeneRange("source_min_area", 3, 20, "int", "Minimum source area (pixels)"),
    GeneRange("deblend_nthresh", 16, 64, "int", "Deblending sub-thresholds"),
    GeneRange("deblend_mincont", 0.001, 0.1, "float", "Min contrast for deblending"),
    # Gabor filters
    GeneRange("gabor_freq_min", 0.02, 0.1, "float", "Min Gabor frequency"),
    GeneRange("gabor_freq_max", 0.2, 0.8, "float", "Max Gabor frequency"),
    GeneRange("gabor_n_freqs", 2, 6, "int", "Number of Gabor frequencies"),
    GeneRange("gabor_n_orient", 4, 16, "int", "Number of Gabor orientations"),
    # Anomaly detection
    GeneRange("anomaly_contamination", 0.01, 0.15, "float", "Isolation Forest contamination"),
    GeneRange("anomaly_n_estimators", 50, 300, "int", "Number of trees in Isolation Forest"),
    # Lens detection
    GeneRange("lens_arc_min_length", 8, 30, "int", "Min arc length (pixels)"),
    GeneRange("lens_arc_max_width", 3, 15, "int", "Max arc width (pixels)"),
    GeneRange("lens_ring_min_r", 5, 30, "int", "Min ring radius"),
    GeneRange("lens_ring_max_r", 30, 120, "int", "Max ring radius"),
    GeneRange("lens_snr_threshold", 1.5, 5.0, "float", "Lens detection SNR threshold"),
    # Morphology
    GeneRange("morph_smoothing", 1.0, 5.0, "float", "Smoothness kernel sigma"),
    GeneRange("morph_threshold_pctile", 60, 95, "float", "Mask threshold percentile"),
    # Distribution
    GeneRange("dist_overdensity_sigma", 2.0, 5.0, "float", "Overdensity threshold (sigma)"),
    # Galaxy detection
    GeneRange("galaxy_tidal_threshold", 0.1, 0.8, "float", "Tidal feature detection threshold"),
    GeneRange("galaxy_color_sigma", 1.5, 5.0, "float", "Color outlier threshold (sigma)"),
    GeneRange("galaxy_asymmetry_threshold", 0.1, 0.6, "float", "Merger asymmetry threshold"),
    # Kinematic detection
    GeneRange("kinematic_pm_min", 1.0, 20.0, "float", "Min proper motion (mas/yr)"),
    GeneRange("kinematic_cluster_eps", 0.5, 10.0, "float", "DBSCAN eps for PM clustering"),
    GeneRange("kinematic_cluster_min", 3, 15, "int", "Min members for kinematic group"),
    GeneRange("kinematic_stream_min_length", 5, 20, "int", "Min members for stream"),
    # Transient detection
    GeneRange("transient_noise_threshold", 1.0, 10.0, "float", "Astro noise threshold"),
    GeneRange("transient_parallax_snr", 2.0, 10.0, "float", "Parallax SNR threshold"),
    # Sersic profile fitting
    GeneRange("sersic_max_radius_frac", 0.3, 0.95, "float", "Max radius fraction for Sersic fit"),
    GeneRange("sersic_residual_sigma", 1.5, 5.0, "float", "Residual significance threshold"),
    # Wavelet analysis
    GeneRange("wavelet_n_scales", 3, 7, "int", "Number of wavelet scales"),
    GeneRange("wavelet_significance", 1.5, 5.0, "float", "Wavelet significance threshold"),
    # Stellar population / CMD
    GeneRange("population_ms_width", 0.1, 0.6, "float", "Main sequence width in color"),
    GeneRange("population_bs_offset", 0.1, 0.8, "float", "Blue straggler color offset"),
    # Variability / time-domain detection
    GeneRange("variability_min_epochs", 5, 30, "int", "Min epochs for variability"),
    GeneRange("variability_significance", 1.5, 5.0, "float", "Variability chi2 threshold"),
    GeneRange("variability_period_min", 0.01, 1.0, "float", "Min period search (days)"),
    GeneRange("variability_period_max", 100, 1000, "float", "Max period search (days)"),
    # Ensemble weights
    GeneRange("weight_classical", 0.0, 1.0, "float", "Weight for classical detection"),
    GeneRange("weight_morphology", 0.0, 1.0, "float", "Weight for morphology"),
    GeneRange("weight_anomaly", 0.0, 1.0, "float", "Weight for anomaly detection"),
    GeneRange("weight_distribution", 0.0, 1.0, "float", "Weight for distribution analysis"),
    GeneRange("weight_galaxy", 0.0, 1.0, "float", "Weight for galaxy detection"),
    GeneRange("weight_kinematic", 0.0, 1.0, "float", "Weight for kinematic detection"),
    GeneRange("weight_transient", 0.0, 1.0, "float", "Weight for transient detection"),
    GeneRange("weight_sersic", 0.0, 1.0, "float", "Weight for Sersic profile analysis"),
    GeneRange("weight_wavelet", 0.0, 1.0, "float", "Weight for wavelet analysis"),
    GeneRange("weight_population", 0.0, 1.0, "float", "Weight for CMD/population analysis"),
    GeneRange("weight_variability", 0.0, 1.0, "float", "Weight for variability detection"),
    # Detector enable/disable gates (>= 0.5 = enabled)
    GeneRange("enable_classical", 0, 1, "bool", "Enable classical detector"),
    GeneRange("enable_morphology", 0, 1, "bool", "Enable morphology detector"),
    GeneRange("enable_anomaly", 0, 1, "bool", "Enable anomaly detector"),
    GeneRange("enable_lens", 0, 1, "bool", "Enable lens detector"),
    GeneRange("enable_distribution", 0, 1, "bool", "Enable distribution detector"),
    GeneRange("enable_galaxy", 0, 1, "bool", "Enable galaxy detector"),
    GeneRange("enable_kinematic", 0, 1, "bool", "Enable kinematic detector"),
    GeneRange("enable_transient", 0, 1, "bool", "Enable transient detector"),
    GeneRange("enable_sersic", 0, 1, "bool", "Enable sersic detector"),
    GeneRange("enable_wavelet", 0, 1, "bool", "Enable wavelet detector"),
    GeneRange("enable_population", 0, 1, "bool", "Enable population detector"),
    GeneRange("enable_variability", 0, 1, "bool", "Enable variability detector"),
    # Meta-detector (Phase 2)
    GeneRange("meta_blend_weight", 0.0, 1.0, "float", "Meta-detector blend weight (0=linear, 1=learned)"),
    GeneRange("meta_gbm_depth", 2, 6, "int", "Meta-detector GBM max depth"),
    GeneRange("meta_gbm_estimators", 50, 300, "int", "Meta-detector GBM n_estimators"),
    # Representation learning (Phase 3)
    GeneRange("repr_anomaly_contamination", 0.01, 0.15, "float", "Embedding anomaly contamination"),
    GeneRange("repr_weight", 0.0, 1.0, "float", "Embedding representation weight"),
    # Compositional detection (Phase 4)
    GeneRange("composed_weight", 0.0, 1.0, "float", "Composed pipeline score weight"),
    # Temporal detection (Phase 5)
    GeneRange("temporal_snr_threshold", 3.0, 10.0, "float", "Temporal residual SNR threshold"),
    GeneRange("temporal_min_epochs", 2, 10, "int", "Min epochs for temporal analysis"),
    GeneRange("temporal_max_baseline", 30, 2000, "float", "Max baseline in days"),
    GeneRange("temporal_min_baseline", 1, 30, "float", "Min baseline in days"),
    GeneRange("temporal_dipole_max_sep", 1.0, 10.0, "float", "Max dipole separation in arcsec"),
    GeneRange("weight_temporal", 0.0, 1.0, "float", "Weight for temporal detection"),
]


class DetectionGenome:
    """An evolvable set of detection parameters.

    Each genome is a vector of ~22 floats, each mapped to a detection parameter.
    """

    def __init__(
        self,
        genes: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.gene_defs = GENE_DEFINITIONS
        self.n_genes = len(self.gene_defs)
        self.rng = rng or np.random.default_rng()

        if genes is not None:
            self.genes = genes.copy()
        else:
            self.genes = np.array([g.random(self.rng) for g in self.gene_defs])

        self.fitness: float = 0.0
        self.fitness_components: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}

    def get(self, name: str) -> float:
        """Get a gene value by name."""
        for i, gdef in enumerate(self.gene_defs):
            if gdef.name == name:
                if i < len(self.genes):
                    return float(self.genes[i])
                return float(gdef.random(self.rng))
        raise KeyError(f"Unknown gene: {name}")

    def _safe_get(self, name: str, default: float) -> float:
        """Get a gene value by name, returning default if not present.

        Handles backward compatibility when loading old genomes that
        lack newer genes.
        """
        for i, gdef in enumerate(self.gene_defs):
            if gdef.name == name:
                if i < len(self.genes):
                    return float(self.genes[i])
                return default
        return default

    def to_detection_config(self) -> dict[str, Any]:
        """Convert genome to detection configuration dict."""
        g = self.get

        # Gate weights by enable genes: disabled detectors get weight=0.
        # Note: "lens" has an enable gene but no weight gene -- its score
        # contributes through the anomaly detector and ensemble scoring.
        _WEIGHT_NAMES = [
            "classical", "morphology", "anomaly", "distribution",
            "galaxy", "kinematic", "transient", "sersic", "wavelet",
            "population", "variability",
        ]
        _ENABLE_NAMES = [
            "classical", "morphology", "anomaly", "lens", "distribution",
            "galaxy", "kinematic", "transient", "sersic", "wavelet",
            "population", "variability",
        ]
        enabled_detectors = {}
        for name in _ENABLE_NAMES:
            enable_val = self._safe_get(f"enable_{name}", 1.0)
            enabled_detectors[name] = enable_val >= 0.5

        raw_weights = {}
        for name in _WEIGHT_NAMES:
            is_enabled = enabled_detectors.get(name, True)
            raw_weights[name] = g(f"weight_{name}") if is_enabled else 0.0

        # Temporal has no enable gene (always on when temporal images exist)
        raw_weights["temporal"] = self._safe_get("weight_temporal", 0.0)

        w_sum = sum(raw_weights.values())
        if w_sum < 1e-10:
            w_sum = 1.0

        return {
            "source_extraction": {
                "threshold": g("source_threshold"),
                "min_area": int(g("source_min_area")),
                "deblend_nthresh": int(g("deblend_nthresh")),
                "deblend_mincont": g("deblend_mincont"),
            },
            "gabor": {
                "frequencies": np.linspace(
                    g("gabor_freq_min"), g("gabor_freq_max"), int(g("gabor_n_freqs"))
                ).tolist(),
                "n_orientations": int(g("gabor_n_orient")),
            },
            "anomaly": {
                "contamination": g("anomaly_contamination"),
                "n_estimators": int(g("anomaly_n_estimators")),
            },
            "lens": {
                "arc_min_length": int(g("lens_arc_min_length")),
                "arc_max_width": int(g("lens_arc_max_width")),
                "ring_min_radius": int(g("lens_ring_min_r")),
                "ring_max_radius": int(g("lens_ring_max_r")),
                "snr_threshold": g("lens_snr_threshold"),
            },
            "morphology": {
                "smoothing_sigma": g("morph_smoothing"),
                "threshold_percentile": g("morph_threshold_pctile"),
            },
            "distribution": {
                "overdensity_sigma": g("dist_overdensity_sigma"),
            },
            "galaxy": {
                "tidal_threshold": g("galaxy_tidal_threshold"),
                "color_sigma": g("galaxy_color_sigma"),
                "asymmetry_threshold": g("galaxy_asymmetry_threshold"),
            },
            "kinematic": {
                "pm_min": g("kinematic_pm_min"),
                "cluster_eps": g("kinematic_cluster_eps"),
                "cluster_min": int(g("kinematic_cluster_min")),
                "stream_min_length": int(g("kinematic_stream_min_length")),
            },
            "transient": {
                "noise_threshold": g("transient_noise_threshold"),
                "parallax_snr": g("transient_parallax_snr"),
            },
            "sersic": {
                "max_radius_frac": g("sersic_max_radius_frac"),
                "residual_sigma": g("sersic_residual_sigma"),
            },
            "wavelet": {
                "n_scales": int(g("wavelet_n_scales")),
                "significance": g("wavelet_significance"),
            },
            "population": {
                "ms_width": g("population_ms_width"),
                "blue_straggler_offset": g("population_bs_offset"),
            },
            "variability": {
                "min_epochs": int(g("variability_min_epochs")),
                "significance": g("variability_significance"),
                "period_min": g("variability_period_min"),
                "period_max": g("variability_period_max"),
            },
            "ensemble_weights": {
                name: raw_weights[name] / w_sum
                for name in list(_WEIGHT_NAMES) + ["temporal"]
            },
            "enabled_detectors": enabled_detectors,
            "meta": {
                "blend_weight": self._safe_get("meta_blend_weight", 0.0),
                "gbm_max_depth": int(self._safe_get("meta_gbm_depth", 3)),
                "gbm_n_estimators": int(self._safe_get("meta_gbm_estimators", 100)),
            },
            "representation": {
                "anomaly_contamination": self._safe_get("repr_anomaly_contamination", 0.05),
                "weight": self._safe_get("repr_weight", 0.0),
            },
            "compositional": {
                "weight": self._safe_get("composed_weight", 0.0),
            },
            "temporal": {
                "snr_threshold": self._safe_get("temporal_snr_threshold", 5.0),
                "min_epochs": int(self._safe_get("temporal_min_epochs", 2)),
                "max_baseline": self._safe_get("temporal_max_baseline", 2000.0),
                "min_baseline": self._safe_get("temporal_min_baseline", 1.0),
                "dipole_max_sep": self._safe_get("temporal_dipole_max_sep", 5.0),
            },
        }

    def mutate(self, rate: float = 0.15) -> DetectionGenome:
        """Create a mutated copy."""
        new_genes = self.genes.copy()
        for i, gdef in enumerate(self.gene_defs):
            if self.rng.random() < rate:
                # Gaussian mutation with 10% of range
                range_size = gdef.max_val - gdef.min_val
                delta = self.rng.normal(0, 0.1 * range_size)
                new_genes[i] = gdef.clip(new_genes[i] + delta)
                if gdef.dtype == "int":
                    new_genes[i] = round(new_genes[i])
                elif gdef.dtype == "bool":
                    new_genes[i] = float(new_genes[i] > 0.5)

        child = DetectionGenome(genes=new_genes, rng=self.rng)
        return child

    def crossover(self, other: DetectionGenome) -> tuple[DetectionGenome, DetectionGenome]:
        """Two-point crossover with another genome."""
        pt1, pt2 = sorted(self.rng.choice(self.n_genes, size=2, replace=False))

        child1_genes = self.genes.copy()
        child2_genes = other.genes.copy()

        child1_genes[pt1:pt2] = other.genes[pt1:pt2]
        child2_genes[pt1:pt2] = self.genes[pt1:pt2]

        return (
            DetectionGenome(genes=child1_genes, rng=self.rng),
            DetectionGenome(genes=child2_genes, rng=self.rng),
        )

    def distance(self, other: DetectionGenome) -> float:
        """Genetic distance to another genome."""
        # Normalize each gene to [0,1] range
        normalized_self = np.zeros(self.n_genes)
        normalized_other = np.zeros(self.n_genes)
        for i, gdef in enumerate(self.gene_defs):
            r = gdef.max_val - gdef.min_val
            if r > 0:
                normalized_self[i] = (self.genes[i] - gdef.min_val) / r
                normalized_other[i] = (other.genes[i] - gdef.min_val) / r

        return float(np.linalg.norm(normalized_self - normalized_other))

    def to_dict(self) -> dict[str, Any]:
        return {
            "genes": self.genes.tolist(),
            "fitness": self.fitness,
            "fitness_components": self.fitness_components,
            "config": self.to_detection_config(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionGenome:
        raw_genes = np.array(d["genes"])
        n_expected = len(GENE_DEFINITIONS)
        # Pad old genomes that have fewer genes than current definitions
        if len(raw_genes) < n_expected:
            rng = np.random.default_rng()
            padded = np.zeros(n_expected)
            padded[: len(raw_genes)] = raw_genes
            for i in range(len(raw_genes), n_expected):
                padded[i] = GENE_DEFINITIONS[i].random(rng)
            raw_genes = padded
        genome = cls(genes=raw_genes)
        genome.fitness = d.get("fitness", 0)
        genome.fitness_components = d.get("fitness_components", {})
        return genome

    def __repr__(self) -> str:
        return f"DetectionGenome(fitness={self.fitness:.4f}, genes={self.n_genes})"
