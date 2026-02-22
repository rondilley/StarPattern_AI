"""Tests for detection genome."""

import numpy as np
import pytest

from star_pattern.discovery.genome import DetectionGenome, GENE_DEFINITIONS


class TestDetectionGenome:
    def test_create_random(self):
        genome = DetectionGenome()
        assert len(genome.genes) == len(GENE_DEFINITIONS)

    def test_gene_count(self):
        assert len(GENE_DEFINITIONS) == 54

    def test_gene_ranges(self):
        rng = np.random.default_rng(42)
        genome = DetectionGenome(rng=rng)
        for i, gdef in enumerate(GENE_DEFINITIONS):
            assert gdef.min_val <= genome.genes[i] <= gdef.max_val

    def test_get_by_name(self):
        genome = DetectionGenome()
        threshold = genome.get("source_threshold")
        assert 1.5 <= threshold <= 10.0

    def test_get_unknown_raises(self):
        genome = DetectionGenome()
        with pytest.raises(KeyError):
            genome.get("nonexistent_param")

    def test_new_gene_names_exist(self):
        genome = DetectionGenome()
        new_genes = [
            "galaxy_tidal_threshold",
            "galaxy_color_sigma",
            "galaxy_asymmetry_threshold",
            "kinematic_pm_min",
            "kinematic_cluster_eps",
            "kinematic_cluster_min",
            "kinematic_stream_min_length",
            "transient_noise_threshold",
            "transient_parallax_snr",
            "weight_galaxy",
            "weight_kinematic",
            "weight_transient",
        ]
        for name in new_genes:
            val = genome.get(name)
            assert isinstance(val, float)

    def test_to_detection_config(self):
        genome = DetectionGenome()
        config = genome.to_detection_config()
        assert "source_extraction" in config
        assert "gabor" in config
        assert "anomaly" in config
        assert "lens" in config
        assert "galaxy" in config
        assert "kinematic" in config
        assert "transient" in config
        assert "ensemble_weights" in config
        # Weights should sum to ~1
        weights = config["ensemble_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
        # All 11 weight keys present
        assert len(weights) == 11
        for key in ["classical", "morphology", "anomaly", "distribution",
                     "galaxy", "kinematic", "transient",
                     "sersic", "wavelet", "population", "variability"]:
            assert key in weights

    def test_mutate(self):
        rng = np.random.default_rng(42)
        genome = DetectionGenome(rng=rng)
        original = genome.genes.copy()
        mutated = genome.mutate(rate=1.0)  # Mutate all genes
        # At least some genes should change
        assert not np.allclose(original, mutated.genes)
        # All genes should be in range
        for i, gdef in enumerate(GENE_DEFINITIONS):
            assert gdef.min_val <= mutated.genes[i] <= gdef.max_val

    def test_crossover(self):
        rng = np.random.default_rng(42)
        parent1 = DetectionGenome(rng=rng)
        parent2 = DetectionGenome(rng=rng)
        child1, child2 = parent1.crossover(parent2)
        assert len(child1.genes) == len(parent1.genes)
        assert len(child2.genes) == len(parent2.genes)

    def test_distance(self):
        rng = np.random.default_rng(42)
        g1 = DetectionGenome(rng=rng)
        g2 = DetectionGenome(rng=rng)
        d = g1.distance(g2)
        assert d >= 0
        assert g1.distance(g1) < 0.01  # Self-distance ~0

    def test_serialization_roundtrip(self):
        rng = np.random.default_rng(42)
        genome = DetectionGenome(rng=rng)
        genome.fitness = 0.75
        d = genome.to_dict()
        restored = DetectionGenome.from_dict(d)
        np.testing.assert_allclose(genome.genes, restored.genes)
        assert restored.fitness == 0.75


class TestPresets:
    def test_preset_count(self):
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes()
        assert len(presets) == 11  # 5 original + kinematic + transient + sersic + wavelet + population + variability

    def test_presets_valid(self):
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes(rng=np.random.default_rng(42))
        for preset in presets:
            assert len(preset.genes) == 54
            config = preset.to_detection_config()
            weights = config["ensemble_weights"]
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_kinematic_preset_weights(self):
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes(rng=np.random.default_rng(42))
        kinematic_preset = presets[5]  # Index 5 = kinematic
        config = kinematic_preset.to_detection_config()
        weights = config["ensemble_weights"]
        # Kinematic preset should have highest kinematic weight
        assert weights["kinematic"] > weights["classical"]
        assert weights["kinematic"] > weights["morphology"]

    def test_transient_preset_weights(self):
        from star_pattern.discovery.presets import get_preset_genomes
        presets = get_preset_genomes(rng=np.random.default_rng(42))
        transient_preset = presets[6]  # Index 6 = transient
        config = transient_preset.to_detection_config()
        weights = config["ensemble_weights"]
        # Transient preset should have highest transient weight
        assert weights["transient"] > weights["classical"]
        assert weights["transient"] > weights["morphology"]
