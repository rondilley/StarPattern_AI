"""Tests for evolutionary search."""

import numpy as np
import pytest

from star_pattern.core.config import PipelineConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.discovery.genome import DetectionGenome
from star_pattern.discovery.evolutionary import EvolutionaryDiscovery
from star_pattern.discovery.presets import get_preset_genomes


class TestPresets:
    def test_preset_count(self):
        presets = get_preset_genomes()
        assert len(presets) >= 3

    def test_presets_valid(self):
        for genome in get_preset_genomes():
            config = genome.to_detection_config()
            assert "source_extraction" in config


class TestEvolutionaryDiscovery:
    def test_initialize_population(self, sample_config: PipelineConfig):
        sample_config.evolution.population_size = 10
        engine = EvolutionaryDiscovery(sample_config)
        engine.initialize_population()
        assert len(engine.population) == 10

    def test_tournament_select(self, sample_config: PipelineConfig):
        sample_config.evolution.population_size = 10
        engine = EvolutionaryDiscovery(sample_config)
        engine.initialize_population()
        # Give different fitness values
        for i, g in enumerate(engine.population):
            g.fitness = float(i) / 10
        selected = engine.tournament_select()
        assert isinstance(selected, DetectionGenome)

    def test_evolve_generation(self, sample_config: PipelineConfig):
        sample_config.evolution.population_size = 10
        engine = EvolutionaryDiscovery(sample_config)
        engine.initialize_population()
        for g in engine.population:
            g.fitness = np.random.random()
        engine.population.sort(key=lambda g: g.fitness, reverse=True)
        engine.best_genome = engine.population[0]
        engine.evolve_generation()
        assert len(engine.population) == 10
        assert engine.generation == 1

    def test_short_run(self, sample_config: PipelineConfig, synthetic_image: FITSImage):
        sample_config.evolution.population_size = 5
        sample_config.evolution.generations = 2
        engine = EvolutionaryDiscovery(sample_config, images=[synthetic_image])
        best = engine.run()
        assert isinstance(best, DetectionGenome)
        assert best.fitness >= 0
        assert len(engine.history) == 2

    def test_evolution_with_real_fitness(
        self, sample_config: PipelineConfig, synthetic_image: FITSImage,
    ):
        """Initialize population, evaluate on a synthetic image, verify fitness values."""
        sample_config.evolution.population_size = 5
        engine = EvolutionaryDiscovery(sample_config, images=[synthetic_image])
        engine.initialize_population()
        engine.evaluate_population()

        # All genomes should have been evaluated
        for genome in engine.population:
            assert genome.fitness >= 0
            assert genome.fitness_components.get("fitness", -1) >= 0

        # Best genome should be the first (sorted by fitness)
        assert engine.best_genome is not None
        assert engine.best_genome.fitness == engine.population[0].fitness
