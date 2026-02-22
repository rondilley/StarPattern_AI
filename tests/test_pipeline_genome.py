"""Tests for PipelineGenome: variable-length evolved detection pipelines."""

import numpy as np
import pytest

from star_pattern.detection.compositional import OperationSpec, ALL_OPERATIONS
from star_pattern.discovery.pipeline_genome import PipelineGenome


class TestPipelineGenome:
    """Tests for PipelineGenome."""

    def test_random_creation(self):
        """Random genome has valid number of operations."""
        rng = np.random.default_rng(42)
        genome = PipelineGenome.random(rng=rng, min_ops=2, max_ops=5)

        assert 2 <= len(genome.operations) <= 5
        assert genome.score_method in ["component_count", "max_residual", "area_fraction"]
        for op in genome.operations:
            assert op.name in ALL_OPERATIONS

    def test_mutation_preserves_validity(self):
        """Mutation always produces a valid genome (2-5 ops)."""
        rng = np.random.default_rng(42)
        genome = PipelineGenome.random(rng=rng, min_ops=2, max_ops=5)

        for _ in range(50):
            mutated = genome.mutate(rate=0.5)
            assert 2 <= len(mutated.operations) <= 5
            for op in mutated.operations:
                assert op.name in ALL_OPERATIONS

    def test_crossover_preserves_validity(self):
        """Crossover always produces valid genomes."""
        rng = np.random.default_rng(42)
        g1 = PipelineGenome.random(rng=rng, min_ops=2, max_ops=5)
        g2 = PipelineGenome.random(rng=rng, min_ops=2, max_ops=5)

        for _ in range(20):
            c1, c2 = g1.crossover(g2)
            assert 2 <= len(c1.operations) <= 5
            assert 2 <= len(c2.operations) <= 5

    def test_describe_readable(self):
        """describe() produces a human-readable string."""
        genome = PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 95}),
            ],
            score_method="component_count",
        )

        desc = genome.describe()
        assert "subtract_model" in desc
        assert "threshold_image" in desc
        assert "score:component_count" in desc

    def test_serialization_roundtrip(self):
        """to_dict / from_dict preserves all data."""
        rng = np.random.default_rng(42)
        genome = PipelineGenome.random(rng=rng)
        genome.fitness = 0.75

        d = genome.to_dict()
        restored = PipelineGenome.from_dict(d, rng=rng)

        assert len(restored.operations) == len(genome.operations)
        assert restored.score_method == genome.score_method
        assert restored.fitness == genome.fitness

        for orig, rest in zip(genome.operations, restored.operations):
            assert orig.name == rest.name
            assert orig.params == rest.params

    def test_to_pipeline_spec(self):
        """to_pipeline_spec creates valid PipelineSpec."""
        genome = PipelineGenome(
            operations=[
                OperationSpec("mask_sources", {"radius_px": 5}),
                OperationSpec("wavelet_residual", {"scale": 3}),
            ],
            score_method="max_residual",
        )

        spec = genome.to_pipeline_spec()
        assert len(spec.operations) == 2
        assert spec.score_method == "max_residual"

    def test_mutation_structural_add(self):
        """Mutation can add operations (stays within bounds)."""
        rng = np.random.default_rng(42)
        genome = PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 90}),
            ],
            rng=rng,
            min_ops=2,
            max_ops=5,
        )

        # Try many mutations, some should add ops
        lengths = set()
        for _ in range(100):
            mutated = genome.mutate(rate=0.8)
            lengths.add(len(mutated.operations))

        # Should see variation in length
        assert len(lengths) > 1

    def test_mutation_structural_remove(self):
        """Mutation can remove operations (stays above minimum)."""
        rng = np.random.default_rng(42)
        genome = PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 5.0}),
                OperationSpec("threshold_image", {"percentile": 90}),
                OperationSpec("edge_detect", {"method": "sobel"}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            rng=rng,
            min_ops=2,
            max_ops=5,
        )

        for _ in range(100):
            mutated = genome.mutate(rate=0.8)
            assert len(mutated.operations) >= 2

    def test_crossover_single_op_genomes(self):
        """Crossover handles edge case of minimum-length genomes."""
        rng = np.random.default_rng(42)
        g1 = PipelineGenome(
            operations=[
                OperationSpec("subtract_model", {"smooth_sigma": 3.0}),
                OperationSpec("threshold_image", {"percentile": 90}),
            ],
            rng=rng,
            min_ops=2,
            max_ops=5,
        )
        g2 = PipelineGenome(
            operations=[
                OperationSpec("edge_detect", {"method": "sobel"}),
                OperationSpec("region_statistics", {"stat_type": "count"}),
            ],
            rng=rng,
            min_ops=2,
            max_ops=5,
        )

        c1, c2 = g1.crossover(g2)
        assert 2 <= len(c1.operations) <= 5
        assert 2 <= len(c2.operations) <= 5
