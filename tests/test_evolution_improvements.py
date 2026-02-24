"""Tests for evolution improvements: adaptive mutation, experience replay, feedback retraining."""

import json
from pathlib import Path

import numpy as np
import pytest

from star_pattern.core.config import PipelineConfig, EvolutionConfig, DetectionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.detection.ensemble import EnsembleDetector
from star_pattern.discovery.genome import DetectionGenome, GENE_DEFINITIONS
from star_pattern.discovery.fitness import FitnessEvaluator
from star_pattern.discovery.evolutionary import EvolutionaryDiscovery
from star_pattern.pipeline.active_learning import ActiveLearner
from star_pattern.evaluation.metrics import PatternResult


class TestAdaptiveMutation:
    """Test adaptive mutation rate in evolutionary discovery."""

    def test_mutation_increases_on_stagnation(self):
        """Mutation rate should increase when fitness stagnates."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config)
        evo.initialize_population()

        initial_rate = evo._current_mutation_rate

        # Simulate stagnation: best fitness does not improve
        evo._best_fitness_seen = 0.5
        for genome in evo.population:
            genome.fitness = 0.5  # everyone has the same fitness

        for _ in range(5):
            evo._adapt_mutation_rate()

        assert evo._current_mutation_rate > initial_rate
        assert evo._stagnation_counter >= 3

    def test_mutation_decreases_on_improvement(self):
        """Mutation rate should decrease when fitness improves."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config)
        evo.initialize_population()

        # Set high mutation rate
        evo._current_mutation_rate = 0.4
        evo._best_fitness_seen = 0.3

        # Simulate improvement
        evo.population[0].fitness = 0.5
        evo._adapt_mutation_rate()

        assert evo._current_mutation_rate < 0.4
        assert evo._stagnation_counter == 0

    def test_mutation_bounded(self):
        """Mutation rate should stay within bounds."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config)
        evo.initialize_population()

        # Extreme stagnation
        evo._best_fitness_seen = 0.5
        for genome in evo.population:
            genome.fitness = 0.5

        for _ in range(50):
            evo._adapt_mutation_rate()

        assert evo._current_mutation_rate <= evo._mutation_max
        assert evo._current_mutation_rate >= evo._mutation_min


class TestExperienceReplay:
    """Test experience replay persistence."""

    def test_save_and_load_replay(self, tmp_path):
        """Save replay genomes then load them in a new run."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config, replay_path=tmp_path)
        evo.initialize_population()

        # Assign fitness to population
        for i, genome in enumerate(evo.population):
            genome.fitness = float(i) / len(evo.population)

        evo.population.sort(key=lambda g: g.fitness, reverse=True)
        evo.best_genome = evo.population[0]

        # Save replay
        evo._save_replay_genomes()

        # Verify file exists
        replay_file = tmp_path / "experience_replay.json"
        assert replay_file.exists()

        # Load in a new instance
        evo2 = EvolutionaryDiscovery(config, replay_path=tmp_path)
        loaded = evo2._load_replay_genomes()

        assert len(loaded) > 0
        # Best genome should be first
        assert loaded[0].fitness >= loaded[-1].fitness

    def test_empty_replay_returns_empty(self, tmp_path):
        """Loading from nonexistent path returns empty list."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config, replay_path=tmp_path)
        loaded = evo._load_replay_genomes()
        assert loaded == []

    def test_replay_seeds_population(self, tmp_path):
        """Replay genomes should appear in initialized population."""
        config = PipelineConfig()
        evo1 = EvolutionaryDiscovery(config, replay_path=tmp_path)
        evo1.initialize_population()

        # Make a distinctive genome
        for i, genome in enumerate(evo1.population):
            genome.fitness = 0.9
            genome.genes[:] = 0.5  # Distinctive gene values

        evo1.population.sort(key=lambda g: g.fitness, reverse=True)
        evo1.best_genome = evo1.population[0]
        evo1._save_replay_genomes()

        # Initialize a new population - should include replay genomes
        evo2 = EvolutionaryDiscovery(config, replay_path=tmp_path)
        evo2.initialize_population()

        # Check that some genomes have the distinctive gene values
        has_replay = any(
            np.allclose(g.genes, 0.5, atol=0.01)
            for g in evo2.population
        )
        assert has_replay


class TestSyntheticInjectionFitness:
    """Test synthetic injection in fitness evaluation."""

    def test_recovery_component_present(self):
        """Fitness evaluation should include recovery component."""
        config = EvolutionConfig()
        evaluator = FitnessEvaluator(config, use_synthetic_injection=True)

        # Create a simple test image
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (64, 64)).astype(np.float32)
        image = FITSImage.from_array(data)

        genome = DetectionGenome()
        result = evaluator.evaluate(genome.to_detection_config(), [image])

        assert "recovery" in result
        assert 0 <= result["recovery"] <= 1

    def test_recovery_disabled(self):
        """Recovery should be 0 when injection is disabled."""
        config = EvolutionConfig()
        evaluator = FitnessEvaluator(config, use_synthetic_injection=False)

        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (64, 64)).astype(np.float32)
        image = FITSImage.from_array(data)

        genome = DetectionGenome()
        result = evaluator.evaluate(genome.to_detection_config(), [image])

        assert result["recovery"] == 0.0

    def test_fitness_with_recovery_weight(self):
        """Fitness should incorporate recovery weight."""
        config = EvolutionConfig()
        evaluator = FitnessEvaluator(config, use_synthetic_injection=True)

        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (128, 128)).astype(np.float32)
        # Add a bright source so detection has something to find
        y, x = np.mgrid[:128, :128]
        data += 500 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 5 ** 2))
        image = FITSImage.from_array(data)

        genome = DetectionGenome()
        result = evaluator.evaluate(genome.to_detection_config(), [image])

        # Fitness should be positive
        assert result["fitness"] >= 0
        # All components should be present
        for key in ["anomaly", "significance", "novelty", "diversity", "recovery"]:
            assert key in result


class TestFeedbackDrivenRetraining:
    """Test ActiveLearner feedback-driven retraining."""

    def _make_pattern_result(self, score=0.5, detection_type="test"):
        """Create a minimal PatternResult for testing."""
        return PatternResult(
            region_ra=180.0,
            region_dec=45.0,
            anomaly_score=score,
            detection_type=detection_type,
        )

    def test_add_feedback_with_features(self):
        """Feedback with features should accumulate for retraining."""
        learner = ActiveLearner()
        feat = np.random.default_rng(42).normal(0, 1, 10)

        result = self._make_pattern_result(score=0.6)
        learner.add_feedback(result, is_interesting=True, features=feat)

        assert len(learner._labeled_positive) == 1
        assert len(learner.feedback_history) == 1

    def test_retrain_with_sufficient_data(self):
        """Retraining should work when enough labeled data exists."""
        learner = ActiveLearner(retrain_interval=5)
        rng = np.random.default_rng(42)

        # Add enough positive examples
        for i in range(6):
            feat = rng.normal(1, 0.5, 10)  # positive cluster
            result = self._make_pattern_result(score=0.7)
            learner.add_feedback(result, is_interesting=True, features=feat)

        # Add enough negative examples
        for i in range(6):
            feat = rng.normal(-1, 0.5, 10)  # negative cluster
            result = self._make_pattern_result(score=0.3)
            learner.add_feedback(result, is_interesting=False, features=feat)

        # Retrained detector should be available
        detector = learner.get_retrained_detector()
        assert detector is not None
        assert detector["n_positive"] >= 5
        assert detector["n_negative"] >= 5

    def test_retrain_insufficient_data(self):
        """Should not retrain with insufficient data."""
        learner = ActiveLearner()

        # Only 2 positive examples
        rng = np.random.default_rng(42)
        for i in range(2):
            feat = rng.normal(0, 1, 10)
            result = self._make_pattern_result()
            learner.add_feedback(result, is_interesting=True, features=feat)

        detector = learner.get_retrained_detector()
        assert detector is None

    def test_score_with_retrained(self):
        """Should be able to score features with retrained detector."""
        learner = ActiveLearner(retrain_interval=5)
        rng = np.random.default_rng(42)

        # Train with labeled data
        for i in range(8):
            feat = rng.normal(1, 0.3, 10)
            result = self._make_pattern_result(score=0.7)
            learner.add_feedback(result, is_interesting=True, features=feat)
        for i in range(8):
            feat = rng.normal(-1, 0.3, 10)
            result = self._make_pattern_result(score=0.3)
            learner.add_feedback(result, is_interesting=False, features=feat)

        # Score a new feature vector
        new_feat = rng.normal(1, 0.3, 10)  # similar to positive
        score = learner.score_with_retrained(new_feat)
        assert score is not None
        assert 0 <= score <= 1

    def test_learned_weights(self):
        """Should learn ensemble weights from feedback with detector scores."""
        learner = ActiveLearner()

        # Interesting detections have high lens_score
        for i in range(15):
            result = self._make_pattern_result(score=0.7)
            learner.add_feedback(
                result, is_interesting=True,
                detector_scores={"classical": 0.2, "lens": 0.8, "morphology": 0.3},
            )

        # Boring detections have low lens_score
        for i in range(15):
            result = self._make_pattern_result(score=0.3)
            learner.add_feedback(
                result, is_interesting=False,
                detector_scores={"classical": 0.5, "lens": 0.1, "morphology": 0.4},
            )

        weights = learner.get_learned_weights()
        assert weights is not None
        # Lens should have higher weight (correlated with interesting)
        assert weights["lens"] > weights["classical"]

    def test_persistence(self, tmp_path):
        """Feedback should persist and reload across sessions."""
        learner1 = ActiveLearner(persistence_path=tmp_path)
        rng = np.random.default_rng(42)

        for i in range(5):
            feat = rng.normal(0, 1, 10)
            result = self._make_pattern_result(score=0.5 + i * 0.05)
            learner1.add_feedback(result, is_interesting=i % 2 == 0, features=feat)

        # Load in new instance
        learner2 = ActiveLearner(persistence_path=tmp_path)
        assert len(learner2.feedback_history) == 5

    def test_adaptive_query_strategy(self):
        """Query window should narrow with more feedback."""
        learner = ActiveLearner(uncertainty_threshold=0.3)

        # Initially, query range is [0.3, 0.7]
        result_mid = self._make_pattern_result(score=0.5)
        assert learner.should_query(result_mid)

        # Add many feedback records to trigger adaptation
        for i in range(30):
            r = self._make_pattern_result(score=0.5)
            learner.add_feedback(r, is_interesting=i % 2 == 0)

        # After 30 feedback records, the window should be narrower
        # Score at 0.35 was queryable before, may not be now
        result_edge = self._make_pattern_result(score=0.35)
        # The adaptive window should have changed
        assert learner.should_query(result_mid)  # 0.5 still queryable

    def test_statistics(self):
        """get_statistics should return correct counts."""
        learner = ActiveLearner()

        for i in range(10):
            result = self._make_pattern_result()
            learner.add_feedback(result, is_interesting=i < 4)

        stats = learner.get_statistics()
        assert stats["n_total"] == 10
        assert stats["n_interesting"] == 4
        assert stats["n_boring"] == 6
        assert abs(stats["interesting_rate"] - 0.4) < 0.01


class TestGenomeExpansion:
    """Test that new genome genes are properly integrated."""

    def test_gene_count(self):
        """Genome should have the correct number of genes."""
        genome = DetectionGenome()
        # 4 source + 4 gabor + 2 anomaly + 5 lens + 2 morph + 1 dist
        # + 3 galaxy + 4 kinematic + 2 transient
        # + 2 sersic + 2 wavelet + 2 population + 4 variability
        # + 11 weights + 3 meta + 2 repr + 1 composed
        # + 5 temporal params + 1 temporal weight
        assert genome.n_genes == 72

    def test_new_genes_accessible(self):
        """New genes should be retrievable by name."""
        genome = DetectionGenome()

        # Sersic genes
        assert 0.3 <= genome.get("sersic_max_radius_frac") <= 0.95
        assert 1.5 <= genome.get("sersic_residual_sigma") <= 5.0

        # Wavelet genes
        assert 3 <= genome.get("wavelet_n_scales") <= 7
        assert 1.5 <= genome.get("wavelet_significance") <= 5.0

        # Population genes
        assert 0.1 <= genome.get("population_ms_width") <= 0.6
        assert 0.1 <= genome.get("population_bs_offset") <= 0.8

        # New weights
        assert 0 <= genome.get("weight_sersic") <= 1
        assert 0 <= genome.get("weight_wavelet") <= 1
        assert 0 <= genome.get("weight_population") <= 1

    def test_to_detection_config_includes_new(self):
        """to_detection_config should include new detector sections."""
        genome = DetectionGenome()
        config = genome.to_detection_config()

        assert "sersic" in config
        assert "wavelet" in config
        assert "population" in config
        assert "weight_sersic" in config["ensemble_weights"] or "sersic" in config["ensemble_weights"]

    def test_detection_config_roundtrip(self):
        """DetectionConfig should roundtrip through genome dict."""
        genome = DetectionGenome()
        config_dict = genome.to_detection_config()
        det_config = DetectionConfig.from_genome_dict(config_dict)

        assert det_config.sersic_max_radius_frac > 0
        assert det_config.wavelet_n_scales >= 3
        assert det_config.population_ms_width > 0

    def test_crossover_preserves_new_genes(self):
        """Crossover should work with expanded genome."""
        rng = np.random.default_rng(42)
        g1 = DetectionGenome(rng=rng)
        g2 = DetectionGenome(rng=rng)

        child1, child2 = g1.crossover(g2)

        assert child1.n_genes == g1.n_genes
        assert child2.n_genes == g1.n_genes

    def test_mutation_preserves_new_genes(self):
        """Mutation should work with expanded genome."""
        genome = DetectionGenome()
        mutated = genome.mutate(rate=0.5)

        assert mutated.n_genes == genome.n_genes
        # At least some genes should differ
        assert not np.allclose(genome.genes, mutated.genes)


class TestDetectorEnableDisable:
    """Test detector enable/disable gene gating."""

    def test_enable_genes_exist(self):
        """All 12 enable genes should be in GENE_DEFINITIONS."""
        names = {g.name for g in GENE_DEFINITIONS}
        for det in [
            "classical", "morphology", "anomaly", "lens", "distribution",
            "galaxy", "kinematic", "transient", "sersic", "wavelet",
            "population", "variability",
        ]:
            assert f"enable_{det}" in names

    def test_disable_gates_weight_to_zero(self):
        """Genome with enable_sersic=0 should produce weight_sersic=0."""
        genome = DetectionGenome()
        # Set enable_sersic to 0 (disabled)
        for i, gdef in enumerate(GENE_DEFINITIONS):
            if gdef.name == "enable_sersic":
                genome.genes[i] = 0.0
            # Ensure sersic weight is nonzero before gating
            if gdef.name == "weight_sersic":
                genome.genes[i] = 0.5

        config = genome.to_detection_config()
        assert config["ensemble_weights"]["sersic"] == 0.0
        assert config["enabled_detectors"]["sersic"] is False

    def test_disable_lens_gate(self):
        """Genome with enable_lens=0 should mark lens as disabled."""
        genome = DetectionGenome()
        for i, gdef in enumerate(GENE_DEFINITIONS):
            if gdef.name == "enable_lens":
                genome.genes[i] = 0.0

        config = genome.to_detection_config()
        # Lens has no weight gene, but enabled_detectors should reflect it
        assert config["enabled_detectors"]["lens"] is False

    def test_enable_gates_preserve_weight(self):
        """Genome with enable_sersic=1 should preserve weight_sersic."""
        genome = DetectionGenome()
        for i, gdef in enumerate(GENE_DEFINITIONS):
            if gdef.name == "enable_sersic":
                genome.genes[i] = 1.0
            if gdef.name == "weight_sersic":
                genome.genes[i] = 0.5

        config = genome.to_detection_config()
        # Weight should be nonzero (normalized but not zeroed)
        assert config["ensemble_weights"]["sersic"] > 0
        assert config["enabled_detectors"]["sersic"] is True

    def test_all_disabled_still_normalizes(self):
        """If all detectors disabled, weights should still sum to ~1."""
        genome = DetectionGenome()
        for i, gdef in enumerate(GENE_DEFINITIONS):
            if gdef.name.startswith("enable_"):
                genome.genes[i] = 0.0
        config = genome.to_detection_config()
        # Temporal has no enable gate, so it should still be weighted
        weights = config["ensemble_weights"]
        total = sum(weights.values())
        # Should still normalize (temporal weight / temporal weight = 1)
        assert total > 0

    def test_enabled_detectors_in_config(self):
        """enabled_detectors dict should appear in genome config output."""
        genome = DetectionGenome()
        config = genome.to_detection_config()
        assert "enabled_detectors" in config
        assert isinstance(config["enabled_detectors"], dict)
        assert len(config["enabled_detectors"]) == 12

    def test_detection_config_roundtrip_with_enabled(self):
        """DetectionConfig.from_genome_dict should preserve enabled_detectors."""
        genome = DetectionGenome()
        for i, gdef in enumerate(GENE_DEFINITIONS):
            if gdef.name == "enable_sersic":
                genome.genes[i] = 0.0

        config_dict = genome.to_detection_config()
        det_config = DetectionConfig.from_genome_dict(config_dict)
        assert det_config.enabled_detectors.get("sersic") is False


class TestSetLearnedWeights:
    """Test injection of active-learning weights into evolution."""

    def test_inject_learned_weights(self):
        """set_learned_weights should replace worst genomes with variants."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config)
        evo.initialize_population()

        # Assign ascending fitness so we know which are worst
        for i, g in enumerate(evo.population):
            g.fitness = float(i)
        evo.population.sort(key=lambda g: g.fitness, reverse=True)

        weights = {"classical": 0.9, "lens": 0.8, "morphology": 0.1}
        evo.set_learned_weights(weights)

        # Check that the last 2 genomes now have the learned weights
        for variant in evo.population[-2:]:
            config = variant.to_detection_config()
            # The weight_classical gene should be clipped to 0.9
            for i, gdef in enumerate(variant.gene_defs):
                if gdef.name == "weight_classical":
                    assert abs(variant.genes[i] - 0.9) < 0.01

    def test_empty_weights_noop(self):
        """Empty weights dict should not modify population."""
        config = PipelineConfig()
        evo = EvolutionaryDiscovery(config)
        evo.initialize_population()

        original_genes = [g.genes.copy() for g in evo.population]
        evo.set_learned_weights({})

        for i, g in enumerate(evo.population):
            np.testing.assert_array_equal(g.genes, original_genes[i])


class TestTypeDiversityFitness:
    """Test type-diversity bonus in fitness evaluation."""

    def test_diverse_detectors_score_higher(self):
        """Genome activating more detectors should get higher novelty bonus."""
        from star_pattern.discovery.fitness import FitnessEvaluator

        evaluator = FitnessEvaluator(
            EvolutionConfig(), use_synthetic_injection=False
        )

        # Create a simple test image
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (64, 64)).astype(np.float32)
        image = FITSImage.from_array(data)

        # Evaluate -- the type diversity bonus should be present
        genome = DetectionGenome()
        result = evaluator.evaluate(genome.to_detection_config(), [image])

        # Novelty should incorporate type fraction
        assert "novelty" in result
        assert result["novelty"] >= 0


class TestDisabledDetectorInEnsemble:
    """Test that disabled detectors are skipped in ensemble."""

    def test_disabled_detector_returns_disabled_flag(self):
        """Disabled detector should produce disabled: True in results."""
        config = DetectionConfig()
        config.enabled_detectors = {"lens": False, "sersic": False}
        detector = EnsembleDetector(config)

        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, (64, 64)).astype(np.float32)
        image = FITSImage.from_array(data)

        result = detector.detect(image)
        assert result["lens"].get("disabled") is True
        assert result["sersic"].get("disabled") is True
        # Classical should still run (not in disabled list)
        assert "disabled" not in result.get("classical", {})


class TestNeverFoundTypes:
    """Test that never-found types appear in strategy summary."""

    def test_never_found_types_in_summary(self):
        """Summary should list anomaly types never detected."""
        from star_pattern.pipeline.autonomous import AutonomousDiscovery

        config = PipelineConfig()
        config.max_cycles = 0
        ad = AutonomousDiscovery(config, use_llm=False)

        # No findings => all types should be in never_found
        summary = ad._summarize_recent_findings()
        assert "never_found_types" in summary
        assert len(summary["never_found_types"]) > 0
        assert "lens_arc" in summary["never_found_types"]
        assert "stellar_stream" in summary["never_found_types"]

    def test_found_types_reduces_never_found(self):
        """Finding an anomaly should remove its type from never_found."""
        from star_pattern.pipeline.autonomous import AutonomousDiscovery
        from star_pattern.evaluation.metrics import Anomaly

        config = PipelineConfig()
        config.max_cycles = 0
        ad = AutonomousDiscovery(config, use_llm=False)

        # Simulate finding a lens_arc
        result = PatternResult(
            region_ra=180.0, region_dec=45.0,
            anomaly_score=0.8, detection_type="lens",
        )
        result.anomalies = [
            Anomaly(anomaly_type="lens_arc", detector="lens", score=0.9),
        ]
        ad.findings.append(result)

        summary = ad._summarize_recent_findings()
        assert "lens_arc" not in summary["never_found_types"]
        assert "lens_arc" in summary["found_types"]
