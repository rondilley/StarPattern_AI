"""Evolutionary discovery engine: GA with tournament selection.

Includes:
- Adaptive mutation: rate increases when fitness stagnates, decreases when improving
- Experience replay: persists best genomes to disk and reloads as seeds in future runs
- Synthetic injection fitness: ground-truth evaluation via injected patterns
- LLM-guided population seeding: uses StrategyResult to create informed genome variants
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from star_pattern.core.config import PipelineConfig, EvolutionConfig
from star_pattern.core.fits_handler import FITSImage
from star_pattern.discovery.genome import DetectionGenome
from star_pattern.discovery.fitness import FitnessEvaluator
from star_pattern.discovery.presets import get_preset_genomes
from star_pattern.utils.run_manager import RunManager
from star_pattern.utils.logging import get_logger

if TYPE_CHECKING:
    from star_pattern.llm.strategy import StrategyResult

logger = get_logger("discovery.evolutionary")

# Default path for experience replay storage
_EXPERIENCE_REPLAY_FILE = "experience_replay.json"
_MAX_REPLAY_GENOMES = 50


class EvolutionaryDiscovery:
    """Genetic algorithm engine for detection parameter optimization.

    Features:
    - Adaptive mutation rate that increases during fitness stagnation
    - Experience replay that persists top genomes across runs
    - Synthetic injection fitness for ground-truth evaluation
    """

    def __init__(
        self,
        config: PipelineConfig,
        run_manager: RunManager | None = None,
        images: list[FITSImage] | None = None,
        replay_path: Path | None = None,
    ):
        self.config = config
        self.evo = config.evolution
        self.run_manager = run_manager
        self.images = images or []
        self.replay_path = replay_path
        self.n_workers = getattr(config, "evolve_workers", 4)

        self.rng = np.random.default_rng()
        self.fitness_evaluator = FitnessEvaluator(self.evo)

        self.population: list[DetectionGenome] = []
        self.generation: int = 0
        self.best_genome: DetectionGenome | None = None
        self.history: list[dict[str, Any]] = []

        # Adaptive mutation state
        self._current_mutation_rate = self.evo.mutation_rate
        self._stagnation_counter = 0
        self._best_fitness_seen = -np.inf
        self._mutation_min = 0.05
        self._mutation_max = 0.5

    def initialize_population(self) -> None:
        """Create initial population with presets, replay genomes, and random."""
        self.population = []

        # Add presets
        presets = get_preset_genomes(self.rng)
        n_presets = min(len(presets), self.evo.population_size // 4)
        self.population.extend(presets[:n_presets])

        # Add experience replay genomes (best from previous runs)
        replay_genomes = self._load_replay_genomes()
        n_replay = min(len(replay_genomes), self.evo.population_size // 4)
        if replay_genomes:
            self.population.extend(replay_genomes[:n_replay])
            logger.info(f"Loaded {n_replay} genomes from experience replay")

        # Fill rest with random genomes
        n_random = 0
        while len(self.population) < self.evo.population_size:
            self.population.append(DetectionGenome(rng=self.rng))
            n_random += 1

        logger.info(
            f"Initialized population: {len(self.population)} "
            f"({n_presets} presets, {n_replay} replay, {n_random} random)"
        )

    def evaluate_population(self) -> None:
        """Evaluate fitness for all genomes.

        Uses ThreadPoolExecutor to evaluate genomes in parallel.
        Process-level parallelism is avoided due to pickling overhead
        with FITSImage objects; thread-level is sufficient because
        numpy/scipy release the GIL.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _eval_genome(genome: DetectionGenome) -> dict[str, float]:
            config = genome.to_detection_config()
            return self.fitness_evaluator.evaluate(config, self.images)

        n_workers = min(self.n_workers, len(self.population))

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_eval_genome, genome): genome
                    for genome in self.population
                }
                for future in as_completed(futures):
                    genome = futures[future]
                    try:
                        result = future.result()
                        genome.fitness = result["fitness"]
                        genome.fitness_components = result
                    except Exception as e:
                        logger.warning(f"Genome evaluation failed: {e}")
                        genome.fitness = 0.0
                        genome.fitness_components = {"fitness": 0.0}
        else:
            for genome in self.population:
                config = genome.to_detection_config()
                result = self.fitness_evaluator.evaluate(config, self.images)
                genome.fitness = result["fitness"]
                genome.fitness_components = result

        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        if self.population:
            self.best_genome = self.population[0]

    def tournament_select(self) -> DetectionGenome:
        """Tournament selection: pick best from random subset."""
        candidates = self.rng.choice(
            self.population, size=min(self.evo.tournament_size, len(self.population)), replace=False
        )
        return max(candidates, key=lambda g: g.fitness)

    def _adapt_mutation_rate(self) -> None:
        """Adapt mutation rate based on fitness progress.

        Increases mutation when fitness stagnates (explore more).
        Decreases mutation when fitness is improving (exploit good regions).
        """
        if not self.population:
            return

        current_best = self.population[0].fitness

        if current_best > self._best_fitness_seen + 0.001:
            # Improvement: reduce mutation rate (exploit)
            self._stagnation_counter = 0
            self._best_fitness_seen = current_best
            self._current_mutation_rate = max(
                self._mutation_min,
                self._current_mutation_rate * 0.9,
            )
        else:
            # Stagnation: increase mutation rate (explore)
            self._stagnation_counter += 1
            if self._stagnation_counter >= 3:
                self._current_mutation_rate = min(
                    self._mutation_max,
                    self._current_mutation_rate * 1.2,
                )

    def evolve_generation(self) -> None:
        """Produce the next generation with adaptive mutation."""
        new_population: list[DetectionGenome] = []

        # Adapt mutation rate
        self._adapt_mutation_rate()

        # Elitism: keep top genomes
        elite = self.population[: self.evo.elite_count]
        new_population.extend(elite)

        # Fill rest through selection + crossover + mutation
        while len(new_population) < self.evo.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            if self.rng.random() < self.evo.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = DetectionGenome(genes=parent1.genes.copy(), rng=self.rng)
                child2 = DetectionGenome(genes=parent2.genes.copy(), rng=self.rng)

            child1 = child1.mutate(self._current_mutation_rate)
            child2 = child2.mutate(self._current_mutation_rate)

            new_population.append(child1)
            if len(new_population) < self.evo.population_size:
                new_population.append(child2)

        self.population = new_population
        self.generation += 1

    def run(self, max_seconds: float = 600) -> DetectionGenome:
        """Run the full evolutionary search.

        Args:
            max_seconds: Maximum wall-clock time in seconds (default 600 = 10 min).
                Set to 0 or negative to disable the time limit.

        Returns:
            Best genome found.
        """
        self.initialize_population()

        run_start = time.monotonic()

        logger.info(
            f"Starting evolution: {self.evo.generations} generations, "
            f"{self.evo.population_size} population, "
            f"{len(self.images)} images, "
            f"time_limit={max_seconds:.0f}s"
        )

        completed_gens = 0
        for gen in range(self.evo.generations):
            gen_start = time.monotonic()
            self.evaluate_population()

            best = self.population[0]
            mean_fitness = np.mean([g.fitness for g in self.population])
            std_fitness = np.std([g.fitness for g in self.population])

            elapsed_total = time.monotonic() - run_start
            gen_elapsed = time.monotonic() - gen_start

            gen_stats = {
                "generation": gen,
                "best_fitness": best.fitness,
                "mean_fitness": float(mean_fitness),
                "std_fitness": float(std_fitness),
                "best_components": best.fitness_components,
                "mutation_rate": self._current_mutation_rate,
                "stagnation_count": self._stagnation_counter,
            }
            self.history.append(gen_stats)

            logger.info(
                f"Gen {gen + 1}/{self.evo.generations} "
                f"[{elapsed_total:.0f}s elapsed, gen took {gen_elapsed:.1f}s]: "
                f"best={best.fitness:.4f}, mean={mean_fitness:.4f}+/-{std_fitness:.4f}, "
                f"mut_rate={self._current_mutation_rate:.3f}"
            )

            completed_gens = gen + 1

            # Check time budget
            if max_seconds > 0 and elapsed_total > max_seconds:
                logger.info(
                    f"Evolution time limit reached ({elapsed_total:.0f}s > {max_seconds:.0f}s) "
                    f"after {completed_gens}/{self.evo.generations} generations"
                )
                break

            # Checkpoint
            if self.run_manager and (gen + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gen)

            # Evolve
            if gen < self.evo.generations - 1:
                self.evolve_generation()
        else:
            # Loop completed without break -- run final evaluation
            self.evaluate_population()

        # Save experience replay (persist best genomes for future runs)
        self._save_replay_genomes()

        if self.run_manager:
            self._save_checkpoint(completed_gens - 1)
            self.run_manager.save_result(
                "evolution_history",
                {"history": self.history, "best_genome": self.best_genome.to_dict()},
            )

        total_time = time.monotonic() - run_start
        logger.info(
            f"Evolution complete: {completed_gens}/{self.evo.generations} generations "
            f"in {total_time:.1f}s. Best fitness: {self.best_genome.fitness:.4f}"
        )
        return self.best_genome

    def set_learned_weights(self, weights: dict[str, float]) -> None:
        """Inject active-learning-derived weights into the population.

        Creates 2 genome variants with the learned weights and replaces
        the worst genomes in the population with them. This connects the
        active learning feedback loop to the evolutionary search.

        Args:
            weights: Dict of detector_name -> weight from ActiveLearner.
        """
        if not self.population or not weights:
            return

        # Create 2 variants: one from best genome, one random
        bases = [self.population[0]]
        if len(self.population) > 1:
            bases.append(DetectionGenome(rng=self.rng))

        injected = 0
        for base in bases:
            variant = DetectionGenome(genes=base.genes.copy(), rng=self.rng)
            applied = 0
            for i, gdef in enumerate(variant.gene_defs):
                if gdef.name.startswith("weight_"):
                    det_name = gdef.name[len("weight_"):]
                    if det_name in weights:
                        variant.genes[i] = gdef.clip(float(weights[det_name]))
                        applied += 1
            if applied > 0 and len(self.population) >= 2:
                # Replace worst genome
                self.population[-1 - injected] = variant
                injected += 1

        if injected > 0:
            logger.info(
                f"Injected {injected} active-learning weight variants into population"
            )

    def apply_strategy_to_population(self, strategy: StrategyResult) -> None:
        """Create genome variants from LLM strategy suggestions.

        Takes the current best genome and applies the LLM's suggested
        parameter adjustments, adding the result to the population.
        No LLM call here -- uses the already-computed StrategyResult.

        Args:
            strategy: StrategyResult from a StrategyAdvisor session.
        """
        if not strategy or not strategy.detector_adjustments:
            return

        if not self.best_genome:
            return

        # Create variant genome from best + LLM adjustments
        variant = DetectionGenome(
            genes=self.best_genome.genes.copy(), rng=self.rng
        )

        adjustments_applied = 0
        for adj in strategy.detector_adjustments:
            param = adj.get("parameter", "")
            value = adj.get("suggested")
            if param and value is not None:
                try:
                    # Find the gene index and set it
                    for i, gdef in enumerate(variant.gene_defs):
                        if gdef.name == param:
                            variant.genes[i] = gdef.clip(float(value))
                            adjustments_applied += 1
                            break
                except (ValueError, TypeError):
                    continue

        if adjustments_applied > 0 and self.population:
            # Replace worst genome in population
            self.population[-1] = variant
            logger.info(
                f"Applied {adjustments_applied} LLM strategy adjustments "
                f"to population variant"
            )

    def merge_strategy_weights(
        self, genome: DetectionGenome, strategy: StrategyResult
    ) -> None:
        """Blend evolved weights with LLM-suggested weights.

        Uses 70% evolved + 30% LLM suggestion to prevent LLM from
        dominating the optimization process.

        Args:
            genome: The genome to modify in-place.
            strategy: StrategyResult containing weight adjustments.
        """
        if not strategy or not strategy.weight_adjustments:
            return

        # Map weight names to gene indices
        weight_gene_prefix = "weight_"
        for i, gdef in enumerate(genome.gene_defs):
            if gdef.name.startswith(weight_gene_prefix):
                detector_name = gdef.name[len(weight_gene_prefix):]
                if detector_name in strategy.weight_adjustments:
                    suggested = strategy.weight_adjustments[detector_name]
                    current = genome.genes[i]
                    # 70% evolved + 30% LLM suggestion
                    blended = 0.7 * current + 0.3 * float(suggested)
                    genome.genes[i] = gdef.clip(blended)

        logger.debug("Merged LLM strategy weights with evolved genome")

    def _save_checkpoint(self, generation: int) -> None:
        """Save evolution state checkpoint."""
        if not self.run_manager:
            return

        data = {
            "generation": generation,
            "population": [g.to_dict() for g in self.population],
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "history": self.history,
            "mutation_rate": self._current_mutation_rate,
            "stagnation_counter": self._stagnation_counter,
            "best_fitness_seen": self._best_fitness_seen,
        }
        self.run_manager.save_checkpoint(f"evolution_gen{generation}", data)

    def resume_from_checkpoint(self, checkpoint_name: str) -> None:
        """Resume evolution from a checkpoint."""
        if not self.run_manager:
            raise ValueError("RunManager required for checkpoint resume")

        data = self.run_manager.load_checkpoint(checkpoint_name)
        if data is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

        self.generation = data["generation"]
        self.population = [
            DetectionGenome.from_dict(g) for g in data["population"]
        ]
        if data["best_genome"]:
            self.best_genome = DetectionGenome.from_dict(data["best_genome"])
        self.history = data.get("history", [])

        # Restore adaptive mutation state
        self._current_mutation_rate = data.get("mutation_rate", self.evo.mutation_rate)
        self._stagnation_counter = data.get("stagnation_counter", 0)
        self._best_fitness_seen = data.get("best_fitness_seen", -np.inf)

        logger.info(f"Resumed from generation {self.generation}")

    # --- Experience Replay ---

    def _get_replay_path(self) -> Path | None:
        """Get path for experience replay file."""
        if self.replay_path:
            return self.replay_path / _EXPERIENCE_REPLAY_FILE
        if self.run_manager:
            return Path(self.run_manager.run_dir).parent / _EXPERIENCE_REPLAY_FILE
        return None

    def _load_replay_genomes(self) -> list[DetectionGenome]:
        """Load best genomes from previous runs."""
        path = self._get_replay_path()
        if path is None or not path.exists():
            return []

        try:
            data = json.loads(path.read_text())
            genomes = [DetectionGenome.from_dict(g) for g in data.get("genomes", [])]
            logger.info(f"Loaded {len(genomes)} replay genomes from {path}")
            return genomes
        except Exception as e:
            logger.debug(f"Failed to load replay genomes: {e}")
            return []

    def evolve_pipelines(
        self,
        pipeline_genomes: list[Any],
        images: list[Any] | None = None,
        generations: int = 5,
    ) -> list[Any]:
        """Co-evolve a population of pipeline genomes alongside detection genomes.

        Args:
            pipeline_genomes: List of PipelineGenome objects.
            images: Images to evaluate on (uses self.images if None).
            generations: Number of pipeline evolution generations.

        Returns:
            Evolved pipeline genome population, sorted by fitness.
        """
        from star_pattern.discovery.pipeline_genome import PipelineGenome
        from star_pattern.detection.compositional import ComposedPipeline

        eval_images = images or self.images
        if not eval_images or not pipeline_genomes:
            return pipeline_genomes

        for gen in range(generations):
            # Evaluate
            for genome in pipeline_genomes:
                pipeline = ComposedPipeline(genome.to_pipeline_spec())
                scores = []
                for img in eval_images[:15]:
                    try:
                        result = pipeline.run(img)
                        scores.append(result.get("composed_score", 0))
                    except Exception:
                        scores.append(0)
                genome.fitness = float(np.mean(scores)) if scores else 0.0

            pipeline_genomes.sort(key=lambda g: g.fitness, reverse=True)

            if gen < generations - 1:
                # Evolve
                elite = pipeline_genomes[:2]
                new_pop = list(elite)
                while len(new_pop) < len(pipeline_genomes):
                    idx = self.rng.integers(min(5, len(pipeline_genomes)))
                    child = pipeline_genomes[idx].mutate(0.2)
                    new_pop.append(child)
                pipeline_genomes = new_pop

        return pipeline_genomes

    def _save_replay_genomes(self) -> None:
        """Persist top genomes for future runs."""
        path = self._get_replay_path()
        if path is None:
            return

        try:
            # Load existing replay genomes
            existing = self._load_replay_genomes()

            # Add current top genomes
            all_genomes = existing + self.population[:self.evo.elite_count]

            # Deduplicate and keep top by fitness
            all_genomes.sort(key=lambda g: g.fitness, reverse=True)

            # Keep only top N
            keep = all_genomes[:_MAX_REPLAY_GENOMES]

            data = {
                "genomes": [g.to_dict() for g in keep],
                "n_runs": len(existing) // max(self.evo.elite_count, 1) + 1,
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(keep)} replay genomes to {path}")

        except Exception as e:
            logger.debug(f"Failed to save replay genomes: {e}")
