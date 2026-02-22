"""Variable-length genome for composed detection pipelines.

Evolves the structure and parameters of detection pipelines
(2-5 operations). Separate from DetectionGenome.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from star_pattern.detection.compositional import (
    OperationSpec,
    PipelineSpec,
    ALL_OPERATIONS,
    OPERATION_PARAMS,
    _CHOICE_MAPS,
)
from star_pattern.utils.logging import get_logger

logger = get_logger("discovery.pipeline_genome")

_SCORE_METHODS = ["component_count", "max_residual", "area_fraction"]


class PipelineGenome:
    """Evolvable genome for composed detection pipelines.

    Variable-length (2-5 operations) with both structural and parametric mutation.
    """

    def __init__(
        self,
        operations: list[OperationSpec] | None = None,
        score_method: str = "component_count",
        rng: np.random.Generator | None = None,
        min_ops: int = 2,
        max_ops: int = 5,
    ):
        self.rng = rng or np.random.default_rng()
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.score_method = score_method
        self.fitness: float = 0.0

        if operations is not None:
            self.operations = list(operations)
        else:
            self.operations = []

    @classmethod
    def random(
        cls,
        rng: np.random.Generator | None = None,
        min_ops: int = 2,
        max_ops: int = 5,
    ) -> PipelineGenome:
        """Create a random pipeline genome."""
        rng = rng or np.random.default_rng()
        n_ops = rng.integers(min_ops, max_ops + 1)

        operations = []
        for _ in range(n_ops):
            op_name = rng.choice(ALL_OPERATIONS)
            params = _random_params(op_name, rng)
            operations.append(OperationSpec(name=op_name, params=params))

        score_method = rng.choice(_SCORE_METHODS)

        return cls(
            operations=operations,
            score_method=score_method,
            rng=rng,
            min_ops=min_ops,
            max_ops=max_ops,
        )

    def to_pipeline_spec(self) -> PipelineSpec:
        """Convert to PipelineSpec for execution."""
        return PipelineSpec(
            operations=list(self.operations),
            score_method=self.score_method,
        )

    def mutate(self, rate: float = 0.2) -> PipelineGenome:
        """Create a mutated copy with structural and parametric mutations.

        Structural: swap, add, or remove operations.
        Parametric: modify operation parameters.
        """
        new_ops = [
            OperationSpec(name=op.name, params=dict(op.params))
            for op in self.operations
        ]
        new_score = self.score_method

        # Structural mutation: swap an operation
        if self.rng.random() < rate and len(new_ops) >= 1:
            idx = self.rng.integers(len(new_ops))
            op_name = self.rng.choice(ALL_OPERATIONS)
            new_ops[idx] = OperationSpec(
                name=op_name, params=_random_params(op_name, self.rng)
            )

        # Structural mutation: add an operation
        if self.rng.random() < rate * 0.5 and len(new_ops) < self.max_ops:
            op_name = self.rng.choice(ALL_OPERATIONS)
            pos = self.rng.integers(len(new_ops) + 1)
            new_ops.insert(
                pos, OperationSpec(name=op_name, params=_random_params(op_name, self.rng))
            )

        # Structural mutation: remove an operation
        if self.rng.random() < rate * 0.3 and len(new_ops) > self.min_ops:
            idx = self.rng.integers(len(new_ops))
            new_ops.pop(idx)

        # Parametric mutation: tweak params of existing ops
        for op in new_ops:
            if self.rng.random() < rate:
                _mutate_params(op, self.rng)

        # Score method mutation
        if self.rng.random() < rate * 0.3:
            new_score = self.rng.choice(_SCORE_METHODS)

        child = PipelineGenome(
            operations=new_ops,
            score_method=new_score,
            rng=self.rng,
            min_ops=self.min_ops,
            max_ops=self.max_ops,
        )
        return child

    def crossover(
        self, other: PipelineGenome
    ) -> tuple[PipelineGenome, PipelineGenome]:
        """Single-point crossover aligned by position.

        Swaps a contiguous segment between two genomes.
        """
        ops1 = [
            OperationSpec(name=op.name, params=dict(op.params))
            for op in self.operations
        ]
        ops2 = [
            OperationSpec(name=op.name, params=dict(op.params))
            for op in other.operations
        ]

        # Crossover point
        min_len = min(len(ops1), len(ops2))
        if min_len <= 1:
            return (
                PipelineGenome(ops1, self.score_method, self.rng, self.min_ops, self.max_ops),
                PipelineGenome(ops2, other.score_method, self.rng, self.min_ops, self.max_ops),
            )

        pt = self.rng.integers(1, min_len)

        child1_ops = ops1[:pt] + ops2[pt:]
        child2_ops = ops2[:pt] + ops1[pt:]

        # Enforce length constraints
        child1_ops = child1_ops[: self.max_ops]
        child2_ops = child2_ops[: self.max_ops]
        while len(child1_ops) < self.min_ops:
            child1_ops.append(
                OperationSpec(
                    name=self.rng.choice(ALL_OPERATIONS),
                    params=_random_params(self.rng.choice(ALL_OPERATIONS), self.rng),
                )
            )
        while len(child2_ops) < self.min_ops:
            child2_ops.append(
                OperationSpec(
                    name=self.rng.choice(ALL_OPERATIONS),
                    params=_random_params(self.rng.choice(ALL_OPERATIONS), self.rng),
                )
            )

        return (
            PipelineGenome(child1_ops, self.score_method, self.rng, self.min_ops, self.max_ops),
            PipelineGenome(child2_ops, other.score_method, self.rng, self.min_ops, self.max_ops),
        )

    def describe(self) -> str:
        """Human-readable description of the pipeline."""
        parts = []
        for op in self.operations:
            if op.params:
                param_str = ", ".join(f"{k}={v}" for k, v in op.params.items())
                parts.append(f"{op.name}({param_str})")
            else:
                parts.append(op.name)
        parts.append(f"score:{self.score_method}")
        return " -> ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "operations": [
                {"name": op.name, "params": op.params}
                for op in self.operations
            ],
            "score_method": self.score_method,
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], rng: np.random.Generator | None = None
    ) -> PipelineGenome:
        """Deserialize from dict."""
        operations = [
            OperationSpec(name=op["name"], params=op.get("params", {}))
            for op in d.get("operations", [])
        ]
        genome = cls(
            operations=operations,
            score_method=d.get("score_method", "component_count"),
            rng=rng,
        )
        genome.fitness = d.get("fitness", 0.0)
        return genome

    def __repr__(self) -> str:
        return (
            f"PipelineGenome(ops={len(self.operations)}, "
            f"fitness={self.fitness:.4f}, "
            f"desc={self.describe()!r})"
        )


def _random_params(
    op_name: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Generate random parameters for an operation."""
    param_defs = OPERATION_PARAMS.get(op_name, {})
    params: dict[str, Any] = {}

    for param_name, (lo, hi, dtype) in param_defs.items():
        choice_key = f"{op_name}.{param_name}"
        if dtype == "choice":
            choices = _CHOICE_MAPS.get(choice_key, [])
            if choices:
                params[param_name] = rng.choice(choices)
            else:
                params[param_name] = int(rng.integers(int(lo), int(hi) + 1))
        elif dtype == "int":
            params[param_name] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            params[param_name] = float(rng.uniform(lo, hi))

    return params


def _mutate_params(op: OperationSpec, rng: np.random.Generator) -> None:
    """Mutate parameters of an operation in-place."""
    param_defs = OPERATION_PARAMS.get(op.name, {})

    for param_name, (lo, hi, dtype) in param_defs.items():
        if rng.random() < 0.5:
            continue

        choice_key = f"{op.name}.{param_name}"
        if dtype == "choice":
            choices = _CHOICE_MAPS.get(choice_key, [])
            if choices:
                op.params[param_name] = rng.choice(choices)
        elif dtype == "int":
            current = op.params.get(param_name, (lo + hi) / 2)
            delta = rng.integers(-2, 3)
            op.params[param_name] = int(np.clip(float(current) + delta, lo, hi))
        else:
            current = op.params.get(param_name, (lo + hi) / 2)
            delta = rng.normal(0, (hi - lo) * 0.1)
            op.params[param_name] = float(np.clip(float(current) + delta, lo, hi))
