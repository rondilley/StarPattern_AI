"""Tests for create_evolution_summary visualization."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from star_pattern.visualization.pattern_overlay import create_evolution_summary


def _single_gen_data(n_cycles: int = 5):
    """Build test data where each cycle has exactly 1 generation."""
    generation_histories = []
    evolution_history = []
    for c in range(n_cycles):
        generation_histories.append([{
            "generation": 0,
            "best_fitness": 0.3 + c * 0.05,
            "mean_fitness": 0.2 + c * 0.04,
            "mutation_rate": 0.15 - c * 0.01,
        }])
        evolution_history.append({
            "cycle": c,
            "fitness": 0.3 + c * 0.05,
            "components": {
                "anomaly": 0.1, "significance": 0.08,
                "novelty": 0.05, "diversity": 0.03, "recovery": 0.04,
            },
        })
    return generation_histories, evolution_history


def _multi_gen_data(n_cycles: int = 3, n_gens: int = 5):
    """Build test data where each cycle has multiple generations."""
    generation_histories = []
    evolution_history = []
    for c in range(n_cycles):
        gens = []
        for g in range(n_gens):
            gens.append({
                "generation": g,
                "best_fitness": 0.2 + c * 0.05 + g * 0.02,
                "mean_fitness": 0.15 + c * 0.04 + g * 0.01,
                "mutation_rate": 0.15 - g * 0.01,
            })
        generation_histories.append(gens)
        evolution_history.append({
            "cycle": c,
            "fitness": gens[-1]["best_fitness"],
            "components": {
                "anomaly": 0.1, "significance": 0.08,
                "novelty": 0.05, "diversity": 0.03, "recovery": 0.04,
            },
        })
    return generation_histories, evolution_history


class TestEvolutionSummarySingleGen:
    """Tests for the single-generation-per-cycle fallback view."""

    def test_creates_figure(self):
        """Single-gen data produces a valid Figure with 4 axes."""
        gen_hist, evo_hist = _single_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_switches_to_per_cycle_titles(self):
        """Left panels should switch to per-cycle titles."""
        gen_hist, evo_hist = _single_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        assert "per-cycle" in axes[0].get_title().lower() or "1 gen/cycle" in axes[0].get_title()
        assert "1 gen/cycle" in axes[2].get_title()
        plt.close(fig)

    def test_axes_have_data(self):
        """All four panels should have plotted content (lines or bars)."""
        gen_hist, evo_hist = _single_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        # [0,0] per-cycle fitness: should have lines
        assert len(axes[0].get_lines()) > 0
        # [0,1] best fitness trend: should have lines
        assert len(axes[1].get_lines()) > 0
        # [1,0] mutation rate: should have lines
        assert len(axes[2].get_lines()) > 0
        # [1,1] component breakdown: should have patches (bars)
        assert len(axes[3].patches) > 0
        plt.close(fig)

    def test_xlabel_is_pipeline_cycle(self):
        """Left panels should use 'Pipeline cycle' as x-label."""
        gen_hist, evo_hist = _single_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        assert "cycle" in axes[0].get_xlabel().lower()
        assert "cycle" in axes[2].get_xlabel().lower()
        plt.close(fig)


class TestEvolutionSummaryMultiGen:
    """Tests for the standard multi-generation view."""

    def test_creates_figure(self):
        """Multi-gen data produces a valid Figure with 4 axes."""
        gen_hist, evo_hist = _multi_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_keeps_per_generation_title(self):
        """Left panels should keep original per-generation titles."""
        gen_hist, evo_hist = _multi_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        assert "Per-generation" in axes[0].get_title()
        assert "Mutation rate adaptation" in axes[2].get_title()
        plt.close(fig)

    def test_axes_have_data(self):
        """All four panels should have plotted content."""
        gen_hist, evo_hist = _multi_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        assert len(axes[0].get_lines()) > 0
        assert len(axes[1].get_lines()) > 0
        assert len(axes[2].get_lines()) > 0
        assert len(axes[3].patches) > 0
        plt.close(fig)

    def test_markers_present(self):
        """Line plots should have markers (non-None marker style)."""
        gen_hist, evo_hist = _multi_gen_data()
        fig = create_evolution_summary(gen_hist, evo_hist)
        axes = fig.get_axes()
        # Check [0,0] lines have markers
        for line in axes[0].get_lines():
            marker = line.get_marker()
            assert marker is not None and marker != "None" and marker != ""
        plt.close(fig)


class TestEvolutionSummaryEdgeCases:
    """Edge cases."""

    def test_empty_histories(self):
        """Empty inputs should not crash."""
        fig = create_evolution_summary([], [])
        assert fig is not None
        plt.close(fig)

    def test_mixed_empty_cycles(self):
        """Some empty cycles should still work."""
        gen_hist, evo_hist = _single_gen_data(3)
        gen_hist.insert(1, [])  # Insert empty cycle
        evo_hist.insert(1, {"cycle": 99, "fitness": 0})
        fig = create_evolution_summary(gen_hist, evo_hist)
        assert fig is not None
        assert len(fig.get_axes()) == 4
        plt.close(fig)

    def test_no_mutation_rate_data(self):
        """Missing mutation_rate keys should show fallback text."""
        gen_hist = [[{"generation": 0, "best_fitness": 0.3, "mean_fitness": 0.2}]]
        evo_hist = [{"cycle": 0, "fitness": 0.3, "components": {}}]
        fig = create_evolution_summary(gen_hist, evo_hist)
        assert fig is not None
        plt.close(fig)
