"""Tests for the LLM strategy advisor."""

import json

import pytest

from star_pattern.llm.strategy import StrategyAdvisor, StrategyResult
from star_pattern.llm.token_tracker import TokenTracker, estimate_tokens
from star_pattern.llm.cache import LLMCache
from star_pattern.llm.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    """Fake LLM provider for testing."""

    def __init__(self, response: str = ""):
        self._response = response

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(
        self, prompt, system_prompt=None, max_tokens=2048, temperature=0.7
    ) -> str:
        return self._response


class TestStrategyResult:
    def test_empty_result(self):
        result = StrategyResult()
        assert result.detector_adjustments == []
        assert result.weight_adjustments == {}
        assert result.focus_regions == []
        assert result.detection_strategy == ""
        assert result.token_cost == 0

    def test_to_dict(self):
        result = StrategyResult(
            detector_adjustments=[{"parameter": "lens_snr_threshold", "suggested": 3.0}],
            weight_adjustments={"lens": 0.25},
            detection_strategy="Focus on lens detection",
            stop_doing="Low-quality regions",
            token_cost=1000,
            strategy_id=1,
        )
        d = result.to_dict()
        assert d["strategy_id"] == 1
        assert d["token_cost"] == 1000
        assert len(d["detector_adjustments"]) == 1


class TestStrategyAdvisor:
    def test_build_summary_compact(self):
        """Summary should be under 600 tokens."""
        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        findings = {
            "n_total": 12,
            "n_high_confidence": 3,
            "n_artifacts": 5,
            "type_counts": {"lens": 4, "morphology": 3, "kinematic": 2},
            "n_regions": 25,
        }
        genome = {
            "ensemble_weights": {
                "lens": 0.20, "morphology": 0.15, "anomaly": 0.12,
            },
        }
        al_stats = {
            "n_total": 8,
            "interesting_rate": 0.62,
        }
        evo_history = [{"fitness": 0.72, "stagnation_count": 3}]

        summary = advisor._build_summary(
            findings, genome, al_stats, evo_history
        )

        # Check content
        assert "DETECTIONS" in summary
        assert "12 total" in summary
        assert "TOP TYPES" in summary
        assert "ACTIVE LEARNING" in summary
        assert "EVOLUTION" in summary
        assert "CURRENT WEIGHTS" in summary
        assert "REGIONS SEARCHED" in summary

        # Check size (under ~600 tokens)
        tokens = estimate_tokens(summary)
        assert tokens < 600

    def test_strategy_result_parsing(self):
        """JSON response is parsed correctly."""
        response_json = json.dumps({
            "detector_adjustments": [
                {"parameter": "lens_snr_threshold", "current": 2.0, "suggested": 3.0, "reason": "test"}
            ],
            "weight_adjustments": {"lens": 0.25, "morphology": 0.15},
            "focus_regions": [{"ra": 180, "dec": 45, "reason": "cluster field"}],
            "detection_strategy": "Prioritize lens detection",
            "stop_doing": "Low galactic latitude regions",
        })

        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        result = advisor._parse_strategy_response(response_json)

        assert len(result.detector_adjustments) == 1
        assert result.detector_adjustments[0]["parameter"] == "lens_snr_threshold"
        assert result.weight_adjustments["lens"] == 0.25
        assert len(result.focus_regions) == 1
        assert result.focus_regions[0].ra == 180
        assert "lens" in result.detection_strategy.lower()

    def test_strategy_result_parsing_with_markdown(self):
        """JSON wrapped in markdown code block is parsed."""
        response = '```json\n{"detector_adjustments": [], "weight_adjustments": {}, "focus_regions": [], "detection_strategy": "test", "stop_doing": "nothing"}\n```'

        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        result = advisor._parse_strategy_response(response)
        assert result.detection_strategy == "test"

    def test_outcome_tracking(self):
        """Pre/post metrics are recorded and compared."""
        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        # Simulate a strategy result
        strategy = StrategyResult(strategy_id=0)
        strategy.pre_metrics = {
            "n_findings": 10,
            "n_high_confidence": 3,
            "interesting_rate": 0.5,
        }
        advisor._strategy_history.append(strategy)

        # Record outcome
        outcome = advisor.record_outcome(0, {
            "n_total": 15,
            "n_high_confidence": 5,
            "interesting_rate": 0.6,
        })

        assert outcome is not None
        assert outcome["findings_delta"] == 5
        assert outcome["high_confidence_delta"] == 2
        assert outcome["improved"] is True

    def test_outcome_tracking_no_improvement(self):
        """Correctly identifies when strategy did not improve results."""
        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        strategy = StrategyResult(strategy_id=0)
        strategy.pre_metrics = {
            "n_findings": 10,
            "n_high_confidence": 5,
            "interesting_rate": 0.7,
        }
        advisor._strategy_history.append(strategy)

        outcome = advisor.record_outcome(0, {
            "n_total": 12,
            "n_high_confidence": 4,
            "interesting_rate": 0.5,
        })

        assert outcome is not None
        assert outcome["improved"] is False

    def test_review_session_with_fake_provider(self):
        """Full review session with a fake provider."""
        response_json = json.dumps({
            "detector_adjustments": [],
            "weight_adjustments": {},
            "focus_regions": [],
            "detection_strategy": "Focus on morphology",
            "stop_doing": "Nothing",
        })

        provider = FakeProvider(response=response_json)
        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([provider], tracker)

        result = advisor.review_session(
            findings_summary={"n_total": 5, "n_high_confidence": 1, "n_regions": 10},
            current_genome={"ensemble_weights": {}},
            active_learning_stats={"n_total": 3, "interesting_rate": 0.5},
            evolution_history=[],
        )

        assert result.detection_strategy == "Focus on morphology"
        assert result.strategy_id == 0
        assert result.timestamp != ""

    def test_review_session_budget_check(self):
        """Session skipped when budget is insufficient."""
        provider = FakeProvider(response="{}")
        tracker = TokenTracker(budget_tokens=100)  # Very small budget
        tracker.record("test", "test", 50, 50)  # Use most budget

        advisor = StrategyAdvisor([provider], tracker)
        result = advisor.review_session(
            findings_summary={},
            current_genome={},
            active_learning_stats={},
            evolution_history=[],
        )

        # Should return empty result (budget too low)
        assert result.detection_strategy == ""

    def test_cache_hit(self, tmp_path):
        """Identical summaries return cached response."""
        response_json = json.dumps({
            "detector_adjustments": [],
            "weight_adjustments": {},
            "focus_regions": [],
            "detection_strategy": "cached result",
            "stop_doing": "",
        })

        provider = FakeProvider(response=response_json)
        tracker = TokenTracker(budget_tokens=100_000)
        cache = LLMCache(tmp_path / "cache")
        advisor = StrategyAdvisor([provider], tracker, cache=cache)

        inputs = dict(
            findings_summary={"n_total": 5, "n_high_confidence": 1, "n_regions": 10},
            current_genome={"ensemble_weights": {}},
            active_learning_stats={"n_total": 3, "interesting_rate": 0.5},
            evolution_history=[],
        )

        # First call
        result1 = advisor.review_session(**inputs)
        calls_after_first = len(tracker.calls)

        # Second call (should use cache)
        result2 = advisor.review_session(**inputs)
        calls_after_second = len(tracker.calls)

        assert result1.detection_strategy == result2.detection_strategy
        # Second call should have recorded a cached call
        assert calls_after_second > calls_after_first

    def test_batch_review_parsing(self):
        """Batch review response is parsed correctly."""
        response = json.dumps([
            {"verdict": "real", "classification": "lens", "brief_hypothesis": "Arc morphology"},
            {"verdict": "artifact", "classification": "PSF", "brief_hypothesis": "Diffraction spike"},
        ])

        tracker = TokenTracker(budget_tokens=100_000)
        advisor = StrategyAdvisor([], tracker)

        reviews = advisor._parse_batch_review(response, 2)
        assert len(reviews) == 2
        assert reviews[0]["verdict"] == "real"
        assert reviews[1]["verdict"] == "artifact"
