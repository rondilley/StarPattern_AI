"""Tests for the LLM token tracking and budget system."""

import json
import time

import pytest

from star_pattern.llm.token_tracker import (
    TokenTracker,
    TokenBudgetExceeded,
    LLMCall,
    estimate_tokens,
)


class TestEstimateTokens:
    def test_basic_estimate(self):
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # Minimum 1

    def test_short_string(self):
        assert estimate_tokens("hi") >= 1

    def test_realistic_text(self):
        text = "This is a prompt with several words and sentences."
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)


class TestTokenTracker:
    def test_initial_state(self):
        tracker = TokenTracker(budget_tokens=10_000)
        assert tracker.total_tokens == 0
        assert tracker.remaining() == 10_000
        assert len(tracker.calls) == 0

    def test_record_and_summary(self):
        tracker = TokenTracker(budget_tokens=100_000)
        tracker.record("openai", "strategy", 500, 300)
        tracker.record("claude", "evaluation", 200, 100)

        assert tracker.total_tokens == 1100
        assert len(tracker.calls) == 2

        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["total_input_tokens"] == 700
        assert summary["total_output_tokens"] == 400
        assert summary["total_tokens"] == 1100

        assert "strategy" in summary["by_purpose"]
        assert summary["by_purpose"]["strategy"]["calls"] == 1
        assert summary["by_purpose"]["strategy"]["tokens"] == 800

        assert "openai" in summary["by_provider"]
        assert "claude" in summary["by_provider"]

    def test_budget_enforcement(self):
        tracker = TokenTracker(budget_tokens=1000)
        tracker.record("openai", "test", 500, 400)

        # Should still have room
        assert tracker.can_afford(100)
        assert tracker.remaining() == 100

        # Should not afford 200
        assert not tracker.can_afford(200)

        # require_budget should raise
        with pytest.raises(TokenBudgetExceeded):
            tracker.require_budget(200)

    def test_can_afford_exact_boundary(self):
        tracker = TokenTracker(budget_tokens=1000)
        tracker.record("openai", "test", 500, 400)

        # Exactly at boundary
        assert tracker.can_afford(100)
        assert not tracker.can_afford(101)

    def test_cached_calls_not_counted(self):
        tracker = TokenTracker(budget_tokens=10_000)
        tracker.record("openai", "strategy", 500, 300, cached=True)

        # Cached calls don't count toward budget
        assert tracker.total_tokens == 0
        assert tracker.remaining() == 10_000
        assert len(tracker.calls) == 1

        summary = tracker.summary()
        assert summary["cached_calls"] == 1

    def test_save_and_load(self, tmp_path):
        tracker = TokenTracker(budget_tokens=50_000, session_id="test-session")
        tracker.record("openai", "strategy", 500, 300)
        tracker.record("claude", "evaluation", 200, 100, cached=True)

        save_path = tmp_path / "token_usage.json"
        tracker.save(save_path)

        assert save_path.exists()

        # Verify JSON content
        data = json.loads(save_path.read_text())
        assert data["summary"]["total_calls"] == 2
        assert data["summary"]["session_id"] == "test-session"
        assert len(data["calls"]) == 2

        # Load and verify
        loaded = TokenTracker.load(save_path)
        assert loaded.total_tokens == tracker.total_tokens
        assert loaded.budget_tokens == 50_000
        assert len(loaded.calls) == 2

    def test_remaining_never_negative(self):
        tracker = TokenTracker(budget_tokens=100)
        tracker.record("openai", "test", 200, 200)
        assert tracker.remaining() == 0

    def test_multiple_purposes_tracked(self):
        tracker = TokenTracker(budget_tokens=100_000)
        tracker.record("openai", "strategy", 500, 300)
        tracker.record("openai", "batch_review", 400, 200)
        tracker.record("openai", "strategy", 300, 200)

        summary = tracker.summary()
        assert summary["by_purpose"]["strategy"]["calls"] == 2
        assert summary["by_purpose"]["batch_review"]["calls"] == 1
