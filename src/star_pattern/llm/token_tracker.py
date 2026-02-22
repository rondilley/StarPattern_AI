"""Token tracking, budgeting, and cost accounting for LLM calls."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("llm.token_tracker")


class TokenBudgetExceeded(Exception):
    """Raised when an LLM call would exceed the token budget."""


@dataclass
class LLMCall:
    """Record of a single LLM call."""

    timestamp: str
    provider: str
    purpose: str  # "strategy", "evaluation", "guidance", "cached"
    input_tokens: int
    output_tokens: int
    cached: bool


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using char/4 approximation.

    This is a rough estimate suitable for budget pre-checks.
    Actual usage is recorded from provider responses.
    """
    return max(1, len(text) // 4)


class TokenTracker:
    """Track and budget LLM token usage across a session.

    Records every LLM call with provider, purpose, and token counts.
    Enforces a per-session budget and provides usage summaries.
    """

    def __init__(
        self,
        budget_tokens: int = 500_000,
        session_id: str | None = None,
    ):
        self.budget_tokens = budget_tokens
        self.session_id = session_id or str(int(time.time()))
        self.calls: list[LLMCall] = []
        self._total_input = 0
        self._total_output = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self._total_input + self._total_output

    def record(
        self,
        provider: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> None:
        """Record an LLM call.

        Args:
            provider: Provider name (e.g., 'openai', 'claude').
            purpose: Call purpose (e.g., 'strategy', 'evaluation').
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
            cached: Whether this was a cache hit.
        """
        call = LLMCall(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            provider=provider,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached=cached,
        )
        self.calls.append(call)

        if not cached:
            self._total_input += input_tokens
            self._total_output += output_tokens

        logger.debug(
            f"Token usage: {provider}/{purpose} "
            f"in={input_tokens} out={output_tokens} "
            f"total={self.total_tokens}/{self.budget_tokens}"
        )

    def remaining(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.budget_tokens - self.total_tokens)

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if budget allows a call of this size.

        Args:
            estimated_tokens: Estimated total tokens for the call.

        Returns:
            True if the call fits within the remaining budget.
        """
        return self.total_tokens + estimated_tokens <= self.budget_tokens

    def require_budget(self, estimated_tokens: int) -> None:
        """Raise TokenBudgetExceeded if budget is insufficient.

        Args:
            estimated_tokens: Estimated total tokens for the call.

        Raises:
            TokenBudgetExceeded: If the call would exceed the budget.
        """
        if not self.can_afford(estimated_tokens):
            raise TokenBudgetExceeded(
                f"Budget exceeded: need ~{estimated_tokens} tokens, "
                f"only {self.remaining()} remaining "
                f"({self.total_tokens}/{self.budget_tokens} used)"
            )

    def summary(self) -> dict[str, Any]:
        """Usage breakdown by purpose and provider.

        Returns:
            Dict with total counts, per-purpose breakdown, and
            per-provider breakdown.
        """
        by_purpose: dict[str, dict[str, int]] = {}
        by_provider: dict[str, dict[str, int]] = {}
        n_cached = 0

        for call in self.calls:
            total = call.input_tokens + call.output_tokens

            # By purpose
            if call.purpose not in by_purpose:
                by_purpose[call.purpose] = {"calls": 0, "tokens": 0}
            by_purpose[call.purpose]["calls"] += 1
            by_purpose[call.purpose]["tokens"] += total

            # By provider
            if call.provider not in by_provider:
                by_provider[call.provider] = {"calls": 0, "tokens": 0}
            by_provider[call.provider]["calls"] += 1
            by_provider[call.provider]["tokens"] += total

            if call.cached:
                n_cached += 1

        return {
            "session_id": self.session_id,
            "total_calls": len(self.calls),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "remaining_tokens": self.remaining(),
            "cached_calls": n_cached,
            "by_purpose": by_purpose,
            "by_provider": by_provider,
        }

    def save(self, path: Path) -> None:
        """Persist usage log to JSON.

        Args:
            path: File path for the JSON output.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.summary(),
            "calls": [
                {
                    "timestamp": c.timestamp,
                    "provider": c.provider,
                    "purpose": c.purpose,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cached": c.cached,
                }
                for c in self.calls
            ],
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Token usage saved to {path}")

    @classmethod
    def load(cls, path: Path) -> TokenTracker:
        """Load a TokenTracker from a saved JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed TokenTracker with recorded calls.
        """
        data = json.loads(path.read_text())
        summary = data.get("summary", {})

        tracker = cls(
            budget_tokens=summary.get("budget_tokens", 500_000),
            session_id=summary.get("session_id"),
        )

        for call_data in data.get("calls", []):
            call = LLMCall(
                timestamp=call_data["timestamp"],
                provider=call_data["provider"],
                purpose=call_data["purpose"],
                input_tokens=call_data["input_tokens"],
                output_tokens=call_data["output_tokens"],
                cached=call_data.get("cached", False),
            )
            tracker.calls.append(call)
            if not call.cached:
                tracker._total_input += call.input_tokens
                tracker._total_output += call.output_tokens

        return tracker
