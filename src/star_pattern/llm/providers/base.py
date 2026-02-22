"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from star_pattern.llm.token_tracker import TokenTracker
    from star_pattern.llm.cache import LLMCache


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'claude', 'gemini')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum response length.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        ...

    def generate_tracked(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        tracker: TokenTracker | None = None,
        purpose: str = "general",
    ) -> str:
        """Generate a response with token tracking and budget enforcement.

        Args:
            prompt: User prompt.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum response length.
            temperature: Sampling temperature.
            tracker: Optional TokenTracker for budget enforcement.
            purpose: Call purpose label for tracking.

        Returns:
            Generated text response.

        Raises:
            TokenBudgetExceeded: If the call would exceed the token budget.
        """
        from star_pattern.llm.token_tracker import estimate_tokens

        # Pre-check budget
        if tracker is not None:
            estimated_input = estimate_tokens(prompt)
            if system_prompt:
                estimated_input += estimate_tokens(system_prompt)
            estimated_total = estimated_input + max_tokens
            tracker.require_budget(estimated_total)

        # Make the actual call
        response = self.generate(prompt, system_prompt, max_tokens, temperature)

        # Record actual usage
        if tracker is not None:
            input_tokens = estimate_tokens(prompt)
            if system_prompt:
                input_tokens += estimate_tokens(system_prompt)
            output_tokens = estimate_tokens(response)
            tracker.record(
                provider=self.name,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return response

    def generate_cached(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        cache: LLMCache | None = None,
        tracker: TokenTracker | None = None,
        purpose: str = "general",
    ) -> str:
        """Generate with cache check, tracking, and budget enforcement.

        Checks the cache first. On a miss, calls generate_tracked()
        and stores the result.

        Args:
            prompt: User prompt.
            system_prompt: Optional system-level instructions.
            max_tokens: Maximum response length.
            temperature: Sampling temperature.
            cache: Optional LLMCache for response caching.
            tracker: Optional TokenTracker for budget enforcement.
            purpose: Call purpose label for tracking.

        Returns:
            Generated text response (from cache or fresh).
        """
        # Check cache
        if cache is not None:
            key = cache.hash_prompt(prompt, system_prompt or "")
            cached = cache.get(key)
            if cached is not None:
                if tracker is not None:
                    tracker.record(
                        provider=self.name,
                        purpose=purpose,
                        input_tokens=0,
                        output_tokens=0,
                        cached=True,
                    )
                return cached

        # Cache miss: generate with tracking
        response = self.generate_tracked(
            prompt, system_prompt, max_tokens, temperature,
            tracker=tracker, purpose=purpose,
        )

        # Store in cache
        if cache is not None:
            key = cache.hash_prompt(prompt, system_prompt or "")
            cache.put(key, response, {"provider": self.name, "purpose": purpose})

        return response

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate structured (JSON) output. Fallback: parse text response."""
        import json

        response = self.generate(prompt, system_prompt, max_tokens, temperature)

        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                return {"text": response}

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return {"text": response}

    def is_available(self) -> bool:
        """Check if this provider is usable."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"
