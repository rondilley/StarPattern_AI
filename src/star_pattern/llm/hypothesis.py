"""Hypothesis generation from pattern detection results."""

from __future__ import annotations

import json
import re
from typing import Any

from star_pattern.core.config import LLMConfig
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.prompts import SYSTEM_ASTRONOMER, HYPOTHESIS_PROMPT
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.hypothesis")


def _sanitize_external_text(text: str, max_length: int = 500) -> str:
    """Sanitize text from external catalogs before injecting into LLM prompts.

    Strips control characters, prompt injection patterns, and enforces
    length limits on strings sourced from SIMBAD/NED/TNS cross-matches.
    """
    # Remove control characters (keep newlines and tabs for readability)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Strip common prompt injection delimiters
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL)
    text = re.sub(r"</?system>", "", text, flags=re.IGNORECASE)
    # Truncate
    return text[:max_length]


class HypothesisGenerator:
    """Generate physical hypotheses for detected patterns using LLMs."""

    def __init__(self, provider: LLMProvider, config: LLMConfig | None = None):
        self.provider = provider
        self.config = config or LLMConfig()

    def generate(self, pattern_data: dict[str, Any]) -> dict[str, Any]:
        """Generate a hypothesis for a pattern detection result.

        Args:
            pattern_data: Detection result dict with ra, dec, type, scores, details.

        Returns:
            Dict with hypothesis, mechanism, confidence, follow-ups.
        """
        # Format detection details
        details = json.dumps(pattern_data.get("details", {}), indent=2, default=str)
        cross_refs = json.dumps(pattern_data.get("cross_matches", []), indent=2, default=str)

        # Sanitize external catalog data before prompt injection
        cross_refs = _sanitize_external_text(cross_refs, max_length=500)
        details = _sanitize_external_text(details, max_length=2000)

        prompt = HYPOTHESIS_PROMPT.format(
            ra=pattern_data.get("ra", 0),
            dec=pattern_data.get("dec", 0),
            detection_type=pattern_data.get("type", "unknown"),
            anomaly_score=pattern_data.get("anomaly_score", 0),
            significance=pattern_data.get("significance", 0),
            details=details,
            cross_refs=cross_refs,
        )

        logger.info(f"Generating hypothesis using {self.provider.name}")

        result = self.provider.generate_structured(
            prompt=prompt,
            system_prompt=SYSTEM_ASTRONOMER,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Ensure required fields
        if "text" in result and "hypothesis" not in result:
            result = {
                "hypothesis": result["text"],
                "physical_mechanism": "See hypothesis text",
                "confidence": 0.5,
                "classification": "unknown",
                "follow_up_observations": [],
            }

        result["provider"] = self.provider.name
        result["model"] = self.provider.model_name

        logger.info(
            f"Hypothesis generated: {result.get('classification', 'unknown')} "
            f"(confidence={result.get('confidence', 0):.2f})"
        )
        return result

    def generate_batch(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate hypotheses for multiple patterns."""
        results = []
        for i, pattern in enumerate(patterns):
            logger.info(f"Generating hypothesis {i + 1}/{len(patterns)}")
            try:
                result = self.generate(pattern)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed: {e}")
                results.append({"error": str(e), "pattern": pattern})
        return results
