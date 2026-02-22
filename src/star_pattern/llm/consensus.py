"""Multi-LLM consensus scoring using Borda count."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from star_pattern.core.config import LLMConfig
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.prompts import SYSTEM_ASTRONOMER, CONSENSUS_PROMPT
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.consensus")


class PatternConsensus:
    """Multi-LLM significance rating with Borda count aggregation."""

    def __init__(
        self,
        providers: list[LLMProvider],
        config: LLMConfig | None = None,
    ):
        self.providers = providers
        self.config = config or LLMConfig()

    def rate(self, pattern_data: dict[str, Any], hypothesis: str = "") -> dict[str, Any]:
        """Get consensus significance rating from multiple LLMs.

        Args:
            pattern_data: Detection result dict.
            hypothesis: Previously generated hypothesis text.

        Returns:
            Dict with individual ratings, consensus score, and category.
        """
        detection_summary = json.dumps(
            {k: v for k, v in pattern_data.items() if k != "details"},
            indent=2,
            default=str,
        )[:1500]

        prompt = CONSENSUS_PROMPT.format(
            detection_summary=detection_summary,
            hypothesis=hypothesis[:500] or "No hypothesis generated yet.",
        )

        individual_ratings: list[dict[str, Any]] = []

        for provider in self.providers:
            logger.info(f"Getting rating from {provider.name}")
            try:
                result = provider.generate_structured(
                    prompt=prompt,
                    system_prompt=SYSTEM_ASTRONOMER,
                    max_tokens=512,
                    temperature=0.3,
                )

                rating = result.get("rating", 5)
                if isinstance(rating, str):
                    try:
                        rating = int(rating)
                    except ValueError:
                        rating = 5

                individual_ratings.append(
                    {
                        "provider": provider.name,
                        "model": provider.model_name,
                        "rating": int(np.clip(rating, 1, 10)),
                        "rationale": result.get("rationale", ""),
                        "category": result.get("category", "unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Rating failed for {provider.name}: {e}")
                individual_ratings.append(
                    {
                        "provider": provider.name,
                        "rating": 5,
                        "rationale": f"Error: {e}",
                        "category": "unknown",
                    }
                )

        if not individual_ratings:
            return {"consensus_rating": 0, "category": "unknown", "ratings": []}

        # Compute consensus via mean (simple) and Borda count
        ratings = [r["rating"] for r in individual_ratings]
        mean_rating = float(np.mean(ratings))
        std_rating = float(np.std(ratings))

        # Borda count for categories
        categories = [r["category"] for r in individual_ratings]
        category_counts: dict[str, int] = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        consensus_category = max(category_counts, key=category_counts.get)  # type: ignore[arg-type]

        # Agreement metric
        agreement = 1.0 - std_rating / 5.0  # Higher = more agreement

        result = {
            "consensus_rating": mean_rating,
            "std_rating": std_rating,
            "agreement": float(np.clip(agreement, 0, 1)),
            "consensus_category": consensus_category,
            "category_votes": category_counts,
            "n_providers": len(individual_ratings),
            "ratings": individual_ratings,
        }

        logger.info(
            f"Consensus: {mean_rating:.1f}/10 ({consensus_category}), "
            f"agreement={agreement:.2f}"
        )
        return result

    def rank_patterns(
        self, patterns: list[dict[str, Any]], hypotheses: list[str] | None = None
    ) -> list[tuple[int, float]]:
        """Rank multiple patterns by consensus significance.

        Returns:
            List of (pattern_index, consensus_score) sorted by score descending.
        """
        scores = []
        for i, pattern in enumerate(patterns):
            hyp = hypotheses[i] if hypotheses and i < len(hypotheses) else ""
            result = self.rate(pattern, hyp)
            scores.append((i, result["consensus_rating"]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
