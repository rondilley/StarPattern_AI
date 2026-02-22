"""LLM-guided search strategy: suggest regions and parameter adjustments."""

from __future__ import annotations

import json
from typing import Any

from star_pattern.core.config import LLMConfig
from star_pattern.core.sky_region import SkyRegion
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.prompts import SYSTEM_ASTRONOMER, SEARCH_GUIDE_PROMPT
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.search_guide")


class LLMSearchGuide:
    """Use LLMs to guide the discovery search strategy."""

    def __init__(self, provider: LLMProvider, config: LLMConfig | None = None):
        self.provider = provider
        self.config = config or LLMConfig()

    def suggest(
        self,
        findings: list[dict[str, Any]],
        searched_regions: list[SkyRegion],
        current_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get search suggestions from the LLM.

        Args:
            findings: List of pattern results found so far.
            searched_regions: Regions already explored.
            current_params: Current detection parameters.

        Returns:
            Dict with suggested regions, parameter adjustments, and reasoning.
        """
        # Summarize findings
        findings_summary = ""
        for i, f in enumerate(findings[:10]):
            findings_summary += (
                f"{i + 1}. ({f.get('ra', 0):.2f}, {f.get('dec', 0):.2f}): "
                f"{f.get('type', 'unknown')} score={f.get('anomaly_score', 0):.3f}\n"
            )

        # Summarize searched regions
        regions_text = ""
        for r in searched_regions[-20:]:
            regions_text += f"  ({r.ra:.2f}, {r.dec:.2f}) r={r.radius}'\n"

        # Format params
        params_text = json.dumps(current_params or {}, indent=2, default=str)[:500]

        prompt = SEARCH_GUIDE_PROMPT.format(
            findings_summary=findings_summary or "No findings yet.",
            searched_regions=regions_text or "None yet.",
            best_params=params_text,
        )

        logger.info(f"Getting search guidance from {self.provider.name}")

        result = self.provider.generate_structured(
            prompt=prompt,
            system_prompt=SYSTEM_ASTRONOMER,
            max_tokens=self.config.max_tokens,
            temperature=0.8,
        )

        # Parse suggested regions
        suggested_regions = []
        for region_data in result.get("suggested_regions", []):
            try:
                ra = float(region_data.get("ra", 0))
                dec = float(region_data.get("dec", 0))
                if 0 <= ra <= 360 and -90 <= dec <= 90:
                    suggested_regions.append(
                        {
                            "region": SkyRegion(ra=ra, dec=dec, radius=3.0),
                            "rationale": region_data.get("rationale", ""),
                        }
                    )
            except (TypeError, ValueError):
                continue

        result["parsed_regions"] = suggested_regions
        logger.info(f"LLM suggested {len(suggested_regions)} regions")
        return result
