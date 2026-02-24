"""LLM-guided strategic advisor for the detection pipeline.

Called periodically (every strategy_interval cycles) with a batch
summary of recent findings. Provides detector parameter adjustments,
ensemble weight recommendations, and region selection guidance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from star_pattern.core.sky_region import SkyRegion
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.token_tracker import TokenTracker, estimate_tokens
from star_pattern.llm.cache import LLMCache
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.strategy")


@dataclass
class StrategyResult:
    """Actionable output from a strategy session."""

    detector_adjustments: list[dict[str, Any]] = field(default_factory=list)
    weight_adjustments: dict[str, float] = field(default_factory=dict)
    focus_regions: list[SkyRegion] = field(default_factory=list)
    detection_strategy: str = ""
    stop_doing: str = ""
    token_cost: int = 0
    strategy_id: int = 0
    timestamp: str = ""

    # Detector enable/disable from LLM strategy
    disable_detectors: list[str] = field(default_factory=list)
    enable_detectors: list[str] = field(default_factory=list)
    pipeline_suggestion: str = ""

    # Pre-strategy metrics snapshot for outcome tracking
    pre_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_adjustments": self.detector_adjustments,
            "weight_adjustments": self.weight_adjustments,
            "focus_regions": [
                {"ra": r.ra, "dec": r.dec, "radius": r.radius}
                for r in self.focus_regions
            ],
            "detection_strategy": self.detection_strategy,
            "stop_doing": self.stop_doing,
            "disable_detectors": self.disable_detectors,
            "enable_detectors": self.enable_detectors,
            "pipeline_suggestion": self.pipeline_suggestion,
            "token_cost": self.token_cost,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp,
        }


class StrategyAdvisor:
    """LLM-guided strategic advisor for the detection pipeline.

    Called periodically with a batch summary of recent findings.
    Provides detector parameter adjustments, ensemble weight
    recommendations, new detection approaches, and region
    selection guidance.

    Token budget: ~2,000 tokens per session (1-2 LLM calls).
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        tracker: TokenTracker,
        cache: LLMCache | None = None,
    ):
        self.providers = providers
        self.tracker = tracker
        self.cache = cache
        self._strategy_history: list[StrategyResult] = []
        self._next_strategy_id = 0

    def review_session(
        self,
        findings_summary: dict[str, Any],
        current_genome: dict[str, Any],
        active_learning_stats: dict[str, Any],
        evolution_history: list[dict[str, Any]],
        previous_strategy_outcome: dict[str, Any] | None = None,
    ) -> StrategyResult:
        """Conduct a strategy review session.

        Args:
            findings_summary: Compact summary of recent detections.
            current_genome: Current detection parameters.
            active_learning_stats: Feedback statistics.
            evolution_history: Recent evolution performance.
            previous_strategy_outcome: Outcome of the last strategy.

        Returns:
            StrategyResult with actionable adjustments.
        """
        from star_pattern.llm.prompts import STRATEGY_PROMPT

        summary_text = self._build_summary(
            findings_summary, current_genome,
            active_learning_stats, evolution_history,
            previous_strategy_outcome,
        )

        prompt = STRATEGY_PROMPT.format(summary=summary_text)

        # Estimate cost and check budget
        estimated_tokens = estimate_tokens(prompt) + 500  # ~500 output
        if not self.tracker.can_afford(estimated_tokens):
            logger.warning(
                f"Insufficient token budget for strategy session "
                f"(need ~{estimated_tokens}, have {self.tracker.remaining()})"
            )
            return StrategyResult()

        # Try each provider until one succeeds
        response_text = None
        for provider in self.providers:
            try:
                response_text = provider.generate_cached(
                    prompt=prompt,
                    system_prompt=None,
                    max_tokens=800,
                    temperature=0.4,
                    cache=self.cache,
                    tracker=self.tracker,
                    purpose="strategy",
                )
                break
            except Exception as e:
                logger.warning(f"Strategy session failed with {provider.name}: {e}")
                continue

        if response_text is None:
            logger.warning("All providers failed for strategy session")
            return StrategyResult()

        # Parse response
        result = self._parse_strategy_response(response_text)
        result.strategy_id = self._next_strategy_id
        self._next_strategy_id += 1
        result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        result.token_cost = estimate_tokens(prompt) + estimate_tokens(response_text)

        # Snapshot pre-strategy metrics for outcome tracking
        result.pre_metrics = {
            "n_findings": findings_summary.get("n_total", 0),
            "n_high_confidence": findings_summary.get("n_high_confidence", 0),
            "interesting_rate": active_learning_stats.get("interesting_rate", 0),
        }

        self._strategy_history.append(result)

        logger.info(
            f"Strategy session {result.strategy_id}: "
            f"{len(result.detector_adjustments)} parameter adjustments, "
            f"{len(result.weight_adjustments)} weight changes, "
            f"{len(result.focus_regions)} focus regions, "
            f"cost={result.token_cost} tokens"
        )

        return result

    def review_flagged_findings(
        self,
        flagged: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Review flagged findings in a single batched LLM call.

        Items flagged needs_llm_review by LocalClassifier/LocalEvaluator
        are reviewed in batch instead of per-item.

        Args:
            flagged: List of findings needing LLM review (max 5).

        Returns:
            List of dicts with verdict, classification, brief_hypothesis.
        """
        from star_pattern.llm.prompts import BATCH_REVIEW_PROMPT

        if not flagged:
            return []

        # Limit to 5 items per batch
        batch = flagged[:5]

        summaries = [self._compact_finding_summary(f) for f in batch]
        prompt = BATCH_REVIEW_PROMPT.format(findings="\n".join(summaries))

        # Estimate cost
        estimated_tokens = estimate_tokens(prompt) + 600
        if not self.tracker.can_afford(estimated_tokens):
            logger.warning("Insufficient budget for batch review")
            return []

        # Try each provider
        response_text = None
        for provider in self.providers:
            try:
                response_text = provider.generate_cached(
                    prompt=prompt,
                    system_prompt=None,
                    max_tokens=800,
                    temperature=0.3,
                    cache=self.cache,
                    tracker=self.tracker,
                    purpose="batch_review",
                )
                break
            except Exception as e:
                logger.warning(f"Batch review failed with {provider.name}: {e}")
                continue

        if response_text is None:
            return []

        return self._parse_batch_review(response_text, len(batch))

    def record_outcome(
        self,
        strategy_id: int,
        findings_after: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record pipeline performance after applying a strategy.

        Compares pre-strategy and post-strategy metrics to evaluate
        whether the LLM's suggestions improved detection quality.
        Included in the next strategy session's summary.

        Args:
            strategy_id: ID of the strategy to evaluate.
            findings_after: Current pipeline metrics.

        Returns:
            Outcome dict with improvement metrics, or None if not found.
        """
        # Find the strategy
        strategy = None
        for s in self._strategy_history:
            if s.strategy_id == strategy_id:
                strategy = s
                break

        if strategy is None or not strategy.pre_metrics:
            return None

        pre = strategy.pre_metrics
        outcome = {
            "strategy_id": strategy_id,
            "findings_delta": (
                findings_after.get("n_total", 0)
                - pre.get("n_findings", 0)
            ),
            "high_confidence_delta": (
                findings_after.get("n_high_confidence", 0)
                - pre.get("n_high_confidence", 0)
            ),
            "interesting_rate_delta": (
                findings_after.get("interesting_rate", 0)
                - pre.get("interesting_rate", 0)
            ),
            "improved": (
                findings_after.get("interesting_rate", 0)
                > pre.get("interesting_rate", 0)
            ),
        }

        logger.info(
            f"Strategy {strategy_id} outcome: "
            f"findings_delta={outcome['findings_delta']}, "
            f"improved={outcome['improved']}"
        )

        return outcome

    def get_latest_outcome(self) -> dict[str, Any] | None:
        """Get the most recent strategy outcome for inclusion in next session."""
        if not self._strategy_history:
            return None
        latest = self._strategy_history[-1]
        if latest.pre_metrics:
            return {
                "strategy_id": latest.strategy_id,
                "pre_metrics": latest.pre_metrics,
            }
        return None

    def _build_summary(
        self,
        findings: dict[str, Any],
        genome: dict[str, Any],
        al_stats: dict[str, Any],
        evo_history: list[dict[str, Any]],
        prev_outcome: dict[str, Any] | None = None,
    ) -> str:
        """Build a compact summary for LLM review (~500 tokens input).

        Formats pipeline state into a minimal-token representation
        suitable for a strategy review prompt.
        """
        lines = []

        # Findings summary
        n_total = findings.get("n_total", 0)
        n_high = findings.get("n_high_confidence", 0)
        lines.append(f"DETECTIONS (recent): {n_total} total, {n_high} high-confidence")

        # Top detection types
        type_counts = findings.get("type_counts", {})
        if type_counts:
            top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            types_str = ", ".join(f"{t}({c})" for t, c in top_types[:5])
            lines.append(f"TOP TYPES: {types_str}")

        # False positive info
        n_fp = findings.get("n_artifacts", 0)
        if n_fp > 0:
            lines.append(f"FALSE POSITIVES: {n_fp}")

        # Active learning stats
        n_labeled = al_stats.get("n_total", 0)
        int_rate = al_stats.get("interesting_rate", 0)
        lines.append(f"ACTIVE LEARNING: {n_labeled} labeled, {int_rate:.0%} interesting rate")

        # Evolution stats
        if evo_history:
            latest = evo_history[-1]
            best_fit = latest.get("fitness", 0)
            stag = latest.get("stagnation_count", 0)
            lines.append(f"EVOLUTION: best_fitness={best_fit:.3f}, stagnation={stag} cycles")

        # Current ensemble weights
        weights = genome.get("ensemble_weights", {})
        if weights:
            weight_str = ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(weights.items())
            )
            lines.append(f"CURRENT WEIGHTS: {weight_str}")

        # Regions searched
        n_regions = findings.get("n_regions", 0)
        lines.append(f"REGIONS SEARCHED: {n_regions}")

        # Enabled/disabled detectors
        enabled = genome.get("enabled_detectors", {})
        if enabled:
            disabled_list = [k for k, v in enabled.items() if not v]
            if disabled_list:
                lines.append(f"DISABLED DETECTORS: {', '.join(disabled_list)}")

        # Types never found (help LLM suggest new directions)
        never_found = findings.get("never_found_types", [])
        if never_found:
            lines.append(f"NEVER FOUND: {', '.join(never_found[:10])}")

        # Previous strategy outcome
        if prev_outcome:
            improved = prev_outcome.get("improved", False)
            delta = prev_outcome.get("findings_delta", 0)
            label = "improved" if improved else "no improvement"
            lines.append(
                f"PREVIOUS STRATEGY: {label}, "
                f"findings_delta={delta}"
            )

        return "\n".join(lines)

    def _compact_finding_summary(self, finding: dict[str, Any]) -> str:
        """Build a compact summary of a single finding for batch review."""
        parts = [
            f"Type: {finding.get('classification', finding.get('type', 'unknown'))}",
            f"Score: {finding.get('anomaly_score', finding.get('confidence', 0)):.2f}",
            f"RA/Dec: ({finding.get('ra', 0):.2f}, {finding.get('dec', 0):.2f})",
        ]

        rationale = finding.get("rationale", "")
        if rationale:
            parts.append(f"Rationale: {rationale[:100]}")

        verdict = finding.get("verdict", "")
        if verdict:
            parts.append(f"Local verdict: {verdict}")

        return " | ".join(parts)

    def _parse_strategy_response(self, response: str) -> StrategyResult:
        """Parse LLM response into a StrategyResult."""
        result = StrategyResult()

        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                result.detection_strategy = response.strip()
                return result

            data = json.loads(json_str)

            # Parse detector adjustments
            result.detector_adjustments = data.get("detector_adjustments", [])

            # Parse weight adjustments
            result.weight_adjustments = data.get("weight_adjustments", {})

            # Parse focus regions
            for region_data in data.get("focus_regions", []):
                try:
                    ra = float(region_data.get("ra", 0))
                    dec = float(region_data.get("dec", 0))
                    result.focus_regions.append(
                        SkyRegion(ra=ra, dec=dec, radius=3.0)
                    )
                except (ValueError, TypeError):
                    continue

            result.detection_strategy = data.get("detection_strategy", "")
            result.stop_doing = data.get("stop_doing", "")

            # Parse new fields (detector enable/disable, pipeline suggestion)
            result.disable_detectors = data.get("disable_detectors", [])
            result.enable_detectors = data.get("enable_detectors", [])
            result.pipeline_suggestion = data.get("pipeline_suggestion", "")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse strategy response: {e}")
            result.detection_strategy = response.strip()[:200]

        return result

    def _parse_batch_review(
        self, response: str, n_expected: int
    ) -> list[dict[str, Any]]:
        """Parse batch review response into per-finding verdicts."""
        try:
            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                json_str = response[start:end]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = f"[{response[start:end]}]"
            else:
                return []

            reviews = json.loads(json_str)
            if isinstance(reviews, dict):
                reviews = [reviews]

            return [
                {
                    "verdict": r.get("verdict", "inconclusive"),
                    "classification": r.get("classification", "unknown"),
                    "brief_hypothesis": r.get("brief_hypothesis", ""),
                }
                for r in reviews[:n_expected]
            ]

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse batch review response: {e}")
            return []
