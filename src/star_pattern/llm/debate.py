"""D3-style adversarial debate: is this pattern real or artifact?"""

from __future__ import annotations

import json
from typing import Any

from star_pattern.core.config import LLMConfig
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.prompts import (
    SYSTEM_ASTRONOMER,
    DEBATE_ADVOCATE_PROMPT,
    DEBATE_CHALLENGER_PROMPT,
    DEBATE_JUDGE_PROMPT,
)
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.debate")


class PatternDebate:
    """Adversarial debate to evaluate whether a detection is real.

    Uses advocate/challenger/judge pattern with multiple LLMs.
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        config: LLMConfig | None = None,
    ):
        if len(providers) < 2:
            raise ValueError("Need at least 2 providers for debate")
        self.providers = providers
        self.config = config or LLMConfig()

    def run(self, pattern_data: dict[str, Any]) -> dict[str, Any]:
        """Run a full adversarial debate on a pattern detection.

        Args:
            pattern_data: Detection result dict.

        Returns:
            Dict with debate transcript and final verdict.
        """
        detection_summary = json.dumps(
            {
                k: pattern_data[k]
                for k in ["ra", "dec", "type", "anomaly_score", "significance"]
                if k in pattern_data
            },
            indent=2,
        )
        details = json.dumps(pattern_data.get("details", {}), indent=2, default=str)[:1000]
        detection_summary += f"\n\nDetails:\n{details}"

        # Assign roles
        advocate = self.providers[0]
        challenger = self.providers[1 % len(self.providers)]
        judge = self.providers[2 % len(self.providers)] if len(self.providers) > 2 else self.providers[0]

        transcript: list[dict[str, Any]] = []
        advocate_args = ""
        challenger_args = ""

        n_rounds = self.config.debate_rounds
        logger.info(f"Starting debate: {n_rounds} rounds, advocate={advocate.name}, challenger={challenger.name}")

        for round_num in range(n_rounds):
            previous = "\n".join(
                f"{t['role']} ({t['provider']}): {t['argument'][:300]}"
                for t in transcript
            )

            # Advocate argues
            advocate_prompt = DEBATE_ADVOCATE_PROMPT.format(
                detection_summary=detection_summary,
                previous_arguments=previous or "None yet.",
            )
            advocate_response = advocate.generate(
                advocate_prompt,
                system_prompt=SYSTEM_ASTRONOMER,
                max_tokens=self.config.max_tokens // 2,
                temperature=0.7,
            )
            transcript.append(
                {
                    "round": round_num + 1,
                    "role": "advocate",
                    "provider": advocate.name,
                    "argument": advocate_response,
                }
            )
            advocate_args += f"\nRound {round_num + 1}: {advocate_response}"

            # Challenger argues
            previous = "\n".join(
                f"{t['role']} ({t['provider']}): {t['argument'][:300]}"
                for t in transcript
            )
            challenger_prompt = DEBATE_CHALLENGER_PROMPT.format(
                detection_summary=detection_summary,
                previous_arguments=previous,
            )
            challenger_response = challenger.generate(
                challenger_prompt,
                system_prompt=SYSTEM_ASTRONOMER,
                max_tokens=self.config.max_tokens // 2,
                temperature=0.7,
            )
            transcript.append(
                {
                    "round": round_num + 1,
                    "role": "challenger",
                    "provider": challenger.name,
                    "argument": challenger_response,
                }
            )
            challenger_args += f"\nRound {round_num + 1}: {challenger_response}"

            logger.info(f"Round {round_num + 1}/{n_rounds} complete")

        # Judge renders verdict -- try each available judge, falling back on failure
        judge_candidates = [judge] + [p for p in self.providers if p is not judge]
        verdict = None
        judge_used = None

        for candidate in judge_candidates:
            judge_prompt = DEBATE_JUDGE_PROMPT.format(
                detection_summary=detection_summary,
                advocate_args=advocate_args[:2000],
                challenger_args=challenger_args[:2000],
            )
            try:
                verdict = candidate.generate_structured(
                    judge_prompt,
                    system_prompt=SYSTEM_ASTRONOMER,
                    max_tokens=self.config.max_tokens,
                    temperature=0.3,
                )
                judge_used = candidate
                break
            except Exception as e:
                logger.warning(f"Judge {candidate.name} failed: {e}. Trying next provider.")
                continue

        if verdict is None:
            raise RuntimeError("All providers failed as judge")

        if "text" in verdict and "verdict" not in verdict:
            verdict = {
                "verdict": "inconclusive",
                "reasoning": verdict["text"],
                "confidence": 0.5,
                "significance_rating": 5,
            }

        verdict["judge_provider"] = judge_used.name
        verdict["advocate_provider"] = advocate.name
        verdict["challenger_provider"] = challenger.name

        logger.info(
            f"Verdict: {verdict.get('verdict', 'unknown')} "
            f"(significance={verdict.get('significance_rating', 0)})"
        )

        return {
            "transcript": transcript,
            "verdict": verdict,
            "n_rounds": n_rounds,
        }
