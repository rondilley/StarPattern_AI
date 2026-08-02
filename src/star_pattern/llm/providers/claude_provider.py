"""Anthropic Claude LLM provider."""

from __future__ import annotations

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.providers.models import (
    CLAUDE_DEFAULT_MODEL,
    accepts_sampling_params,
)
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("llm.claude")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = CLAUDE_DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        import anthropic

        client = self._get_client()

        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Current Anthropic models reject temperature outright. Sending it
        # returns 400, so the parameter is dropped rather than translated.
        if accepts_sampling_params(self._model):
            kwargs["temperature"] = temperature
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = client.messages.create(**kwargs)
        except anthropic.NotFoundError:
            # Almost always a retired or misspelled model identifier.
            logger.error(
                "Claude model %s not found. Check the identifier in " "llm/providers/models.py.",
                self._model,
            )
            raise
        except anthropic.AuthenticationError:
            logger.error("Claude API key rejected")
            raise
        except anthropic.BadRequestError as exc:
            logger.error("Claude rejected the request: %s", exc)
            raise

        return self._first_text_block(response)

    @staticmethod
    def _first_text_block(response) -> str:
        """Extract the first text block, tolerating other block types.

        A response can legitimately carry no text block at all: a turn cut
        off by max_tokens may hold only a thinking block, and a refusal
        carries an empty content list. Indexing content[0].text blindly
        raises IndexError or AttributeError on both.
        """
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            logger.warning("Claude declined the request")
            return ""

        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return block.text

        logger.warning("Claude returned no text block (stop_reason=%s)", stop_reason)
        return ""

    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)
