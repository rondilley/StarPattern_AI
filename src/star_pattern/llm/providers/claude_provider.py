"""Anthropic Claude LLM provider."""

from __future__ import annotations

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.utils.retry import retry_with_backoff
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.claude")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
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
        client = self._get_client()

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def is_available(self) -> bool:
        try:
            from anthropic import Anthropic

            return bool(self._api_key)
        except ImportError:
            return False
