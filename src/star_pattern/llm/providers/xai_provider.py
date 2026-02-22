"""xAI (Grok) LLM provider."""

from __future__ import annotations

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.utils.retry import retry_with_backoff
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.xai")


class XAIProvider(LLMProvider):
    """xAI Grok provider (OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str = "grok-2-latest"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.x.ai/v1",
            )
        return self._client

    @property
    def name(self) -> str:
        return "xai"

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        try:
            from openai import OpenAI

            return bool(self._api_key)
        except ImportError:
            return False
