"""xAI (Grok) LLM provider."""

from __future__ import annotations

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.providers.models import XAI_DEFAULT_MODEL
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("llm.xai")


class XAIProvider(LLMProvider):
    """xAI Grok provider (OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str = XAI_DEFAULT_MODEL):
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

        import openai

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.NotFoundError:
            logger.error(
                "xAI model %s not found. Check the identifier in " "llm/providers/models.py.",
                self._model,
            )
            raise
        except openai.AuthenticationError:
            logger.error("xAI API key rejected")
            raise
        except openai.BadRequestError as exc:
            logger.error("xAI rejected the request: %s", exc)
            raise

        if not response.choices:
            logger.warning("xAI returned no choices")
            return ""
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)
