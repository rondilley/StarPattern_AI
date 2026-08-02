"""Google Gemini LLM provider."""

from __future__ import annotations

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.providers.models import GEMINI_DEFAULT_MODEL
from star_pattern.utils.logging import get_logger
from star_pattern.utils.retry import retry_with_backoff

logger = get_logger("llm.gemini")


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str = GEMINI_DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._client_cache: dict[str | None, object] = {}

    def _get_client(self, system_prompt: str | None = None):
        if system_prompt not in self._client_cache:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            kwargs = {}
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            self._client_cache[system_prompt] = genai.GenerativeModel(self._model, **kwargs)
        return self._client_cache[system_prompt]

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    # Gemini 2.5 models use internal reasoning tokens that count against
    # max_output_tokens. We enforce a floor so the model has enough budget
    # for both thinking and the visible response.
    _MIN_OUTPUT_TOKENS = 1024

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        client = self._get_client(system_prompt)

        from google.api_core import exceptions as google_exceptions

        effective_tokens = max(max_tokens, self._MIN_OUTPUT_TOKENS)
        try:
            response = client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": effective_tokens,
                    "temperature": temperature,
                },
            )
        except google_exceptions.NotFound:
            logger.error(
                "Gemini model %s not found. Check the identifier in " "llm/providers/models.py.",
                self._model,
            )
            raise
        except google_exceptions.PermissionDenied:
            logger.error("Gemini API key rejected")
            raise
        except google_exceptions.InvalidArgument as exc:
            logger.error("Gemini rejected the request: %s", exc)
            raise

        # Handle empty responses from safety filters or token exhaustion
        if not response.candidates:
            logger.warning("Gemini returned no candidates")
            return ""
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            reason = getattr(candidate, "finish_reason", "unknown")
            logger.warning(f"Gemini returned no content (finish_reason={reason})")
            return ""

        return response.text

    def is_available(self) -> bool:
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False
        return bool(self._api_key)
