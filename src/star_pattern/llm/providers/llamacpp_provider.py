"""Local LLM provider via llama.cpp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.llamacpp")

DEFAULT_MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
DEFAULT_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"


class LlamaCppProvider(LLMProvider):
    """Local LLM using llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
        return self._llm

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def model_name(self) -> str:
        return Path(self._model_path).stem

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        llm = self._get_llm()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"[INST] {prompt} [/INST]"

        output = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama

            return Path(self._model_path).exists()
        except ImportError:
            return False

    @classmethod
    def setup_default(cls, models_dir: str = "models") -> LlamaCppProvider:
        """Download and set up a default local model."""
        from huggingface_hub import hf_hub_download

        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {DEFAULT_MODEL}...")
        model_path = hf_hub_download(
            repo_id=DEFAULT_MODEL,
            filename=DEFAULT_FILENAME,
            local_dir=str(models_path),
        )

        logger.info(f"Model saved to: {model_path}")
        return cls(model_path=model_path)
