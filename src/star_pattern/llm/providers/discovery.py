"""Auto-discover available LLM providers from key files."""

from __future__ import annotations

from pathlib import Path

from star_pattern.llm.providers.base import LLMProvider
from star_pattern.utils.logging import get_logger

logger = get_logger("llm.discovery")

# Map key file patterns to provider classes
KEY_FILE_MAP = {
    "openai.key.txt": ("star_pattern.llm.providers.openai_provider", "OpenAIProvider"),
    "claude.key.txt": ("star_pattern.llm.providers.claude_provider", "ClaudeProvider"),
    "gemini.key.txt": ("star_pattern.llm.providers.gemini_provider", "GeminiProvider"),
    "xai.key.txt": ("star_pattern.llm.providers.xai_provider", "XAIProvider"),
}


class ProviderDiscovery:
    """Scan for API key files and instantiate available providers."""

    def __init__(self, key_dir: str | Path = "."):
        self.key_dir = Path(key_dir)

    def discover(self) -> list[LLMProvider]:
        """Discover and instantiate all available LLM providers.

        Scans key_dir for *.key.txt files matching known providers.

        Returns:
            List of instantiated LLMProvider objects.
        """
        providers: list[LLMProvider] = []

        for key_file, (module_path, class_name) in KEY_FILE_MAP.items():
            key_path = self.key_dir / key_file
            if not key_path.exists():
                continue

            api_key = key_path.read_text().strip()
            if not api_key:
                continue

            try:
                import importlib

                module = importlib.import_module(module_path)
                provider_class = getattr(module, class_name)
                provider = provider_class(api_key=api_key)

                if provider.is_available():
                    providers.append(provider)
                    logger.info(f"Discovered provider: {provider.name} ({provider.model_name})")
                else:
                    logger.warning(f"Provider {class_name} not available (missing deps?)")
            except Exception as e:
                logger.warning(f"Failed to init {class_name}: {e}")

        # Check for local models
        try:
            from star_pattern.llm.providers.llamacpp_provider import LlamaCppProvider

            models_dir = self.key_dir / "models"
            if models_dir.exists():
                for gguf in models_dir.glob("*.gguf"):
                    try:
                        provider = LlamaCppProvider(model_path=str(gguf))
                        if provider.is_available():
                            providers.append(provider)
                            logger.info(f"Discovered local model: {gguf.name}")
                    except Exception:
                        pass
        except ImportError:
            pass

        logger.info(f"Total providers discovered: {len(providers)}")
        return providers

    def discover_by_name(self, name: str) -> LLMProvider | None:
        """Find a specific provider by name."""
        for provider in self.discover():
            if provider.name == name:
                return provider
        return None
