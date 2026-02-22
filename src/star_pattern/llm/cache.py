"""SHA256-keyed cache for LLM responses.

Avoids redundant LLM calls by caching responses keyed by prompt content.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from star_pattern.utils.logging import get_logger

logger = get_logger("llm.cache")


class LLMCache:
    """SHA256-keyed cache for LLM responses.

    Identical prompts return cached responses instead of making
    new API calls. Entries expire after a configurable TTL.
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 168):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cached responses.
            ttl_hours: Time-to-live in hours (default: 168 = 1 week).
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0

    def get(self, prompt_hash: str) -> str | None:
        """Retrieve cached response if exists and not expired.

        Args:
            prompt_hash: SHA256 hash of the prompt content.

        Returns:
            Cached response string, or None on miss/expiry.
        """
        cache_file = self.cache_dir / f"{prompt_hash}.json"

        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(cache_file.read_text())

            # Check TTL
            cached_time = data.get("timestamp", 0)
            age_hours = (time.time() - cached_time) / 3600
            if age_hours > self.ttl_hours:
                logger.debug(f"Cache expired for {prompt_hash[:12]}...")
                cache_file.unlink(missing_ok=True)
                self._misses += 1
                return None

            self._hits += 1
            logger.debug(f"Cache hit for {prompt_hash[:12]}...")
            return data.get("response")

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Cache read error for {prompt_hash[:12]}: {e}")
            self._misses += 1
            return None

    def put(
        self,
        prompt_hash: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store response with metadata.

        Args:
            prompt_hash: SHA256 hash of the prompt content.
            response: The LLM response to cache.
            metadata: Optional metadata (provider, tokens, etc.).
        """
        cache_file = self.cache_dir / f"{prompt_hash}.json"

        data = {
            "response": response,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        try:
            cache_file.write_text(json.dumps(data))
            logger.debug(f"Cached response for {prompt_hash[:12]}...")
        except OSError as e:
            logger.debug(f"Cache write error for {prompt_hash[:12]}: {e}")

    @staticmethod
    def hash_prompt(prompt: str, system_prompt: str = "") -> str:
        """Deterministic SHA256 hash of prompt content.

        Args:
            prompt: The user prompt.
            system_prompt: The system prompt.

        Returns:
            Hex-encoded SHA256 hash string.
        """
        content = f"{system_prompt}\n---\n{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
            "total_cached": len(list(self.cache_dir.glob("*.json"))),
        }

    def clear_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                cached_time = data.get("timestamp", 0)
                age_hours = (time.time() - cached_time) / 3600
                if age_hours > self.ttl_hours:
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, OSError):
                continue

        if removed:
            logger.info(f"Cleared {removed} expired cache entries")
        return removed
