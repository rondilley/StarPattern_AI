"""Tests for the LLM response cache."""

import json
import time

import pytest

from star_pattern.llm.cache import LLMCache


class TestLLMCache:
    def test_cache_miss_then_hit(self, tmp_path):
        """First access misses, second hits."""
        cache = LLMCache(tmp_path / "cache")

        key = cache.hash_prompt("test prompt", "system")

        # Miss
        assert cache.get(key) is None

        # Store
        cache.put(key, "cached response", {"provider": "test"})

        # Hit
        result = cache.get(key)
        assert result == "cached response"

    def test_ttl_expiry(self, tmp_path):
        """Expired entries are not returned."""
        cache = LLMCache(tmp_path / "cache", ttl_hours=0)  # Expire immediately

        key = cache.hash_prompt("expiring prompt", "")

        # Store with a timestamp in the past
        cache_file = cache.cache_dir / f"{key}.json"
        data = {
            "response": "old response",
            "timestamp": time.time() - 3600,  # 1 hour ago
            "metadata": {},
        }
        cache_file.write_text(json.dumps(data))

        # Should miss (expired)
        assert cache.get(key) is None

    def test_hash_deterministic(self):
        """Same prompt produces same hash."""
        h1 = LLMCache.hash_prompt("hello world", "system")
        h2 = LLMCache.hash_prompt("hello world", "system")
        assert h1 == h2

    def test_hash_different_prompts(self):
        """Different prompts produce different hashes."""
        h1 = LLMCache.hash_prompt("prompt A", "system")
        h2 = LLMCache.hash_prompt("prompt B", "system")
        assert h1 != h2

    def test_hash_different_system_prompts(self):
        """Different system prompts produce different hashes."""
        h1 = LLMCache.hash_prompt("prompt", "system A")
        h2 = LLMCache.hash_prompt("prompt", "system B")
        assert h1 != h2

    def test_stats(self, tmp_path):
        """Hit/miss statistics are tracked."""
        cache = LLMCache(tmp_path / "cache")

        key = cache.hash_prompt("test", "")

        cache.get(key)  # Miss
        cache.put(key, "response", {})
        cache.get(key)  # Hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_cached"] >= 1

    def test_clear_expired(self, tmp_path):
        """Expired entries are removed by clear_expired."""
        cache = LLMCache(tmp_path / "cache", ttl_hours=1)

        # Create an expired entry
        key = LLMCache.hash_prompt("old", "")
        cache_file = cache.cache_dir / f"{key}.json"
        data = {
            "response": "old",
            "timestamp": time.time() - 7200,  # 2 hours ago
            "metadata": {},
        }
        cache_file.write_text(json.dumps(data))

        # Create a fresh entry
        key2 = LLMCache.hash_prompt("fresh", "")
        cache.put(key2, "fresh response", {})

        removed = cache.clear_expired()
        assert removed == 1

        # Fresh entry should still exist
        assert cache.get(key2) == "fresh response"

    def test_metadata_stored(self, tmp_path):
        """Metadata is stored alongside response."""
        cache = LLMCache(tmp_path / "cache")

        key = cache.hash_prompt("meta test", "")
        cache.put(key, "response", {"provider": "openai", "tokens": 100})

        # Read raw file to verify metadata
        cache_file = cache.cache_dir / f"{key}.json"
        data = json.loads(cache_file.read_text())
        assert data["metadata"]["provider"] == "openai"
        assert data["metadata"]["tokens"] == 100

    def test_corrupt_cache_file_handled(self, tmp_path):
        """Corrupt cache files are handled gracefully."""
        cache = LLMCache(tmp_path / "cache")

        key = "corrupt_key"
        cache_file = cache.cache_dir / f"{key}.json"
        cache_file.write_text("not valid json{{{")

        # Should return None (not crash)
        assert cache.get(key) is None

    def test_cache_dir_created(self, tmp_path):
        """Cache directory is created on initialization."""
        cache_dir = tmp_path / "new" / "cache" / "dir"
        cache = LLMCache(cache_dir)
        assert cache_dir.exists()
