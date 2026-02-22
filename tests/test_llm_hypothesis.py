"""Tests for LLM integration using real providers from *.key.txt files."""

import pytest
from pathlib import Path

from star_pattern.core.config import LLMConfig
from star_pattern.llm.providers.base import LLMProvider
from star_pattern.llm.hypothesis import HypothesisGenerator
from star_pattern.llm.debate import PatternDebate
from star_pattern.llm.consensus import PatternConsensus
from star_pattern.llm.providers.discovery import ProviderDiscovery


@pytest.fixture
def pattern_data():
    return {
        "ra": 180.0,
        "dec": 45.0,
        "type": "lens",
        "anomaly_score": 0.75,
        "significance": 0.6,
        "details": {"lens_score": 0.7, "n_arcs": 2},
        "cross_matches": [],
    }


class TestHypothesisGenerator:
    def test_generate(self, first_provider, pattern_data):
        gen = HypothesisGenerator(first_provider)
        result = gen.generate(pattern_data)
        assert "hypothesis" in result or "text" in result
        assert "provider" in result
        assert result["provider"] == first_provider.name

    def test_generate_batch(self, first_provider, pattern_data):
        gen = HypothesisGenerator(first_provider)
        results = gen.generate_batch([pattern_data])
        assert len(results) == 1
        assert "provider" in results[0]


class TestPatternDebate:
    def test_debate_requires_2_providers(self, first_provider):
        with pytest.raises(ValueError, match="at least 2"):
            PatternDebate([first_provider])

    def test_debate_runs(self, llm_providers, pattern_data):
        if len(llm_providers) < 2:
            pytest.skip("Need at least 2 LLM providers for debate")
        config = LLMConfig(debate_rounds=1)
        debate = PatternDebate(llm_providers[:3], config)
        result = debate.run(pattern_data)
        assert "transcript" in result
        assert "verdict" in result
        assert len(result["transcript"]) == 2  # 1 round = 1 advocate + 1 challenger


class TestPatternConsensus:
    def test_rate(self, llm_providers, pattern_data):
        if len(llm_providers) < 2:
            pytest.skip("Need at least 2 LLM providers for consensus")
        consensus = PatternConsensus(llm_providers[:2])
        result = consensus.rate(pattern_data)
        assert "consensus_rating" in result
        assert 1 <= result["consensus_rating"] <= 10
        assert "agreement" in result
        assert result["n_providers"] == 2


class TestProviderDiscovery:
    def test_discover_no_keys(self, tmp_path):
        discovery = ProviderDiscovery(key_dir=tmp_path)
        providers = discovery.discover()
        assert len(providers) == 0

    def test_discover_real_keys(self, project_root):
        discovery = ProviderDiscovery(key_dir=project_root)
        providers = discovery.discover()
        # We know there are key files in the project root
        assert len(providers) > 0
        for p in providers:
            assert p.name in ("openai", "claude", "gemini", "xai", "llamacpp")
            assert p.model_name  # Non-empty
