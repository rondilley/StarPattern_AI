"""Central registry of default model identifiers, one per provider.

Model identifiers go stale. Keeping them in one file means a refresh
touches one place instead of five provider modules, and makes it obvious
when a default was last checked.

Each entry records the date the identifier was last verified against the
vendor's own documentation. Do not edit these from memory: a plausible
but wrong identifier fails at call time with a 404 that reads like a
network problem, which is how `claude-sonnet-4-20250514` sat broken in
this repository until a test run caught it.
"""

from __future__ import annotations

# Anthropic. Verified 2026-08-01.
# The previous default, claude-sonnet-4-20250514, is retired and returns
# 404. claude-sonnet-5 is the documented replacement for it.
CLAUDE_DEFAULT_MODEL = "claude-sonnet-5"

# Anthropic models from Opus 4.7 onward reject temperature, top_p and
# top_k outright: a non-default value returns 400. The providers below
# therefore omit the sampling parameter for these models rather than
# passing the caller's value through.
CLAUDE_MODELS_WITHOUT_SAMPLING_PARAMS: tuple[str, ...] = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)

# OpenAI. Verified 2026-08-01 by a live completion with the project key.
OPENAI_DEFAULT_MODEL = "gpt-4o"

# Google. Model identifier NOT verified as of 2026-08-01: the key in
# gemini.key.txt is rejected with API_KEY_INVALID, so no call reaches the
# model. The identifier itself is unchanged from the last working
# configuration. Re-verify once a valid key is in place.
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# xAI. Verified 2026-08-01 by listing GET /v1/models with the project key
# and confirming a live completion. The previous default, grok-2-latest,
# no longer exists: the API answers "Model not found". Despite the
# "-latest" suffix it was not a rolling alias that followed the line
# forward. To re-check, list the models rather than guessing a version:
#     from openai import OpenAI
#     OpenAI(api_key=..., base_url="https://api.x.ai/v1").models.list()
XAI_DEFAULT_MODEL = "grok-4.5"

# Local llama.cpp. Repository identifier, not an API model name.
LLAMACPP_DEFAULT_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"


def accepts_sampling_params(model: str) -> bool:
    """Return False when the Anthropic model rejects sampling parameters."""
    return not any(model.startswith(prefix) for prefix in CLAUDE_MODELS_WITHOUT_SAMPLING_PARAMS)
