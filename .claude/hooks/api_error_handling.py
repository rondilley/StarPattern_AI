"""Hook: Rule 5 - APIs Must Have Error Handling.

PreToolUse hook for Edit|Write tools.
Checks that code with API/network calls includes error handling.
"""

import json
import re
import sys

# Patterns indicating API/network calls
API_PATTERNS = [
    r"requests\.(get|post|put|delete|patch|head)\s*\(",
    r"httpx\.\w+\s*\(",
    r"urllib\.request\.\w+\s*\(",
    r"aiohttp\.\w+\s*\(",
    r"urlopen\s*\(",
    r"fetch_images\s*\(",
    r"fetch_catalog\s*\(",
    r"fetch_region\s*\(",
    r"\.query_region\s*\(",
    r"\.query_object\s*\(",
    r"Gaia\.launch_job\s*\(",
    r"openai\.\w+\.create\s*\(",
    r"anthropic\.\w+\.create\s*\(",
    r"genai\.\w+\s*\(",
]

COMPILED_API = [re.compile(pat) for pat in API_PATTERNS]

# Patterns indicating error handling is present
ERROR_HANDLING = [
    r"\btry\b",
    r"\bexcept\b",
    r"\braise\b",
    r"\.raise_for_status\(\)",
    r"timeout\s*=",
]

COMPILED_ERROR = [re.compile(pat) for pat in ERROR_HANDLING]


def has_api_call(text):
    """Check if text contains API/network calls."""
    for pat in COMPILED_API:
        if pat.search(text):
            return pat.pattern
    return None


def has_error_handling(text):
    """Check if text contains error handling."""
    for pat in COMPILED_ERROR:
        if pat.search(text):
            return True
    return False


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})

    text = tool_input.get("new_string", "") or tool_input.get("content", "")
    if not text:
        sys.exit(0)

    # Only check Python files
    file_path = tool_input.get("file_path", "")
    if not file_path.endswith(".py"):
        sys.exit(0)

    api_pattern = has_api_call(text)
    if api_pattern and not has_error_handling(text):
        print(
            f"RULE 5 WARNING: API/network call detected (pattern: {api_pattern}) "
            "but no error handling (try/except/raise/timeout) found in the same edit. "
            "Every API call must have proper error handling.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
