"""Hook: Rules 6 & 7 - Update CLAUDE.md and VIBE_HISTORY.md.

PostToolUse hook for Edit|Write tools.
When architecture-significant files are modified, reminds to update docs.
When bug-fix or test-fix files are modified, reminds about VIBE_HISTORY.md
for the self-improvement loop.
"""

import json
import sys

# Files/directories that indicate architecture changes
ARCHITECTURE_INDICATORS = [
    "__init__.py",
    "config.py",
    "base.py",
    "cli.py",
    "pipeline/",
    "pyproject.toml",
]

# Patterns that suggest a bug fix or lesson-worthy change
LESSON_INDICATORS = [
    "test_",
    "tests/",
    "conftest",
    "fix",
    "bugfix",
    "hotfix",
    "patch",
    "workaround",
]


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        sys.exit(0)

    is_architecture_change = any(
        indicator in file_path for indicator in ARCHITECTURE_INDICATORS
    )

    is_lesson_worthy = any(
        indicator in file_path.lower() for indicator in LESSON_INDICATORS
    )

    if is_architecture_change:
        msg = (
            "RULES 6-7 REMINDER: Architecture-significant file modified. "
            "Remember to update CLAUDE.md (why changes were made) and "
            "VIBE_HISTORY.md (approaches, lessons learned) if applicable."
        )
        output = {"systemMessage": msg}
        json.dump(output, sys.stdout)
    elif is_lesson_worthy:
        msg = (
            "SELF-IMPROVEMENT REMINDER: Bug fix or test change detected. "
            "If this was a correction or hard-won insight, capture the "
            "pattern and lesson in VIBE_HISTORY.md to prevent recurrence."
        )
        output = {"systemMessage": msg}
        json.dump(output, sys.stdout)

    sys.exit(0)


if __name__ == "__main__":
    main()
