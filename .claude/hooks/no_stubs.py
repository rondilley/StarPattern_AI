"""Hook: Rule 3 - No Stubs/Placeholders/Fake Data.

PreToolUse hook for Edit|Write tools.
Checks for stub functions, placeholder implementations, mock usage, and fake data.
"""

import json
import re
import sys

# Patterns that indicate stubs or placeholders
STUB_PATTERNS = [
    (r"\bmock\b", "mock usage"),
    (r"\bMock\b", "Mock usage"),
    (r"\bMagicMock\b", "MagicMock usage"),
    (r"\bunittest\.mock\b", "unittest.mock import"),
    (r"\bfrom\s+mock\s+import\b", "mock library import"),
    (r"\b@patch\b", "@patch decorator (mocking)"),
    (r"\[INSERT\b", "placeholder bracket"),
    (r"\[TODO\b", "placeholder bracket"),
    (r"\[ADD\b", "placeholder bracket"),
    (r"\[DESCRIBE\b", "placeholder bracket"),
    (r"#\s*placeholder", "placeholder comment"),
    (r"#\s*stub", "stub comment"),
    (r"raise\s+NotImplementedError\s*\(\s*\)", "NotImplementedError stub"),
]

# Pattern for bare `pass` in function/method bodies (not in except/class blocks)
# This is a heuristic - checks for def followed by pass on next line
BARE_PASS_PATTERN = re.compile(
    r"def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:\s*\n\s*(?:#[^\n]*)?\n?\s*pass\b"
)

# Compiled patterns
COMPILED = [(re.compile(pat, re.IGNORECASE if pat.startswith(r"\[") else 0), desc)
            for pat, desc in STUB_PATTERNS]


def check_text(text):
    """Return list of violations found."""
    violations = []

    for pattern, desc in COMPILED:
        matches = list(pattern.finditer(text))
        if matches:
            for m in matches[:2]:
                line_num = text[:m.start()].count("\n") + 1
                violations.append(f"line {line_num}: {desc}")

    # Check for bare pass in function bodies
    for m in BARE_PASS_PATTERN.finditer(text):
        line_num = text[:m.start()].count("\n") + 1
        violations.append(f"line {line_num}: bare 'pass' in function body (stub)")

    return violations


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})

    text = tool_input.get("new_string", "") or tool_input.get("content", "")
    if not text:
        sys.exit(0)

    # Skip check for test files that legitimately test error paths
    file_path = tool_input.get("file_path", "")
    if not file_path:
        sys.exit(0)

    violations = check_text(text)
    if violations:
        detail = "; ".join(violations[:5])
        msg = f"RULE 3 VIOLATION: No stubs/placeholders/fake data/mocks. Found: {detail}"
        if len(violations) > 5:
            msg += f" (and {len(violations) - 5} more)"
        print(msg, file=sys.stderr)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
