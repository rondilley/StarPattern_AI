"""Hook: Rule 4 - Never Push Secrets to Git.

PreToolUse hook for Bash tool.
Checks git commands for operations that might commit or push secret files.
"""

import json
import re
import sys

# File patterns that likely contain secrets
SECRET_PATTERNS = [
    r"\.key\.txt",
    r"\.key$",
    r"\.secret",
    r"\.env$",
    r"\.env\.",
    r"credentials",
    r"\.pem$",
    r"\.p12$",
    r"\.pfx$",
    r"api_key",
    r"apikey",
    r"token\.txt",
]

COMPILED_SECRETS = [re.compile(pat, re.IGNORECASE) for pat in SECRET_PATTERNS]

# Git commands that stage or commit files
GIT_ADD_PATTERN = re.compile(r"git\s+add\s+(.*)", re.IGNORECASE)
GIT_COMMIT_ALL = re.compile(r"git\s+commit\s+.*-a", re.IGNORECASE)


def has_secret_pattern(text):
    """Check if text matches any secret file pattern."""
    for pat in COMPILED_SECRETS:
        if pat.search(text):
            return pat.pattern
    return None


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    # Check git add commands for secret files
    add_match = GIT_ADD_PATTERN.search(command)
    if add_match:
        files_str = add_match.group(1).strip()

        # Block "git add -A" and "git add ." as they risk including secrets
        if files_str in ("-A", ".", "--all"):
            print(
                "RULE 4 VIOLATION: 'git add -A' / 'git add .' risks committing secrets. "
                "Add specific files by name instead.",
                file=sys.stderr,
            )
            sys.exit(2)

        # Check individual file arguments for secret patterns
        secret = has_secret_pattern(files_str)
        if secret:
            print(
                f"RULE 4 VIOLATION: git add may include secret files "
                f"(matched pattern: {secret}). Check .gitignore first.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Check git commit -a (stages all changes, could include secrets)
    if GIT_COMMIT_ALL.search(command):
        print(
            "RULE 4 VIOLATION: 'git commit -a' stages all changes and risks "
            "committing secrets. Stage files explicitly with git add.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
