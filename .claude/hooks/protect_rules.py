"""Hook: Rule 8 - Don't Remove These Rules.

PreToolUse hook for Edit|Write tools.
Prevents deletion or weakening of rules in claude-code-rules.md.
"""

import json
import sys


RULES_FILE = "claude-code-rules.md"


def main():
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if RULES_FILE not in file_path:
        sys.exit(0)

    # For Write tool: block complete overwrites of the rules file
    content = tool_input.get("content", "")
    if content:
        # Count rule headers in new content
        import re
        rule_headers = re.findall(r"###\s+\d+\.", content)
        if len(rule_headers) < 16:
            print(
                f"RULE 8 VIOLATION: Cannot weaken or remove rules from {RULES_FILE}. "
                f"Found only {len(rule_headers)} rules (expected 16+). "
                "Rules can only be clarified or expanded, never removed.",
                file=sys.stderr,
            )
            sys.exit(2)

    # For Edit tool: check if old_string contains rule headers being removed
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    if old_string and new_string is not None:
        import re
        old_rules = re.findall(r"###\s+\d+\.", old_string)
        new_rules = re.findall(r"###\s+\d+\.", new_string)
        if len(old_rules) > len(new_rules):
            print(
                f"RULE 8 VIOLATION: Cannot remove rules from {RULES_FILE}. "
                f"Edit removes {len(old_rules) - len(new_rules)} rule header(s). "
                "Rules can only be clarified or expanded, never removed.",
                file=sys.stderr,
            )
            sys.exit(2)

        # Check for weakening keywords being removed
        strength_words = ["NEVER", "ALWAYS", "EVERY", "absolute"]
        for word in strength_words:
            old_count = old_string.upper().count(word.upper())
            new_count = new_string.upper().count(word.upper())
            if old_count > new_count:
                print(
                    f"RULE 8 VIOLATION: Cannot weaken rules. "
                    f"Edit reduces '{word}' count from {old_count} to {new_count}. "
                    "Rules can only be clarified or expanded, never weakened.",
                    file=sys.stderr,
                )
                sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
