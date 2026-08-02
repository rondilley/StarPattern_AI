"""Hook: Rule 2 - Never Declare Success Without Testing.

Stop hook that checks if the response claims code is working/complete/production-ready
without evidence of actual testing.
"""

import json
import re
import sys


# Patterns that indicate success claims
SUCCESS_CLAIMS = [
    r"production.?ready",
    r"works?\s+(perfectly|flawlessly|correctly|great)",
    r"implementation\s+is\s+(complete|done|finished|ready)",
    r"everything\s+(works|is\s+working)",
    r"fully\s+(functional|working|operational)",
    r"code\s+is\s+(ready|complete|working)",
]

COMPILED_CLAIMS = [re.compile(pat, re.IGNORECASE) for pat in SUCCESS_CLAIMS]

# Patterns that indicate testing was actually done
TESTING_EVIDENCE = [
    r"test.*pass",
    r"ran\s+tests?",
    r"pytest",
    r"all\s+\d+\s+tests?\s+pass",
    r"verified\s+(with|by|using)",
    r"tested\s+(with|by|against|using)",
    r"output\s+(shows|confirms|demonstrates)",
    r"untested",
    r"implementation\s+complete\s+but\s+untested",
]

COMPILED_TESTING = [re.compile(pat, re.IGNORECASE) for pat in TESTING_EVIDENCE]


def main():
    data = json.load(sys.stdin)
    # For Stop hooks, the transcript contains the conversation
    # We check the last assistant message
    transcript_path = data.get("transcript_path", "")

    if not transcript_path:
        sys.exit(0)

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            # Read last few lines to find last assistant message
            lines = f.readlines()

        # Find the last assistant message content
        last_response = ""
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                if entry.get("role") == "assistant":
                    content = entry.get("content", "")
                    if isinstance(content, str):
                        last_response = content
                    elif isinstance(content, list):
                        last_response = " ".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                    break
            except (json.JSONDecodeError, AttributeError):
                continue

        if not last_response:
            sys.exit(0)

        # Check for success claims
        has_claim = any(pat.search(last_response) for pat in COMPILED_CLAIMS)
        has_evidence = any(pat.search(last_response) for pat in COMPILED_TESTING)

        if has_claim and not has_evidence:
            reason = (
                "RULE 2 REMINDER: Response claims code is working/complete but shows "
                "no evidence of testing. Either run tests first or qualify the claim "
                "with 'Implementation complete but untested'."
            )
            output = {"decision": "block", "reason": reason}
            json.dump(output, sys.stdout)
            sys.exit(0)

    except Exception:
        # Don't block on hook errors
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
