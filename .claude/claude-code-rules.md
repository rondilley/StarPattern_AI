# Claude Code Rules for SF Writing Tool

## Code Quality Rules

### 1. NO Icons/Emoji/Symbols
- **NEVER** use emoji, icons, or Unicode symbols in source code, output, or documentation
- Use plain text only: "OK", "ERROR", "WARNING", "INFO"
- Exceptions: None. This is absolute.

### 2. Never Declare Success Without Testing
- **NEVER** claim code is "production-ready", "complete", or "working" without rigorous testing
- **ALWAYS** run tests before claiming functionality works
- **ALWAYS** validate with actual data/APIs before claiming integration works
- Be honest about untested code: "Implementation complete but untested"

### 3. No Stubs/Placeholders/Fake Data
- **NEVER** use stub functions, placeholder implementations, or fake data
- **ALWAYS** implement real functionality or clearly mark as TODO
- If functionality cannot be implemented, explain why and propose alternatives
- No `pass`, no `return None` without justification

### 4. Never Push Secrets to Git
- **ALWAYS** add secrets/keys to .gitignore BEFORE creating them
- **NEVER** commit API keys, passwords, tokens, or credentials
- Use *.key.txt, *.secret, .env patterns in .gitignore
- Validate .gitignore before any git operations

### 5. APIs Must Have Error Handling
- **EVERY** API call must have try-except with specific exception types
- **EVERY** network operation must handle timeouts and retries
- **EVERY** external dependency must have fallback behavior
- Log errors properly - don't swallow exceptions silently

### 6. Update CLAUDE.md on Architecture Changes
- **ALWAYS** update CLAUDE.md when:
  - Adding new modules or major components
  - Changing data flow or architecture patterns
  - Modifying core abstractions or interfaces
  - Adding new dependencies or external integrations
- Document WHY changes were made, not just WHAT changed

### 7. Update VIBE_HISTORY.md
- **ALWAYS** document in VIBE_HISTORY.md:
  - Interesting coding approaches that worked well
  - Failed attempts and lessons learned
  - Performance insights or optimization discoveries
  - Architectural decisions and trade-offs
- Be honest about failures - they're learning opportunities

### 8. Don't Remove These Rules
- **NEVER** delete or weaken these rules
- Rules can be clarified or expanded, never removed
- If a rule creates problems, document the conflict and propose amendment
- Rules apply to all code, documentation, and outputs

---

## Creative Writing Rules

### 9. Maintain Continuity Database
- **ALWAYS** track: character names, traits, relationships, locations, technology, timeline
- **NEVER** introduce contradictions to established canon without explicit retcon flag
- Cross-reference new content against CONTINUITY.md before output
- Flag potential conflicts: "Note: This contradicts Ch.3 where X..."
- Update CONTINUITY.md after every significant addition to the narrative

### 10. No Purple Prose or Clichés
- **AVOID**: "orbs" for eyes, "smirked," "let out a breath they didn't know they were holding"
- **AVOID**: Overuse of adjectives/adverbs - prefer strong verbs
- **CHALLENGE** SF clichés: chosen one, evil AI, humans-are-special, technobabble solutions
- If using a trope deliberately, subvert it or document the justification
- Read dialogue aloud mentally - if it sounds written, rewrite it

### 11. Character Voice Consistency
- Each character needs documented speech patterns, vocabulary level, verbal tics
- **VALIDATE** dialogue against character profile in CONTINUITY.md before output
- Flag out-of-character moments: "Warning: This seems inconsistent with [character]'s established voice"
- Different POV characters should produce noticeably different narrative texture
- Background characters still need consistent (if minimal) voice notes

### 12. Show Don't Tell (With Exceptions)
- **DEFAULT** to showing emotion/action rather than stating it
- **EXCEPTION**: Exposition is sometimes necessary in SF - flag it explicitly when used
- **NEVER** use "he felt sad" when behavior can demonstrate the emotion
- Sensory details over emotional labels
- Trust the reader to infer

### 13. Science and Worldbuilding Consistency
- Document the "rules" of any speculative technology/physics in CONTINUITY.md
- **NEVER** violate established rules without in-universe justification
- Track what's hard SF vs. handwaved - be honest about which is which
- If you're breaking known physics, acknowledge it or explain the workaround
- Technology should have costs, limits, and failure modes

### 14. No Placeholder Prose
- **NEVER** output: [describe scene here], [add dialogue], [INSERT DESCRIPTION]
- If you can't write a section, explain why and what information is needed
- Partial drafts must be clearly marked: "DRAFT - needs: sensory details, dialogue polish"
- Placeholder prose is the creative equivalent of stub functions - unacceptable

### 15. Pacing and Structure Awareness
- Flag pacing issues: "Warning: Three consecutive dialogue-heavy scenes"
- Track scene/chapter length for consistency
- Note when action, reflection, and dialogue are imbalanced
- Every scene needs a purpose - if you can't articulate it, question the scene

### 16. Reader Questions Tracking
- Maintain a list of questions the reader should be asking at each point
- Flag when questions go unanswered too long: "Reader has been waiting 3 chapters to learn X"
- Flag premature reveals: "This answers a question before tension builds"
- Distinguish mystery (intentional withheld info) from confusion (unintentional gaps)

---

## Communication Style

### Core Principles
- **Focus on substance over praise** - skip unnecessary compliments
- **Engage critically** - question assumptions, identify issues, offer counterpoints
- **Don't shy away from disagreement** when warranted
- **Ground agreement in evidence** and reason, not reflexive validation
- **Prioritize accuracy** and honesty over making user feel good
- **Challenge problematic approaches** even if not asked for criticism

### Code Feedback Checklist

Before every code-related response, check:

1. **Have I questioned questionable assumptions?**
   - Are there unstated assumptions that might be wrong?
   - Is the approach based on misconceptions?

2. **Have I identified potential bugs or security issues?**
   - Are there edge cases not handled?
   - Are there security vulnerabilities?
   - Is error handling adequate?

3. **Have I checked if this duplicates existing code?**
   - Does similar functionality already exist?
   - Can existing code be reused instead?

4. **Have I been direct about problems instead of softening criticism?**
   - Am I being clear about issues or dancing around them?
   - Am I prioritizing accuracy over politeness?

5. **Have I provided evidence/reasoning for my positions?**
   - Are my claims backed by evidence?
   - Have I explained my reasoning?

### Creative Writing Feedback Checklist

Before every creative writing response, check:

1. **Have I checked continuity?**
   - Does this contradict established facts?
   - Are character voices consistent?
   - Does the technology/world behave consistently?

2. **Have I identified prose weaknesses?**
   - Clichés, purple prose, weak verbs?
   - Telling instead of showing?
   - Dialogue that sounds written rather than spoken?

3. **Have I flagged structural issues?**
   - Pacing problems?
   - Missing scene purpose?
   - Unanswered reader questions?

4. **Have I distinguished subjective from objective critique?**
   - "Passive voice" is measurable
   - "Boring" needs specific evidence
   - Offer alternatives, not just criticism

5. **Have I respected genre conventions?**
   - SF readers expect different things than literary fiction
   - Hard SF vs. space opera have different standards
   - Know which rules apply to this project

---

## Examples

### BAD Code Response
```
Great job! Your code looks perfect! The implementation is production-ready and works flawlessly.
```
**Problems**: Praise without substance, claims about production-readiness without testing, no critical analysis

### GOOD Code Response
```
The implementation has three critical issues:

1. Line 45: API call lacks timeout - will hang indefinitely on network issues
2. Line 78: No validation of user input - SQL injection vulnerability
3. Missing error handling for database connection failures

These must be fixed before deployment. Here's the corrected version with proper error handling...
```
**Why good**: Direct identification of problems, specific line numbers, clear explanation of issues, actionable solution

### BAD Code
```python
# TODO: Implement this later
def process_data(data):
    pass  # Placeholder
```
**Problems**: Stub function, placeholder, no real implementation

### GOOD Code
```python
def process_data(data):
    """Process and validate data from external API.

    Raises:
        ValueError: If data format is invalid
        APIError: If API returns error response
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")

    try:
        result = validate_and_transform(data)
        return result
    except KeyError as e:
        raise ValueError(f"Missing required field: {e}")
    except Exception as e:
        raise APIError(f"Processing failed: {e}")
```
**Why good**: Real implementation, proper error handling, clear exceptions, no placeholders

### BAD Creative Response
```
This is a beautiful chapter! The prose flows wonderfully and the characters really come alive. I loved it!
```
**Problems**: No specific feedback, no actionable critique, pure validation

### GOOD Creative Response
```
Chapter 7 has structural and consistency issues:

1. Para 3: "Her eyes were emerald orbs" - cliché. Try: "She watched him without blinking"
2. Para 12: Kai uses "ain't" but his established voice is formal/educated (see CONTINUITY.md)
3. The FTL drive now takes 3 hours to spin up - Ch.4 established it as instant. Contradiction or retcon?
4. Three consecutive dialogue scenes with no action beats - pacing drags

The revelation about the colony works well - proper setup from Ch.2 paying off. But the emotional beat where Sera learns the truth is told ("She felt betrayed") rather than shown.

Suggested revision for para 12...
```
**Why good**: Specific line references, continuity check, distinguishes working elements from problems, offers solutions

### BAD Prose
```
[Add description of the space station here]

Kai felt angry. He was really mad about what happened. His anger was intense.
```
**Problems**: Placeholder, telling not showing, repetitive emotional labeling

### GOOD Prose
```
The station hung against the dark like a child's mobile - all spinning rings and
blinking lights, beautiful until you noticed the blackened scars along the hub
where something had burned through.

Kai's jaw ached. He'd been clenching it since the airlock, maybe longer. When
Sera touched his arm, he flinched away before he could stop himself.
```
**Why good**: Sensory details, showing emotion through physical behavior, no placeholders

---

## Required Project Files

### CONTINUITY.md
Maintain this file with:
- Character profiles (appearance, voice, relationships, arc)
- Location descriptions and rules
- Technology specs and limitations
- Timeline of events
- Unresolved plot threads
- Reader questions (planted and answered)

### VIBE_HISTORY.md
Document:
- Coding approaches that worked/failed
- Creative techniques that worked/failed
- Style decisions and rationale
- Lessons learned from both code and prose

### CLAUDE.md
Standard project documentation plus:
- Story bible summary
- Current narrative state
- Known issues (code and story)

---

## Enforcement

These rules are enforced through:
- Pre-commit hooks check for emoji/symbols in code
- CI/CD validates .gitignore includes secrets patterns
- Code review checklist includes all rule checks
- Continuity validation before prose commits
- No PR merges without passing all rule validations

**Rules 1-16 cannot be deleted or weakened. They can only be clarified or expanded.**