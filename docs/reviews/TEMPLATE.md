# Review XX — <Topic Title>

Provenance
- Topic: XX — <topic name>
- Author: <name or agent>
- Date: <YYYY-MM-DD>
- Commit: <git sha>
- Timebox: <e.g., 60–120 minutes>
- Scope: <what was in/out>
- Version: v0.1

Context
- Briefly state the focus and why this review is being run now.

Inputs & Method
- List the main sources you consulted and the exact commands you ran.
- Examples: `uv run mkdocs build --strict`, `rg -n "pattern"`, `just checks`.

Findings (unsorted)
- Long, unfiltered, unsorted list of observations. Redundancy OK. Be specific and actionable where possible.

Actions (first pass)
- 3–7 concrete, bite-sized actions you propose. Keep scope small; consolidation happens later.

Cross-References
- Topics: T01, T03 (if relevant)
- Files: `path/to/file.md:line`, `mkdocs.yml:line`

Validation
- Record commands and outcomes, e.g., `uv run mkdocs build --strict` — OK; `just checks` — OK.

Limitations
- What you did not review; assumptions; potential blind spots.

Status
- Draft | Finalized

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews
- `uv run mkdocs build --strict` green
- `just checks` green (or N/A for docs-only)
- Actions extracted (even if minimal)
- Cross-refs noted (topics/files)
- Provenance filled
- Title matches pattern `Review XX — <Topic Title>`

