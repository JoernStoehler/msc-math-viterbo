Title: <concise, agent-friendly summary>

Executive Summary (3–6 bullets)
- <what changed and why>
- <outcomes and user-visible effects>
- <notable risks/mitigations>

Scope
- Goal: <one feature OR one fix OR one refactor>
- Diff size: <approx LOC>; keep small and focused (≈ ≤300 LOC when practical)

Files Touched
- `path/to/file.py`
- `docs/xyz.md`

What I Read (evidence)
- Pointers that informed the change (files, docs, issues). Use file references like `path/to/file.py:42`.

Changes
- <file/path>: <one-sentence rationale>
- <file/path>: <one-sentence rationale>

Testing
- Commands run locally (paste short summaries):
  - `just format && just lint && just typecheck && just test`
  - Optional full: `just ci`
- Add/updated tests: <brief list>

Performance (if perf-critical code)
- Bench command: `pytest tests/performance -q --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks`
- Delta summary: <short note or link to artifact>; waiver if applicable.

Limitations
- <known gaps or trade-offs>

Clarifications Needed (numbered)
1. <question>. Assumption used: <assumption>.

Follow-ups (numbered, prioritized H/M/L)
1. <task> — <H/M/L>

Checklist
- [ ] `just setup` (first-time env) or environment already prepared
- [ ] `just format && just lint && just typecheck && just test` pass locally
- [ ] `just ci` green locally (recommended before merge)
- [ ] No secrets in code or logs; configuration via env vars

Reviewer Notes (optional)
- <review guidance, noteworthy decisions, or alternative paths considered>
