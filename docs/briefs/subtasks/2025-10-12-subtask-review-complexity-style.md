---
status: draft
created: 2025-10-12
workflow: task
summary: Scope a repository-wide review targeting complexity hotspots and style policy alignment.
---

# Subtask: Review code complexity and style alignment

## Context

- Policies (e.g. no `__all__`, JAX-first design) are codified in `AGENTS.md`, but legacy modules may predate them.
- The repository has grown across modules (`capacity`, `volume`, `spectrum`, `experiments`, notebooks) with varying levels of documentation and structure.
- We need a systematic sweep to flag simplification opportunities and decide whether to update code or adjust conventions.

## Objectives (initial draft)

- Audit the codebase to identify complexity hotspots, dead code, and deviations from current policy.
- Propose refactorings or policy updates to resolve the gaps, prioritised by impact.
- Provide concrete examples illustrating the recommended changes.

## Deliverables (tentative)

- Written report (brief or ADR) summarising findings, recommendations, and proposed follow-up tasks.
- Annotated inventory of affected modules/files with severity tags.
- Optional prototype diffs or scripts demonstrating quick wins (if within scope).

## Dependencies

- Relies on current policy definitions in `AGENTS.md`, lint/typecheck configurations, and waivers to understand expectations.
- Should coordinate with active refactor subtasks to avoid duplicated effort or conflicting edits.
- May reference metrics or insights from existing tests/benchmarks to substantiate findings.

## Acceptance criteria (to validate completion)

- The review document enumerates complexity/style issues by module, classifying severity and citing evidence (code paths, metrics, or policy references).
- Recommendations include actionable next steps (new subtasks, policy adjustments) with suggested priority levels.
- Any opportunistic cleanups performed during the review are documented and limited in scope.
- The report is comprehensible without prior meeting notes and includes a summary for quick triage.

## Decisions and constraints

- Scope spans the entire repository: code, tests, notebooks, and docs.
- Capture recommendations in the report; we will triage into subtasks together afterward.
- Judge complexity via standard engineering heuristics (API size, coupling, readability, maintainability) rather than rigid metrics.
- Small, low-risk cleanups (e.g. single-line fixes) are allowed when they clarify findings—avoid deeper refactors during the review.
- Cross-check other subtasks for planned changes but no special waivers exist beyond that.
- Aim for full policy alignment, calling out contradictions or required updates explicitly.

## Open Questions

1. None currently.

## Notes

- Consider running `just lint`, `just typecheck`, and targeted smoke tests during the review to capture reproducible evidence for identified issues.
