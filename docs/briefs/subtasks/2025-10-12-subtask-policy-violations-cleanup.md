---
status: backlog
created: 2025-10-12
workflow: task
summary: Capture follow-up work to resolve repository policy violations identified in AGENTS.md sweep.
---

# Subtask: Eliminate remaining AGENTS.md policy violations

## Context

- Recent audit flagged multiple legacy patterns contradicting the standing policies in `AGENTS.md`.
- Two violations require coordinated remediation across several modules: prohibited `__all__` re-export lists and disallowed `TypeAlias` typedefs.
- The contradictions are spread across math and datasets namespaces; scheduling a dedicated cleanup will unblock future enforcement work.

## Objectives (initial draft)

1. Inventory every module that still declares an `__all__` list and plan replacements with explicit imports/exports.
2. Locate custom `TypeAlias` definitions and redesign the relevant APIs to rely on direct jaxtyping annotations instead.
3. Prioritize fixes by module criticality (core math first, then datasets/helpers) and quantify the expected diff footprint.

## Deliverables (tentative)

- A checklist enumerating offending files for both violation categories with owners or suggested leads.
- Proposed sequencing for the cleanup, including any prerequisite refactors or tests that need to land first.
- Optional draft diffs or code snippets illustrating the preferred replacement patterns.

## Dependencies

- Requires up-to-date guidance from `AGENTS.md` to verify that fixes align with policy intent.
- Coordinate with maintainers to avoid conflicting edits on modules under active development.
- Ensure lint/typecheck coverage stays green after removing aliases and re-export lists.

## Acceptance criteria (to validate completion)

- Every `__all__` declaration in the codebase is either removed or migrated to a compliant structure.
- All custom `TypeAlias` typedefs are eliminated in favor of direct type annotations using jaxtyping semantics.
- Follow-up issues/PRs document the resolution and include references back to this subtask brief.
- Tests and linting confirm the cleanup did not regress functionality.

## Open Questions

1. Should we enforce the ban on `__all__` via lint rules once the cleanup lands?
2. Do any of the current typedefs encode semantics (e.g., unit conventions) that merit dedicated lightweight classes instead of aliases?

## Notes

- Consider batching removals per namespace to keep review manageable while maintaining atomic commits.
- Capture before/after examples to socialize the updated patterns with contributors.
