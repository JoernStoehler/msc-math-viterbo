---
status: done
created: 2025-10-12
workflow: task
summary: Document closure of the AGENTS.md policy violations sweep.
---

# Subtask: Eliminate remaining AGENTS.md policy violations

## Resolution

- The sweep for banned `__all__` re-export lists and `TypeAlias` typedefs is complete; neither pattern exists in the current tree under `src/` or `tests/`.
- Existing linting and typing gates are sufficient to prevent regressions, so no ongoing work remains under this brief.
- Future violations should be handled via small targeted patches and cross-referenced back to AGENTS.md rather than reviving this subtask.
