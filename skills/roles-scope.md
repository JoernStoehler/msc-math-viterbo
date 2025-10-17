---
name: roles-scope
description: This skill should be used when clarifying governance, maintainer responsibilities, and escalation triggers.
last-updated: 2025-10-17
---

# Roles & Scope

## Ownership

- **Project Owner/Maintainer — Jörn Stöhler**
  - Owns the technical vision, research roadmap, and repository scope.
  - Approves task briefs and any directional changes.
  - Manages DevOps/CI and merges pull requests (may delegate merges explicitly).
- **Academic Advisor — Kai Cieliebak**
  - Provides scientific guidance and week-scale reorientation.
  - Reviews and signs off on the final thesis text.
  - Does not gate merges and rarely interacts with the repository directly.
- **Codex Agents (ephemeral)**
  - Implement focused, incremental changes scoped to individual tasks.
  - Escalate uncertainties early; keep the maintainer in the loop.
  - Ensure CI is green before handoff and avoid destructive commands unless instructed.

## Escalation Triggers

Use `Needs-Unblock: <topic>` in the task, or raise an issue/maintainer ping, when encountering:

- Ambiguous acceptance criteria or conflicting policies.
- Architectural decisions spanning multiple tasks or layers.
- Performance regressions beyond documented thresholds.
- Environment/CI changes that affect other collaborators.

## Expectations

- Maintain clear task notes summarizing progress, open questions, and validation steps.
- Respect existing worktree changes; never revert files you did not edit.
- Always confirm with the maintainer before adopting experimental tooling or altering core workflows.

## Related Skills

- `repo-onboarding` — first-action checklist before implementing changes.
- `collaboration-reporting` — communication hygiene and reporting cadence.
