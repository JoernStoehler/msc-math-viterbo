---
name: understanding-roles-and-scope
description: This skill should be used when clarifying governance, maintainer responsibilities, and escalation triggers.
last-updated: 2025-10-18
---

# Understanding Roles & Scope

## Instructions
- Use this skill to align responsibilities before planning cross-cutting changes or policy updates.
- Capture escalation triggers directly in tasks using `Needs-Unblock: <topic>` and notify the maintainer on blocking decisions.
- Confirm with the maintainer before altering CI/devcontainer workflows or introducing experimental tooling.

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
- Performance regressions observed in benchmarks or user-facing runtime.
- Environment/CI changes that affect other collaborators.

## Expectations

- Maintain clear task notes summarizing progress, open questions, and validation steps.
- Respect existing worktree changes; never revert files you did not edit.
- Always confirm with the maintainer before adopting experimental tooling or altering core workflows.

## Related Skills

- `always` — first-action checklist before implementing changes.
- `collaborating-and-reporting` — communication hygiene and reporting cadence.
