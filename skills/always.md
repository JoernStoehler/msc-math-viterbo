---
name: always
description: This skill should be used every time — core norms, handoff, minimal commands, and how to orient in this repo.
last-updated: 2025-10-18
relevance: always
---

# Always-On Knowledge

Use this guidance on every task. Keep work focused, avoid ceremony, and hand off cleanly.

## Instructions

- Boot sequence
  - Read `AGENTS.md` and use the embedded Skills Overview to open only the skills needed for your task.
  - Draft a short plan unless the task is trivial; keep exactly one step `in_progress` and update after major steps.

- What you do not need to do
  - No manual Git workflow is required (commit, PR, merge, rebase) unless explicitly asked.
  - No manual Kanban status changes are required; updates happen implicitly via handoff and maintainer actions.

- Handoff expectations
  - Provide a concise final message that states: goal, what changed, how to validate, and open questions.
  - Leave the workspace in a runnable state; prefer `just checks` passing or clearly call out deviations.
  - Use `Needs-Unblock: <topic>` early if you’re blocked or scope is ambiguous.

- Commands
  - Use the quick gate and lint workflow described in `skills/operating-environment.md` and `skills/testing-and-troubleshooting.md`. Provisioning keeps AGENTS.md in sync; you typically don’t need to run setup manually.

- Repository orientation (quick facts)
  - Code lives under `src/`, tests in `tests/`, docs in `docs/`, skills in `skills/`.
  - See `skills/navigating-the-repository.md` for canonical locations and sources of truth.
  - Store large outputs under `artefacts/` (git-ignored); avoid committing binaries.

## Guardrails & Working Norms

- Avoid destructive commands (resets, wide deletes) without maintainer approval.
- Keep changes minimal and scoped; do not fix unrelated issues unless requested.
- Prefer high-signal diffs and clear summaries over long narratives.
- Prefer `rg` for searches and `uv run` for Python entry points.
- Preserve notebook front-matter for Jupytext-managed `.py` notebooks.
- Use `skills/authoring-skills.md` when adding or modifying skills; keep metadata updated.

## Quick Checklist

1. Read task in context and skim AGENTS.md overview.
2. Select and open only the relevant skills.
3. Draft a short plan; update after major steps.
4. Check `git status -sb`; do not revert files you didn’t touch.
5. Keep large outputs under `artefacts/`; avoid committing binaries.
6. Capture questions and add `Needs-Unblock: <topic>` early if blocked.

## Pointers

- Commands and environment: `skills/operating-environment.md`
- Testing workflows: `skills/testing-and-troubleshooting.md`
- Repository structure: `skills/navigating-the-repository.md`

## Related Skills

 - `skills/navigating-the-repository.md` — directory structure and sources of truth.
- `skills/executing-daily-workflow.md` — planning cadence and handoff-first expectations.
- `skills/interacting-with-vibekanban.md` — explicit board interaction guidance.
