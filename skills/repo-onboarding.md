---
name: repo-onboarding
description: This skill should be used when starting a new task or resuming work to follow the repository’s startup sequence.
last-updated: 2025-10-17
---

# Repo Onboarding

## Instructions
- Read `AGENTS.md` to ground on policies; then run `uv run python scripts/load_skills_metadata.py` to load skills context.
- Follow the checklist below to validate environment, searches, and planning before code changes.
- Raise `Needs-Unblock: onboarding` if scope, environment, or access issues prevent progress.

## Purpose

Ensure new agents follow the mandatory startup sequence before making code changes, grounding every task in shared policies and fresh repository context.

## Checklist

1. Review the task brief on VibeKanban (`Msc Math Viterbo`) and confirm scope with the maintainer notes.
2. Read `AGENTS.md` fully and capture escalation triggers relevant to the task.
3. Run `uv run python scripts/load_skills_metadata.py` to load current skill metadata into context.
4. Load additional skills based on task type:
   - Environment or container change → `skills/devcontainer-ops.md`
   - Golden commands & repo nav → `skills/basic-environment.md`
   - Library or math code change → `skills/good-code-loop.md`
   - Math-layer geometry work → `skills/math-layer.md`
   - Data batching/collation → `skills/data-collation.md`
   - Cross-team communication or weekly updates → `skills/collaboration-reporting.md`
   - Performance analysis or benchmarking → `skills/performance-discipline.md`
   - Notebook or exploratory analysis → `skills/experiments-notebooks-artefacts.md`
5. Inspect `git status -sb` to check for existing worktree changes; do not revert files you did not touch.
6. If the task spans notebooks or artefacts, verify whether the files are under git ignore before editing.
7. Capture outstanding questions for the maintainer early; flag blockers in the task description using `Needs-Unblock: <topic>` when required.

## Working Norms

- Draft a 4–7 step plan unless the task is trivial; update the plan after each major step, aligning with the planning policy spelled out in `AGENTS.md`.
- Defer destructive commands (`git reset --hard`, removing caches) unless the maintainer explicitly approves them.
- Prefer `rg` for searches and `uv run` for Python entry points to match the shared devcontainer environment.
- Use `skills/skill-authoring.md` when adding or modifying skills, keeping metadata accurate.
- Reference `docs/environment.md` to mirror the project owner’s golden-path setup when environment differences arise.
- Preserve notebook front-matter when editing `.py` notebooks maintained by Jupytext.

## Command Quick Reference

- `just checks` — format + lint + type + smoke tests.
- `just fix` — apply Ruff auto-fixes before rerunning checks.
- `just test` — run incremental smoke tests; combine with `INC_ARGS="--debug"` for selector details.
- `just bench` — execute smoke benchmarks; see `skills/performance-discipline.md`.
- `just ci` — full CI parity run before handoff.
- `.devcontainer/bin/owner-up.sh` / `owner-down.sh` — host lifecycle scripts; see `skills/devcontainer-ops.md`.

## Roles and Expectations Recap

- Project Owner/Maintainer (Jörn Stöhler) owns scope, roadmap, DevOps, and merge decisions.
- Academic Advisor (Kai Cieliebak) reviews research outcomes but does not gate code merges.
- Codex agents implement incremental changes, escalate uncertainties promptly, and ensure CI is green before handoff.
- Escalate immediately when encountering ambiguous acceptance criteria, policy conflicts, architecture changes, or potential performance regressions beyond documented thresholds.

## Exit Criteria

- Plans reflect actual progress (updated after first execution).
- Skill metadata script runs without warnings.
- Task notes capture any open questions for the maintainer prior to implementation.
- Relevant downstream skills are loaded and referenced within the task scratchpad or updates.
- Any deviations from standard workflow (e.g., skipped tests) are documented alongside justification.

## Related Skills

- `roles-scope` — governance, responsibilities, and escalation triggers.
- `basic-environment` — golden command palette and shell practices.
- `environment-tooling` — troubleshooting deviations from the golden path.
- `devcontainer-ops` — container lifecycle and troubleshooting.
- `vibekanban` — backlog usage and coordination.
