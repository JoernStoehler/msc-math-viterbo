---
name: repo-onboarding
description: Establish a reliable first workflow when joining the Viterbo project branch and align with the board-driven process within the first task.
last-updated: 2025-10-17
---

# Repo Onboarding

## Purpose

Ensure new agents follow the mandatory startup sequence before making code changes, grounding every task in shared policies and fresh repository context.

## Checklist

1. Review the task brief on VibeKanban (`Msc Math Viterbo`) and confirm scope with the maintainer notes.
2. Read `AGENTS.md` fully and capture escalation triggers relevant to the task.
3. Run `uv run python scripts/load_skills_metadata.py` to load current skill metadata into context.
4. Load additional skills based on task type:
   - Environment or container change → `skills/devcontainer-ops.md`
   - Library or math code change → `skills/coding-standards.md`
   - Math-layer geometry work → `skills/math-layer.md`
   - Test or CI related change → `skills/testing-workflow.md`
   - Cross-team communication or weekly updates → `skills/collaboration-reporting.md`
   - Performance analysis or benchmarking → `skills/performance-discipline.md`
   - Notebook or exploratory analysis → `skills/notebook-etiquette.md`
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
