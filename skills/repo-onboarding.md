---
name: repo-onboarding
description: Establish a reliable first workflow when joining the Viterbo project branch and align with the board-driven process within the first task.
last-updated: 2025-10-17
---

# Repo Onboarding

## Purpose

Ensure new agents understand the mandatory startup sequence before making code changes, grounding every task in shared policies and fresh repository context.

## Checklist

1. Review the task brief on VibeKanban (`Msc Math Viterbo`) and confirm scope with the maintainer notes.
2. Read `AGENTS.md` fully and capture escalation triggers relevant to the task.
3. Run `uv run python scripts/load_skills_metadata.py` to load current skill metadata into context.
4. Load additional skills based on task type:
   - Environment or container change → `skills/devcontainer-ops.md`
   - Library or math code change → `skills/coding-standards.md`
   - Test or CI related change → `skills/testing-workflow.md`
5. Inspect `git status -sb` to check for existing worktree changes; do not revert files you did not touch.
6. If the task spans notebooks or artefacts, verify whether the files are under git ignore before editing.

## Working Norms

- Draft a 4–7 step plan unless the task is trivial; update the plan after each major step, aligning with the planning policy spelled out in `AGENTS.md`.
- Defer destructive commands (`git reset --hard`, removing caches) unless the maintainer explicitly approves them.
- Prefer `rg` for searches and `uv run` for Python entry points to match the shared devcontainer environment.
- Use `docs/creating_skills.md` when adding or modifying skills, keeping metadata accurate.

## Exit Criteria

- Plans reflect actual progress (updated after first execution).
- Skill metadata script runs without warnings.
- Task notes capture any open questions for the maintainer prior to implementation.
