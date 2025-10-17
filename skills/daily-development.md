---
name: daily-development
description: This skill should be used when executing the daily workflow, planning tasks, and preparing PRs.
last-updated: 2025-10-17
---

# Daily Development Workflow

## Task Kickoff

1. Read the VibeKanban task (project `Msc Math Viterbo`) and capture scope notes.
2. Load required skills after running `uv run python scripts/load_skills_metadata.py`.
3. Draft a 4–7 step plan unless the task is trivial. Update the plan after major steps per planning policy.

## Execution Loop

1. Implement cohesive changes in small increments.
2. Run `just checks` before handoff or whenever significant code is ready.
3. Keep math code pure; push I/O and side effects to adapters/datasets/models.
4. Avoid destructive Git commands (`git reset --hard`, etc.) unless the maintainer explicitly approves.

## Validation & PR Prep

- For parity, run `just ci` before requesting review or opening a PR.
- Capture any skipped validations in task notes with justification.
- Summarize work in the PR body:
  - Feature changes and scope.
  - Files touched and validation evidence (Ruff/Pyright/Pytest, benchmarks if relevant).
  - Performance delta (if measured), limitations, and follow-ups.
 - Open PRs with GitHub CLI: `gh pr create --body-file docs/PR_TEMPLATE.md` (prefer this over `--body`).

## Related Skills

- `repo-onboarding` — first-action checklist.
- `testing-workflow` — details on lint/test commands and troubleshooting.
- `collaboration-reporting` — communication cadence and weekly reporting.
