name: executing-daily-workflow
description: This skill should be used when executing the daily workflow, planning work, and handing off changes.
last-updated: 2025-10-18
---

# Executing the Daily Workflow

## Instructions
- Start from the task text in your context; skim AGENTS.md’s Skills Overview and open only the relevant skills. Draft a short plan (4–7 steps) before coding.
- Implement in small increments; run `just checks` regularly and document deviations.
- Handoff with a concise final message (goal, changes, validation, open questions). The maintainer can open a PR if needed.

## Task Kickoff

1. Read the task message injected from VibeKanban and capture scope notes.
2. Use AGENTS.md’s Skills Overview to load required skills (provisioning auto-refreshes sections).
3. Draft a 4–7 step plan unless the task is trivial. Update the plan after major steps per planning policy.
4. When posting to VK, follow `skills/interacting-with-vibekanban.md` VK‑Safe Formatting: use backticks for identifiers, escape underscores in prose, prefer Unicode math.

## Execution Loop

1. Implement cohesive changes in small increments.
2. Run `just checks` before handoff or whenever significant code is ready.
3. Keep math code pure; push I/O and side effects to adapters/datasets/models.
4. Avoid destructive Git commands (`git reset --hard`, etc.) unless the maintainer explicitly approves.

## Validation & PR Prep

- For parity, run `just ci` before handoff when changes affect core modules, infra, or cross-cutting behavior.
- Capture any skipped validations in task notes with justification.
- Summarize work in the PR body:
  - Feature changes and scope.
  - Files touched and validation evidence (Ruff/Pyright/Pytest, benchmarks if relevant).
  - Performance delta (if measured), limitations, and follow-ups.
 - Open PRs with GitHub CLI: `gh pr create --body-file docs/PR_TEMPLATE.md` (prefer this over `--body`).

## Related Skills

- `always` — first-action checklist.
- `testing-and-troubleshooting` — details on lint/test commands and troubleshooting.
- `collaborating-and-reporting` — communication cadence and weekly reporting.
