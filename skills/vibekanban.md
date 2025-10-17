---
name: vibekanban
description: This skill should be used when working with the Msc Math Viterbo Kanban board for scoping, updates, and escalation.
last-updated: 2025-10-17
---

# VibeKanban Workflow

## Board Basics

- Canonical backlog: project `Msc Math Viterbo` on VibeKanban.
- Keep task descriptions concise; link supporting docs or skills when additional context is required.
- Use board columns to encode status—avoid duplicating state strings inside descriptions.

## Task Updates

- Summarize progress and blockers directly in the task comments or description updates.
- Add `Needs-Unblock: <topic>` to the description when escalation is required.
- Cross-link artefacts or benchmark summaries stored under `artefacts/` or `mail/` for reviewer reference.

## Keywords & Searchability

- Optionally add a `Keywords:` line near the top (e.g., `Keywords: prio:3, math, tests`) when it improves searchability.
- Remove keywords once they stop adding value to avoid drift.

## Coordination Tips

- Sync weekly summaries with `skills/collaboration-reporting.md` to keep the maintainer and advisor aligned.
- Reference the relevant skills in task notes so future agents understand prior context.

## Related Skills

- `repo-onboarding` — initial checklist before touching code.
- `daily-development` — planning cadence and validation expectations.
- `collaboration-reporting` — broader communication policies.
