---
name: collaboration-reporting
description: Handle communication hygiene, weekly reporting, and information sharing norms with maintainers and advisors.
last-updated: 2025-10-17
---

# Collaboration & Reporting

## Communication Channels

- **Primary backlog:** VibeKanban project `Msc Math Viterbo`. Keep task descriptions concise and link to supporting docs when context is needed.
- **Escalations:** Use `Needs-Unblock: <topic>` in task descriptions or open an issue when acceptance criteria are ambiguous, policies conflict, or cross-task architecture changes are required.
- **Maintainer updates:** Summaries and PR notes go to the project owner; the Academic Advisor receives curated updates via meetings or email, not via the repo.

## Email & Artefacts Hygiene

- Never commit verbatim third-party emails. Summarize them under `mail/archive/` with context and references.
- Store private attachments under `mail/private/` (git-ignored). For public references, prefer paraphrased notes.
- Treat unpublished preprints as private unless clearly released (e.g., arXiv). Avoid redistribution; cite conservatively.

## Weekly Reporting Loop

1. Gather completed tasks and outstanding questions from the board.
2. Capture progress summaries under `mail/` using the existing naming pattern. Coordinate with the project owner before opening PRs for reports.
3. Ensure artefacts live under `artefacts/` when outputs are large; keep repository history focused on source changes and concise summaries.

## Collaborating Inside the Repo

- Respect existing worktree changes; do not revert files unless you introduced the modifications or have explicit approval.
- When working on notebooks, preserve Jupytext metadata headers to maintain round-trip safety.
- Cross-reference new skills or docs from `AGENTS.md` once they become canonical, keeping the onboarding funnel in sync.

## Related Skills

- `repo-onboarding` — aligns startup workflow before collaboration activities.
- `performance-discipline` — reference when reporting benchmark results or performance regressions.
