---
name: vibekanban
description: This skill should be used when working with the Msc Math Viterbo Kanban board for scoping, updates, and escalation.
last-updated: 2025-10-18
---

# VibeKanban Workflow

## Instructions
- Track work on project `Msc Math Viterbo`, updating the ticket description/comments with concise progress and blockers.
- Use `Needs-Unblock: <topic>` in the ticket for escalations; link artefacts from `artefacts/` or `mail/`.
- Keep ticket text lean; let board columns encode status and link out to skills/docs for context.

## Board Basics

- Canonical backlog: project `Msc Math Viterbo` on VibeKanban.
- Keep task descriptions concise; link supporting docs or skills when additional context is required.
- Use board columns to encode status—avoid duplicating state strings inside descriptions.

## Task Updates

- Summarize progress and blockers directly in the task comments or description updates.
- Add `Needs-Unblock: <topic>` to the description when escalation is required.
- Cross-link artefacts or benchmark summaries stored under `artefacts/` or `mail/` for reviewer reference.

## VK-Safe Formatting

- Wrap identifiers and paths in backticks (`snake_case`, `path/to/file.py`, `ENV_VARS`) so underscores are not interpreted as italics.
- Escape stray underscores in prose (`\_`) when backticks are impractical.
- Prefer Unicode math symbols (ℝ, ℤ, ≤, ≥, ∑) over LaTeX. If you must include LaTeX, fence it with a language hint:
  ```latex
  \int_0^1 f(x)\,dx \leq 1
  ```
- Avoid raw HTML; assume the renderer sanitizes or strips it.
- Draft longer updates locally with these rules, then paste into VK to avoid formatting glitches.

## Renderer Caveats

- Intraword underscores become italics (`_foo_bar_`); backticks or escapes prevent this.
- `$...$` and `$$...$$` math delimiters are not guaranteed to render; fall back to Unicode or fenced `latex`.
- Lists and fenced blocks survive best; avoid mixing inline formatting that confuses the renderer.

## Debugging Rendering Behaviour

- Inspect markdown handling in `bloop/vibe-kanban` when behaviour changes rather than reverse-engineering minified assets.
- Use small, incremental ticket edits when experimenting; document findings in task notes.

## Upstream Recommendations (for VK maintainers)

- Disable intraword underscore emphasis in the renderer (e.g., GitHub-flavoured Markdown settings).
- Sanitize HTML explicitly (e.g., `html: false` in markdown-it or DOMPurify post-processing).
- Consider optional KaTeX/MathJax support or clearly document the absence of LaTeX rendering in the UI.

## Keywords & Searchability

- Optionally add a `Keywords:` line near the top (e.g., `Keywords: prio:3, math, tests`) when it improves searchability.
- Remove keywords once they stop adding value to avoid drift.

## Coordination Tips

- Sync weekly summaries with `skills/collaboration-reporting.md` to keep the maintainer and advisor aligned.
- Reference the relevant skills in task notes so future agents understand prior context.

## Related Skills

- `repo-onboarding` — initial checklist before touching code.
- `good-code-loop` — planning cadence and validation expectations.
- `collaboration-reporting` — broader communication policies.
