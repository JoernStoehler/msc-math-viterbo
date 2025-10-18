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

## VK‑Safe Formatting

- Wrap identifiers and paths in backticks: use `snake_case`, `path/to/file.py`, and `ENV_VARS` to avoid underscore‑as‑italics.
- Escape underscores in plain prose when not using backticks: write `\_` instead of `_`.
- Prefer Unicode math (ℝ, ℤ, ≤, ≥, ×, ∑) over LaTeX.
- LaTeX: not guaranteed. If you must include it, fence it: 
  - Inline: prefer `` `x_i ≤ y_j` `` using Unicode subscripts when possible.
  - Block: use fenced code with a language hint for readability:
    ```
    ```latex
    \int_0^1 f(x)\,dx \leq 1
    ```
    ```
- Avoid raw HTML; assume HTML is sanitized or stripped by VK.
- Lists and code blocks are safest for structured content; avoid ambiguous mixes like `_id_ is set to x_i` outside code.

Tip: For longer notes, draft in Markdown locally with these rules, then paste into VK. The API sanitizer preserves URLs and link destinations while escaping only intraword underscores elsewhere.

## Renderer Caveats

- Underscore emphasis can eat characters in identifiers (e.g., `_foo_bar_` → italics). Use backticks or escapes.
- LaTeX delimiters `$...$`/`$$...$$` may not render; prefer Unicode math. If needed, fenced `latex` blocks display as monospaced text.
- HTML is not supported; assume server‑side sanitization removes tags.

Debugging rendering behavior
- Prefer inspecting the upstream frontend Markdown code in the official `bloop/vibe-kanban` repo over curling minified assets.
- Use small, targeted ticket updates to test changes; avoid bulk edits.

## Upstream Recommendations (for VK maintainers)

- Disable intraword underscore emphasis in the Markdown renderer (e.g., marked: `gfm: true, pedantic: false`; markdown‑it: treat underscores strictly or use a rule that ignores intraword `_`).
- Sanitize HTML (`html: false` in markdown‑it or DOMPurify post‑processing).
- Consider an opt‑in KaTeX/MathJax for `$...$` and `$$...$$`, or document non‑support explicitly in the UI help.

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
