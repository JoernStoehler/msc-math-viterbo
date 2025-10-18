name: interacting-with-vibekanban
description: This skill should be used when tasks require explicit VibeKanban interaction; most tasks do not need the board because the task text is in context and Git ops are handled by the workflow.
last-updated: 2025-10-18
---

# Interacting with VibeKanban

## Instructions
- Tasks are managed on VibeKanban (`Msc Math Viterbo`), but the active task’s text is injected into your context as a user message. You usually do not need to open the board to read or manage the task.
- Most Git operations (commits, PRs, merges) are handled by the project workflow. Focus on making scoped changes and handing off with a concise final message; the maintainer can one‑click open a PR when needed.
- If you explicitly need to manage the board (rare), use this skill as the guide; otherwise, keep working and escalate via notes.

## Board Basics

- Canonical backlog: project `Msc Math Viterbo` on VibeKanban.
- Board interaction is typically unnecessary for agents; status changes are driven by handoff and maintainer actions.
- If editing, keep descriptions concise; link supporting docs when additional context is required.

## Task Updates

- Summarize progress and blockers in your assistant messages and final handoff.
- Use `Needs-Unblock: <topic>` in notes when escalation is required.
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

- Optional: add a `Keywords:` line near the top (e.g., `Keywords: prio:3, math, tests`) only when it materially improves searchability. Remove once it stops adding value.

## Coordination Tips

- Sync weekly summaries with `skills/collaborating-and-reporting.md` to keep the maintainer and advisor aligned.
- Reference specific skills in your assistant updates only when it helps readers; avoid ticket-level micromanagement lists.

## Related Skills

- `always` — initial checklist before touching code.
- `executing-daily-workflow` — planning cadence and validation expectations.
- `collaborating-and-reporting` — broader communication policies.
