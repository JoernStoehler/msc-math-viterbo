# Working with Codex

This document summarizes practices to help Codex agents work productively in this Julia project.

## Write for Codex

- Codex reads fast; reasoned bandwidth is precious. Prefer clear, explicit, and predictable writing.
- Document context, assumptions, implications, and workflows in-repo so agents can self-serve.
- Use precise terminology and common patterns so agents can leverage prior knowledge.

## Structure for Codex

- Keep repository layout conventional and explicit (see `AGENTS.md`).
- Prefer standard tools (Julia 1.11, Test, JuliaFormatter, Aqua) and well-known CI patterns.
- Provide short, copyable commands in `AGENTS.md`.

## Ownership & Collaboration

- Human owner sets direction; Codex executes tasks and can propose improvements.
- Agents should clarify ambiguous requirements and avoid scope creep; suggest follow-ups instead.
- Capture non-obvious decisions and learnings as short docs or notes linked from issues/PRs.

## Continuous Improvement

- When agents encounter surprises or unclear documentation, file an issue with a concrete suggestion.
- Prefer small, incremental changes that increase predictability and reduce ambiguity.

## Short-Form Modes

- READ ONLY — investigation/explanation only; no edits
- IMMEDIATE — act quickly, minimal commands, respond once confident
- TALK ONLY — reason in chat; avoid opening files/commands
