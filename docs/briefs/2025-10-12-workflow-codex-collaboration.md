---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Capture collaboration patterns and authoring norms for Codex agents on this project.
---

# Workflow: Working with Codex

## Context

- Codex agents execute scoped tasks under the maintainer’s direction; predictability and explicit instructions keep iteration fast.
- Repository layout, tooling, and automation are documented in `AGENTS.md`; this brief focuses on collaboration habits and writing norms.

## Objectives

- Encourage concise, explicit communication tailored to Codex agents’ high read-through speed.
- Reinforce conventions that keep the repository conventional and automation friendly.
- Provide quick reminders on ownership boundaries and continuous-improvement expectations.

## Execution

- **Write for agents.** Capture assumptions, workflows, and decisions in-repo; use precise terminology and predictable patterns so agents can reuse context.
- **Structure the project.** Stick to standard Python tooling (Python 3.12, pytest, Ruff, Pyright). Surface copyable commands in `AGENTS.md` and avoid bespoke wrappers.
- **Clarify ownership.** Maintainer sets direction and reviews; agents implement focused changes, clarify ambiguities, and flag scope creep rather than self-expanding tasks.
- **Improve continuously.** When gaps appear, open issues or briefs with concrete suggestions instead of ad-hoc fixes.
- **Short-form modes.** Recognise request modes such as READ ONLY, IMMEDIATE, and TALK ONLY to align response style and command usage.

## Acceptance

- Collaboration guidance in this brief mirrors `AGENTS.md` and current maintainer expectations.
- Agents can point to this document when requesting clarifications or proposing documentation updates.

## Status Log

- 2025-02-14 — Migrated legacy `docs/05-working-with-codex.md` into the briefs tree; emphasised short-form modes and ownership split.
