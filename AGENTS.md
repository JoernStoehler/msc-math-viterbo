# AGENTS.md

This file is an index. Load the right skill guides for your task.

## Boot Sequence

1. Read your task on VibeKanban (`Msc Math Viterbo`).
2. Run `uv run python scripts/load_skills_metadata.py` (or `--quiet` when scripted) to surface skill summaries.
3. Open the skills that match your scenario using the map below (or see `skills.index.md` for clusters). Treat those files as the authoritative source.

## Always-On Essentials

- Golden commands: `just checks`, `just test`, `just ci`, `just fix`, `just bench`; use `uv run ...` for Python entry points.
- Navigation: prefer `rg` for searches; stream reads ≤250 lines in the shell.
- Escalation: add `Needs-Unblock: <topic>` to tickets; Owner: Jörn Stöhler; Advisor: Kai Cieliebak.
- Safety: do not revert files you didn’t edit; keep device/dtype choices explicit.
- PDF to Markdown (quick): `pdftotext -layout -nopgbrk input.pdf output.md` (store summaries under `mail/private/`).

## Skill Map

### Governance & Collaboration

- `skills/roles-scope.md` — ownership, responsibilities, escalation triggers.
- `skills/collaboration-reporting.md` — communication hygiene, weekly reporting, artefact handling.
- `skills/vibekanban.md` *(if present)* — board workflows (fallback to `skills/daily-development.md` otherwise).

### Maintaining Skills & Docs

- `skills/skill-authoring.md` — how to create or update skills to Anthropic spec (authoritative).

### Environment & Tooling

- `skills/basic-environment.md` — golden command palette and shell practices.
- `skills/devcontainer-ops.md` — host/container lifecycle scripts.
- `skills/environment-tooling.md` — troubleshooting environment/tooling deviations.
- `skills/experiments-notebooks-artefacts.md` — experiments, notebooks, and artefact storage.

### Daily Execution

- `skills/repo-onboarding.md` — startup checklist and command quick reference.
- `skills/good-code-loop.md` — coding standards, architecture guardrails, PR hygiene.
- `skills/testing-and-ci.md` — detailed testing, static analysis, incremental selection, CI parity.
- `skills/vibekanban.md` — planning cadence and escalation on the board.

### Code & Architecture

- `skills/good-code-loop.md` — source+tests loop and architecture guardrails.
- `skills/data-collation.md` — ragged batching and collate helpers.
- `skills/math-layer.md` — implementation guidance for geometry modules.
- `skills/math-theory.md` — invariants and conventions that guide implementation.
- `skills/architecture-overview.md` — condensed layering map (quick reference).
- `skills/performance-discipline.md` — measure/profile, bottlenecks, and escalation to C++.

### Repository Layout

- `skills/repo-layout.md` — canonical locations for configs, docs, tests, and artefacts.

## Maintaining Skills

- When adding or updating skill files, follow `skills/skill-authoring.md` and ensure metadata stays current.
- `just lint` validates frontmatter via `scripts/load_skills_metadata.py --quiet`; fix warnings before handoff.
