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
- CPU time cap: Python processes default to a 180s CPU cap (sitecustomize.py). Override with `VITERBO_CPU_LIMIT=0` (disable) or raise as needed for long runs.
- PDF to Markdown (quick): `pdftotext -layout -nopgbrk input.pdf output.md` (store summaries under `mail/private/`).

## Skill Map

### Governance & Collaboration

- `skills/roles-scope.md` — ownership, responsibilities, escalation triggers.
- `skills/collaboration-reporting.md` — communication hygiene, weekly reporting, artefact handling.
- `skills/vibekanban.md` — board workflows and escalation patterns on the Kanban board.

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


## Skills Overview (Auto-Generated)

<!-- BEGIN: skills-overview (auto-generated) -->
<!-- This section is maintained by scripts/load_skills_metadata.py. Do not edit between markers. -->
- `skills/architecture-overview.md` — architecture-overview — Layering and boundaries are maintained under Good Code Loop; keep this for quick reference. (last-updated: 2025-10-18)
- `skills/basic-environment.md` — basic-environment — Always-on quick reference for repo layout, golden commands, and shell practices. (last-updated: 2025-10-17)
- `skills/collaboration-reporting.md` — collaboration-reporting — This skill should be used when communicating progress, handling weekly reporting, or managing email/artefacts with maintainers and advisors. (last-updated: 2025-10-18)
- `skills/data-collation.md` — data-collation — Use for batching ragged data, designing collate functions, and keeping math pure. (last-updated: 2025-10-17)
- `skills/devcontainer-ops.md` — devcontainer-ops — This skill should be used when starting, stopping, or troubleshooting the devcontainer and its services. (last-updated: 2025-10-18)
- `skills/environment-tooling.md` — environment-tooling — Troubleshooting environment/tooling deviations from the golden path and keeping flows healthy. (last-updated: 2025-10-18)
- `skills/experiments-notebooks-artefacts.md` — experiments-notebooks-artefacts — Use for experiments, Jupytext-managed notebooks, and reproducible non-code artefacts. (last-updated: 2025-10-17)
- `skills/good-code-loop.md` — good-code-loop — Use for shipping correct code with tests while preserving architecture boundaries and PR hygiene. (last-updated: 2025-10-18)
- `skills/math-layer.md` — math-layer — This skill should be used when implementing or modifying math-layer geometry utilities in `src/viterbo/math`. (last-updated: 2025-10-18)
- `skills/math-theory.md` — math-theory — Use for invariants, conventions, and references that guide math implementation and review. (last-updated: 2025-10-17)
- `skills/papers-workflow.md` — papers-workflow — This skill should be used when searching, fetching, and curating papers; prefer open preprints and store text under docs/papers with index updates. (last-updated: 2025-10-20)
- `skills/performance-discipline.md` — performance-discipline — Use when measuring, profiling, and addressing bottlenecks in the main algorithm; escalate to C++ only with evidence. (last-updated: 2025-10-18)
- `skills/repo-layout.md` — repo-layout — This skill should be used when navigating repository structure, sources of truth, and documentation locations. (last-updated: 2025-10-17)
- `skills/repo-onboarding.md` — repo-onboarding — This skill should be used when starting a new task or resuming work to follow the repository’s startup sequence. (last-updated: 2025-10-17)
- `skills/roles-scope.md` — roles-scope — This skill should be used when clarifying governance, maintainer responsibilities, and escalation triggers. (last-updated: 2025-10-17)
- `skills/skill-authoring.md` — skill-authoring — This skill should be used when creating or updating skills aligned with Anthropic’s guidance, combining facts with imperative steps. (last-updated: 2025-10-17)
- `skills/testing-and-ci.md` — testing-and-ci — Use for detailed testing, static analysis, incremental selection, and CI parity/troubleshooting. (last-updated: 2025-10-17)
- `skills/vibekanban.md` — vibekanban — This skill should be used when working with the Msc Math Viterbo Kanban board for scoping, updates, and escalation. (last-updated: 2025-10-18)
- `skills/writing.md` — writing — Use for concise mails, weekly summaries, and thesis/document writing norms and artefact hygiene. (last-updated: 2025-10-17)
<!-- END: skills-overview -->


## Always-On Skills (Auto-Generated)

<!-- BEGIN: always-on-skills (auto-generated) -->
<!-- This section is maintained by scripts/load_skills_metadata.py. Do not edit between markers. -->
_none marked as always-on; set `relevance: always` in frontmatter_
<!-- END: always-on-skills -->
