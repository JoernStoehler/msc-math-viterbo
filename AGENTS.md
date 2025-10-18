# AGENTS.md

This file is an index. Load the right skill guides for your task.

## Boot Sequence

1. Read your task on VibeKanban (`Msc Math Viterbo`).
2. Run `uv run python scripts/load_skills_metadata.py` (or `--quiet` when scripted) to surface skill summaries.
3. Open the skills that match your scenario using the map below. Treat those files as the authoritative source.
4. For any text you will post to VibeKanban, follow the VK‑Safe Formatting guidance in `skills/vibekanban.md`. The Cloudflare API sanitizer on `/api/*` ensures intraword underscores don’t become italics. For rationale and ops, see `.devcontainer/cloudflare/README.md`. Deploy both Workers via `just cf` (in container) or `bash .devcontainer/bin/admin cf` (host).

## Skill Map

### Governance & Collaboration

- `skills/roles-scope.md` — ownership, responsibilities, escalation triggers.
- `skills/collaboration-reporting.md` — communication hygiene, weekly reporting, artefact handling.
- `skills/vibekanban.md` *(if present)* — board workflows (fallback to `skills/daily-development.md` otherwise).

### Maintaining Skills & Docs

- `skills/skill-authoring.md` — how to create or update skills to Anthropic spec (authoritative).

### Environment & Tooling

- `skills/devcontainer-ops.md` — host/container lifecycle scripts.
- `skills/environment-tooling.md` — command palette, PDF ingestion, shell practices.
- `skills/notebook-etiquette.md` — reproducible notebook workflows and artefact storage.

### Daily Execution

- `skills/repo-onboarding.md` — startup checklist and command quick reference.
- `skills/daily-development.md` — planning cadence, just-command flow, PR expectations.
- `skills/testing-workflow.md` — lint/test commands, incremental selector, troubleshooting.

### Code & Architecture

- `skills/coding-standards.md` — PyTorch-first conventions, typing, docstrings, C++ guidance.
- `skills/math-layer.md` — symplectic geometry focus, tensor semantics, invariants.
- `skills/architecture-overview.md` — layer responsibilities and extension strategy.
- `skills/performance-discipline.md` — benchmarking, profiling, and optimization policy.

### Repository Layout

- `skills/repo-layout.md` — canonical locations for configs, docs, tests, and artefacts.

## Maintaining Skills

- When adding or updating skill files, follow `skills/skill-authoring.md` and ensure metadata stays current.
- `just lint` validates frontmatter via `scripts/load_skills_metadata.py --quiet`; fix warnings before handoff.
