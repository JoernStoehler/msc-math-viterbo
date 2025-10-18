---
name: navigating-the-repository
description: This skill should be used when navigating repository structure, sources of truth, and documentation locations.
last-updated: 2025-10-18
---

# Navigating the Repository

## Instructions
- Read `AGENTS.md` to align on sources of truth, then use this skill to locate files and policies quickly.
- Prefer `rg` for navigation (e.g., `rg --files`, `rg -n <term>`); keep reads ≤250 lines in the shell.
- When changing structure, preserve layering and update cross-links (skills and docs) in the same PR; escalate larger reorganizations.
- If you find conflicting guidance across files, add `Needs-Unblock: repo-layout` to the task and propose a single canonical location.

## Primary References

- `AGENTS.md` — top-level index; load relevant skills after running the metadata script.
- `skills/` — task-scoped guides with YAML frontmatter metadata. Select entries based on task needs.
- Git history — contains legacy migration notes; rely on current skills for onboarding.

## Key Configuration Files

- `pyproject.toml` — runtime dependencies, Ruff configuration, package metadata.
- `pyrightconfig.json`, `pyrightconfig.strict.json` — static type-check settings.
- `pytest.ini` — default smoke-test markers and options.
- `.github/workflows/ci.yml` — CI pipeline definition.
- `.devcontainer/` — environment setup scripts and host/container lifecycle utilities.
- `Justfile` — canonical task runner commands.

## Library Structure (`src/viterbo/`)

- `math/` — pure geometry and math utilities (Torch tensors). No I/O or hidden state.
- `datasets/` — adapters and collate utilities for ragged data; thin wrappers over math primitives.
- `models/` — experiments and training loops; optional CUDA usage.
- `_cpp/` — C++/pybind11 extensions with Python fallbacks when available.

## Tests & Artefacts

- `tests/` — smoke tests named `test_*.py`; performance benchmarks reside in `tests/performance/`.
- `.benchmarks/` — auto-generated benchmark artefacts (ignored by linting, tracked in Git if relevant).
- `artefacts/` — stored outputs (ignored by Git); reference from task notes instead of committing large binaries.

## Documentation & Notes

- `docs/` — public-facing site content, briefs, workflows.
- `notebooks/` — Jupytext-managed `.py` notebooks; retain metadata headers (see `skills/working-with-notebooks.md`).
- `mail/` — archived summaries, private attachments, and weekly reports (follow `skills/collaborating-and-reporting.md`).

## Related Skills

- `operating-environment` — environment setup, quick commands, and PDF ingestion workflow.
- `understanding-architecture` — high-level dependency layering and design principles.
