---
name: repo-layout
description: Navigate sources of truth, repository structure, and documentation locations for the project.
last-updated: 2025-10-17
---

# Repository Layout & Sources of Truth

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
- `notebooks/` — Jupytext-managed `.py` notebooks; retain metadata headers (see `skills/notebook-etiquette.md`).
- `mail/` — archived summaries, private attachments, and weekly reports (follow `skills/collaboration-reporting.md`).

## Related Skills

- `environment-tooling` — environment setup, quick commands, and PDF ingestion workflow.
- `architecture-overview` — high-level dependency layering and design principles.
