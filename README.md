# Viterbo Conjecture — Python Project

[![Docs](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml/badge.svg)](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://joernstoehler.github.io/msc-math-viterbo)

Numerical experiments around the Viterbo conjecture built on a JAX-first Python stack. Public APIs
operate on JAX arrays, with NumPy/SciPy interop isolated in thin adapters under
`src/viterbo/_wrapped/`.

## Policy & Onboarding

- `AGENTS.md` is the single source of truth for roles, conventions, and workflows. Treat this README
  as a convenience overview and defer to `AGENTS.md` whenever guidance differs.
- Task briefs, RFCs, and background notes live in `docs/`. Start with `docs/02-project-roadmap.md`
  and `docs/tasks/02-task-portfolio.md` for current priorities.
- A MkDocs site is published from the `docs/` tree; the badge above links to the latest build.

## Environment

- The repository ships a devcontainer; agents start with dependencies pre-installed and
  `JAX_ENABLE_X64=1` enabled (see `.devcontainer/post-start.sh`).
- Dependency management uses `uv`. The `Makefile` wraps common workflows and keeps the golden path
  in sync with CI.
- To bootstrap locally, run `make setup` (idempotent) from the repo root.

## Command Reference

```
make setup       # install project and dev dependencies via uv
make format      # Ruff format + Prettier for Markdown/YAML/JSON
make lint        # Ruff lint + Prettier --check
make typecheck   # Pyright strict
make test        # Pytest suite (unit + integration)
make bench       # Pytest benchmarks in tests/performance/
make ci          # Full CI sequence: waivers → format → lint → typecheck → tests
make docs-build  # Build MkDocs site with strict checks
```

Additional helpers include profiling targets (`make profile`, `make profile-line`) and the logistic
regression experiment pipeline (`make train-logreg`, `make evaluate-logreg`, `make publish-logreg`).
The training command expects `WANDB_API_KEY` in your environment (see Makefile for details).

## Typing & Linting

- Pyright runs in strict mode with repository-local stubs under `typings/jax/`. Unknown or missing
  types surface as errors; keep signatures accurate and prefer jaxtyping annotations with explicit
  shape tokens.
- Ruff enforces the Google docstring convention (with curated exceptions) and bans relative imports.
- Optional runtime jaxtyping checks can be enabled during tests via `JAXTYPING_CHECKS=1 make test`.

## Tests & Benchmarks

- Unit and integration tests reside in `tests/viterbo/`; performance benchmarks live in
  `tests/performance/viterbo/`.
- CI mirrors the golden path (`.github/workflows/ci.yml`): waivers, Ruff format/lint, Prettier,
  Pyright, and Pytest. A scheduled workflow runs weekly performance benchmarks and uploads
  `.benchmarks/` artefacts.
- Stick to deterministic seeds; tolerances default to `rtol=1e-9`, `atol=0.0` via the shared pytest
  fixture (`tests/conftest.py`).

## Docs & Knowledge Base

- Architecture and module overviews live in `docs/22-code-structure.md` and related notes.
- Experiment briefs follow the templates in `docs/tasks/`; drafts and scheduled work appear under
  the corresponding subdirectories.
- `waivers.toml` tracks approved policy exceptions. CI runs `scripts/check_waivers.py` before other
  checks.

## Experiments & Tooling

- Scripts under `scripts/` coordinate reproducible experiments (e.g., the logistic regression toy
  pipeline). Prefer `uv run python scripts/<name>.py` when invoking them directly.
- ML experiment tracking defaults to Weights & Biases; ensure secrets are sourced via `.env` or your
  environment. See `docs/tasks/rfc/2025-10-05-experimental-training-artifacts.md` for conventions.

## License

Distributed under the MIT License (`LICENSE`).
