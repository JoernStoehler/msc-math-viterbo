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
- Dependency management uses `uv`. The `Justfile` wraps common workflows and keeps the golden path
  in sync with CI.
- To bootstrap locally, run `just setup` (idempotent) from the repo root.

## Command Reference

```
just checks      # Fast loop: lint-fast → type → test-fast
just test        # Smoke-tier tests (parallel unless testmon enabled)
just test-fast   # FAST-mode tests (CPU/no-JIT; single-process with testmon)
just type        # Pyright basic over src/ (fast loop)
just type-strict # Pyright strict across the repository
just lint        # Ruff lint + metadata check (CI parity)
just format      # Ruff format
just fix         # Ruff format + autofix
just sync        # Install project and dev dependencies via uv
just ci          # CI parity (sync → waivers → lint → type-strict → pytest)
just build       # Build sdist/wheel into dist/
just publish     # Publish dist/* to index (requires PYPI_TOKEN)
just release L   # Bump semver (patch|minor|major), commit, tag
just bench       # Benchmarks (smoke tier)
just docs-build  # Build MkDocs site with strict checks
```

Additional helpers include profiling targets (`just profile`, `just profile-line`) and the logistic
regression experiment pipeline (`just train-logreg`, `just evaluate-logreg`, `just publish-logreg`).
The training command expects `WANDB_API_KEY` in your environment (see the Justfile for details).

## Typing & Linting

- The default Pyright profile (`pyrightconfig.json`) uses basic mode for day-to-day loops; CI flips
  to strict mode via `pyrightconfig.strict.json`. Repository-local stubs live under
  `typings/jax/`—keep signatures accurate and prefer jaxtyping annotations with explicit shape
  tokens.
- Ruff enforces the Google docstring convention (with curated exceptions) and bans relative imports.
- Optional runtime jaxtyping checks can be enabled during tests via `JAXTYPING_CHECKS=1 just test`.

## Tests & Benchmarks

- Unit and integration tests reside in `tests/viterbo/`; performance benchmarks live in
  `tests/performance/viterbo/`.
- Export `FAST=1` (or run `just quick`) to force CPU/no-JIT defaults and automatically skip tests
  marked `slow`, `gpu`, `jit`, or `integration`.
- CI delegates to `just ci` (see `.github/workflows/ci.yml`) for parity with the local loop. A
  scheduled workflow runs weekly performance benchmarks and uploads `.benchmarks/` artefacts.
- Stick to deterministic seeds; tolerances default to `rtol=1e-9`, `atol=0.0` via the shared pytest
  fixture (`tests/conftest.py`).

### Testing Policy

- Every test declares exactly one goal marker: `@pytest.mark.goal_math`, `@pytest.mark.goal_code`,
  or `@pytest.mark.goal_performance`.
- Each test starts with a short docstring stating the invariant or behaviour under test.
- Quick inspection commands:
  - `just test-metadata` to list tests with goal markers and docstrings.
  - `just lint` runs the same metadata check used in CI.

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
