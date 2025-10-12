# Viterbo Conjecture — Python Project

[![Docs](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml/badge.svg)](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://joernstoehler.github.io/msc-math-viterbo)

Numerical experiments around the Viterbo conjecture built on a JAX-first Python stack. Public APIs
operate on JAX arrays, with NumPy/SciPy interop isolated in thin adapters under
`src/viterbo/_wrapped/`.

## Modern namespace overview

The production solvers, generators, and artefact helpers live under `viterbo`. Capacity,
cycle, and spectrum entry points consume the shared tolerance policy in
[`src/viterbo/numerics.py`](src/viterbo/numerics.py) so downstream experiments observe
consistent behaviour. The legacy `viterbo.symplectic` package has been removed; upgrade consumers to
the flat modules via:

- `viterbo.capacity` for Haim–Kislev subset search, Chaidez–Hutchings graph wrappers, and
  Minkowski billiard routines,
- `viterbo.volume` for deterministic volume estimators,
- `viterbo.atlas` for parquet schema helpers once the atlas pipeline is implemented.

Higher-dimensional (≥6D) capacities, cycles, and spectra remain on the documented backlog while the
team scopes combinatorial limits and CI/runtime budgets. See
[`docs/briefs/2025-10-12-task-modernization-roadmap.md`](docs/briefs/2025-10-12-task-modernization-roadmap.md)
for the release note and follow-up plan.

## Policy & Onboarding

- `AGENTS.md` is the single source of truth for roles, conventions, and workflows. Treat this README
  as a convenience overview and defer to `AGENTS.md` whenever guidance differs.
- Task briefs and background notes live in `docs/`. Start with
  `docs/briefs/2025-10-07-task-systolic-overview.md` for the current exploration programme and use
  the neighbouring briefs for execution details. Draft new briefs using the conventions described in
  `docs/briefs/2025-10-08-workflow-brief-authoring.md`.
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

- Core references live in `docs/` (project goal, reading list, symplectic quantity catalogues, and
  capacity/volume surveys).
- Planning, ADRs, and workflow notes live under `docs/briefs/` as dated Markdown files with YAML
  front matter (see `docs/briefs/2025-10-08-workflow-brief-authoring.md`).
- `waivers.toml` tracks approved policy exceptions. CI runs `scripts/check_waivers.py` before other
  checks.

## Experiments & Tooling

- Scripts under `scripts/` coordinate reproducible experiments (e.g., the logistic regression toy
  pipeline). Prefer `uv run python scripts/<name>.py` when invoking them directly.
- ML experiment tracking defaults to Weights & Biases; ensure secrets are sourced via `.env` or your
  environment. Capture reproducibility details inside the relevant brief (e.g.,
  `docs/briefs/2025-10-07-workflow-task-evaluation.md`).

## License

Distributed under the MIT License (`LICENSE`).
