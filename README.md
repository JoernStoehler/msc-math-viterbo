# Viterbo Conjecture — Python Project

This repository explores the Viterbo conjecture and related symplectic capacities using a
lightweight Python stack. The focus is on fast iteration with NumPy and small, testable helpers.

## Getting Started

Onboarding & conventions live in `AGENTS.md` (single source of truth).

Preferred workflow (`make` targets, backed by `uv`):

- Install deps: `make setup`
- Run tests: `make test`
- Format code: `make format`
- Lint: `make lint`
- Typecheck: `make typecheck`
- Full CI mirror: `make ci`

## Repository Layout

- `src/viterbo/`: Python package with numerical helpers and experiments.
- `tests/`: Pytest suite that exercises the functional core.
- `docs/`: Project goal, roadmap, math/reference notes (onboarding lives in `AGENTS.md`).
- `.devcontainer/`: Devcontainer config and lifecycle scripts (Local + Codex Cloud).
- `.github/`: CI (ruff, pyright, pytest) and issue/PR templates.
- `tmp/`: Ignored scratch area for notes before they graduate into docs.

## Docs

- Overview: `docs/README.md`
- Reading list: `docs/12-math-reading-list.md`
- Thesis topics (incl. systolic-ratio brief): `docs/11-math-thesis-topics.md`

## CI

GitHub Actions runs Ruff format checks, Ruff lint, Pyright, and pytest on pushes and pull requests
(see `.github/workflows/ci.yml`).

### Performance benchmarks

- A dedicated `performance-benchmarks` job runs weekly (Mondays at 05:00 UTC) or on-demand via
  the **Run workflow** button. It installs the dev dependencies and executes
  `pytest tests/performance --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks`.
- Benchmark outputs are uploaded as a `benchmark-history` workflow artifact so regressions can be
  tracked locally. Download the latest artifact and unpack `.benchmarks/` to compare against your
  branch, or regenerate locally with `make bench` for a quick spot check.

## License

MIT License — see `LICENSE`.
