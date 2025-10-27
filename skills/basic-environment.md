---
name: basic-environment
description: Always-on quick reference for repo layout, golden commands, and shell practices.
last-updated: 2025-10-27
---

# Basic Environment & Repo Use

## Instructions
- Use the golden command palette below; prefer `uv run` for Python entry points.
- Keep reads ≤250 lines in shell output and prefer `rg` for navigation.
- Consult `repo-layout` for structure; see `devcontainer-ops` for lifecycle.

## Golden Commands

- `just checks` — format, lint, type, smoke tests.
- `just fix` — Ruff auto-fixes before re-running checks.
- `just test` — incremental tests; debug selector with `INC_ARGS="--debug"`.
- `just ci` — CI parity (lint, type, tests) before PR/handoff.
- `just bench` — smoke-tier benchmarks when performance matters.
- `uv run ...` — execute Python with locked deps.
- `just notebooks-md 'pattern'` — execute and render Jupytext notebooks to Markdown under `docs/notebooks/`.
- `just notebooks-html 'pattern'` — render single-file HTML under `docs/notebooks/html/` (use when a one-file handout is needed).

## Shell Practices

- Prefer `rg --files` and `rg -n <term>` for speed and concise output.
- Stream contents in chunks (≤250 lines) to avoid truncation.
- Avoid manual `pip` inside the repo; rely on `uv` and devcontainer scripts.

## References

- `repo-layout` — canonical structure and sources of truth.
- `environment-tooling` — troubleshooting and deviations from the golden path.
- `devcontainer-ops` — start/stop/rebuild and status checks.
