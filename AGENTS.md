# AGENTS.md — Onboarding & Conventions (Source of Truth)

This file is the single source of truth for onboarding, workflows, and conventions. If anything elsewhere contradicts this file, update those docs to match this.

**Quick Commands**
- `make setup` — install deps (instantiate)
- `make test` — run tests (incl. Aqua)
- `make format` — run pinned JuliaFormatter
- `make ci` — CI mirror: pinned format diff + tests

**Workflows**
- Setup
  - Open in the devcontainer (VS Code) or run `make setup` locally.
  - Devcontainer installs Julia via Juliaup, symlinks to `~/.local/bin`, then precompiles.
- Daily Dev
  - Implement small, composable, mostly‑pure functions. Keep side effects explicit.
  - Add/adjust unit tests for any new or changed behavior.
  - Add docstrings for public APIs; include brief examples when helpful.
  - Run `make test`; run `make format`.
- Local CI Check
  - Run `make ci` to enforce pinned formatting and run tests before opening a PR.
  - Recommendation: run `make ci` before push/PR to catch issues early.
- Devcontainer (Codex Cloud)
  - One‑time: `bash .devcontainer/post-create.sh`
  - Every boot: `bash .devcontainer/post-start.sh`
  - Mounts are intentionally hardcoded to `/home/codespace/...` (see comments in `.devcontainer/devcontainer.json`).
- Before PR
  - Ensure tests cover changes and public APIs have docstrings.
  - Keep diffs minimal and scoped.
  - No secrets in logs; env‑only auth.

**Conventions**
- Tech Stack
  - Julia 1.11; Linux‑only assumption; single project with committed `Manifest.toml`.
  - Testing via `Test`; Aqua runs with the test target.
  - Formatting via JuliaFormatter with `indent = 2` (pinned in CI).
  - Devcontainer: base image `mcr.microsoft.com/vscode/devcontainers/universal:2`; Julia installed via Juliaup; symlinks to `~/.local/bin`; no rc‑file edits.
- Coding
  - Prefer small, composable, pure functions for math core.
  - Use explicit types only when they clarify intent or performance.
  - Docstrings for public APIs; examples where practical.
  - Unit tests are required for new or changed behavior; keep tests small and focused.
  - Emphasize tests for math correctness and type behavior; Julia’s dynamic typing can hide edge cases across numeric types.
- Project Layout
  - `src/` — module `ViterboConjecture`
  - `test/` — unit tests + Aqua
  - `docs/` — overview + numbered references (project, math, tooling). Onboarding lives in this file.
  - `tmp/` — ignored scratch area for copied papers/notes you intend to incorporate later; do not commit raw notes.
  - `.devcontainer/` — container config and lifecycle scripts
  - `.github/` — workflows and templates
- CI
  - GitHub Actions on push/PR (Linux only): checkout, setup Julia 1.11, cache, instantiate, pinned format check, tests with coverage, produce lcov, upload to Codecov without a token (public repo).
  - Concurrency cancels in‑progress runs per ref.
  - Coverage policy: upload-only; no hard threshold. We monitor and adjust if needed.
  - No version matrix; keep CI simple and fast for the thesis.
- Security
  - Never log secrets; environment variables only; avoid `set -x` and echoing env.

**Release**
- No releases/semver process planned; this is a MSc thesis repository.

**References**
- Project goal: `docs/01-project-goal.md`
- Roadmap: `docs/02-project-roadmap.md`
- Docs overview: `docs/README.md`
- Reading list: `docs/12-math-reading-list.md`
- Thesis topics: `docs/11-math-thesis-topics.md`
- Working with Codex: `docs/21-tool-working-with-codex-agents.md`

**Continuous Improvement**
- If you encounter ambiguity or friction, file an issue with a concrete change proposal. When you fix a discrepancy between this file and another doc, prefer aligning the other doc to this file.
