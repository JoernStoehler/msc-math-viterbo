# Review 14 — Misc

Provenance
- Topic: 14 — Misc
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563
- Timebox: ~75 minutes (scan → draft → validate)
- Scope: repository hygiene, small risks, QoL improvements, gaps in docs/tests, minor ergonomics
- Version: v0.1

Context
- Purpose: Catch‑all finishing pass to capture observations that didn’t fit other topics (hygiene, small risks, ergonomics). Unfiltered, unsorted list; redundancy OK.
- Why now: Close the loop before consolidation so minor paper cuts and latent risks are visible and can be triaged.

Inputs & Method
- Read: `AGENTS.md`, `mkdocs.yml`, `Justfile`, `.pre-commit-config.yaml`, `.gitignore`, `pyproject.toml`, `.github/workflows/*`, `docs/reviews/*`, selected `src/` and `tests/` modules.
- Commands:
  - `uv run python scripts/load_skills_metadata.py --quiet`
  - `uv run mkdocs build --strict`
  - `just checks`
  - `rg -n "TODO|FIXME|HACK|XXX|TBD|BUG|NOTE\(|NOCOMMIT" -S`
  - `git rev-parse --short HEAD`

Findings (unsorted)
- Critical secret exposure: `.env` is committed with a real‑looking `WANDB_API_KEY` value. `.gitignore` correctly ignores `.env`, but since it’s tracked, ignore rules do not help. Immediate action: remove `.env` from history (filter‑repo or BFG), rotate the API key, and replace the file with placeholders only. Keep `.env.example` as the canonical template.
- Pre‑commit hook deps mismatch: `.pre-commit-config.yaml` local hook runs `pytest --testmon -n auto`, but `pyproject.toml` lacks `pytest-testmon` and `pytest-xdist` in dev extras. Result: pre‑push hook will fail on a fresh dev setup. Either add both plugins to `dev` or drop `--testmon`/`-n auto` in the hook.
- Reviews index small formatting nit: inconsistent bullet indentation under “Published reviews” in `docs/reviews/README.md` (some bullets have a leading space). Cosmetic only; can be harmonized during consolidation.
- Review titles: several review pages include a frontmatter `title:` and also an H1 header (e.g., `04-code.md`). Consistent pattern is fine; consider ensuring all review pages include the frontmatter title for uniform MkDocs rendering.
- MkDocs strict build: baseline is green; logs list many orphan pages not in `nav` (expected per policy). Confirmed that `nav` intentionally keeps only the Reviews index; individual review pages remain discoverable. No broken links observed.
- Skills “always‑on” gap: `AGENTS.md` auto‑generated “Always‑On Skills” section is empty. Consider marking `basic-environment` (and optionally `repo-onboarding`) with `relevance: always` to improve agent defaults and reduce cold‑start friction.
- CI ergonomics are strong: `just ci` mirrors GH Actions, `ci-cpu` installs CPU Torch wheels via pinned indexes. Good separation between uv‑managed venvs and CI system‑Python flow.
- Docs workflow is healthy: `docs.yml` builds with `uv sync --extra dev` and `mkdocs build --strict`, then uploads Pages artifact. No link‑check plugin, but strict mode already catches common issues.
- Site customization: `sitecustomize.py` caps CPU time by default via `RLIMIT_CPU` (180s) unless overridden by `VITERBO_CPU_LIMIT`. Good guardrail, but surprising for long local runs or builds; document prominently in `AGENTS.md` or reduce default for tests only.
- Optional JAX tweak in `sitecustomize.py` enables float64 by default if JAX is present. Benign given try/except; note for users to avoid confusion if they import JAX in notebooks.
- Tests posture: smoke tier is fast and thorough; deep/longhaul gated. Many future tests are pre‑skipped in `tests/math/test_polytope.py` (volume backends) — clear road map; no action needed now.
- Benchmarks saved to `.benchmarks/` and ignored by Git; recipes exist for deep/longhaul — good hygiene.
- Devcontainer: robust bootstrap with idempotent `uv sync`, ownership fixups, and Cloudflare helpers. Docs explain volumes and public exposure via named tunnels; good operational clarity.
- Cloudflare Workers helpers: present under `.devcontainer/cloudflare/` with sanitizer and font injector. Ensure no secrets land in worker config; current files look safe (route/domain docs only).
- `Justfile` commentary drift: comments suggest “checks: removed; prefer precommit”, but `checks` is implemented and used widely (and helpful). Consider updating comments for consistency.
- `train-logreg` recipe: fails fast if `WANDB_API_KEY` is unset (good). Once `.env` is untracked and sanitized, this flow remains correct.
- TODOs in math layer: stubs for `incidence.py` and `convex_hull.py` are clearly marked; keep visible in roadmap to avoid user surprise.
- Documentation stubs: `docs/math/similarity.md` and `docs/math/polar.md` explicitly labeled as stubs with titles — good expectation setting; consider adding a one‑line banner pattern for all stub pages.
- Reviews pre‑submit checklist vs. current constraints: index suggests adding a bullet per review; current task explicitly forbids updating index/nav for now. Capture this tension so consolidation can link later.
- Git attributes/line endings: `.editorconfig` present; Ruff format enforces LF and double quotes. Cross‑platform friction likely low.
- Type checking posture: Pyright basic by default; strict config included for sweeps. Healthy two‑tier approach.
- PyTorch extension shim: safe fallbacks present; caches clearable via `clear_extension_caches` for tests. Good testability design.
- Minor nit: `docs/workflows-project-owner-vibekanban.md` is long; consider adding a short index/TOC at top or splitting operational sections, though not urgent.

Actions (first pass)
- Remove `.env` from Git history; rotate the leaked `WANDB_API_KEY`; keep only `.env.example` committed. Add a one‑time scan with `git secrets` or `trufflehog` to CI.
- Fix pre‑commit hook deps: add `pytest-xdist` and `pytest-testmon` to `dev` extras, or simplify the hook to drop `--testmon -n auto`.
- Mark at least one skill as always‑on (e.g., `basic-environment`) to seed the AGENTS.md “Always‑On Skills” block.
- Normalize review page headers: ensure each review has frontmatter `title:` and a consistent H1, or choose one pattern.
- Harmonize bullet indentation in `docs/reviews/README.md` during the consolidation pass (not in this change).
- Document `VITERBO_CPU_LIMIT` prominently (e.g., AGENTS.md “Always‑On Essentials”) with an example to disable or raise for long jobs.

Cross-References
- Topics: T02, T05, T08, T11, T12 (onboarding, docs, reviews process, owner workflows, project management)
- Files: `.pre-commit-config.yaml:1`, `.gitignore:4`, `.env:6`, `pyproject.toml:23`, `AGENTS.md:90`, `mkdocs.yml:1`, `Justfile:300`, `sitecustomize.py:33`

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK (no output).
- `uv run mkdocs build --strict` — OK (built; orphan pages listed; no errors).
- `just checks` — OK (format, lint, type, smoke tests). Note: warning about empty Always‑On skills in AGENTS.md.
- `rg -n "TODO|FIXME|HACK|XXX|TBD|BUG" -S` — surfaced known stubs/TODOs in `src/viterbo/math/*` and thesis notes.

Limitations
- Did not run deep/longhaul tests or benchmarks; focused on smoke tier and docs.
- Did not edit any CI or review index files per task constraints; suggested changes deferred to consolidation.
- Secret rotation and history rewriting not performed here — flagged for owner action.

Status
- Draft

Pre-Submit Checklist
- Not linked from `docs/reviews/README.md` per constraints (will be linked during consolidation).
- `uv run mkdocs build --strict` green.
- `just checks` green (docs-only change).
- Actions extracted.
- Cross-refs noted.
- Provenance filled.
- Title matches pattern `Review 14 — Misc`.
