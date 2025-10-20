# Combined Review — Problem Areas and Decisions Needed

Provenance
- Source: Consolidation of all files under `docs/reviews/`
- Author: Codex CLI Agent
- Date: 2025-10-20
- Scope: Aggregate problematic spots; highlight decisions/approvals needed; propose safe quick wins. This page guides where your input is required before changes.
- Status: Draft

Summary
- Overall foundation is strong: clear layering, Torch-first numerics, good docs and tests posture. Most items are polish or sequencing decisions. The few critical issues are security/ops and policy choices that affect CI and tooling.

Decisions Required (Owner Approval/Input)
- Secrets history rewrite and rotation — critical
  - Ask: Approve purging `.env` from git history and rotating the leaked key; enable secret scanning (pre-commit and CI).
  - Why: `.env` with a real-looking `WANDB_API_KEY` is committed; treat as compromised. `.gitignore` does not protect tracked files.
  - Proposed: Use `git filter-repo`/BFG to remove `.env` across history; replace with sanitized placeholder; rotate API key at provider; add `detect-secrets` or `gitleaks` to pre-commit and a GH Action.
  - Source: `.env:6`, `.gitignore:4`, `.pre-commit-config.yaml:1`, `.github/workflows/ci.yml:1`.

- Pre-commit test runner policy (plugins vs simplification)
  - Ask: Choose one: (A) add `pytest-xdist` and `pytest-testmon` to dev deps, or (B) simplify the pre-commit test hook to drop `--testmon -n auto`.
  - Why: Current hook references plugins not in `dev` extras, causing failures on fresh setups.
  - Proposed: Prefer (B) for simplicity now; revisit once plugin value proven.
  - Source: `.pre-commit-config.yaml:1`, `pyproject.toml:1`.

- CI coverage enforcement thresholds and scope
  - Ask: Approve adding coverage floors to fail CI when below agreed thresholds.
  - Why: Charter sets floors (e.g., Math ≥95%, Datasets ≥60%) but CI does not enforce them today.
  - Proposed: Add `--cov-fail-under` to `Justfile` coverage target and wire an optional `ci-with-coverage`; or set `[tool.coverage.report] fail_under` and run coverage in CI.
  - Trade-off: Slight CI runtime increase; clearer quality bar.
  - Source: `docs/charter.md:1`, `Justfile:198`, `.github/workflows/ci.yml:1`, `pyproject.toml:100`.

- Pyright strict adoption scope
  - Ask: Decide whether to enforce strict typing for select math/public APIs (vs keeping repo-wide basic).
  - Why: Strict config exists but not used by default; could catch more issues at the cost of friction.
  - Proposed: Run strict on `src/viterbo/math/**` in `just type-math-strict` and gate PRs touching math with it; keep basic for the rest.
  - Source: `pyrightconfig.json:1`, `pyrightconfig.strict.json:1`, `Justfile:1`.

- C++ extension build policy (Ninja default and verbosity)
  - Ask: Choose policy: remove `ninja` dependency and keep `USE_NINJA=0`, or honor Ninja by default when present and document override.
  - Why: Declares `ninja` in deps but disables it via env — inconsistent; also no opt-in verbose logs for build failures.
  - Proposed: Honor Ninja when available; add `VITERBO_CPP_VERBOSE=1` to toggle build logs; expand safe exception list for clean fallbacks.
  - Source: `src/viterbo/_cpp/__init__.py:24`, `pyproject.toml:16`.

- Runtime CPU time cap default
  - Ask: Keep or change default `RLIMIT_CPU` (180s) set in `sitecustomize.py` for all Python invocations.
  - Why: Good safety guard, but can surprise during long local runs/builds.
  - Proposed: Document prominently in AGENTS.md and allow easy override; optionally scope default cap to tests only.
  - Source: `sitecustomize.py:33`, `AGENTS.md:1`.

- Advisor-facing overview and “Atlas Tiny” short doc priority
  - Ask: Prioritize creating a one-page advisor overview and a short `docs/datasets/atlas_tiny.md` before the next demo.
  - Why: Improves advisor onboarding and aligns with Charter acceptance criteria.
  - Source: `docs/charter.md:1`, `README.md:1`, `docs/README.md:1`.

- Oriented‑edge 4D spec/impl wording and defaults
  - Ask: Clarify whether to change docs or code: spec says “deterministic, CPU‑only,” but implementation is largely device‑agnostic except for the certified C* builder (CPU).
  - Proposed: Update docs to “deterministic; CPU path for C* builder” and keep current code; also decide defaults for `use_cF_budgets` and `use_memo` (safer off by default).
  - Source: `docs/math/oriented_edge_spectrum_4d.md:1`, `src/viterbo/math/capacity_ehz/stubs.py:241`, `:361`.


Clarifications Requested (Options/Policy)
- DataLoader multi‑worker RNG policy
  - Question: Standardize `worker_init_fn` to avoid RNG replay across workers for datasets using internal Generators?
  - Impact: Doc-only vs helper function; avoids subtle duplication with `num_workers>0`.
  - Source: `src/viterbo/datasets/core.py:22`, `:157`.

- Volume backend sequencing
  - Question: Implement order: triangulation → Lawrence → Monte Carlo, or prioritize MC as a dev oracle first?
  - Impact: Test enablement path and early coverage.
  - Source: `docs/math/volume.md:1`, `src/viterbo/math/volume.py:79`.

- Coverage artifacts in CI
  - Question: Upload HTML/XML coverage as PR artifacts or keep local-only?
  - Impact: Transparency vs CI time/storage.
  - Source: `.github/workflows/ci.yml:1`, `Justfile:198`.

- Reviews in nav
  - Question: Keep only the Reviews index in nav (current policy) or add per-topic entries?
  - Impact: Discoverability for non-authors vs nav sprawl.
  - Source: `mkdocs.yml:21`, `docs/reviews/README.md:1`.


Quick Wins (Safe to Implement Without Decisions)
- Add numeric tolerance helper and use consistently
  - Why: Today `max(sqrt(eps), 1e-9)` is duplicated; central helper improves consistency.
  - Source: `src/viterbo/math/volume.py:68`, `src/viterbo/math/polytope.py:116`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:31`.

- Expand error messages to include actual shapes consistently
  - Why: Some modules already do; standardize across math/datasets.
  - Source: `src/viterbo/math/constructions.py:80`, `src/viterbo/datasets/core.py:157`.

- Add doctest-style examples for key Math APIs in docs
  - Why: Improves readability and builds confidence for new readers.
  - Source: `docs/math/polytope.md:1`, `docs/math/volume.md:1`.

- Mark at least one skill as Always‑On
  - Why: Populate AGENTS.md “Always‑On Skills” block to improve cold-start defaults.
  - Source: `AGENTS.md:90`, `skills/basic-environment.md:1`.

- CI polish: add `--durations=10` to smoke tier
  - Why: Surfaces slowest tests for easy triage without heavy overhead.
  - Source: `pytest.ini:1`, `.github/workflows/ci.yml:1`, `Justfile:1`.

- C++ shim ergonomics
  - Add `VITERBO_CPP_VERBOSE` env gate; provide `just ext-clean` to clear `~/.cache/torch_extensions`.
  - Source: `src/viterbo/_cpp/__init__.py:24`, `Justfile:1`.

- Dataset docs note on multi‑worker seeding pattern
  - Why: Prevents repeated sequences across workers; low-cost doc tweak.
  - Source: `src/viterbo/datasets/core.py:22`.

- Tests: targeted additions
  - Add property test: `random_symplectic_matrix` preserves J.
  - Add two tiny 4D oriented‑edge cases (product and non‑product) toggling `use_memo` and caps.
  - Source: `src/viterbo/math/symplectic.py:1`, `src/viterbo/math/capacity_ehz/stubs.py:241`.


Optional/Controversial (Not inherently “bad”) — flag for discussion
- Keep reviews out of MkDocs nav (index only)
  - Rationale: Reduces nav noise; discoverable via the index and search. Fine to keep as-is.
  - Source: `mkdocs.yml:21`, `docs/reviews/README.md:1`.

- Default CPU time cap in `sitecustomize.py`
  - Rationale: Great safety default; surprising for long local runs. Might scope to tests only, but leaving as-is is defensible.
  - Source: `sitecustomize.py:33`.

- Oriented‑edge memoisation and budgets default off
  - Rationale: Heuristic pruning can compromise completeness; keeping defaults conservative is reasonable.
  - Source: `src/viterbo/math/capacity_ehz/stubs.py:303`.


Deferred Workstreams (Plan & Sequence After Decisions)
- Secrets remediation and guardrails
  - Purge `.env` from history; rotate keys; add secret scans (pre‑commit + CI); add server-side protections.

- CI quality gates
  - Add coverage floors and optional coverage artifacts; wire `SMOKE_TEST_TIMEOUT` properly or remove.

- Volume backends implementation sequence
  - Implement triangulation; enable inter‑algorithm agreement tests; follow with Lawrence (+ rational cert mode); add Monte Carlo estimator.

- C++ policy alignment and ergonomics
  - Decide Ninja default; add verbosity env; add `ext-clean`/`ext-smoke` helpers; consider project-local build dir for easier cleanup.

- Advisor-facing docs
  - Add a one-page overview and `docs/datasets/atlas_tiny.md` to support near-term demos.


Appendix — Source Notes by Topic
- Security & Ops
  - `.env:6` — leaked key in repo
  - `.pre-commit-config.yaml:1` — missing secret scan hooks; testmon/xdist mismatch
  - `.github/workflows/ci.yml:1`, `.github/workflows/docs.yml:1` — permissions and CI posture
  - `.devcontainer/bin/container-admin:1`, `.devcontainer/README.md:1` — cloudflared suite pinning; wrangler version pinning

- Testing & CI
  - `pyproject.toml:100`, `Justfile:198`, `pytest.ini:1`, `scripts/inc_select.py:1` — coverage config without enforcement; incremental selector wiring

- Code & Math
  - `src/viterbo/math/polytope.py:116`, `src/viterbo/math/volume.py:68`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:31` — tolerance duplication
  - `src/viterbo/math/capacity_ehz/stubs.py:241`, `:361` — oriented‑edge CPU path for C*; memo/budgets defaults
  - `src/viterbo/_cpp/__init__.py:24` — Ninja default; verbosity controls

- Docs & Advisor
  - `docs/charter.md:1`, `docs/README.md:1`, `README.md:1` — advisor‑facing gaps (overview; Atlas Tiny short page)

Next Steps
- Please reply with decisions on the “Decisions Required” bullets. I will then:
  1) implement the approved quick wins and policies;
  2) open targeted tickets for the larger workstreams with scoped plans and acceptance checks.

