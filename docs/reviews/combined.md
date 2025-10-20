# Combined Review — Problem Areas and Decisions Needed

Provenance
- Source: Consolidation of all files under `docs/reviews/`
- Author: Codex CLI Agent
- Date: 2025-10-20
- Scope: Aggregate problematic spots; highlight decisions/approvals needed; propose safe quick wins. This page guides where your input is required before changes.
- Status: In progress (statuses added below)

Summary
- Overall foundation is strong: clear layering, Torch-first numerics, good docs and tests posture. Most items are polish or sequencing decisions. The few critical issues are security/ops and policy choices that affect CI and tooling.

Decisions Required (Owner Approval/Input)
(Status markers: Done · Approved · Rejected · Deferred · N/A)
- Secrets history rewrite and rotation — critical
  - Status: N/A (verified `.env` is ignored and not in history); minimal guard added (Done)
  - Ask: Approve purging `.env` from git history and rotating the leaked key; enable secret scanning (pre-commit and CI).
  - Why: `.env` with a real-looking `WANDB_API_KEY` is committed; treat as compromised. `.gitignore` does not protect tracked files.
  - Proposed: Use `git filter-repo`/BFG to remove `.env` across history; replace with sanitized placeholder; rotate API key at provider; add `detect-secrets` or `gitleaks` to pre-commit and a GH Action.
  - Source: `.env:6`, `.gitignore:4`, `.pre-commit-config.yaml:1`, `.github/workflows/ci.yml:1`.

- Pre-commit test runner policy (plugins vs simplification)
  - Status: Done (simplified to our incremental selector; dropped testmon/xdist)
  - Ask: Choose one: (A) add `pytest-xdist` and `pytest-testmon` to dev deps, or (B) simplify the pre-commit test hook to drop `--testmon -n auto`.
  - Why: Current hook references plugins not in `dev` extras, causing failures on fresh setups.
  - Proposed: Prefer (B) for simplicity now; revisit once plugin value proven.
  - Source: `.pre-commit-config.yaml:1`, `pyproject.toml:1`.

- CI coverage enforcement thresholds and scope
  - Status: Done (global floor 85% via `ci-cpu`; no coverage artifact uploads by default)
  - Ask: Approve adding coverage floors to fail CI when below agreed thresholds.
  - Why: Charter sets floors (e.g., Math ≥95%, Datasets ≥60%) but CI does not enforce them today.
  - Proposed: Add `--cov-fail-under` to `Justfile` coverage target and wire an optional `ci-with-coverage`; or set `[tool.coverage.report] fail_under` and run coverage in CI.
  - Trade-off: Slight CI runtime increase; clearer quality bar.
  - Source: `docs/charter.md:1`, `Justfile:198`, `.github/workflows/ci.yml:1`, `pyproject.toml:100`.

- Pyright strict adoption scope
  - Status: Deferred (keep repo-wide basic; strict available on demand)
  - Ask: Decide whether to enforce strict typing for select math/public APIs (vs keeping repo-wide basic).
  - Why: Strict config exists but not used by default; could catch more issues at the cost of friction.
  - Proposed: Run strict on `src/viterbo/math/**` in `just type-math-strict` and gate PRs touching math with it; keep basic for the rest.
  - Source: `pyrightconfig.json:1`, `pyrightconfig.strict.json:1`, `Justfile:1`.

- C++ extension build policy (Ninja default and verbosity)
  - Status: Done (honor Ninja in devcontainer; `VITERBO_CPP_VERBOSE` added)
  - Ask: Choose policy: remove `ninja` dependency and keep `USE_NINJA=0`, or honor Ninja by default when present and document override.
  - Why: Declares `ninja` in deps but disables it via env — inconsistent; also no opt-in verbose logs for build failures.
  - Proposed: Honor Ninja when available; add `VITERBO_CPP_VERBOSE=1` to toggle build logs; expand safe exception list for clean fallbacks.
  - Source: `src/viterbo/_cpp/__init__.py:24`, `pyproject.toml:16`.

- Runtime CPU time cap default
  - Status: Done (keep 180s; override documented in AGENTS.md)
  - Ask: Keep or change default `RLIMIT_CPU` (180s) set in `sitecustomize.py` for all Python invocations.
  - Why: Good safety guard, but can surprise during long local runs/builds.
  - Proposed: Document prominently in AGENTS.md and allow easy override; optionally scope default cap to tests only.
  - Source: `sitecustomize.py:33`, `AGENTS.md:1`.

- Advisor-facing overview and “Atlas Tiny” short doc priority
  - Status: Deferred (add only if it gates an advisor demo)
  - Ask: Prioritize creating a one-page advisor overview and a short `docs/datasets/atlas_tiny.md` before the next demo.
  - Why: Improves advisor onboarding and aligns with Charter acceptance criteria.
  - Source: `docs/charter.md:1`, `README.md:1`, `docs/README.md:1`.

- Oriented‑edge 4D spec/impl wording and defaults
  - Status: Done (docs updated to reflect CPU path for C*; defaults conservative)
  - Ask: Clarify whether to change docs or code: spec says “deterministic, CPU‑only,” but implementation is largely device‑agnostic except for the certified C* builder (CPU).
  - Proposed: Update docs to “deterministic; CPU path for C* builder” and keep current code; also decide defaults for `use_cF_budgets` and `use_memo` (safer off by default).
  - Source: `docs/math/oriented_edge_spectrum_4d.md:1`, `src/viterbo/math/capacity_ehz/stubs.py:241`, `:361`.


Clarifications Requested (Options/Policy)
- DataLoader multi‑worker RNG policy
  - Status: N/A (no multi-worker today)
  - Question: Standardize `worker_init_fn` to avoid RNG replay across workers for datasets using internal Generators?
  - Impact: Doc-only vs helper function; avoids subtle duplication with `num_workers>0`.
  - Source: `src/viterbo/datasets/core.py:22`, `:157`.

- Volume backend sequencing
  - Status: Deferred (recommended order: triangulation → Lawrence → Monte Carlo)
  - Question: Implement order: triangulation → Lawrence → Monte Carlo, or prioritize MC as a dev oracle first?
  - Impact: Test enablement path and early coverage.
  - Source: `docs/math/volume.md:1`, `src/viterbo/math/volume.py:79`.

- Coverage artifacts in CI
  - Status: Rejected for now (keep CI lean; reconsider weekly on main)
  - Question: Upload HTML/XML coverage as PR artifacts or keep local-only?
  - Impact: Transparency vs CI time/storage.
  - Source: `.github/workflows/ci.yml:1`, `Justfile:198`.

- Reviews in nav
  - Status: Done (keep index-only; reviews are transient)
  - Question: Keep only the Reviews index in nav (current policy) or add per-topic entries?
  - Impact: Discoverability for non-authors vs nav sprawl.
  - Source: `mkdocs.yml:21`, `docs/reviews/README.md:1`.


Quick Wins (Safe to Implement Without Decisions)
- Add numeric tolerance helper and use consistently
  - Status: Deferred (revisit with volume backends)
  - Why: Today `max(sqrt(eps), 1e-9)` is duplicated; central helper improves consistency.
  - Source: `src/viterbo/math/volume.py:68`, `src/viterbo/math/polytope.py:116`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:31`.

- Expand error messages to include actual shapes consistently
  - Status: Deferred
  - Why: Some modules already do; standardize across math/datasets.
  - Source: `src/viterbo/math/constructions.py:80`, `src/viterbo/datasets/core.py:157`.

- Add doctest-style examples for key Math APIs in docs
  - Status: Deferred
  - Why: Improves readability and builds confidence for new readers.
  - Source: `docs/math/polytope.md:1`, `docs/math/volume.md:1`.

- Mark at least one skill as Always‑On
  - Status: Rejected (keep Always‑On empty per policy)

- CI polish: add `--durations=10` to smoke tier
  - Status: Done
  - Why: Surfaces slowest tests for easy triage without heavy overhead.
  - Source: `pytest.ini:1`, `.github/workflows/ci.yml:1`, `Justfile:1`.

- C++ shim ergonomics
  - Status: Done (`VITERBO_CPP_VERBOSE` + `just ext-clean`)
  - Source: `src/viterbo/_cpp/__init__.py:24`, `Justfile:1`.

- Dataset docs note on multi‑worker seeding pattern
  - Status: N/A (no multi-worker)

- Tests: targeted additions
  - Status: Partial — J-invariance test added; oriented‑edge memo/cap smoke tests added (finite outputs)
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
  - Verified `.env` not tracked; keep pre-commit guard; rotate only if leaked externally.

- CI quality gates
  - Coverage floor in place; optional artifact uploads deferred; timeout wiring optional.

- Volume backends implementation sequence
  - Implement triangulation; enable inter‑algorithm agreement tests; follow with Lawrence (+ rational cert mode); add Monte Carlo estimator.

- C++ policy alignment and ergonomics
  - Ninja default decided; verbosity env and `ext-clean` added; local build dir optional.

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
