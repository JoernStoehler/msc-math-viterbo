status: adopted
version: 0.1
owners: [Project Owner Jörn]
start: 2025-10-01
end: 2026-03-31
last-updated: 2025-10-20
---

# Project Charter — MSc “Viterbo” (PyTorch + C++)

Purpose & Scope
- Purpose: Build a trustworthy, reproducible, and efficient toolkit to compute and study the systolic ratio for 4D convex polytopes at scale, and produce a high‑quality MSc thesis by the end of March (archival hand‑in).
- In scope: EHZ/systolic computations for convex polytopes (general and Lagrangian products), dataset construction and analysis (classical DS/ML toolbox), rigorous algorithm cutoffs with proofs or citations, reproducible experiments.
- Out of scope (for this thesis): Publishing the library to PyPI; API stability for external dependents; inventing novel ML architectures (use existing components); CUDA perf work unless strictly necessary for scale targets.

Non‑negotiables
- End date: Thesis submitted and repo archived by end of March 2026.
- Linear progress: Writeups track work as it happens; lag ≤ 7 days. Drafts first, refinement later.
- Trustworthy results: Any algorithmic cutoff must be backed by a theorem (cited or our own proof) guaranteeing adequacy. Unit tests cover invariants and sanity checks (transformations, special cases, bounds, agreements across algorithms).
- Reproducibility: Experiments may go stale as code evolves, but must record the last commit hash for which they ran successfully.

Goals (What we will deliver)
- Efficient systolic‑ratio computation for polytopes up to a size constraint (distinct cutoffs for general vs. Lagrangian products), with rigorous justification of any pruning/budget parameters.
- A large dataset (> 1e5 polytopes, ~25 vertices typical), with computed invariants (volume, capacity, systolic ratio, minimal cycles when available) and metadata.
- A DS/ML analysis toolbox and results applied to the dataset (predictors, clustering, correlations), using existing ML techniques.
- A complete MSc thesis that documents background, methods, results, and limitations, with reproducibility instructions.

Success Metrics (measurable)
- Onboarding SLO (agents, provisioned environment):
  - One‑shot orientation from AGENTS.md + minimal reads; single iteration before first code change.
  - Median time to first green `just checks`: ≤ 10 minutes on a fresh daily worktree (provisioning ~10s + reading ~7s + coding/tests). Stretch: ≤ 7 minutes for small tasks.
- CI SLO (GitHub runners): Main CI (lint+type+smoke+docs) p95 ≤ 7 minutes.
- Coverage floors:
  - Math layer (`src/viterbo/math/**`): ≥ 95% lines (target nearly 100%).
  - Datasets (`src/viterbo/datasets/**`): ≥ 60% lines.
  - Models/experiments/notebooks: not required (use reproducible runs instead).
- Type‑checking:
  - Pyright basic: zero errors repo‑wide.
  - Pyright strict: optional; enable selectively for math modules where it adds value. No hard gate on strict.
- Docs freshness: MkDocs strict build (0 warnings) on main. API‑affecting changes update docs within ≤ 48 hours.
- Reproducibility: Smoke tier flake rate 0 over rolling 20 CI runs.
- Dataset scale & throughput:
  - “Atlas Tiny” immediate: demonstrable tomorrow (see Milestones) with benchmarks and computed properties.
  - Main dataset build (>1e5 polytopes): acceptable if it requires a one‑time batch run ≤ 24 core‑hours; status and commit hash recorded.

 Milestones & Timeline
- Project start: 2025-10-01
- Project end (hard): 2026-03-31
- 2025-10-21 — Atlas Tiny demo (tomorrow): dataset preview with benchmarks and computed properties (all currently implemented algorithms; literature values where available).
- TBD — Detailed month-by-month milestones to be authored separately.

Acceptance Criteria — 2025-10-21 Atlas Tiny demo
 - Tests: `tests/datasets/test_atlas_tiny.py` all green; optionally include a micro‑benchmark assertion for per‑row compute if helpful.
 - Tests: `tests/datasets/test_atlas_tiny.py` all green; add a micro‑benchmark assertion for per‑row compute if helpful.
 - Docs: a short page under `docs/` describing schema, invariants computed, and known references.
 - Artefact: example batch with computed invariants; recorded commit hash.
 

Quality Bar & Policies
- Layering: `math` (pure, Torch‑first, no I/O/state) ← `datasets` (adapters, ragged collate) ← `models` (experiments). C++ with Python fallbacks.
- Numerics: float64 in math; dtype/device preserved; no implicit device moves; tolerances scale ~sqrt(eps).
- Correctness: Theorem‑backed cutoffs; unit tests check transformation behavior, special cases, bounds, and inter‑algorithm agreement.
- Experiments: Must record commit hash; allowed to be stale; keep scripts/notebooks deterministic where feasible.

Testing & CI
- Tiers: smoke (default), deep (opt‑in), longhaul (scheduled). Benchmarks separated; skipped if plugin missing.
- Gates: CI requires lint (Ruff), type (Pyright basic), smoke tests, docs strict build. Coverage floors enforced for math/datasets.
- Performance: No global per‑test hard caps; derive budgets from dataset scale goals. One‑time dataset build allowance ≤ 24 core‑hours.

Governance & Roles
- Project Owner: sets scope, approves milestone content, and arbitrates changes to success metrics.
- Maintainers: own code quality and CI health; review PRs; ensure documentation alignment.
- Agents: follow skills and task scopes; keep changes minimal; surface blockers via “Needs‑Unblock: …”.

Risk Register (top)
- Solver scalability (oriented‑edge) on hard polytopes → Mitigation: certified budgets, memo heuristics (flagged), and dataset constraints.
- Exact backends (e.g., volume via Lawrence) slip → Mitigation: document stubs; keep 2D/4D parity strong; plan relaxations.
- Devcontainer drift/perf → Mitigation: owner runbook; cache co‑location notes; OK to tolerate minor uv copy penalties.

Communication & Reporting
- Weekly: brief status (progress, risks, next steps) and updated thesis draft sections as they evolve.
- Monthly: advisor‑facing consolidated report (slides or doc) with current findings.
- Artefacts: keep under `artefacts/` (git‑ignored); reference commit hashes and environment details.

Change Control
- Propose metric/milestone changes via short PR edits to this charter; include rationale and impact; owner approval required.

Revision History
- 2025-10-20 — v0.1 (draft): Initial charter created from owner inputs and repository design principles.
