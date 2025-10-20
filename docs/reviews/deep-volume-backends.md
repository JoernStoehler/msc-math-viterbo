# Review — Volume Backends Readiness Deep‑Dive

Provenance
- Topic: Volume Backends Readiness Deep‑Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc
- Timebox: ~90 minutes
- Scope: docs/math/volume.md, src/viterbo/math/volume.py, src/viterbo/math/polytope.py, tests/math/*volume* and related briefs; mkdocs build
- Version: v0.1

Context
- Goal: Unfiltered, unsorted findings on volume backends with emphasis on exact triangulation, Lawrence sign‑decomposition, Monte Carlo estimator specs/roles; stubs and acceptance; inter‑algorithm agreement tests; numerics/tolerance policies.
- Sources used: skills in AGENTS.md map (math‑layer, math‑theory, testing‑and‑ci), Math API docs, code under `src/viterbo/math`, tests under `tests/math`.

Inputs & Method
- Commands run (representative) and outcomes:
  - `uv run python scripts/load_skills_metadata.py` → surfaced skill summaries (17 entries) for quick routing (skills index updated in AGENTS.md scope).
  - `rg -n "volume|Lawrence|triangul|Monte|polytope|simplex" -S` → located docs and code for volume, triangulation/Lawrence mentions, and test stubs.
  - `sed -n '1,200p' docs/math/volume.md` → confirmed implemented helper and planned backends; stubs explicitly listed (docs/math/volume.md:1).
  - `sed -n '1,200p' src/viterbo/math/volume.py` → verified current backend and stubs; tol policy; device/dtype handling (src/viterbo/math/volume.py:56).
  - `uv run mkdocs build --strict` (pre/post‑write) → success; site built; INFO notes list pages not in nav; strict build passed locally (build ~1.5–2.0s).
  - `git rev-parse --short HEAD` → 87fdebc.

Findings (unsorted)
- Implemented baseline matches docs
  - The shipped backend is a hybrid: D=1 interval length; D=2 shoelace on convex‑hull order; D≥3 facet‑height accumulation from centroid with recursive facet measure (src/viterbo/math/volume.py:56). This aligns with the Math page description (docs/math/volume.md:1).
  - H‑representation conversion relies on SciPy `ConvexHull` and returns normalized outward normals; device/dtype restored on exit (src/viterbo/math/polytope.py:121).

- Triangulation backend — target shape and acceptance
  - Stub present: `volume_via_triangulation(vertices)` with intended “oriented‑hull triangulation and signed simplex accumulation” (docs/math/volume.md:24, src/viterbo/math/volume.py:93).
  - Acceptance (tests documented): exact match against determinant/simplex formula on a rational‑coordinate simplex (tests/math/test_polytope.py:78).
  - Suggested approach: beneath–beyond or incremental hull triangulation; accumulate signed simplex volumes w.r.t. a fixed reference point; validate orientation consistency via lexicographic ordering and SVD‑based rank checks similar to `_facet_measure`.
  - Degeneracy handling: apply `rank` gates and `torch.isclose` masks using repo‑wide tol policy; de‑duplicate sliver simplices by hull ridge adjacency.
  - Complexity expectations: practical for D≤5 and few hundred vertices; reuses H/V conversions for robustness.

- Lawrence sign‑decomposition — target shape and acceptance
  - Stub present: `volume_via_lawrence(normals, offsets, *, basis=None)` (docs/math/volume.md:26, src/viterbo/math/volume.py:98).
  - Acceptance (tests documented): certify unit cube exactly from its six halfspaces; expose sign contributions/certificates for audit (tests/math/test_polytope.py:87).
  - Algorithm notes: per‑vertex tangent cone decomposition; integrate signed contributions using active sets `A_v x = b_v`; robust to facet permutations; canonicalize facet bases to fix signs.
  - Exactness path: offer two modes — float (fast) and rational certificate (slow) using integer/rational arithmetic on `A_v` minors to reproduce Büeler–Enge–Fukuda‑style results; return certificate payload for tests.
  - Degeneracy: detect non‑simple vertices via rank/tie checks; either symbolic perturb (lexicographic) or block‑pivot rules; surface “non‑simple” as structured error with remediation hints.

- Monte Carlo estimator — target shape, role, acceptance
  - Stub present: `volume_via_monte_carlo(vertices, normals, offsets, *, samples, generator)` (docs/math/volume.md:29, src/viterbo/math/volume.py:105).
  - Target role: fallback for moderate/high D (≥6) where exact backends are costly; also a cross‑check oracle for random shapes during development.
  - Acceptance (tests documented): quasi‑MC convergence rate — empirical error halves as `samples` doubles on 6D cross‑polytope with known volume; deterministic seeds (tests/math/test_polytope.py:102).
  - Implementation sketch: Sobol engine (`torch.quasirandom.SobolEngine`) over bounding box; rejection via H‑rep residuals; control variates using inscribed/circumscribed ellipsoids optional; stream accumulation to avoid memory spikes.
  - Reporting: return `(estimate, stderr)` or expose variance via auxiliary API; plumb `generator` for reproducibility; document time‑budget sensitivity.

- Inter‑algorithm agreement — tests posture
  - Present day: skip‑documented tests outline the agreement matrix (triangulation ↔ facet‑height on simplices/zonotopes; Lawrence ↔ triangulation on cubes/zonotopes; MC ↔ exact within tolerance) (tests/math/test_polytope.py:62, tests/math/test_polytope.py:78, tests/math/test_polytope.py:87, tests/math/test_polytope.py:102).
  - Immediate wins: when triangulation lands, enable cross‑checks for `D=3,4` on small cases drawn from `tests/polytopes.py:STANDARD_POLYTOPES` (tests/polytopes.py:1).
  - Agreement policy: use absolute tolerances only (no rtol) for exact vs float comparisons when inputs are rational/small integers to detect orientation slips; reserve mixed `atol/rtol` only for MC vs exact.

- Numerics and tolerance policy — current conventions
  - Tolerances commonly set to `max(sqrt(eps), 1e-9)` in working dtype (src/viterbo/math/volume.py:68, src/viterbo/math/polytope.py:116, src/viterbo/math/capacity_ehz/stubs.py:189, src/viterbo/math/capacity_ehz/stubs.py:320, tests use `rtol=0.0` in many asserts).
  - Facet selection masks use `torch.isclose(..., atol=tol, rtol=0.0)` to avoid scale‑dependent rtol drift (src/viterbo/math/volume.py:82).
  - Halfspace H/V routines lift to CPU float64 for SciPy then restore dtype/device; callers should not interleave GPU‑only sections around them (src/viterbo/math/polytope.py:141).
  - Recommendation: factor a small util `tolerance(dtype) -> tol` and reuse across modules; keep `rtol=0.0` default for geometry predicates; document in Math API notes.

- Device/dtype discipline
  - All helpers are Torch‑first and preserve dtype/device on return; volume dispatch and H/V conversions honour input dtype; tests pin default dtype to float64 and check float32 path explicitly (tests/math/test_volume_smoke.py:33).
  - SciPy dependency implies a CPU detour; no implicit CUDA synchronisation issues observed in code paths inspected.

- API ergonomics and dispatch
  - Today only `volume(vertices)` is public and chooses the facet/height method for D≥3 (src/viterbo/math/volume.py:56).
  - Consider optional dispatch `volume(..., method="pyramids|triangulation|lawrence|monte_carlo")` once backends exist, keeping dtype/device guarantees; a prior review suggested this pattern (docs/reviews/04-code.md:163).

- Test catalogue coverage (relevant to volume)
  - Canonical shapes with references span 1D segments, planar polygons (area known), 3D tetrahedron/cube, 4D simplex/hypercube, Lagrangian products and randomised 4D polytopes; each carries `volume_reference` when analytic (tests/polytopes.py:25, tests/polytopes.py:194, tests/polytopes.py:214, tests/polytopes.py:237).
  - These provide a solid acceptance bed for exact backends and calibration curves for MC.

- Related briefs and references
  - The workflow brief summarises algorithms and references (Lawrence; beneath–beyond; Barvinok; random walks; Lasserre; specialised 4D) and gives reproducible cues (docs/briefs/2025-10-12-workflow-volume-algorithms.md:1).
  - Math API page lists intended backends and caveats (docs/math/volume.md:1).

- Risks and mitigations
  - Degenerate/near‑degenerate combinatorics (non‑simple vertices, coplanar slivers) — mitigate with rank tests, symbolic perturbation, or exact paths in Lawrence.
  - Scale sensitivity — centre/scale vertices before hull/triangulation; retain original coords for final volumes; continue using `sqrt(eps)` tol.
  - Performance — cache H/V conversions; stream MC; limit triangulations to small/moderate cases; surface time budgets via `viterbo.runtime` if needed.

- Execution log (for provenance)
  - `uv run python scripts/load_skills_metadata.py` → OK; listed skills.
  - `uv run mkdocs build --strict` (pre‑write) → OK; INFO listed non‑nav pages; build successful in ~1.6s.
  - Wrote `docs/reviews/deep-volume-backends.md`.
  - `uv run mkdocs build --strict` (post‑write) → OK; build successful; no errors introduced; file remains unlisted in nav per guardrails.

References (files)
- docs/math/volume.md:1 — Volume helpers and planned backends
- src/viterbo/math/volume.py:56 — Implemented backend and stubs
- src/viterbo/math/polytope.py:121 — H/V conversions (SciPy ConvexHull)
- tests/math/test_polytope.py:78 — Triangulation acceptance (skip)
- tests/math/test_polytope.py:87 — Lawrence acceptance (skip)
- tests/math/test_polytope.py:102 — Monte Carlo acceptance (skip)
- docs/briefs/2025-10-12-workflow-volume-algorithms.md:1 — Algorithm survey and cues

Next Steps (non‑binding)
- Implement triangulation backend with beneath–beyond; enable agreement tests for D=3,4.
- Prototype Lawrence float path, then add rational certificate mode; wire up cube certification test.
- Add Monte Carlo estimator with Sobol sampling and error bars; enforce convergence test with seeded generator.
- Extract `tolerance(dtype)` utility; document numerics policy once centralised.
