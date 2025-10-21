Status: Implemented (scope: 4D oriented-edge solver; caveats: certified C*(X) budgets remain CPU float64-only)

# Oriented-Edge CH Algorithm (R^4)

Audience and prerequisites
- Assumes familiarity with: convex polytopes (H- and V-representations), basic symplectic linear algebra in R^4 with complex structure J, and Chaidez–Hutchings (CH) 4D polytope Reeb dynamics (arXiv:2008.10111).
- Recommended reading before this page: CH Sec. 2 (algorithmic overview), Lemma 5.13 (speed/rotation bound), and Theorem 1.12(v) (per-face budget). Our docs/math/capacity_ehz.md provides project-wide EHZ context.

Goal
- Compute the minimal symplectic action among Type‑1 combinatorial Reeb cycles on a 4D convex polytope X ⊂ R^4 with 0 ∈ int(X), using the oriented-edge method described by CH. We support rotation-number pruning and optional CH per-face budgets.

Inputs
- Vertices `V ∈ R^{M×4}` (row-major); H-rep as facet normals and offsets `(n_i, λ_i)` with `X = {x : ⟨n_i, x⟩ ≤ λ_i}` and all λ_i > 0 (origin strictly inside).
- Numerical dtype: float64 recommended. Solver is Torch‑first (preserves input device);
  the certified budget builder C*(X) runs on CPU float64 for determinism.

High-level outline
- Enumerate 2-faces F ⊂ ∂X from vertex–facet incidence.
- For each facet i, compute Reeb vector `R_i = (2/λ_i) J n_i`.
- Build oriented edges between adjacent 2-faces sharing a facet; each edge carries:
  - target facet index j (the other facet bounding the destination face),
  - affine transfer `(u ↦ A u + b)` in a 2D local frame,
  - linear action coefficients `a(u) = a0 + a1·u` for time-of-flight to the target facet,
  - competition set (other facets) to enforce domain admissibility.
- Depth-first search over cycles with:
  - rotation-number pruning via polar decomposition (cap R in radians),
  - optional CH per-face budgets: accumulate `c_F = C*(X)·θ(F)` where `θ(F)=arccos⟨ν_i, ν_j⟩`.
- Evaluate candidate cycles by solving the fixed-point equation for the 2D affine return map and summing per-edge actions (reject negative/degenerate actions and domain violations).

Detailed steps
1) Face enumeration.
   - For each pair of facets (i, j), collect vertices incident to both; accept as a 2-face if their centered point cloud has rank 2. Orthonormalize a local 2D basis via QR for each accepted 2-face F.
2) Reeb vectors per facet.
   - With J the standard complex structure, set `R_i = (2/λ_i) J n_i` (λ_i > 0 required by 0 ∈ int(X)).
3) Oriented edges.
   - For each facet i and ordered pair of distinct 2-faces (F_from, F_to) adjacent to i, identify the other facet j bounding F_to. Compute:
     - `α_target = ⟨n_j, R_i⟩`. Require `α_target > 0` for outward crossing.
     - Domain check: along samples in F_from, ensure the first positive hitting time is to facet j; reject otherwise.
     - Action coefficients: for point `x = x0 + B u` in F_from (origin x0, basis B), the time to hit j is `t(u) = (λ_j − ⟨n_j, x0⟩ − ⟨n_j, B u⟩)/α_target = a0 + a1·u`.
     - Affine transfer: map `u ↦ A u + b` to coordinates on F_to using the reparametrization with `R_i` and `t(u)`.
4) DFS with pruning.
   - Maintain current face, transfer matrix, and accumulated CH budget.
   - Rotation-number pruning: compute polar rotation angle of 2×2 transfer; prune if exceeds cap R.
   - CH budgets (optional): precompute `c_F` for every 2-face using `C*(X)` (see below), accumulate along visited faces, prune if sum exceeds R/(2π) in rotation-number units.
   - Optionally memoize by quantizing transfer and tracking best coarse budget used.
5) Cycle evaluation.
   - For each closed edge list, solve `(I − A_k … A_1) u = b_tot` in least squares (small regularization) and check residual consistency in world coordinates. Reject non-admissible or negative-action cycles.
   - Minimal admissible action across discovered cycles is the result.

Certified per-face budgets
- The certified builder `compute_cF_constant_certified` constructs `C*(X)` per docs/math/cstar_constant_spec.md, computing D_min/U_max and an adaptive Lipschitz-certified lower bound for `N_ann(X)`.
- In code, `oriented_edge_spectrum_4d(..., use_cF_budgets=True, cF_constant=None)` triggers the builder and uses `c_F(F)=C*(X)·θ(F)`.
- `C*(X)` is device-stable and deterministic; by construction it can only be smaller than the sharp speed constant and thus cannot over-prune. For symmetric cases we typically observe `C*(X) ≈ 1/(4π)`; the cube case achieves `1/(2π)` per the spec example when `D_min=U_max=N_ann=1`.

Differences to Chaidez–Hutchings
- We only implement optimizations explicitly documented in the paper: rotation-number bounds and per-face budgets. Undiscussed additional optimizations from their code are not replicated.
- Our `C*(X)` uses rigorous denominator bounds (pairwise closed forms + safe relaxations) and a conservative lower bound for the annest term. This is certified but may be less sharp than a full Lipschitz-certified search; we plan to implement that to improve pruning.
- We run deterministically on CPU with float64; CH reported a bespoke C implementation with additional optimizations.

API and defaults
- Function: `oriented_edge_spectrum_4d(vertices, normals, offsets, *, rotation_cap=4π, use_cF_budgets=False, cF_constant=None, use_memo=None, memo_grid=1e−6, memo_buckets=32)`.
- Defaults: rotation guard on at 4π; budgets off; memo off unless `use_memo=True` (requires budgets).
- Enabling budgets without `cF_constant` computes a certified `C*(X)` from the H-rep and 2-face bases.

Warning about heuristics
- Dominance memoisation relies on a monotone additive measure; we use the CH per-face budgets for that. Therefore, memo requires `use_cF_budgets=True`. With `use_memo=True` and budgets on, memo can greatly reduce runtime but is not guaranteed complete. For certified results, set `use_memo=None/False` (and keep rotation/budgets as desired).

Complexity, invariants, and checks
- Complexity: worst-case exponential in cycle length; pruning is essential. Rotation cap and per-face budgets dramatically reduce the explored search in typical cases.
- Invariants: scaling, translation, and rotation invariance as per the spec (unit normals, interior origin normalization).
- Guards: numerical tolerances scale with dtype; all matrix ops use stable decompositions; failures surface as ValueError with descriptive messages.

Instrumentation and performance
- The solver is wrapped with a per-call wall-clock budget via `VITERBO_SOLVER_TIMEOUT` (default 30s). For more exhaustive runs, increase this env var. CPU time is also capped to ~180s by our test harness.
- With budgets enabled, small and symmetric test polytopes (e.g., hypercube) complete quickly (≪ 1s). Harder cases (e.g., pentagon product) benefit significantly from enabling memo heuristics (`verified=False`), often finishing well under 30s with budgets active.

Roadmap
- Implement the Lipschitz-certified search for `N_ann(X)` per the spec, which will yield larger certified `C*(X)` and stronger pruning.
- Add optional incremental k_max and face-frequency guards (paper-compatible) as tunables for hard instances.
- Consider multi-start DFS ordering heuristics to prioritize likely minimal cycles.

References
- Chaidez, J.; Hutchings, M. Computing Reeb dynamics on 4d convex polytopes. arXiv:2008.10111.
