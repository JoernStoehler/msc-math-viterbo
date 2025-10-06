# Testing Assessment and Improvement Plan

## 1. Current smoke-tier status

### 1.1 Command outcomes and stability

- `make test` now succeeds with the default configuration thanks to automatic detection of the
  optional `pytest-testmon` plugin; when the plugin is absent the target degrades gracefully and
  finishes in ~38 s.【F:Makefile†L16-L149】【3ff607†L1-L18】
- Smoke coverage runs (`make coverage`) enforce an 80 % floor locally and in CI, producing HTML/XML
  artefacts without manual flags.【F:Makefile†L195-L200】【73e468†L121-L133】
- The project-wide session timeout has been raised to 120 s, eliminating the previous premature
  abort while keeping per-test timeouts at 10 s.【F:pytest.ini†L1-L4】
- Coverage calculations omit the educational `experiments/logreg_toy.py`, reflecting its
  documentation-only purpose while preserving the package for demos.【F:.coveragerc†L1-L3】
  【F:src/viterbo/experiments/logreg_toy.py†L1-L4】

### 1.2 Coverage snapshot (smoke tier)

| Module                         | Statements | Misses | Coverage |
| ------------------------------ | ---------- | ------ | -------- |
| Repository total               | 1,434      | 194    | 86%      |
| `geometry.polytopes.reference` | 290        | 74     | 74%      |
| `optimization.solvers`         | 284        | 50     | 82%      |
| `optimization.search`          | 51         | 5      | 90%      |
| `symplectic.systolic`          | 41         | 8      | 80%      |
| `experiments.__init__`         | 2          | 2      | 0%       |

The remaining low-coverage hotspots are concentrated in the reference polytope constructors,
linear-programming edge cases, and a small experiments stub.【45ead5†L6-L31】

### 1.3 Runtime hotspots inside the smoke marker set

- `tests/viterbo/geometry/volume/test_volume.py::test_hypercube_volume_matches_closed_form`
  dominates at ~3.1 s due to repeated reference/fast comparisons.【3ff607†L19-L37】
  【45ead5†L33-L52】
- `tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py::test_fast_matches_reference[simplex-2d-smoke]`
  costs ~2.3–2.6 s even after restricting smoke to a single 2D instance.【3ff607†L19-L37】
  【45ead5†L33-L52】
- `tests/viterbo/optimization/test_search.py::test_search_space_contains_catalog` still requires
  ~1.7–2.1 s because it materialises the full catalog for assertions.【3ff607†L19-L37】
  【45ead5†L33-L52】

With the random-polytope facet test promoted to the deep tier, the smoke suite completes within the
120 s session budget while retaining the highest-signal
checks.【F:tests/viterbo/geometry/polytopes/test_transforms.py†L80-L89】

## 2. Actions implemented this iteration

1. **Test harness reliability.** Added runtime detection for `pytest-testmon`, improved user
   feedback when the plugin is missing, and introduced coverage gating in both the Makefile and
   GitHub Actions workflow.【F:Makefile†L16-L200】【73e468†L121-L133】
2. **Timeout policy.** Increased the global session timeout to 120 s to accommodate growing smoke
   workloads while keeping short per-test limits for flake control.【F:pytest.ini†L1-L4】
3. **Coverage accounting.** Omitted the toy logistic-regression demo from coverage totals via
   `.coveragerc` and an explicit pragma, aligning metrics with production
   code.【F:.coveragerc†L1-L3】【F:src/viterbo/experiments/logreg_toy.py†L1-L4】
4. **Runtime tiering.** Moved the expensive random-polytope facet assertion to the deep tier and
   limited smoke EHZ comparisons to a single 2D instance, cutting smoke runtime without sacrificing
   representative signal.【F:tests/viterbo/geometry/polytopes/test_transforms.py†L80-L89】
   【F:tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py†L21-L47】
   【F:tests/viterbo/symplectic/capacity/facet_normals/test_fast.py†L21-L47】
5. **Solver safeguards.** Added `_normalize_bounds` regression tests covering scalar expansion,
   sequence validation, and NaN rejection to exercise previously untested
   branches.【F:tests/viterbo/optimization/test_solvers.py†L91-L111】
6. **Search API validation.** Extended the search tests to cover iteration limits, unexpected
   kwargs, and max-dimension handling, driving coverage through the remaining public
   pathways.【F:tests/viterbo/optimization/test_search.py†L46-L72】
   【F:src/viterbo/optimization/search.py†L18-L75】

## 3. Remaining opportunities

1. **Reference polytope coverage (H).** The heavy constructors in `geometry.polytopes.reference`
   remain at 74 %; focused fixtures exercising the uncovered branches (e.g., prism/cross products)
   would boost both coverage and regression fidelity.【45ead5†L6-L18】
   【F:src/viterbo/geometry/polytopes/reference.py†L149-L572】
2. **Solver failure diagnostics (M).** While bound-normalisation is now tested, infeasible HiGHS
   scenarios and status propagation still lack coverage; add synthetic LPs that trigger each status
   code.【45ead5†L24-L28】【F:src/viterbo/optimization/solvers.py†L127-L385】
3. **Search runtime budget (M).** Consider splitting catalog membership checks or caching fixtures
   so the remaining 1.7 s smoke hotspot does not grow with future catalog entries.【3ff607†L19-L37】
   【F:tests/viterbo/optimization/test_search.py†L33-L43】
4. **Higher-order symplectic invariants (L).** Property-based tests (e.g., Hypothesis rotations)
   would further stress `symplectic.systolic` once runtime headroom is secured.【45ead5†L29-L31】
   【F:tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py†L83-L129】

## 4. Target metrics and trade-offs

| Criterion               | Description                                                        | Target / Weight                                                      | Trade-off notes                                                                                     |
| ----------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Signal quality          | Probability a regression (logical or numerical) is caught.         | ≥0.9 detection probability for high-severity invariants; weight 0.4. | Promote high-signal tests to deeper tiers instead of weakening assertions.                          |
| Runtime budget          | Wall-clock time for smoke vs. deep tiers.                          | Smoke ≤ 45 s, Deep ≤ 15 min; weight 0.2.                             | Move tests between tiers as they approach limits; current smoke runtime ~38 s.                      |
| Determinism & stability | Flake-free results under fixed seeds/timeouts.                     | Zero tolerated flakes per 50 runs; weight 0.15.                      | Retain short per-test timeouts and explicit seeds to bound flake risk.                              |
| Coverage depth          | Line coverage for critical modules (`optimization`, `symplectic`). | ≥85 % per module, ≥80 % overall; weight 0.15.                        | Coverage gating now enforces the overall floor; module-level gaps remain in reference constructors. |
| Maintenance cost        | Ease of updating fixtures/baselines.                               | Reviewer effort ≤ 1 person-hour per change; weight 0.1.              | Prefer shared fixtures and documented baselines even if initial setup takes longer.                 |

These guardrails balance fast feedback with math-heavy signal, ensuring future work can iterate
quickly without sacrificing regression fidelity.
