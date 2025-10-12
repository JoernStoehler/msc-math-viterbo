status: complete
created: 2025-10-12
workflow: task
summary: End-to-end modernization roadmap aligning viterbo.modern with production readiness.
---

# Task Brief: Modernization Roadmap for `viterbo.modern`

## 1. Updated Context

- The modern modules now host the production Haim–Kislev facet solver, Chaidez–Hutchings oriented-edge graph, Minkowski billiards, and associated wrappers without falling back to the legacy stack.【F:src/viterbo/modern/capacity/facet_normals.py†L1-L371】【F:src/viterbo/modern/capacity/reeb_cycles.py†L1-L209】【F:src/viterbo/modern/capacity/minkowski_billiards.py†L1-L603】【F:src/viterbo/modern/spectrum.py†L1-L118】
- Regression tests anchor on explicit numeric baselines and deterministic adjacency metadata so they no longer depend on the deprecated ``viterbo.symplectic`` package.【F:tests/viterbo/modern/test_capacity_solvers.py†L1-L163】【F:tests/viterbo/modern/test_spectrum.py†L1-L114】
- The legacy ``src/viterbo/symplectic`` and ``tests/viterbo/symplectic`` trees have been removed; all downstream scripts and performance harnesses import exclusively from ``viterbo.modern``.【F:scripts/profile_ehz.py†L1-L109】【F:tests/performance/viterbo/modern/capacity/facet_normals/test_ehz_capacity_benchmarks.py†L1-L71】【F:src/viterbo/__init__.py†L1-L63】

## 2. Modernization Goals

1. Deliver production-ready capacity, cycle, and spectrum solvers within `viterbo.modern` that match or exceed the legacy reference accuracy envelope.
2. Align documentation, tolerances, and regression tests with the upgraded algorithms so downstream users can rely solely on the modern namespace.
3. Retire or quarantine legacy symplectic modules after parity validation without blocking future feature work.

## 3. Gap Analysis (Modern vs Legacy)

| Surface | Modern State | Legacy Capability | Remediation Signal |
| --- | --- | --- | --- |
| Capacity (EHZ) | Reference and fast solvers share the production tolerance envelope and expose diagnostics for degenerate graphs. | Removed. | Monitor tolerance regressions via deterministic tests and performance benches. |
| Reeb cycles | Chaidez–Hutchings graph builder ported with adjacency diagnostics; limited to 4D by design. | Removed. | Extend to ≥6D in future feature work once combinatorics are tractable. |
| Spectrum | Deterministic enumeration atop modern graph with fixed fixtures; ≥6D remains out-of-scope. | Removed. | Track future feature request for higher-dimensional spectra. |
| Tests/Docs | Baselines rely on numeric fixtures and graph metadata, modernization brief in sync with execution. | Removed. | Continue documenting policy updates and perf deltas alongside code changes. |

## 4. Phased Plan

### Phase A — Source-of-truth reconciliation

1. Catalogue every public function/class in modern vs legacy namespaces; produce a crosswalk table capturing parity gaps and consumers (tests/notebooks/scripts).
2. Lock baseline fixtures (polytope samples, expected capacities, cycles) shared between legacy and modern suites to ease regression tracking.
3. Track modernization progress directly in this brief so downstream contributors stay aligned.

### Phase B — Numeric alignment *(status: complete)*

1. Adopt shared tolerance constants for modern geometry/capacity entrypoints that mirror the legacy defaults.【F:src/viterbo/modern/numerics.py†L5-L22】【F:src/viterbo/geometry/polytopes/reference.py†L70-L95】
2. Update combinatorics helpers to consume the shared tolerances and disable caching while parity work is underway.【F:src/viterbo/modern/capacity/reeb_cycles.py†L72-L187】
3. Document the tolerance policy alongside modernization progress in this brief.

### Phase C — Capacity solver modernization *(status: complete)*

1. Port the Haim–Kislev subset solver into `viterbo.modern.capacity`, preserving dynamic-programming fallback for large permutations and hooking into the modern `Polytope` dataclass.【F:src/viterbo/modern/capacity/facet_normals.py†L198-L295】
2. Implement cycle-consistency validation analogous to `reeb_cycles.reference.compute_ehz_capacity_reference`, emitting actionable diagnostics when the oriented-edge graph is degenerate.【F:src/viterbo/modern/capacity/reeb_cycles.py†L206-L244】
3. Replace heuristic-based tests with fixtures comparing modern results to legacy reference outputs over curated polytopes.

### Phase D — Reeb cycle extraction *(status: complete)*

1. Port the oriented-edge graph builder into the modern namespace without the NetworkX dependency while preserving Chaidez–Hutchings constraints and tolerance handling.【F:src/viterbo/modern/capacity/reeb_cycles.py†L72-L187】
2. Extend `OrientedEdgeGraph` metadata so spectrum builders can remain deterministic across runs.【F:src/viterbo/modern/capacity/reeb_cycles.py†L1-L160】
3. Compare modern edge sets, adjacency, and degeneracy failures with the legacy graph on shared fixtures.

### Phase E — Spectrum rebuild *(status: complete)*

1. Refactor spectrum routines to consume the shared oriented-edge graph with deterministic cycle enumeration.【F:src/viterbo/modern/spectrum.py†L1-L166】
2. Rewrite tests to compare action spectra between modern output and legacy cycle data; document that ≥6D support remains a future feature.

### Phase F — Documentation and legacy sunset *(status: complete)*

1. Updated public-facing docs with the modernization summary, tolerance policy, and the retirement of
   :mod:`viterbo.symplectic`.
2. Notebooks remain placeholders but now describe the intended atlas workflow without shadowing the
   modern modules.
3. Legacy symplectic modules removed; release notes highlight the migration to :mod:`viterbo.modern`.

## 5. Risks & Open Questions

- **Combinatorial explosion**: The subset solver and oriented-edge enumeration can explode beyond 4D; define acceptable limits and potential approximate fallbacks before shipping.
- **CI budget**: Additional solver tests/benchmarks may extend runtimes; coordinate with maintainers on tiering (smoke vs deep vs longhaul).

## 6. Immediate Next Actions

1. [important] Track ≥6D feature planning and revisit runtime budgets before promoting new APIs.
2. [recommended] Keep modernization docs synced with future feature releases (≥6D, atlas pipeline).

## 8. ≥6D Backlog and release follow-up

- **Scope.** Extending capacity, cycle, and spectrum solvers beyond 4D demands new combinatorial
  pruning strategies for the subset solver and oriented-edge enumeration. The current algorithms
  scale factorially; we need to investigate symmetry reductions and heuristic pruning before
  offering ≥6D support.
- **Prerequisites.** Establish tighter memory and runtime envelopes in CI (estimate ≥10× current
  smoke runtime) and budget for deterministic baselines covering representative 6D polytopes. The
  atlas pipeline should emit metadata capturing symmetry classes to aid pruning.
- **Release messaging.** Public docs emphasise that modernization is complete while ≥6D work remains
  planned but unscheduled. Downstream teams should treat ≥6D support as follow-on scope and budget
  experimental runs accordingly.
- **Tracking.** Maintain these items in the modernization brief until a dedicated feature roadmap is
  spun up; update MkDocs pages and weekly mail when we schedule the ≥6D implementation.

## 7. Execution Log (2025-10-12)

- [x] C1 Ported Haim–Kislev facet solver (reference + dynamic-programming fast path) into `viterbo.modern.capacity` and rewired dispatchers.
- [x] D1-D3 Implemented oriented-edge graph builder without NetworkX, validated through new Reeb-cycle wrappers, and exposed graph metadata for reuse.
- [x] E1 Rebuilt the 4D spectrum enumeration on top of the modern oriented-edge graph with deterministic cycle ordering.
- [x] B1-B3 Landed shared tolerance constants, updated helpers, and documented the policy.
- [x] C3 Added parity assertions against legacy facet/reeb/minkowski solvers in the modern tests.
- [x] C2 Integrated oriented-edge diagnostics and surfaced failure details in modern wrappers.
- [x] E2 Locked spectrum fixtures against deterministic action sequences and recorded ≥6D as future work.
- [x] Legacy symplectic modules and tests removed; profiling/benchmark harnesses now target `viterbo.modern`.
- [x] F1 Cleared remaining ``viterbo.symplectic`` references in docs and tooling; pointed profiling helpers at modern kernels.
- [x] F-phase tasks: finalized docs, announced legacy removal, and captured higher-dimensional feature backlog.
