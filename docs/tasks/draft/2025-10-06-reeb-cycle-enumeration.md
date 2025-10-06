# Chaidez–Hutchings Reeb-cycle solver implementation

- **Status**: Draft
- **Last updated**: 2025-10-06
- **Owner / DRI**: Codex agent
- **Reviewers**: Unassigned
- **Related docs**: [Reeb orbit cross-check (E2)](2025-10-04-reeb-cross-check.md)

## 1. Context and intent

The current `reeb_cycles` package builds the oriented-edge graph for 4D polytopes but both the
`reference` and `fast` entry points simply validate connectivity before deferring to the existing
facet-normal solvers.【F:src/viterbo/symplectic/capacity/reeb_cycles/reference.py†L9-L24】
【F:src/viterbo/symplectic/capacity/reeb_cycles/fast.py†L9-L27】 This leaves us unable to
cross-check Chaidez–Hutchings combinatorial cycles against facet normals or to evaluate the
performance gap between a faithful reference algorithm and an optimized version.

Implementing the real Chaidez–Hutchings pipeline requires reconstructing the admissible cycle
enumeration (DFS/Johnson), computing the combinatorial action with exact rational arithmetic, and
enforcing the admissibility filters tied to the polytope facets. The blocker is the absence of a
precise, codified description of the action functional and transition matrices derived from the
paper; the existing code only exposes graph structure without the cycle
weights.【F:src/viterbo/symplectic/capacity/reeb_cycles/graph.py†L13-L118】

## 2. Objectives and non-goals

### In scope

- Implement a readable reference solver that enumerates admissible cycles using NetworkX
  DFS/Johnson’s algorithm and evaluates Chaidez–Hutchings action exactly.
- Implement the fast solver with bitset transitions, pruning heuristics, and Johnson’s algorithm for
  cycle enumeration.
- Validate both solvers against facet-normal capacities on Chaidez–Hutchings benchmark polytopes.
- Document algorithm invariants and numerical assumptions so future maintainers can audit the
  pipeline.

### Out of scope

- Extending the oriented-edge framework beyond 4D polytopes.
- Replacing the existing facet-normal solvers or changing their APIs.
- Shipping GPU/JIT acceleration beyond the requested bitset/Johnson optimizations.

## 3. Deliverables and exit criteria

- Production-quality reference and fast implementations in `reeb_cycles/reference.py` and
  `reeb_cycles/fast.py` that return capacities without delegating to facet-normal solvers.
- Unit and regression tests in `tests/viterbo/symplectic/capacity/reeb_cycles/` covering
  Chaidez–Hutchings fixtures plus parity checks against facet-normal outputs.
- Performance benchmarks quantifying the speed-up of the fast solver relative to the reference
  implementation on representative 4D polytopes.
- Narrative documentation (README or module docstrings) explaining admissibility rules, action
  evaluation, and pruning heuristics.

Exit criteria: both solvers pass the new tests, match facet-normal capacities within tolerance, and
benchmarks demonstrate the intended performance characteristics.

## 4. Dependencies and prerequisites

- Authoritative specification or reference implementation of the Chaidez–Hutchings combinatorial
  action and admissibility filters. Without this, the action weights and pruning rules remain
  ambiguous (current blocker).
- Confirmed dataset of Chaidez–Hutchings benchmark polytopes with facet data and expected
  capacities.
- Validation that the existing oriented-edge graph data structure captures all metadata required for
  action evaluation (e.g., affine transition matrices, orientation signs).

## 5. Execution plan and checkpoints

1. Acquire or derive the explicit formulas for action computation and admissibility, and validate
   them against a single benchmark polytope.
2. Implement the reference solver with exact rational arithmetic (e.g., `fractions.Fraction` or
   `sympy`) and DFS/Johnson enumeration; add focused unit tests.
3. Profile the reference solver on benchmark fixtures to identify performance bottlenecks.
4. Implement the fast solver with bitset encoding, Johnson’s algorithm, and pruning heuristics;
   cross-validate results with the reference solver.
5. Expand regression tests and benchmarks, documenting observed performance deltas and any numeric
   stability considerations.

## 6. Effort and resource estimates

- Agent time: High (deriving the formulas, implementing two solvers, and authoring
  tests/benchmarks).
- Compute budget: Medium (cycle enumeration can be expensive but tractable for benchmark-sized
  polytopes).
- Expert/PI involvement: High (required to provide or validate the Chaidez–Hutchings specification
  and benchmark dataset).

Reevaluate after Step 2; if the specification is still unclear, escalate with a
`Needs-Unblock: Chaidez–Hutchings spec` update.

## 7. Testing, benchmarks, and verification

- CI: `just precommit` (lint, typecheck, smoke tests) must pass with the new solvers.
- Unit tests: targeted suites under `tests/viterbo/symplectic/capacity/reeb_cycles/` covering cycle
  enumeration, action evaluation, and parity with facet-normal capacities.
- Benchmarks: extend `tests/performance/.../reeb_cycles/` to record runtime comparisons; capture
  artefacts via the existing benchmarking harness.

## 8. Risks, mitigations, and escalation triggers

- **Specification ambiguity (current blocker)**: without the explicit action formulas,
  implementations risk being incorrect. Mitigation: obtain PI guidance or external references before
  coding; escalate if unavailable.
- **Combinatorial explosion**: cycle enumeration may be infeasible for larger polytopes. Mitigation:
  introduce early pruning heuristics and document practical limits.
- **Numeric drift**: floating-point approximations may break admissibility checks. Mitigation: use
  rational arithmetic in the reference solver and only downcast in the fast solver with documented
  tolerances.

Escalate if action formulas cannot be validated on at least one benchmark or if runtime exceeds
benchmark budgets by >10× compared to facet-normal solvers.

## 9. Follow-on work

- Generalise the oriented-edge framework to higher dimensions once the 4D pipeline is trusted.
- Explore hybrid solvers that blend facet-normal and Reeb-cycle information for better performance.
- Publish a methodological note summarising lessons from implementing combinatorial Reeb-cycle
  capacities.
