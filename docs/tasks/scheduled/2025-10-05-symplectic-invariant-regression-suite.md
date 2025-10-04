# Task Brief — Symplectic invariants regression suite (T3)

- **Status**: Scheduled
- **Last updated**: 2025-10-06
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`,
  `docs/tasks/completed/2025-10-04-geometry-module-refactor.md`,
  `docs/tasks/scheduled/2025-10-04-testing-benchmark-harness.md`

## 1. Context and intent
The geometry restructure (T1) and harness work (T2) unlock richer property-based
checks, but the project still lacks a consolidated suite asserting the core
symplectic invariants (capacity bounds, monotonicity, scaling laws). This task
curates deterministic fixtures, codifies invariance tests spanning the new
module layout, and produces diagnostics that downstream experiments (E1–E5) can
trust before ingesting larger datasets.

## 2. Objectives and non-goals

### In scope
- Catalogue invariants covering symplectic capacities, Reeb orbit surrogates,
  and geometric transforms relevant to Viterbo's conjecture.
- Implement regression tests (unit + property-style) that exercise reference,
  optimised, and JAX variants via the harness markers.
- Capture baseline metrics (e.g., acceptable tolerance windows, runtime) and
  publish them next to the harness README for future comparisons.
- Document escalation paths when invariants fail (e.g., open task, bisect) so
  agents know how to respond.

### Out of scope
- Proving new invariants beyond existing theory; focus on codifying known ones.
- Rewriting algorithms discovered to be incorrect—log follow-up tasks instead.
- Exhaustive randomised fuzzing beyond a handful of deterministic seeded cases.

## 3. Deliverables and exit criteria
- New test modules under `tests/` (likely `tests/geometry/`) grouped by invariant
  family and registered with `smoke` / `ci` / `deep` markers as appropriate.
- Companion documentation (README or section inside the harness brief) listing
  covered invariants, tolerances, and how to interpret failures.
- Baseline artefacts (JSON/YAML) storing expected numeric ranges for quick diff
  during regression triage.

## 4. Dependencies and prerequisites
- Completion of T1 restructure (quantity-first modules available).
- Maintainer review complete; execute once T1/T2 prerequisites land.
- Availability of harness markers and fixtures from T2; collaborate if both run
  in parallel.
- Consensus on invariant list (can start from algorithm implementation plan and
  maintainer guidance).

## 5. Execution plan and checkpoints
1. **Invariant inventory**: gather authoritative statements from docs and
   existing tests; confirm with maintainer if ambiguity arises.
2. **Fixture curation**: select canonical polytopes (simplex, cube, stretched
   variants) and ensure deterministic reproduction.
3. **Test implementation**: add regression/property tests wired into pytest
   markers; capture tolerances and failure messaging.
4. **Baseline recording**: snapshot expected metrics (e.g., capacity ratios) and
   store them alongside the tests for diffing.
5. **Documentation pass**: update harness README/task briefs with invariant
   coverage and escalation steps.
6. **Validation**: run `make ci` plus deep marker suites to ensure stability and
   record runtimes for the portfolio.

## 6. Effort and resource estimates
- **Agent time**: Medium (≈ 1 agent-week)
- **Compute budget**: Low to Medium (property tests + deep markers)
- **Expert/PI involvement**: Medium (confirm invariant list, interpret issues)

## 7. Testing, benchmarks, and verification
- CI: ensure smoke/CI markers run within budgets; add targeted invariance tests
  to GitHub Actions once stable.
- Local: execute deep marker suite before landing substantial geometry changes.
- Manual: when invariants fail, document reproduction steps and open follow-up
  tasks per escalation guidance.

## 8. Risks, mitigations, and escalation triggers
- **Risk**: Some invariants depend on not-yet-implemented algorithms. **Mitigation**:
  mark with TODO + follow-up task and guard tests with fixtures checking availability.
- **Risk**: Tolerances too tight for floating-point drift. **Mitigation**: start
  with conservative bounds, review with maintainer, and tighten iteratively.
- **Escalation triggers**: Missing theoretical clarity on an invariant, inability
  to stabilise tests within runtime budgets, or detection of discrepancies across
  implementations.

## 9. Follow-on work
- Feed verified invariants into dataset generation (E1) and subsequent analyses
  (E2–E5).
- Future task to automate invariant reports (trend charts) if drift monitoring
  becomes recurring.

